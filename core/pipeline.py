# -----------------------------------------------------------------------------
# Orquesta el flujo principal del análisis de video:
#   1) Abrir lector (VideoCapture) del video de entrada
#   2) Abrir escritor (VideoWriter) del video anotado de salida
#   3) Por cada frame: detección YOLO -> tracking -> reglas -> overlays -> write
#   4) Las reglas que detectan infracciones llaman a EventLogger.log(), el cual
#      guarda una captura del frame completo y un recorte (crop) del objeto.
#   5) Al final, se devuelve la ruta final del video anotado y los eventos leídos
#      desde el CSV generado.
# -----------------------------------------------------------------------------

import contextlib
import yaml, os, shutil
import time
import pandas as pd

from core.utils.video_io import open_video_reader, open_video_writer, release_safely
from core.utils.events import EventLogger
from core.utils.drawing import draw_box, draw_line, draw_hud
from core.detectors.yolo_detector import YoloDetector
from core.detectors.helmet_detector import HelmetDetector
from core.detectors.lane_detector import SimpleLaneDetector
from core.trackers.deepsort_wrapper import DeepSortWrapper
from core.rules.helmet import HelmetRule
from core.rules.speed import SpeedRule
from core.rules.lane_invasion import LaneInvasionRule
from core.utils.model_io import ensure_local_model

def _load_config(path):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    if "include" in cfg:
        with open(cfg["include"], 'r') as f:
            base = yaml.safe_load(f)
        base.update({k:v for k,v in cfg.items() if k!="include"})
        cfg = base
    return cfg

# Borrar recursos previos
def _clean_previous_outputs(output_dir: str, evidence_dir: str):
    csv_path = os.path.join(output_dir, "events.csv")
    with contextlib.suppress(Exception):
        if os.path.exists(csv_path):
            os.remove(csv_path)
    with contextlib.suppress(Exception):
        if os.path.exists(evidence_dir):
            shutil.rmtree(evidence_dir)
    os.makedirs(evidence_dir, exist_ok=True)

class Pipeline:
    def __init__(self, scene_config_path, yolo_imgsz=None, yolo_conf=None):
        self.cfg = _load_config(scene_config_path)
        # Overrides de la UI
        if yolo_imgsz: self.cfg["yolo"]["imgsz"] = yolo_imgsz
        if yolo_conf:  self.cfg["yolo"]["conf"]  = yolo_conf

        # Modelos
        self.detector = YoloDetector(
            model_path=self.cfg["models"]["yolo_path"],
            imgsz=self.cfg["yolo"]["imgsz"],
            conf=self.cfg["yolo"]["conf"]
        )
        # Helmet (opcional): resolver ruta del modelo de casco y descargar si falta.
        # Reglas: preferimos no bloquear; si no hay modelo, lo registramos claro.
        def _resolve_helmet_path(cfg_models) -> str | None:
            # 1) Config / Env
            path = cfg_models.get("helmet_path") or os.environ.get("HELMET_MODEL_PATH")
            if path and os.path.exists(path):
                return path
            # 2) Buscar cualquier .pt en models/helmet
            base_dir = os.path.join(os.getcwd(), "models", "helmet")
            with contextlib.suppress(Exception):
                if os.path.isdir(base_dir):
                    for name in os.listdir(base_dir):
                        if name.lower().endswith(".pt"):
                            return os.path.join(base_dir, name)
            # 3) Devolver lo que haya (aunque no exista) para permitir descarga
            return path

        h_url  = self.cfg["models"].get("helmet_url") or os.environ.get("HELMET_MODEL_URL")
        h_path = _resolve_helmet_path(self.cfg["models"])
        print(f"[Pipeline] Ruta modelo casco resuelta: {h_path or 'Ninguna'}")
        if h_path and (not os.path.exists(h_path)) and h_url:
            try:
                ensure_local_model(h_path, h_url)
            except Exception as e:
                print(f"[Pipeline] No se pudo descargar el modelo de casco: {e}")

        # Instancia detector de casco si el archivo existe finalmente
        if h_path and os.path.exists(h_path):
            try:
                print(f"[Pipeline] Cargando modelo de casco desde: {h_path}")
                hcfg = self.cfg.get("helmet", {})
                helmet_imgsz = hcfg.get("imgsz", self.cfg["yolo"]["imgsz"])  # permitir imgsz distinto para casco
                helmet_conf  = hcfg.get("conf", 0.30)
                print(f"[Pipeline] Modelo de casco: {h_path}")
                self.helmet_detector = HelmetDetector(
                    model_path=h_path,
                    imgsz=helmet_imgsz,
                    conf=helmet_conf,
                )
            except Exception as e:
                print(f"[Pipeline] Error cargando modelo de casco: {e}")
                self.helmet_detector = None
        else:
            print("[Pipeline] Modelo de casco no encontrado. Coloca un .pt en models/helmet o configura HELMET_MODEL_URL.")
            self.helmet_detector = None

        # Lanes (MVP sencillo; luego puedes integrar UFLD sin tocar el resto)
        self.lane_detector = SimpleLaneDetector()

        # Tracker
        self.tracker = DeepSortWrapper(max_age=15)

        # Reglas activas
        self.rules = []
        # Activa regla de casco solo si hay modelo listo (evita falsos positivos)
        if self.cfg["rules"].get("helmet") and self.helmet_detector is not None:
            print("[Pipeline] Regla de casco ACTIVADA")
            self.rules.append(HelmetRule(self.cfg))
        else:
            if self.cfg["rules"].get("helmet"):
                print("[Pipeline] Regla de casco DESACTIVADA (no hay modelo)")
        if self.cfg["rules"].get("speed"):         self.rules.append(SpeedRule(self.cfg))
        if self.cfg["rules"].get("lane_invasion"): self.rules.append(LaneInvasionRule(self.cfg))

        # Logger (se re-crea después de limpiar para reescribir encabezados)
        self.logger = EventLogger(self.cfg["video"]["output_dir"], self.cfg["video"]["evidence_dir"])

    def process_video(self, in_path, out_path, clean_previous=True):
        """
        Ejecuta el análisis del video y produce tres artefactos:
          - Video anotado (bounding boxes, HUD y líneas guía), escrito frame a
            frame por el VideoWriter. La ruta exacta puede ajustar la extensión
            según el códec disponible (ver core/utils/video_io.py).
          - CSV 'data/output/events.csv' con eventos detectados.
          - Evidencias en disco (capturas del frame y recortes del bbox) por
            cada infracción detectada, gestionadas por EventLogger.log().

        Parámetros:
          - in_path: ruta del video fuente
          - out_path: ruta deseada del video de salida (se puede ajustar .mp4/.webm/.avi)
          - clean_previous: si True, limpia CSV y evidencias antes de empezar
        """
        if clean_previous:
            _clean_previous_outputs(self.cfg["video"]["output_dir"], self.cfg["video"]["evidence_dir"])
            self.logger = EventLogger(self.cfg["video"]["output_dir"], self.cfg["video"]["evidence_dir"])


        # Marca de tiempo inicial para medir duración del análisis completo
        t0 = time.perf_counter()

        # core/pipeline.py (dentro de process_video)
        # 1) Abrir lector del video de entrada: devuelve handle + tamaño + FPS
        cap, w, h, fps = open_video_reader(in_path)

        # >>> CAMBIO: open_video_writer ahora devuelve (writer, out_path_final)
        # 2) Abrir escritor del video anotado. Devuelve el writer y la ruta
        #    final del archivo (la extensión puede variar según códec elegido).
        writer, out_path_final = open_video_writer(out_path, fps, (w, h))


        # Geometría (para overlays de referencia)
        geom = self.cfg["geometry"]
        A1,A2 = geom["speed_lines"]["A"]
        B1,B2 = geom["speed_lines"]["B"]
        SL1,SL2 = geom["stop_line"]

        frame_idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok: break
                frame_idx += 1
                ts = frame_idx / fps

                # 1) Detección base (YOLO sobre el frame actual)
                base_dets = self.detector.infer(frame)

                # 2) Tracking (asigna IDs persistentes a las detecciones)
                tracks = self.tracker.update(base_dets, frame)

                # 3) Casco (sólo si hay persona + moto en escena para ahorrar cómputo)
                need_helmet = any(t["label"]=="person" for t in tracks) and any(t["label"]=="motorbike" for t in tracks)
                helmet_dets = self.helmet_detector.infer(frame) if (self.helmet_detector and need_helmet) else []

                # 4) Lanes (MVP con Canny+Hough; usado por reglas de carril)
                lane_info = self.lane_detector.infer(frame)

                # 5) Reglas (helmet / speed / lane invasion)
                #    IMPORTANTE: cuando una regla confirma infracción, llama a
                #    self.logger.log(...), que escribe una foto del frame y el
                #    recorte del bbox a data/output/evidence/<tipo>/...
                for rule in self.rules:
                    if rule.__class__.__name__ == "HelmetRule":
                        rule.update(frame, tracks, ts, self.logger, helmet_dets=helmet_dets)
                    elif rule.__class__.__name__ == "LaneInvasionRule":
                        rule.update(frame, tracks, ts, self.logger, lane_info=lane_info)
                    else:
                        rule.update(frame, tracks, ts, self.logger)

                # 6) Overlays (visual) sobre el frame que será escrito a disco
                for t in tracks:
                    txt = f"ID {t['id']} {t['label']}"
                    draw_box(frame, t["bbox"], text=txt)
                draw_line(frame, SL1, SL2, color=(0,0,255))   # stop line
                draw_line(frame, A1, A2, color=(255,255,0))   # speed A
                draw_line(frame, B1, B2, color=(255,255,0))   # speed B
                draw_hud(frame, f"FPS: {fps:.1f} | Frame: {frame_idx}")

                # 7) Escritura del frame anotado al video de salida
                writer.write(frame)

            # Devuelve DataFrame para integraciones programáticas (la UI lo lee del CSV)
            csv_path = os.path.join(self.cfg["video"]["output_dir"], "events.csv")
            df = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()
            # Medición de rendimiento: duración total y FPS de procesamiento
            t1 = time.perf_counter()
            processing_seconds = max(0.0, t1 - t0)
            processing_fps = (frame_idx / processing_seconds) if processing_seconds > 0 else 0.0
            return {
                "events_df": df,
                "out_path_final": out_path_final,
                "processing_seconds": processing_seconds,
                "processing_fps": processing_fps,
            }

        finally:
            # 8) Liberar recursos de video (lector y escritor)
            release_safely(cap, writer)
