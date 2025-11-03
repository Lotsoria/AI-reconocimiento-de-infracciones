# core/pipeline.py
# -----------------------------------------------------------------------------
# Orquesta el flujo:
#   lectura -> detección -> tracking -> reglas -> overlays -> writer + log
# Importante:
#   - Limpia evidencias y CSV al comenzar un NUEVO análisis (cuando el caller
#     pasa clean_previous=True). La UI sólo llama esto al pulsar 'Analizar'.
# -----------------------------------------------------------------------------

import contextlib
import cv2, yaml, os, shutil
import numpy as np
import pandas as pd

from core.utils.video_io import open_video_reader, open_video_writer, release_safely
from core.utils.events import EventLogger
from core.utils.drawing import draw_box, draw_line, draw_hud
from core.detectors.yolo_detector import YoloDetector
from core.detectors.helmet_detector import HelmetDetector
from core.detectors.lane_detector import SimpleLaneDetector
from core.trackers.deepsort_wrapper import DeepSortWrapper
from core.rules.red_light import RedLightRule
from core.rules.helmet import HelmetRule
from core.rules.speed import SpeedRule
from core.rules.lane_invasion import LaneInvasionRule

def _load_config(path):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    # Soporte 'include' para heredar defaults
    if "include" in cfg:
        with open(cfg["include"], 'r') as f:
            base = yaml.safe_load(f)
        base.update({k:v for k,v in cfg.items() if k!="include"})
        cfg = base
    return cfg

def _clean_previous_outputs(output_dir: str, evidence_dir: str):
    """
    Borra:
      - CSV de eventos (si existe)
      - Carpeta de evidencias (y la recrea vacía)
    Se invoca sólo cuando el caller lo solicita (clean_previous=True).
    """
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
        # Helmet (opcional)
        try:
            self.helmet_detector = HelmetDetector(
                model_path=self.cfg["models"]["helmet_path"],
                imgsz=self.cfg["yolo"]["imgsz"],
                conf=0.35
            )
        except Exception:
            self.helmet_detector = None

        # Lanes (MVP sencillo; luego puedes integrar UFLD sin tocar el resto)
        self.lane_detector = SimpleLaneDetector()

        # Tracker
        self.tracker = DeepSortWrapper(max_age=15)

        # Reglas activas
        self.rules = []
        if self.cfg["rules"].get("red_light"):     self.rules.append(RedLightRule(self.cfg))
        if self.cfg["rules"].get("helmet"):        self.rules.append(HelmetRule(self.cfg))
        if self.cfg["rules"].get("speed"):         self.rules.append(SpeedRule(self.cfg))
        if self.cfg["rules"].get("lane_invasion"): self.rules.append(LaneInvasionRule(self.cfg))

        # Logger (se re-crea después de limpiar para reescribir encabezados)
        self.logger = EventLogger(self.cfg["video"]["output_dir"], self.cfg["video"]["evidence_dir"])

    def process_video(self, in_path, out_path, clean_previous=True):
        """
        Ejecuta el análisis del video y produce:
          - video anotado en 'out_path'
          - CSV 'data/output/events.csv' con encabezados en español
          - evidencias (frames y recortes) en data/output/evidence/
        """
        if clean_previous:
            _clean_previous_outputs(self.cfg["video"]["output_dir"], self.cfg["video"]["evidence_dir"])
            self.logger = EventLogger(self.cfg["video"]["output_dir"], self.cfg["video"]["evidence_dir"])


        # core/pipeline.py (dentro de process_video)
        cap, w, h, fps = open_video_reader(in_path)

        # >>> CAMBIO: open_video_writer ahora devuelve (writer, out_path_final)
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

                # 1) Detección base
                base_dets = self.detector.infer(frame)

                # 2) Casco (si modelo disponible)
                helmet_dets = self.helmet_detector.infer(frame) if self.helmet_detector else []

                # 3) Tracking
                tracks = self.tracker.update(base_dets, frame)

                # 4) Lanes (MVP)
                lane_info = self.lane_detector.infer(frame)

                # 5) Reglas
                for rule in self.rules:
                    if rule.__class__.__name__ == "HelmetRule":
                        rule.update(frame, tracks, ts, self.logger, helmet_dets=helmet_dets)
                    elif rule.__class__.__name__ == "LaneInvasionRule":
                        rule.update(frame, tracks, ts, self.logger, lane_info=lane_info)
                    else:
                        rule.update(frame, tracks, ts, self.logger)

                # 6) Overlays (visual)
                for t in tracks:
                    txt = f"ID {t['id']} {t['label']}"
                    draw_box(frame, t["bbox"], text=txt)
                draw_line(frame, SL1, SL2, color=(0,0,255))   # stop line
                draw_line(frame, A1, A2, color=(255,255,0))   # speed A
                draw_line(frame, B1, B2, color=(255,255,0))   # speed B
                draw_hud(frame, f"FPS: {fps:.1f} | Frame: {frame_idx}")

                # 7) Escritura frame anotado
                writer.write(frame)

            # Devuelve DataFrame para integraciones programáticas (la UI lo vuelve a leer del CSV)
            csv_path = os.path.join(self.cfg["video"]["output_dir"], "events.csv")
            df = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()
            return {"events_df": df, "out_path_final": out_path_final}

        finally:
            release_safely(cap, writer)
