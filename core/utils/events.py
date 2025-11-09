# -----------------------------------------------------------------------------
# EventLogger:
# - CSV con encabezados en ESPAÑOL:
#   fecha_hora, tipo_infraccion, tiempo_seg, id_objeto, x1, y1, x2, y2,
#   ruta_imagen, ruta_recorte, extra
# - Guarda evidencia: frame completo y recorte del bbox con padding.
# -----------------------------------------------------------------------------

import os, csv, cv2, json
from datetime import datetime

def _safe_mkdir(p):
    os.makedirs(p, exist_ok=True)

def _crop_with_padding(frame, bbox, pad=10):
    x1,y1,x2,y2 = map(int, bbox)
    h, w = frame.shape[:2]
    x1p = max(0, x1 - pad); y1p = max(0, y1 - pad)
    x2p = min(w-1, x2 + pad); y2p = min(h-1, y2 + pad)
    return frame[y1p:y2p, x1p:x2p]

class EventLogger:
    def __init__(self, output_dir="data/output", evidence_dir="data/output/evidence"):
        # Directorios donde se escriben CSV y evidencias (imágenes)
        self.output_dir = output_dir
        self.evidence_dir = evidence_dir
        _safe_mkdir(self.output_dir)
        _safe_mkdir(self.evidence_dir)
        # Ruta del CSV de eventos. Se crea con encabezados si no existe.
        self.csv_path = os.path.join(self.output_dir, "events.csv")
        self._init_csv()

    def _init_csv(self):
        # Si no existe o está vacío, crea CSV con encabezados en español.
        if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    "fecha_hora","tipo_infraccion","tiempo_seg","id_objeto",
                    "x1","y1","x2","y2","ruta_imagen","ruta_recorte","extra"
                ])

    def log(self, event_type, ts, track_id, bbox, extra=None, frame=None):
        """
        Registra una fila en el CSV y, si se pasa el frame, crea dos imágenes:
          - imagen completa del frame actual (con overlays incluidos)
          - recorte (crop) alrededor del bbox del objeto infractor
        Las imágenes se guardan en data/output/evidence/<event_type>/
        """
        now = datetime.now().isoformat(timespec="seconds")
        x1,y1,x2,y2 = map(int, bbox)
        extra_json = json.dumps(extra or {}, ensure_ascii=False)

        image_path, crop_path = "", ""

        if frame is not None:
            # 1) Carpeta por tipo de evento (e.g., evidence/no_helmet)
            ev_dir = os.path.join(self.evidence_dir, event_type)
            _safe_mkdir(ev_dir)

            # 2) Nombrar archivos por tipo, id y timestamp en ms
            base = f"{event_type}_id{track_id}_{int(ts*1000)}"
            image_path = os.path.join(ev_dir, f"{base}.jpg").replace("\\","/")
            crop_path  = os.path.join(ev_dir, f"{base}_crop.jpg").replace("\\","/")

            # 3) Guardar frame completo (lo que ves en la GUI en ese momento)
            cv2.imwrite(image_path, frame)

            # 4) Guardar recorte con padding suave alrededor del bbox
            crop = _crop_with_padding(frame, bbox, pad=12)
            if crop.size > 0:
                cv2.imwrite(crop_path, crop)

        # 5) Añadir fila al CSV con las rutas generadas
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                now, event_type, f"{ts:.3f}", track_id, x1,y1,x2,y2,
                image_path, crop_path, extra_json
            ])
