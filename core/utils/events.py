# core/utils/events.py
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
        self.output_dir = output_dir
        self.evidence_dir = evidence_dir
        _safe_mkdir(self.output_dir)
        _safe_mkdir(self.evidence_dir)
        self.csv_path = os.path.join(self.output_dir, "events.csv")
        self._init_csv()

    def _init_csv(self):
        # Si no existe o está vacío, crea con encabezados en español
        if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    "fecha_hora","tipo_infraccion","tiempo_seg","id_objeto",
                    "x1","y1","x2","y2","ruta_imagen","ruta_recorte","extra"
                ])

    def log(self, event_type, ts, track_id, bbox, extra=None, frame=None):
        """
        Registra un evento y (si frame no es None) guarda la evidencia asociada.
        """
        now = datetime.now().isoformat(timespec="seconds")
        x1,y1,x2,y2 = map(int, bbox)
        extra_json = json.dumps(extra or {}, ensure_ascii=False)

        image_path, crop_path = "", ""

        if frame is not None:
            # Evidencias por tipo de evento
            ev_dir = os.path.join(self.evidence_dir, event_type)
            _safe_mkdir(ev_dir)

            # Nombres de archivo
            base = f"{event_type}_id{track_id}_{int(ts*1000)}"
            image_path = os.path.join(ev_dir, f"{base}.jpg").replace("\\","/")
            crop_path  = os.path.join(ev_dir, f"{base}_crop.jpg").replace("\\","/")

            # Guardar frame completo
            cv2.imwrite(image_path, frame)

            # Guardar recorte con padding
            crop = _crop_with_padding(frame, bbox, pad=12)
            if crop.size > 0:
                cv2.imwrite(crop_path, crop)

        # Fila con rutas (la UI puede ocultarlas en la tabla y usarlas solo al seleccionar)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                now, event_type, f"{ts:.3f}", track_id, x1,y1,x2,y2,
                image_path, crop_path, extra_json
            ])