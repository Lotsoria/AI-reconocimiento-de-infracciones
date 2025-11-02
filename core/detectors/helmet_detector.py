# core/detectors/helmet_detector.py
# Carga un modelo YOLO específico para cascos (debes colocar el .pt en models/helmet/)
from ultralytics import YOLO
import os

class HelmetDetector:
    def __init__(self, model_path="models/helmet/helmet_yolo.pt", imgsz=640, conf=0.35):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No se encontró el modelo de casco: {model_path}.\n"
                "Coloca un .pt preentrenado en esa ruta."
            )
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf = conf

    def infer(self, frame):
        res = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, stream=False, verbose=False)[0]
        dets = []
        if res.boxes is None: return dets
        dets.extend(
            {"bbox": b.tolist(), "conf": float(c), "label": "helmet"}
            for b, c, cls in zip(
                res.boxes.xyxy.cpu().numpy(),
                res.boxes.conf.cpu().numpy(),
                res.boxes.cls.cpu().numpy(),
            )
        )
        return dets