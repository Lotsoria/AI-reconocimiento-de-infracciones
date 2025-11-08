"""Detector de casco basado en Ultralytics YOLO.

Uso esperado:
- Proporciona un .pt entrenado con clase 'helmet' en models/helmet/.
- Selecciona GPU automáticamente si está disponible.
"""

from ultralytics import YOLO
import os
import torch


class HelmetDetector:
    def __init__(self, model_path="models/helmet/helmet_yolo.pt", imgsz=768, conf=0.30, device=None):
        # Verifica que el peso exista. Si no, el pipeline debe encargarse de descargarlo.
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No se encontró el modelo de casco: {model_path}.\n"
                "Coloca un .pt preentrenado en esa ruta o configura HELMET_MODEL_URL."
            )

        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf = conf

        # Selección de dispositivo
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        if self.device == 'cuda':
            try:
                name = torch.cuda.get_device_name(0)
            except Exception:
                name = 'CUDA'
            print(f"[HelmetDetector] Usando GPU: {name}")
        else:
            print("[HelmetDetector] Usando CPU para inferencia")

    def infer(self, frame):
        """Devuelve detecciones {bbox, conf, label='helmet'} filtrando por clase 'helmet'."""
        dev_arg = 0 if str(self.device).startswith('cuda') else 'cpu'
        res = self.model.predict(
            frame, imgsz=self.imgsz, conf=self.conf, device=dev_arg, stream=False, verbose=False
        )[0]
        dets = []
        if res.boxes is None:
            return dets
        # Filtra únicamente clases que representen casco
        names = getattr(self.model, 'names', None) or getattr(res, 'names', None) or {}
        allowed = {"helmet", "hardhat", "safety helmet", "safety_helmet"}
        for b, c, cls in zip(
            res.boxes.xyxy.cpu().numpy(),
            res.boxes.conf.cpu().numpy(),
            res.boxes.cls.cpu().numpy(),
        ):
            cls_name = str(names.get(int(cls), str(int(cls)))).lower()
            if cls_name in allowed:
                dets.append({"bbox": b.tolist(), "conf": float(c), "label": "helmet"})
        return dets
