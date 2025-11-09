from ultralytics import YOLO
import torch

CLASS_NAMES = {
    0: 'person', 2: 'car', 3: 'motorbike', 5: 'bus', 7: 'truck', 9: 'traffic light'
}

class YoloDetector:
    """Detector general (vehículos/persona/semáforo) con YOLO Ultralytics.

    - Selecciona GPU automáticamente si está disponible (torch.cuda.is_available()).
    - Loguea en consola el dispositivo que se utilizará para inferencia.
    """
    def __init__(self, model_path="models/yolo/yolo11n.pt", imgsz=640, conf=0.35, device=None):
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf = conf

        # Selección de dispositivo: usa GPU si está disponible (por defecto)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        # Log informativo del dispositivo
        if self.device == 'cuda':
            try:
                dev_name = torch.cuda.get_device_name(0)
            except Exception:
                dev_name = 'CUDA'
            print(f"[YoloDetector] Usando GPU: {dev_name}")
        else:
            print("[YoloDetector] Usando CPU para inferencia")

    def infer(self, frame):
        """Ejecuta inferencia y devuelve lista de dicts {bbox, conf, label}."""
        # Pasa el dispositivo explícitamente a Ultralytics (0 para CUDA, 'cpu' para CPU)
        dev_arg = 0 if str(self.device).startswith('cuda') else 'cpu'
        res = self.model.predict(
            frame, imgsz=self.imgsz, conf=self.conf, device=dev_arg, stream=False, verbose=False
        )[0]
        dets = []
        if res.boxes is None: return dets
        for b, c, cls in zip(res.boxes.xyxy.cpu().numpy(),
                             res.boxes.conf.cpu().numpy(),
                             res.boxes.cls.cpu().numpy()):
            name = CLASS_NAMES.get(int(cls), str(int(cls)))
            dets.append({"bbox": b.tolist(), "conf": float(c), "label": name})
        return dets
