# core/detectors/yolo_detector.py
from ultralytics import YOLO

CLASS_NAMES = {
    0: 'person', 2: 'car', 3: 'motorbike', 5: 'bus', 7: 'truck', 9: 'traffic light'
}

class YoloDetector:
    """Detector general (vehículos/persona/semáforo) con YOLO Ultralytics."""
    def __init__(self, model_path="models/yolo/yolo11n.pt", imgsz=640, conf=0.35, device=None):
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf = conf
        self.device = device

    def infer(self, frame):
        res = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, stream=False, verbose=False)[0]
        dets = []
        if res.boxes is None: return dets
        for b, c, cls in zip(res.boxes.xyxy.cpu().numpy(),
                             res.boxes.conf.cpu().numpy(),
                             res.boxes.cls.cpu().numpy()):
            name = CLASS_NAMES.get(int(cls), str(int(cls)))
            dets.append({"bbox": b.tolist(), "conf": float(c), "label": name})
        return dets