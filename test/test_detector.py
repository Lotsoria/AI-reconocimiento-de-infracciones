# tests/test_detector.py
import cv2
from core.detectors.yolo_detector import YoloDetector

def test_yolo_detector_instantiation():
    det = YoloDetector()
    assert det is not None

def test_yolo_infer_on_blank():
    det = YoloDetector()
    img = (255 * (cv2.UMat(480,640,0).get() if False else 
           cv2.cvtColor(cv2.imread(""), cv2.COLOR_BGR2RGB) if False else 
           cv2.imread("nonexistent.jpg"))).astype('uint8') if False else \
          (255 * (cv2.imread("") if False else 0))
    # Fallback: imagen en blanco
    img = 255 * (cv2.imread("nonexistent.jpg") if False else None)
    import numpy as np
    img = np.ones((480,640,3), dtype='uint8')*255
    dets = det.infer(img)
    assert isinstance(dets, list)