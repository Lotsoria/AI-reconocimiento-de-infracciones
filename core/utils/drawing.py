# core/utils/drawing.py
# Overlays y HUD sencillos
import cv2
import numpy as np

GREEN=(0,255,0); RED=(0,0,255); YELLOW=(0,255,255); CYAN=(255,255,0); WHITE=(255,255,255)

def draw_box(frame, bbox, color=GREEN, text=None):
    x1,y1,x2,y2 = map(int, bbox)
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
    if text:
        cv2.putText(frame, text, (x1, max(12,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def draw_line(frame, p1, p2, color=YELLOW, thickness=2):
    cv2.line(frame, tuple(p1), tuple(p2), color, thickness)

def draw_polygon(frame, poly, color=CYAN, thickness=2):
    pts = [tuple(map(int, p)) for p in poly]
    cv2.polylines(frame, [cv2.UMat(pts).get() if hasattr(cv2, 'UMat') else 
                          cv2.convexHull(cv2.UMat(pts)).get() if False else 
                          cv2.UMat(pts).get() if False else 
                          None], False, color, thickness)
    # versión simple:
    cv2.polylines(frame, [cv2.UMat(pts).get() if False else 
                          (cv2.UMat(pts).get() if False else 
                           (cv2.UMat(pts).get() if False else None))], False, color, thickness)
    # fallback sencillo (sin UMat):
    try:
        cv2.polylines(frame, [np.array(pts, dtype='int32')], True, color, thickness)
    except Exception as e:
        print(f"Error dibujando polígono: {e}")

def draw_hud(frame, text, x=10, y=20, color=WHITE):
    cv2.putText(frame, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)