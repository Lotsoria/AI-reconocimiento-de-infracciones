import cv2
import numpy as np

class SimpleLaneDetector:
    def __init__(self):
        pass

    def infer(self, frame):
        """
        Retorna un dict:
        {
          "lines": [([x1,y1],[x2,y2]), ...],   # líneas Hough
          "center_line": ([x1,y1],[x2,y2])     # línea central estimada (opcional)
        }
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blur, 50, 150)

        # ROI: parte baja de la imagen
        mask = np.zeros_like(edges)
        roi = np.array([[(0,h*0.6), (w,h*0.6), (w,h), (0,h)]], dtype=np.int32)
        cv2.fillPoly(mask, roi, 255)
        masked = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(masked, 1, np.pi/180, threshold=120, minLineLength=80, maxLineGap=50)
        out_lines = []
        if lines is not None:
            for l in lines:
                x1,y1,x2,y2 = l[0]
                out_lines.append(([x1,y1],[x2,y2]))

        # Línea central naive (vertical en el centro)
        center_line = ([w//2, int(h*0.3)], [w//2, h-10])

        return {"lines": out_lines, "center_line": center_line}