# core/utils/color.py
# Clasificación simple del estado del semáforo usando medias de canales en ROI
import numpy as np

def traffic_light_state(frame, roi, red_threshold=1.3):
    x1,y1,x2,y2 = map(int, roi)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0: return "unknown"
    b,g,r = np.mean(crop[:,:,0]), np.mean(crop[:,:,1]), np.mean(crop[:,:,2])
    ratio_rg = (r+1e-6)/(g+1e-6)
    if ratio_rg > red_threshold: return "red"
    return "green" if g > r else "unknown"
