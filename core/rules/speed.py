# core/rules/speed.py
# Mide Δt entre cruce de líneas A y B por track_id -> estima velocidad
from core.utils.geometry import crossed_line

class SpeedRule:
    def __init__(self, cfg):
        self.A = cfg["geometry"]["speed_lines"]["A"]
        self.B = cfg["geometry"]["speed_lines"]["B"]
        self.D_pix = cfg["speed"]["pixel_distance"]
        self.k = cfg["speed"]["k_calibration"]     # m/pixel aprox
        self.limit = cfg["speed"]["limit_kmh"]     # km/h
        self.tsA = {}  # tiempo de cruce por track_id

    def update(self, frame, tracks, ts, logger):
        for t in tracks:
            if t["label"] not in ["car","bus","truck","motorbike"]: 
                continue
            # Cruce A
            if crossed_line(t, self.A[0], self.A[1]) and t["id"] not in self.tsA:
                self.tsA[t["id"]] = ts
            # Cruce B -> medir Δt
            if crossed_line(t, self.B[0], self.B[1]) and t["id"] in self.tsA:
                dt = max(1e-6, ts - self.tsA.pop(t["id"]))
                # v ~ (k * D_pix) / dt  -> m/s
                v_ms = (self.k * self.D_pix) / dt
                v_kmh = v_ms * 3.6
                if v_kmh > self.limit:
                    logger.log("overspeed", ts, t["id"], t["bbox"], extra={"kmh": round(v_kmh,1)}, frame=frame)