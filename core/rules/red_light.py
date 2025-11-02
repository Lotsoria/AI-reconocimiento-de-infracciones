# core/rules/red_light.py
from core.utils.geometry import crossed_line
from core.utils.color import traffic_light_state

class RedLightRule:
    def __init__(self, cfg):
        self.p1, self.p2 = cfg["geometry"]["stop_line"]
        self.roi = cfg["sem_light"]["roi"]
        self.red_thr = cfg["sem_light"]["red_threshold"]
        self.cooldown = {}
        self.min_gap = 3.0

    def update(self, frame, tracks, ts, logger):
        state = traffic_light_state(frame, self.roi, self.red_thr)
        if state != "red": return
        for t in tracks:
            if t["label"] not in ["car","bus","truck","motorbike"]: 
                continue
            if crossed_line(t, self.p1, self.p2):
                last = self.cooldown.get(t["id"], -1e9)
                if ts - last > self.min_gap:
                    self.cooldown[t["id"]] = ts
                    logger.log("red_light", ts, t["id"], t["bbox"], extra={"state":state}, frame=frame)