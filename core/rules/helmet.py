# core/rules/helmet.py
# Lógica: persona↔moto por proximidad; verifica presencia de casco sobre la cabeza.
import numpy as np
from core.utils.geometry import center_of

class HelmetRule:
    def __init__(self, cfg):
        self.min_frames = cfg["helmet"]["min_persistence_frames"]
        self.state = {}  # por track_id: conteo frames sin casco

    @staticmethod
    def _associate_people_to_motos(tracks):
        persons = [t for t in tracks if t["label"]=="person"]
        motos   = [t for t in tracks if t["label"]=="motorbike"]
        pairs = []
        for p in persons:
            pc = center_of(p["bbox"])
            # moto más cercana
            if not motos: continue
            dists = [np.hypot(center_of(m["bbox"])[0]-pc[0], center_of(m["bbox"])[1]-pc[1]) for m in motos]
            j = int(np.argmin(dists))
            if dists[j] < 120:  # umbral px (ajustable según escena)
                pairs.append((p, motos[j]))
        return pairs

    @staticmethod
    def _head_roi(person_bbox):
        # ROI de cabeza: tercio superior del bbox de persona
        x1,y1,x2,y2 = map(int, person_bbox)
        h = y2 - y1
        head_y2 = y1 + int(0.33*h)
        return [x1, y1, x2, head_y2]

    def update(self, frame, tracks, ts, logger, helmet_dets):
        # helmet_dets: lista de bboxes de casco
        pairs = self._associate_people_to_motos(tracks)
        for p, m in pairs:
            roi = self._head_roi(p["bbox"])
            # ¿Algún casco solapa con la ROI?
            has_helmet = False
            x1,y1,x2,y2 = roi
            for hdet in helmet_dets:
                hx1,hy1,hx2,hy2 = map(int, hdet["bbox"])
                ix1, iy1 = max(x1,hx1), max(y1,hy1)
                ix2, iy2 = min(x2,hx2), min(y2,hy2)
                inter = max(0, ix2-ix1)*max(0, iy2-iy1)
                if inter > 0: 
                    has_helmet = True; break

            pid = p["id"]
            if not has_helmet:
                self.state[pid] = self.state.get(pid, 0) + 1
                if self.state[pid] == self.min_frames:
                    logger.log("no_helmet", ts, pid, p["bbox"], extra={"moto_id": m["id"]}, frame=frame)
            else:
                self.state[pid] = 0