# core/rules/lane_invasion.py
# Marca invasión si el centro del vehículo entra al polígono prohibido.
# Evita duplicados mediante persistencia y cooldown.
from core.utils.geometry import center_of, point_in_polygon

class LaneInvasionRule:
    def __init__(self, cfg):
        self.poly = cfg["geometry"]["no_cross_polygon"]
        self.persist = cfg["lane"]["persistence_frames"]  # frames consecutivos para confirmar
        self.state = {}            # track_id -> conteo de frames dentro
        self.active = set()        # tracks actualmente reportados (violación activa)
        self.cooldown = {}         # track_id -> último timestamp reportado
        self.min_gap = 3.0         # segundos entre reportes del mismo track

    def update(self, frame, tracks, ts, logger, lane_info=None):
        for t in tracks:
            # vehículos relevantes
            if t["label"] not in ["car","bus","truck","motorbike"]:
                continue

            tid = t["id"]
            c = center_of(t["bbox"])
            if inside := point_in_polygon(c, self.poly):
                self.state[tid] = self.state.get(tid, 0) + 1
                # Cuando supera persistencia y aún no está activo => reportar si pasó cooldown
                if self.state[tid] >= self.persist and tid not in self.active:
                    last = self.cooldown.get(tid, -1e9)
                    if ts - last >= self.min_gap:
                        logger.log("lane_invasion", ts, tid, t["bbox"], extra={}, frame=frame)
                        self.active.add(tid)          # marcamos como en violación
                        self.cooldown[tid] = ts       # registramos último reporte
            else:
                # Salió de la zona: reset y permitir futuros reportes
                self.state[tid] = 0
                if tid in self.active:
                    self.active.remove(tid)