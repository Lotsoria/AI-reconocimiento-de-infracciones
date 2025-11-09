from core.utils.geometry import crossed_line


class SpeedRule:
    """Regla de velocidad: mide Δt entre líneas A y B y estima km/h.

    Diseño robusto: si falta `cfg['speed']` en la escena, aplica valores por
    defecto razonables en vez de fallar con KeyError.
    """

    def __init__(self, cfg):
        geom = cfg.get("geometry", {})
        sl = geom.get("speed_lines", {}) or {"A": [[0, 0], [0, 0]], "B": [[0, 1], [0, 1]]}
        self.A = sl.get("A")
        self.B = sl.get("B")

        scfg = cfg.get("speed", {})
        self.D_pix = float(scfg.get("pixel_distance", 120))
        self.k = float(scfg.get("k_calibration", 0.18))     # m/pixel aprox
        self.limit = float(scfg.get("limit_kmh", 40))        # km/h

        self.tsA = {}  # tiempo de cruce por track_id

    def update(self, frame, tracks, ts, logger):
        # Si no hay líneas válidas, no hacer nada
        if not (self.A and self.B):
            return
        for t in tracks:
            if t["label"] not in ["car", "bus", "truck", "motorbike"]:
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
                    logger.log(
                        "overspeed",
                        ts,
                        t["id"],
                        t["bbox"],
                        extra={"kmh": round(v_kmh, 1)},
                        frame=frame,
                    )
