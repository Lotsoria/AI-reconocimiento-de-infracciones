"""Regla: detecta motociclistas sin casco con histeresis y verificación de IoU.

Mejoras frente a la versión básica:
- Asociación persona↔moto por proximidad con umbral configurable.
- ROI de cabeza proporcional al alto del bbox de persona (ajustable).
- Verificación con IoU mínimo casco↔cabeza y filtro por confianza del casco.
- Persistencia temporal (frames consecutivos) antes de reportar "sin casco".
"""

import numpy as np
from core.utils.geometry import center_of


def _iou(a, b):
    """Calcula IoU entre dos bboxes [x1,y1,x2,y2]."""
    ax1, ay1, ax2, ay2 = map(int, a)
    bx1, by1, bx2, by2 = map(int, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union


class HelmetRule:
    def __init__(self, cfg):
        hcfg = cfg["helmet"]
        # Frames consecutivos sin casco para reportar. Mantiene compatibilidad
        # con la clave anterior 'min_persistence_frames' si existe.
        self.no_helmet_need = hcfg.get("min_persistence_no_helmet",
                                       hcfg.get("min_persistence_frames", 6))
        # Frames consecutivos con casco para limpiar el estado "activo"
        self.helmet_need = hcfg.get("min_persistence_helmet", 3)
        # Enfría reportes repetidos (segundos mínimos entre logs del mismo id)
        self.min_gap = float(hcfg.get("min_gap_seconds", 3.0))
        # Parámetros de asociación y verificación
        self.max_dist = hcfg.get("max_person_moto_dist", 140)   # px
        self.head_ratio = hcfg.get("head_roi_top_ratio", 0.38)  # 38% superior del bbox
        self.iou_thr = hcfg.get("helmet_iou_thresh", 0.12)      # IoU mínimo casco↔cabeza
        self.conf_min = hcfg.get("helmet_conf_min", 0.25)       # conf mínima detección casco

        # Estado por id de persona
        self.neg = {}              # frames consecutivos sin casco
        self.pos = {}              # frames consecutivos con casco
        self.active = set()        # ids en violación activa (ya reportados)
        self.last_report = {}      # id -> timestamp del último reporte

    def _associate_people_to_motos(self, tracks):
        """Empareja cada persona con su moto más cercana si está dentro de max_dist."""
        persons = [t for t in tracks if t["label"] == "person"]
        motos = [t for t in tracks if t["label"] == "motorbike"]
        pairs = []
        if not motos:
            return pairs
        for p in persons:
            pc = center_of(p["bbox"])
            dists = [np.hypot(center_of(m["bbox"])[0] - pc[0], center_of(m["bbox"])[1] - pc[1]) for m in motos]
            j = int(np.argmin(dists))
            if dists[j] < self.max_dist:
                pairs.append((p, motos[j]))
        return pairs

    def _head_roi(self, person_bbox):
        """ROI de cabeza: franja superior del bbox con ligero margen horizontal."""
        x1, y1, x2, y2 = map(int, person_bbox)
        h = max(1, y2 - y1)
        head_y2 = y1 + int(self.head_ratio * h)
        padx = int(0.08 * (x2 - x1))  # margen para tolerar pequeñas desviaciones
        return [x1 - padx, y1, x2 + padx, head_y2]

    def update(self, frame, tracks, ts, logger, helmet_dets):
        # Filtra detecciones de casco por confianza
        helmet_dets = [h for h in helmet_dets if h.get("conf", 1.0) >= self.conf_min]

        pairs = self._associate_people_to_motos(tracks)
        for p, m in pairs:
            pid = p["id"]
            roi = self._head_roi(p["bbox"])

            # ¿Hay casco con IoU suficiente con la ROI de cabeza?
            has_helmet = False
            for hdet in helmet_dets:
                if _iou(roi, hdet["bbox"]) >= self.iou_thr:
                    has_helmet = True
                    break

            if not has_helmet:
                # Acumula frames negativos y resetea positivos
                self.neg[pid] = self.neg.get(pid, 0) + 1
                self.pos[pid] = 0
                # Al alcanzar persistencia, emite un único reporte por violación
                if self.neg[pid] >= self.no_helmet_need:
                    last = self.last_report.get(pid, -1e9)
                    if (pid not in self.active) and (ts - last >= self.min_gap):
                        logger.log("no_helmet", ts, pid, p["bbox"], extra={"moto_id": m["id"]}, frame=frame)
                        self.active.add(pid)
                        self.last_report[pid] = ts
            else:
                # Vemos casco: acumula positivos y resetea negativos
                self.pos[pid] = self.pos.get(pid, 0) + 1
                self.neg[pid] = 0
                # Si mantiene casco por algunos frames, limpia estado activo
                if self.pos[pid] >= self.helmet_need and pid in self.active:
                    self.active.remove(pid)
