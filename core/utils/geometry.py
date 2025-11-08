import numpy as np

def center_of(bbox):
    x1,y1,x2,y2 = bbox
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def crossed_line(track, p1, p2):
    """
    Retorna True si el centro del track cruzó el segmento p1->p2 entre el frame previo y actual.
    track debe tener 'bbox' y opcionalmente 'prev_center'.
    """
    if "prev_center" not in track or track["prev_center"] is None:
        return False
    c_prev = np.array(track["prev_center"], dtype=float)
    c_curr = np.array(center_of(track["bbox"]), dtype=float)
    p1 = np.array(p1, dtype=float); p2 = np.array(p2, dtype=float)

    # Cruce detectado si los puntos están en lados opuestos de la línea
    def side(p): return np.cross(p2 - p1, p - p1)
    return side(c_prev) * side(c_curr) < 0  # signos opuestos => cruce

def point_in_polygon(point, polygon):
    """Ray casting simple para saber si un punto está dentro de un polígono."""
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1,y1 = polygon[i]
        x2,y2 = polygon[(i+1)%n]
        if ((y1>y) != (y2>y)) and (x < (x2-x1)*(y-y1)/(y2-y1+1e-9)+x1):
            inside = not inside
    return inside

def line_angle(p1, p2):
    v = np.array(p2) - np.array(p1)
    return np.degrees(np.arctan2(v[1], v[0]))