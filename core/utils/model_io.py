"""Descarga de modelos a disco si faltan, con una URL configurable.

Uso:
    ensure_local_model("models/helmet/helmet_yolo.pt", url)

No añade dependencias externas (usa urllib). Crea carpetas si no existen.
"""

import os
import urllib.request


def _makedirs(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def ensure_local_model(dst_path: str, url: str) -> bool:
    """Si `dst_path` no existe y se entrega `url`, descarga el archivo.

    Devuelve True si el archivo existe tras la operación (descargado o ya estaba).
    Lanza excepción si la descarga falla para que el caller pueda manejarla.
    """
    if not dst_path:
        return False
    if os.path.exists(dst_path):
        return True
    if not url:
        return False
    _makedirs(dst_path)
    print(f"[model_io] Descargando modelo desde {url} -> {dst_path}")
    urllib.request.urlretrieve(url, dst_path)
    ok = os.path.exists(dst_path) and os.path.getsize(dst_path) > 0
    print(f"[model_io] Descarga {'OK' if ok else 'FALLÓ'}: {dst_path}")
    return ok

