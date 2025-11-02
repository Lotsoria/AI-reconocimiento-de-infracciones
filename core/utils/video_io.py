# core/utils/video_io.py
import cv2, os

def open_video_reader(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0  # fallback si FPS=0
    return cap, w, h, fps

def _try_writer(out_path, fps, size, fourcc_str, container_ext):
    """Intenta crear un writer con el fourcc y contenedor dados."""
    w, h = size
    # Enforce container extension
    base, _ = os.path.splitext(out_path)
    out = f"{base}{container_ext}"
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    writer = cv2.VideoWriter(out, fourcc, fps, (w, h))
    return writer, out

def open_video_writer(out_path, fps, size):
    """
    Intenta en orden:
      1) MP4/H.264 (avc1)  -> .mp4
      2) MP4/mp4v (MPEG-4 Part 2) -> .mp4
      3) WebM/VP8 (VP80)   -> .webm
    """
    # 1) H.264 (avc1) — ideal para HTML5 players
    writer, final_path = _try_writer(out_path, fps, size, 'avc1', '.mp4')
    if writer.isOpened():
        return writer, final_path

    # 2) mp4v — a veces funciona localmente, pero en navegador puede fallar
    writer, final_path = _try_writer(out_path, fps, size, 'mp4v', '.mp4')
    if writer.isOpened():
        return writer, final_path

    # 3) VP8/WebM — muy compatible en navegadores modernos
    writer, final_path = _try_writer(out_path, fps, size, 'VP80', '.webm')
    if writer.isOpened():
        return writer, final_path

    raise RuntimeError("No se pudo abrir un VideoWriter compatible (avc1/mp4v/VP80).")

def release_safely(cap=None, writer=None):
    try:
        if cap is not None: cap.release()
    finally:
        if writer is not None: writer.release()