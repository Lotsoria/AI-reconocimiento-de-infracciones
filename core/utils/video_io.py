# core/utils/video_io.py
# Utilidades para abrir lectores y escritores de video con fallbacks de códecs.
# En Windows, H.264 (avc1) puede requerir la DLL de OpenH264.
import cv2, os, platform

def open_video_reader(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0  # fallback si FPS=0
    return cap, w, h, fps

def _try_writer(out_path, fps, size, fourcc_str, container_ext):
    """
    Intenta construir un VideoWriter con el FOURCC indicado y devuelve
    (writer, out_path_final). Si falla, `writer.isOpened()` será False.

    - out_path: ruta base de salida (se forzará la extensión del contenedor)
    - fps: frames por segundo deseados
    - size: (ancho, alto) del video
    - fourcc_str: p.ej. 'avc1' (H.264), 'mp4v', 'VP80' (WebM/VP8), 'MJPG', 'XVID'
    - container_ext: '.mp4', '.webm' o '.avi'
    """
    w, h = size
    # Fuerza la extensión del contenedor elegido
    base, _ = os.path.splitext(out_path)
    out = f"{base}{container_ext}"
    # Construye el FOURCC y prueba a abrir el writer
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    writer = cv2.VideoWriter(out, fourcc, fps, (w, h))

    # Log de diagnóstico amigable
    if writer.isOpened():
        print(f"[video_io] OK -> fourcc={fourcc_str} container={container_ext} path={out}")
    else:
        print(f"[video_io] FAIL -> fourcc={fourcc_str} container={container_ext} path={out}")
        # Hint específico para H.264 en Windows
        if platform.system() == 'Windows' and fourcc_str == 'avc1':
            print("[video_io] H.264 en Windows requiere la DLL de OpenH264.\n"
                  "          Descarga 'openh264-1.8.0-win64.dll' (versión del log)\n"
                  "          y colócala en el directorio del proyecto o en una ruta del PATH.")
    return writer, out

def open_video_writer(out_path, fps, size):
    """
    Abre un VideoWriter probando varios códecs/contenedores para maximizar
    compatibilidad, manteniendo H.264 (avc1) como primer intento si así
    funciona mejor en tu entorno, pero con fallbacks y logs claros.

    Secuencia de intentos:
      1) MP4/H.264 (avc1)  -> .mp4  (HTML5-friendly, pero requiere OpenH264 en Windows)
      2) MP4/mp4v          -> .mp4  (sin dependencia de OpenH264)
      3) WebM/VP8 (VP80)   -> .webm (muy compatible en navegadores)
      4) AVI/MJPG          -> .avi  (robusto en Windows)
      5) AVI/XVID          -> .avi  (último recurso)
    """
    # 1) H.264 (avc1)
    writer, final_path = _try_writer(out_path, fps, size, 'avc1', '.mp4')
    if writer.isOpened():
        return writer, final_path

    # 2) mp4v
    writer, final_path = _try_writer(out_path, fps, size, 'mp4v', '.mp4')
    if writer.isOpened():
        return writer, final_path

    # 3) VP8/WebM
    writer, final_path = _try_writer(out_path, fps, size, 'VP80', '.webm')
    if writer.isOpened():
        return writer, final_path

    # 4) MJPG/AVI
    writer, final_path = _try_writer(out_path, fps, size, 'MJPG', '.avi')
    if writer.isOpened():
        return writer, final_path

    # 5) XVID/AVI
    writer, final_path = _try_writer(out_path, fps, size, 'XVID', '.avi')
    if writer.isOpened():
        return writer, final_path

    raise RuntimeError(
        "No se pudo abrir un VideoWriter (probado: avc1/mp4v/VP80/MJPG/XVID).\n"
        "En Windows, instala la DLL 'openh264-1.8.0-win64.dll' para H.264 o usa WebM/MJPG."
    )

def release_safely(cap=None, writer=None):
    try:
        if cap is not None: cap.release()
    finally:
        if writer is not None: writer.release()
