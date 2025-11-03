# app/gui_streamlit.py
# -----------------------------------------------------------------------------
# GUI con Streamlit para detección de infracciones de tránsito
#
# Características:
# - Controla el flujo para evitar reprocesos innecesarios: SOLO procesa al hacer
#   clic en "Analizar video". El resto de interacciones NO reprocesan.
# - Usa rutas ABSOLUTAS ancladas al root del repo para evitar problemas al
#   ejecutar desde distintos directorios.
# - Escribe el video anotado en data/output/annotated_videos/resultado.mp4 y lo
#   reproduce leyendo en BYTES (st.video con bytes es más robusto que con path).
# - Muestra los eventos en una tabla con encabezados en español, sin exponer
#   rutas internas de evidencia.
# - Permite seleccionar un evento y ver su evidencia (recorte o frame completo).
# - Incluye un panel de diagnóstico opcional en la barra lateral.
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import hashlib
import tempfile
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
import streamlit as st

# Importa el pipeline de tu core
from core.pipeline import Pipeline


# ============================
#  RUTAS / UTILIDADES
# ============================

# Root del proyecto: .../<repo_root>/
# Este archivo vive en: .../<repo_root>/app/gui_streamlit.py
ROOT = Path(__file__).resolve().parents[1]

def P(*parts: str) -> str:
    """
    Une partes de ruta relativas al ROOT del repo y devuelve una cadena.
    Ejemplo: P("data", "output", "events.csv")
    """
    return str(ROOT.joinpath(*parts))


# ============================
#  STATE / HELPERS UI
# ============================

def _init_session() -> None:
    """
    Inicializa llaves en session_state si no existen.
    """
    defaults = {
        "uploaded_video_path": None,     # ruta temporal del video subido
        "uploaded_video_hash": None,     # hash del contenido (evita reprocesos accidentales)
        "processed": False,              # ya se analizó (True/False)
        "out_video_path": None,          # ruta del video anotado resultante (ABSOLUTA)
        "events_df": pd.DataFrame(),     # dataframe completo (incluye rutas internas)
        "params": {"imgsz": 640, "conf": 0.35},  # parámetros de inferencia usados
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _save_uploaded_video(uploaded_file) -> Tuple[str, str]:
    """
    Guarda el archivo subido a un archivo temporal y devuelve (path_abs, md5).
    Se calcula un hash para detectar cambios de archivo entre ejecuciones.
    """
    data = uploaded_file.read()
    md5 = hashlib.md5(data).hexdigest()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(data)
        return tmp.name, md5


def _build_df_view_es(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea un DataFrame "visible" en español, SIN rutas internas.
    Columnas esperadas (entrada, en español):
      fecha_hora, tipo_infraccion, tiempo_seg, id_objeto, x1, y1, x2, y2,
      ruta_imagen, ruta_recorte, extra

    - Mapea 'tipo_infraccion' a etiquetas legibles.
    - Renombra encabezados para visualización.
    - Devuelve sólo las columnas relevantes.
    """
    if df.empty:
        return df

    # Mapeo a etiquetas legibles
    map_tipos = {
        "lane_invasion": "Invasión de carril",
        "no_helmet": "Sin casco",
        "red_light": "Paso con luz roja",
        "overspeed": "Exceso de velocidad",
    }

    df = df.copy()

    if "tipo_infraccion" in df.columns:
        df["tipo_infraccion"] = df["tipo_infraccion"].map(
            lambda x: map_tipos.get(str(x), str(x))
        )

    # Renombrar columnas para visualización
    rename_map = {
        "fecha_hora": "Fecha y Hora",
        "tipo_infraccion": "Tipo de infracción",
        "tiempo_seg": "Tiempo (seg)",
        "id_objeto": "ID objeto",
    }
    df.rename(columns=rename_map, inplace=True)

    # Seleccionar columnas visibles (en el orden deseado, si existen)
    cols_visibles = [
        "Fecha y Hora",
        "Tipo de infracción",
        "Tiempo (seg)",
        "ID objeto",
        "x1",
        "y1",
        "x2",
        "y2",
    ]
    return df[[c for c in cols_visibles if c in df.columns]]


# ============================
#  APP
# ============================

st.set_page_config(page_title="Traffic Violations AI", layout="wide")
st.title("Detección de Infracciones de Tránsito")

_init_session()

# --- Panel lateral de diagnóstico (opcional) ---
# with st.sidebar.expander("Diagnóstico", expanded=False):
#     st.write("Root del repo:", f"`{ROOT}`")
#     st.write("Video subido (tmp):", st.session_state.get("uploaded_video_path"))
#     st.write("Video salida:", st.session_state.get("out_video_path"))
#     if st.session_state.get("out_video_path"):
#         try:
#             size = os.path.getsize(st.session_state["out_video_path"])
#             st.write("Tamaño salida (bytes):", size)
#         except Exception:
#             st.write("Tamaño salida (bytes): error al leer")

# Parámetros de inferencia (no disparan procesamiento por sí mismos)
col1, col2 = st.columns(2)
imgsz = col1.slider(
    "Tamaño de imagen YOLO (imgsz)", 416, 960, st.session_state["params"]["imgsz"], 32
)
conf = col2.slider(
    "Confianza mínima detección", 0.1, 0.9, st.session_state["params"]["conf"], 0.05
)

# Subida de video
video_file = st.file_uploader("Sube un video (MP4/MOV/AVI)", type=["mp4", "mov", "avi"])

# Botón de análisis: SOLO aquí se dispara el pipeline
run_clicked = st.button("Analizar video", type="primary", use_container_width=True)

# Config de escena (usa ruta ABSOLUTA)
scene_cfg = P("app", "config", "scenes", "demo_intersection.yaml")

# ============================
#  LÓGICA DE EJECUCIÓN
# ============================

if run_clicked:
    if not video_file:
        st.warning("Primero sube un video para analizar.")
    else:
        # 1) Guardar/actualizar video temporal y hash (para referencia futura)
        in_path, md5 = _save_uploaded_video(video_file)
        st.session_state["uploaded_video_path"] = in_path
        st.session_state["uploaded_video_hash"] = md5

        # 2) Actualizar parámetros actuales en sesión
        st.session_state["params"]["imgsz"] = imgsz
        st.session_state["params"]["conf"] = conf

        # 3) Preparar salida (limpieza se delega al pipeline con clean_previous=True)
        out_dir = P("data", "output", "annotated_videos")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "resultado.mp4")

        # 4) Ejecutar pipeline
        pipe = Pipeline(scene_cfg, yolo_imgsz=imgsz, yolo_conf=conf)
        with st.spinner("Procesando video..."):
            # Ajuste de nombres de parámetros según tu definición real
            
            res = pipe.process_video(in_path, out_path, clean_previous=True)
            st.session_state["processed"]    = True
            st.session_state["out_video_path"] = res.get("out_path_final", out_path)


        # 5) Guardar resultados en sesión
        st.session_state["processed"] = True
        st.session_state["out_video_path"] = out_path

        # 6) Cargar CSV (encabezados en español) desde ruta ABSOLUTA
        csv_path = P("data", "output", "events.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Limpia posibles columnas índice
            df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False)]
        else:
            df = pd.DataFrame()
        st.session_state["events_df"] = df

# ============================
#  RENDER (NO PROCESA)
# ============================

if st.session_state["processed"]:
    # Reproducir el video anotado actual
    
# Reproducción sencilla por ruta
    out_path = st.session_state["out_video_path"]
    if out_path and os.path.exists(out_path) and os.path.getsize(out_path) > 0:

        st.success("¡Análisis completado! Reproduciendo salida…")
        # Leer como BYTES -> más robusto que pasar la ruta
        try:
            st.video(out_path)
        except Exception as e:
            st.info("No se encontró el video anotado.")
    else:
        st.info("No se encontró el video anotado. Vuelve a ejecutar el análisis.")

    # Tabla con encabezados en español (sin rutas)
    df = st.session_state["events_df"]
    df_view = _build_df_view_es(df)
    st.subheader("Eventos detectados")
    st.dataframe(df_view, use_container_width=True)

    # Selección de evento y previsualización de evidencia (NO procesa, sólo lee de disco)
    if not df.empty:
        st.markdown("### Ver evidencia del evento seleccionado")

        map_tipos = {
            "lane_invasion": "Invasión de carril",
            "no_helmet": "Sin casco",
            "red_light": "Paso con luz roja",
            "overspeed": "Exceso de velocidad",
        }

        # Lista legible (índice estable)
        opciones = []
        for idx, row in df.iterrows():
            desc = f"{idx} — {map_tipos.get(str(row.get('tipo_infraccion','?')), str(row.get('tipo_infraccion','?')))} @ {row.get('tiempo_seg','?')}s (ID {row.get('id_objeto','?')})"
            opciones.append(desc)

        sel = st.selectbox("Selecciona un evento", opciones, index=0)
        try:
            selected_idx = int(sel.split("—", 1)[0].strip())
        except Exception:
            selected_idx = 0

        row = None if df.empty else df.iloc[selected_idx]
        if row is not None:
            ruta_recorte = row.get("ruta_recorte", "")
            ruta_imagen = row.get("ruta_imagen", "")

            # Normaliza a rutas ABSOLUTAS si vinieran relativas
            if isinstance(ruta_recorte, str) and ruta_recorte and not os.path.isabs(ruta_recorte):
                ruta_recorte = P(ruta_recorte)
            if isinstance(ruta_imagen, str) and ruta_imagen and not os.path.isabs(ruta_imagen):
                ruta_imagen = P(ruta_imagen)

            # Muestra recorte si existe; si no, el frame completo
            if isinstance(ruta_recorte, str) and ruta_recorte and os.path.exists(ruta_recorte):
                st.image(ruta_recorte, caption="Recorte (evidencia)", use_column_width=True)
            elif isinstance(ruta_imagen, str) and ruta_imagen and os.path.exists(ruta_imagen):
                st.image(ruta_imagen, caption="Frame completo (evidencia)", use_column_width=True)
            else:
                st.info("No se encontró la imagen de evidencia en disco.")

    # Descargar CSV (mantenemos el contenido original; si quieres renombrar encabezados
    # también en el CSV descargado, cambia a df_view y asegúrate de conservar tipos/formatos).
    if not df_view.empty:
        st.download_button(
            "Descargar eventos.csv",
            data=st.session_state["events_df"].to_csv(index=False).encode("utf-8"),
            file_name="eventos.csv",
            mime="text/csv",
            use_container_width=True,
        )
else:
    st.info("Sube un video y presiona **Analizar video** para iniciar el procesamiento.")
