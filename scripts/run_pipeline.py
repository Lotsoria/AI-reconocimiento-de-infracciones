#!/usr/bin/env python
"""Ejecución por CLI del pipeline con parámetros simples.

Uso:
  python scripts/run_pipeline.py --input data/samples/video.mp4 \
      --scene app/config/scenes/demo_intersection.yaml \
      --output data/output/annotated_videos/resultado.mp4

Env vars opcionales (para el modelo de casco):
  HELMET_MODEL_URL, HELMET_MODEL_PATH
"""

import argparse
from core.pipeline import Pipeline


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='Ruta del video de entrada')
    p.add_argument('--output', required=True, help='Ruta del video de salida deseada (.mp4 recomendado)')
    p.add_argument('--scene', default='app/config/scenes/demo_intersection.yaml')
    args = p.parse_args()

    pipe = Pipeline(args.scene)
    res = pipe.process_video(args.input, args.output, clean_previous=True)
    df = res.get('events_df')
    print('OK. Salida:', res.get('out_path_final'))
    print('Eventos detectados:', 0 if df is None else len(df))


if __name__ == '__main__':
    main()

