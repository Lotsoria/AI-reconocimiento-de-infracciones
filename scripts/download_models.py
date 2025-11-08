#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Descarga modelos necesarios para detección de infracciones (casco en moto).
- models/yolo/yolo11n.pt  -> Detector genérico (COCO) de Ultralytics YOLO11.
- models/helmet/helmet_best.pt -> Detector específico casco/no-casco (YOLOv8).

Fuentes:
- YOLO11 docs y pesos (Ultralytics): https://docs.ultralytics.com/models/yolo11/  # [1](https://docs.ultralytics.com/models/yolo11/)
- YOLO11 en Hugging Face (pesos): https://huggingface.co/Ultralytics/YOLO11      # [2](https://huggingface.co/Ultralytics/YOLO11)
- Ultralytics assets releases (mirrors): https://github.com/ultralytics/assets/releases  # [5](https://github.com/ultralytics/assets/releases)
- Modelo casco YOLOv8 (best.pt) en HF: https://huggingface.co/sharathhhhh/safetyHelmet-detection-yolov8  # [3](https://huggingface.co/sharathhhhh/safetyHelmet-detection-yolov8/blob/main/best.pt)[4](https://huggingface.co/sharathhhhh/safetyHelmet-detection-yolov8/blob/main/README.md)
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
HELMET_DIR = MODELS_DIR / "helmet"
YOLO_DIR = MODELS_DIR / "yolo"

HELMET_DIR.mkdir(parents=True, exist_ok=True)
YOLO_DIR.mkdir(parents=True, exist_ok=True)

# Config de descargas
FILES = [
    {
        "name": "yolo11n.pt",
        "dest": YOLO_DIR / "yolo11n.pt",
        # SHA256 publicado para el artefacto de YOLO11n (mirror HF). [6](https://huggingface.co/Ultralytics/YOLO11/blob/a01aaa06caeff788b052e193acb76b3f21571b3a/yolo11n.pt)
        "sha256": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
        "urls": [
            # Hugging Face (Ultralytics/YOLO11)
            "https://huggingface.co/Ultralytics/YOLO11/resolve/main/yolo11n.pt?download=true",  # [2](https://huggingface.co/Ultralytics/YOLO11)
            # Mirror (Ultralytics assets releases – ruta típica; si falla, se intentará el anterior)
            "https://github.com/ultralytics/assets/releases/download/v11/yolo11n.pt",  # [5](https://github.com/ultralytics/assets/releases)
        ],
    },
    {
        "name": "helmet_best.pt",
        "dest": HELMET_DIR / "helmet_best.pt",
        # SHA256 del archivo best.pt publicado en Hugging Face (ver ficha). [3](https://huggingface.co/sharathhhhh/safetyHelmet-detection-yolov8/blob/main/best.pt)
        "sha256": "06297f6c2d27bd157297866e526710e8ffc06b5c04da28ab77db949e805c141c",
        "urls": [
            # Hugging Face – modelo casco/no-casco (YOLOv8)
            "https://huggingface.co/sharathhhhh/safetyHelmet-detection-yolov8/resolve/main/best.pt?download=true",  # [3](https://huggingface.co/sharathhhhh/safetyHelmet-detection-yolov8/blob/main/best.pt)
        ],
    },
]

CHUNK = 1 << 20  # 1 MiB


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def download_with_resume(url: str, dest: Path):
    tmp = dest.with_suffix(f"{dest.suffix}.part")
    existing = tmp.stat().st_size if tmp.exists() else 0

    req = urllib.request.Request(url)
    if existing > 0:
        req.add_header("Range", f"bytes={existing}-")

    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            mode = "ab" if existing > 0 and resp.getcode() in (206, 200) else "wb"
            with open(tmp, mode) as out:
                while True:
                    if chunk := resp.read(CHUNK):
                        out.write(chunk)
                    else:
                        break
    except Exception as e:
        return False, f"{e}"

    tmp.rename(dest)
    return True, None


def ensure_file(file_def) -> bool:
    dest = file_def["dest"]
    # Si existe, validar hash (si lo tenemos)
    if dest.exists():
        if file_def.get("sha256"):
            calc = sha256sum(dest)
            if calc == file_def["sha256"]:
                print(f"✓ {dest} ya existe (SHA256 OK)")
                return True
            else:
                print(f"! {dest} hash inválido; re-descargando...")
                dest.unlink(missing_ok=True)
        else:
            print(f"✓ {dest} ya existe")
            return True

    for i, url in enumerate(file_def["urls"], 1):
        print(f"Descargando {file_def['name']} (intento {i}/{len(file_def['urls'])})\n  URL: {url}")
        ok, err = download_with_resume(url, dest)
        if not ok:
            print(f"  Error: {err}")
            continue

        if file_def.get("sha256"):
            calc = sha256sum(dest)
            if calc != file_def["sha256"]:
                print(f"  Hash incorrecto (esperado {file_def['sha256']}, obtenido {calc}). Probando otro mirror…")
                dest.unlink(missing_ok=True)
                continue

        print(f"  ✓ Guardado en {dest}")
        return True

    print(f"✗ No se pudo descargar {file_def['name']}")
    return False


def main():
    all_ok = True
    for f in FILES:
        ok = ensure_file(f)
        all_ok = all_ok and ok
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()