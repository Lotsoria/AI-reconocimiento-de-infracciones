#!/usr/bin/env bash
# Crea venv e instala requerimientos
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "Entorno listo. Activa con: source .venv/bin/activate"