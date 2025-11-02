# scripts/download_models.py
# Descarga/coloca modelos en /models. Para YOLO base no hace falta (Ultralytics descarga solo).
# Para casco: coloca manualmente tu 'helmet_yolo.pt' en models/helmet/.
import os, shutil

os.makedirs("models/yolo", exist_ok=True)
os.makedirs("models/helmet", exist_ok=True)
os.makedirs("models/lanes", exist_ok=True)

print("Para YOLO base no es necesario descargar manualmente (se descarga al usar).")
print("Coloca tu modelo de casco en: models/helmet/helmet_yolo.pt")
print("Si luego integras UFLD, coloca pesos en: models/lanes/ufld_tusimple_res18.pth")