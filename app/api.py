# app/api.py
# API mínima para procesar un video desde ruta (despliegue opcional)
from fastapi import FastAPI
from pydantic import BaseModel
from core.pipeline import Pipeline

app = FastAPI(title="Traffic Violations API")

class ProcessRequest(BaseModel):
    input_path: str
    output_path: str
    scene_config: str = "app/config/scenes/demo_intersection.yaml"

@app.post("/process")
def process(req: ProcessRequest):
    pipe = Pipeline(req.scene_config)
    res = pipe.process_video(req.input_path, req.output_path, False)
    return {"ok": True, "events_count": len(res["events"]), "events_path": "data/output/events.csv"}
