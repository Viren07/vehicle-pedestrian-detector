"""
Phase 3 — FastAPI backend.
Run: uvicorn app.backend.main:app --reload
"""

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tempfile, shutil, os
from ultralytics import YOLO

app   = FastAPI(title="Vehicle Detector API")
model = YOLO("yolov8n.pt")

@app.get("/")
def root():
    return {"status": "running", "model": "yolov8n"}

@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    results = model(tmp_path, conf=0.4, verbose=False)
    os.unlink(tmp_path)
    detections = [
        {"class": results[0].names[int(b.cls[0])],
         "confidence": round(float(b.conf[0]), 3),
         "bbox": [round(v,1) for v in b.xyxy[0].tolist()]}
        for b in results[0].boxes
    ]
    return JSONResponse({"detections": detections, "count": len(detections)})
