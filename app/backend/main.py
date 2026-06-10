"""
Phase 3 — FastAPI backend.
Run: uvicorn app.backend.main:app --reload
"""
import base64
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tempfile, shutil, os
from ultralytics import YOLO

app   = FastAPI(title="Vehicle Detector API")
model = YOLO("runs/detect/models/vehicle_pedestrian_v1/weights/best.pt")

@app.get("/")
def root():
    return {"status": "running", "model": "yolov8n"}

@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    results = model(tmp_path, conf=0.4, verbose=False)
    annotated_frame = results[0].plot()
    _, buffer = cv2.imencode('.jpg', annotated_frame)
    annotated_b64 = base64.b64encode(buffer).decode('utf-8')
    os.unlink(tmp_path)
    detections = [
        {"class": results[0].names[int(b.cls[0])],
         "confidence": round(float(b.conf[0]), 3),
         "bbox": [round(v,1) for v in b.xyxy[0].tolist()]}
        for b in results[0].boxes
    ]
    return JSONResponse({
    "detections": detections,
    "count": len(detections),
    "annotated_image": annotated_b64
})
