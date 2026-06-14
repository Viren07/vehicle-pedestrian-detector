# Vehicle & Pedestrian Detector

Real-time object detection for autonomous driving perception using YOLOv8.

## Quick Start

### Install
```bash
pip install -r requirements.txt
```

### Run on webcam
```bash
python src/detect.py --source webcam
```

### Run the Web App

You need **two terminals**, both at the project root:

**Terminal 1 — Backend (FastAPI)**
```bash
uvicorn app.backend.main:app --reload
```
Runs at: http://127.0.0.1:8000 — interactive API docs at http://127.0.0.1:8000/docs

**Terminal 2 — Frontend (Streamlit)**
```bash
streamlit run app/frontend/streamlit_app.py
```
Runs at: http://localhost:8501

### Run Detection from Command Line
```bash
python src/detect.py --source webcam
python src/detect.py --source path/to/video.mp4
```

### Run Tracking (experimental)
```bash
python src/track.py --source path/to/video.mp4
```

### Train and Evaluate
```bash
python src/train.py --data data/raw/data.yaml --epochs 50
python src/evaluate.py --weights runs/detect/models/vehicle_pedestrian_v1/weights/best.pt --data data/raw/data.yaml
```

## Key Concepts to Learn
- **IoU** (Intersection over Union): how much two boxes overlap
- **mAP** (mean Average Precision): main evaluation metric
- **NMS** (Non-Maximum Suppression): removes duplicate detections
- **Confidence threshold**: minimum score to show a detection


## Dataset

This project uses the BDD100K dataset hosted on Roboflow.

1. Download version 5 in YOLOv8 format from:
   https://universe.roboflow.com/pedro-azevedo-3c9ol/bdd100k-3zgda/dataset/5
2. Unzip and place the contents into `data/raw/`
3. Keep `data.yaml` paths as `train/images` and `valid/images`


## Model Performance

Fine-tuned YOLOv8n on BDD100K dataset (9,900 images)

| Class | mAP50 |
|---|---|
| Car | 66.5% |
| Traffic Sign | 48.1% |
| Traffic Light | 42.2% |
| Pedestrian | 39.7% |

**Overall mAP50: 49.1%**

## ONNX Optimization

Exported model to ONNX format for faster CPU inference.

| Format | ms per frame | FPS |
|---|---|---|
| PyTorch (.pt) | 114ms | ~9 fps |
| ONNX | 44ms | ~23 fps |

**2.6x speedup on CPU** — same weights, same predictions, leaner inference engine.

### Known Limitations
- Pedestrian detection is weakest due to class imbalance (7x fewer pedestrian instances than cars)
- Tracking IDs can switch during long occlusions — ReID model is person-trained, not vehicle-trained

## Key Concepts to Learn
- **IoU** (Intersection over Union): how much two boxes overlap
- **mAP** (mean Average Precision): main evaluation metric
- **NMS** (Non-Maximum Suppression): removes duplicate detections
- **Confidence threshold**: minimum score to show a detection


## Project Phases

This project was built incrementally in phases:

**Phase 1 — Detection.** Ran YOLOv8 with pretrained COCO weights on webcam and dashcam video.

**Phase 2 — Custom Training.** Fine-tuned YOLOv8n on 9,900 BDD100K road images over 50 epochs on an RTX 4070, reaching 49.1% mAP50 across 4 classes (car, pedestrian, traffic light, traffic sign).

**Phase 3 — Full Stack App.** Built a FastAPI REST backend serving the model and a Streamlit web interface that displays detections and annotated images.

**Phase 4 — Object Tracking (experimental).** Added DeepOcSort tracking with OSNet ReID for persistent object IDs. Works while objects are visible, but IDs can switch during long occlusions since the ReID model is person-trained rather than vehicle-trained.

**Phase 5 — Video Pipeline.** Full video upload, frame-by-frame processing, in-browser playback and download of annotated videos. Redesigned UI with dark theme.

**Phase 6 — Optimization.** ONNX export for faster inference. In progress.

**Phase 7 — Documentation.** Learning guide, architecture diagram, demo media. Planned.


