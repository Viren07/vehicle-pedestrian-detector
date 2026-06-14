# Vehicle & Pedestrian Detector

A full-stack computer vision system for autonomous driving perception. Detects cars, pedestrians, traffic lights, and traffic signs in real time using a YOLOv8 model fine-tuned on 9,900 real driving images from the BDD100K dataset.

Built as a learning project targeting the kind of perception pipelines used at Waymo and Tesla.

---

## Project Phases

**Phase 1 — Detection.** Ran YOLOv8 with pretrained COCO weights on webcam and dashcam video to understand the detection pipeline.

**Phase 2 — Custom Training.** Fine-tuned YOLOv8n on 9,900 BDD100K road images over 50 epochs on an RTX 4070, reaching 49.1% mAP50 across 4 classes (car, pedestrian, traffic light, traffic sign).

**Phase 3 — Full Stack App.** Built a FastAPI REST backend serving the model and a Streamlit web interface that displays detections and annotated images.

**Phase 4 — Object Tracking (experimental).** Added DeepOcSort tracking with OSNet ReID for persistent object IDs. Works while objects are visible but IDs can switch during long occlusions since the ReID model is person-trained rather than vehicle-trained.

**Phase 5 — Video Pipeline.** Full video upload, frame-by-frame processing, in-browser playback and download of annotated videos. Redesigned UI with dark glassmorphism theme.

**Phase 6 — Optimization.** Exported model to ONNX format achieving a 2.6x CPU inference speedup.

**Phase 7 — Documentation.** Learning guide, architecture diagram, demo media.

---

## Quick Start

### Install
```bash
pip install -r requirements.txt
```

### Run the Web App

You need two terminals open at the project root:

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

### Export to ONNX
```bash
python src/export.py
```

---

## Dataset

This project uses the BDD100K dataset hosted on Roboflow — 9,900 dashcam images across day, night, rain, and fog conditions, labeled with bounding boxes for 4 classes.

1. Download version 5 in YOLOv8 format from:
   https://universe.roboflow.com/pedro-azevedo-3c9ol/bdd100k-3zgda/dataset/5
2. Unzip and place the contents into `data/raw/`
3. Ensure `data.yaml` paths read `train/images` and `valid/images`

---

## Model Performance

Fine-tuned YOLOv8n on BDD100K (9,900 images, 50 epochs, RTX 4070)

| Class | mAP50 |
|---|---|
| Car | 66.5% |
| Traffic Sign | 48.1% |
| Traffic Light | 42.2% |
| Pedestrian | 39.7% |

**Overall mAP50: 49.1%** — Precision: 68.5% — Recall: 54.8%

### ONNX Optimization

| Format | ms per frame | FPS |
|---|---|---|
| PyTorch (.pt) | 114ms | ~9 fps |
| ONNX | 44ms | ~23 fps |

**2.6x speedup on CPU** with identical predictions.

### Known Limitations
- Pedestrian detection is weakest due to class imbalance — 7x fewer pedestrian instances than cars in the training data
- Tracking IDs can switch during long occlusions — the ReID model (OSNet) is trained on person re-identification data, not vehicles
- Not real-time on CPU — GPU inference or TensorRT optimization needed for 30+ fps

---

## Tech Stack

Python · PyTorch · YOLOv8 (Ultralytics) · OpenCV · FastAPI · Streamlit · CUDA · Roboflow · BoxMOT · ONNX Runtime · Git

---

## Key Concepts

**IoU** (Intersection over Union) — measures how much a predicted box overlaps with the correct box. Used to decide if a detection counts as correct.

**mAP50** (mean Average Precision at IoU 0.5) — the main evaluation metric. Roughly: how often the model correctly finds and boxes an object.

**NMS** (Non-Maximum Suppression) — removes duplicate boxes when multiple detections overlap the same object.

**Confidence threshold** — minimum score required to show a detection. Lower = more detections, more false positives. Higher = fewer detections, fewer false positives.

**Fine-tuning** — starting from a model already trained on general images (COCO) and retraining it on a specific domain (road scenes) so it specializes without starting from scratch.

**ONNX** — a format for exporting trained models to a leaner inference engine, stripping away training machinery for faster predictions.

---

## Project Structure

```
vehicle-detector/
├── src/
│   ├── detect.py       # Command-line detection on webcam or video
│   ├── track.py        # Object tracking with persistent IDs
│   ├── train.py        # Fine-tune YOLOv8 on custom dataset
│   ├── evaluate.py     # Compute mAP, precision, recall
│   ├── export.py       # Export to ONNX + speed benchmark
│   └── utils.py        # Helper functions
├── app/
│   ├── backend/        # FastAPI REST API
│   └── frontend/       # Streamlit web interface
├── data/
│   └── raw/            # BDD100K dataset (not tracked in git)
├── models/             # Saved weights (not tracked in git)
└── requirements.txt
```


