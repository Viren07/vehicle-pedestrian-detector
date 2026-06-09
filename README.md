# Vehicle & Pedestrian Detector

Real-time object detection for autonomous driving perception using YOLOv8.

## Quick Start

### Install
```bash
pip install -r requirements.txt
```

### Phase 1 — Run on webcam
```bash
python src/detect.py --source webcam
```

### Phase 1 — Run on a video file
```bash
python src/detect.py --source data/raw/dashcam.mp4
```

### Phase 2 — Fine-tune on BDD100K
```bash
python src/train.py --data data/processed/data.yaml --epochs 50
```

### Phase 2 — Evaluate
```bash
python src/evaluate.py --weights models/vehicle_pedestrian_v1/weights/best.pt --data data/processed/data.yaml
```

### Phase 3 — Run the app
```bash
uvicorn app.backend.main:app --reload   # terminal 1
streamlit run app/frontend/streamlit_app.py  # terminal 2
```

## Key Concepts to Learn
- **IoU** (Intersection over Union): how much two boxes overlap
- **mAP** (mean Average Precision): main evaluation metric
- **NMS** (Non-Maximum Suppression): removes duplicate detections
- **Confidence threshold**: minimum score to show a detection


## In order to run the project the required dataset can be found:

- https://universe.roboflow.com/pedro-azevedo-3c9ol/bdd100k-3zgda/dataset/5
