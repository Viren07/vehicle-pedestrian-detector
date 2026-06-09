"""
Phase 2 — Fine-tune YOLOv8 on BDD100K or Roboflow dataset.

Usage:
    python src/train.py --data data/processed/data.yaml --epochs 50
"""

import argparse
from ultralytics import YOLO

def train(data_yaml, model_path="yolov8n.pt", epochs=50, imgsz=640):
    model = YOLO(model_path)
    model.train(
        data=data_yaml, epochs=epochs, imgsz=imgsz,
        batch=16, name="vehicle_pedestrian_v1", project="models",
        exist_ok=True, lr0=0.01, lrf=0.01, flipud=0.0, fliplr=0.5,
    )
    print("\n[INFO] Best weights -> models/vehicle_pedestrian_v1/weights/best.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   required=True)
    parser.add_argument("--model",  default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    train(args.data, args.model, args.epochs)
