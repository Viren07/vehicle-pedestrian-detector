"""
Phase 2 — Evaluate model. Outputs mAP, precision, recall.

Usage:
    python src/evaluate.py --weights models/.../best.pt --data data/processed/data.yaml
"""

import argparse
from ultralytics import YOLO

def evaluate(weights, data_yaml, imgsz=640):
    model = YOLO(weights)
    metrics = model.val(data=data_yaml, imgsz=imgsz, conf=0.25, iou=0.6)
    print("\n===== Evaluation =====")
    print(f"mAP@50:    {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data",    required=True)
    args = parser.parse_args()
    evaluate(args.weights, args.data)
