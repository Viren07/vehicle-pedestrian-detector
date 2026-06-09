"""
Phase 1 — Run YOLOv8 detection on webcam or video file.

Usage:
    python src/detect.py --source webcam
    python src/detect.py --source path/to/video.mp4
"""

import argparse
import cv2
from ultralytics import YOLO

TARGET_CLASSES = {
    0: "person", 2: "car", 5: "bus",
    7: "truck",  1: "bicycle", 3: "motorcycle",
}

def run_detection(source: str, model_path: str = "yolov8n.pt", conf: float = 0.4):
    model = YOLO(model_path)
    cap_source = 0 if source == "webcam" else source
    cap = cv2.VideoCapture(cap_source)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        return

    print(f"[INFO] Running on: {source} | Model: {model_path} | Conf: {conf}")
    print("[INFO] Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf, classes=list(TARGET_CLASSES.keys()), verbose=False)
        annotated = results[0].plot()

        n = len(results[0].boxes)
        cv2.putText(annotated, f"Detections: {n}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Vehicle & Pedestrian Detector", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="webcam")
    parser.add_argument("--model",  default="yolov8n.pt")
    parser.add_argument("--conf",   type=float, default=0.4)
    args = parser.parse_args()
    run_detection(args.source, args.model, args.conf)
