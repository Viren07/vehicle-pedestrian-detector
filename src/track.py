"""
Phase 4 — Vehicle and pedestrian tracking with persistent IDs.

Usage:
    python src/track.py --source webcam
    python src/track.py --source path/to/video.mp4
"""

import argparse
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from boxmot.trackers import DeepOcSort
from boxmot.reid.core.reid import ReID

TARGET_CLASSES = {
    0: "person", 2: "car", 5: "bus",
    7: "truck", 1: "bicycle", 3: "motorcycle",
}

def run_tracking(source: str, model_path: str = "runs/detect/models/vehicle_pedestrian_v1/weights/best.pt", conf: float = 0.4):

    model = YOLO(model_path)

    reid = ReID(
    weights=Path(r"C:\Users\viren\AppData\Local\Programs\Python\Python311\Lib\site-packages\models\osnet_x0_25_msmt17.pt"),
    device="cuda:0",
    half=True,
    )

    tracker = DeepOcSort(
        reid_model=reid.model,
        max_age=500,
        min_hits=3,
    )

    cap_source = 0 if source == "webcam" else source
    cap = cv2.VideoCapture(cap_source)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        return

    print(f"[INFO] Tracking on: {source}")
    print("[INFO] Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf, classes=list(TARGET_CLASSES.keys()), verbose=False)

        dets = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            cls = int(box.cls[0])
            dets.append([x1, y1, x2, y2, confidence, cls])

        dets = np.array(dets) if dets else np.empty((0, 6))
        tracks = tracker.update(dets, frame)

        for track in tracks:
            x1, y1, x2, y2, track_id, track_conf, cls, _ = track
            x1, y1, x2, y2, track_id = int(x1), int(y1), int(x2), int(y2), int(track_id)
            label = results[0].names[cls]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ID:{track_id}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Vehicle & Pedestrian Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="webcam")
    parser.add_argument("--model", default="runs/detect/models/vehicle_pedestrian_v1/weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.4)
    args = parser.parse_args()
    run_tracking(args.source, args.model, args.conf)