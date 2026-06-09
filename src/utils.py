"""Helper functions."""

import cv2, numpy as np

CLASS_COLORS = {
    "person": (255,128,0), "car": (64,200,64),
    "truck": (0,128,255),  "bus": (255,0,128),
    "bicycle": (255,255,0), "motorcycle": (128,0,255),
}

def draw_boxes(frame, boxes, class_names):
    for box in boxes:
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        conf  = float(box.conf[0])
        label = class_names[int(box.cls[0])]
        color = CLASS_COLORS.get(label, (200,200,200))
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return frame

def count_classes(results):
    counts = {}
    for box in results[0].boxes:
        cls = results[0].names[int(box.cls[0])]
        counts[cls] = counts.get(cls, 0) + 1
    return counts
