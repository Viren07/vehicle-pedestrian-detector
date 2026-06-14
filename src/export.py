"""
Phase 6 — Export trained YOLOv8 model to ONNX format for faster inference.

Usage:
    python src/export.py
"""

from ultralytics import YOLO

def export_to_onnx(
    model_path: str = "runs/detect/models/vehicle_pedestrian_v1/weights/best.pt",
    imgsz: int = 640,
):
    model = YOLO(model_path)

    model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=False,
        simplify=True,
    )

    print(f"[INFO] ONNX model saved alongside: {model_path}")
    print("[INFO] Look for best.onnx in the same weights folder")

def benchmark_speed(
    model_pt: str = "runs/detect/models/vehicle_pedestrian_v1/weights/best.pt",
    model_onnx: str = "runs/detect/models/vehicle_pedestrian_v1/weights/best.onnx",
    runs: int = 50,
):
    import time
    import numpy as np
    import cv2

    dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    pt_model = YOLO(model_pt)
    pt_model(dummy, verbose=False, device="cpu")
    start = time.perf_counter()
    for _ in range(runs):
        pt_model(dummy, verbose=False, device="cpu")
    pt_time = (time.perf_counter() - start) / runs * 1000

    onnx_model = YOLO(model_onnx)
    onnx_model(dummy, verbose=False, device="cpu")
    start = time.perf_counter()
    for _ in range(runs):
        onnx_model(dummy, verbose=False, device="cpu")
    onnx_time = (time.perf_counter() - start) / runs * 1000

    print(f"\n===== Speed Benchmark ({runs} runs) =====")
    print(f"PyTorch (.pt):  {pt_time:.1f} ms per frame")
    print(f"ONNX:           {onnx_time:.1f} ms per frame")
    print(f"Speedup:        {pt_time/onnx_time:.2f}x")
    print("==========================================\n")


if __name__ == "__main__":
    export_to_onnx()
    benchmark_speed()


