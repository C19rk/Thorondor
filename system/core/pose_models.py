import os
import numpy as np
from ultralytics import YOLO
from core.config import POSE_MODEL_PATH

_onnx_path  = POSE_MODEL_PATH.replace(".pt", ".onnx")
_model_path = _onnx_path if os.path.exists(_onnx_path) else POSE_MODEL_PATH
MODEL_PATH  = _model_path
_is_onnx    = _model_path.endswith(".onnx")

_mode = "ONNX ✓ fast mode" if _is_onnx else "PyTorch (run export_onnx.py for speedup)"
print(f"[INFO] Pose model   : {os.path.basename(_model_path)} | {_mode} | device: cpu")

pose_model = YOLO(_model_path, task="pose")

if _is_onnx:
    _dummy = np.zeros((256, 256, 3), dtype=np.uint8)
    try:
        pose_model.predict(_dummy, imgsz=256, verbose=False)
        print(f"[INFO] Pose model warmed up")
    except Exception as e:
        print(f"[WARN] Pose model warmup failed: {e}")