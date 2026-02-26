import os
import numpy as np
from ultralytics import YOLO
from core.config import YOLO_DESK_MODEL_PATH

_onnx_path  = YOLO_DESK_MODEL_PATH.replace(".pt", ".onnx")
_model_path = _onnx_path if os.path.exists(_onnx_path) else YOLO_DESK_MODEL_PATH
MODEL_PATH  = _model_path
_is_onnx    = _model_path.endswith(".onnx")

print(f"[INFO] Desk model   : {os.path.basename(_model_path)} | device: cpu")

yolo_desk = YOLO(_model_path, task="detect")

if _is_onnx:
    _dummy = np.zeros((320, 320, 3), dtype=np.uint8)
    try:
        yolo_desk.predict(_dummy, imgsz=320, verbose=False)
        print(f"[INFO] Desk model warmed up")
    except Exception as e:
        print(f"[WARN] Desk model warmup failed: {e}")