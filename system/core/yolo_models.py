import os
import numpy as np
from ultralytics import YOLO
from core.config import YOLO_MODEL_PATH

_onnx_path  = YOLO_MODEL_PATH.replace(".pt", ".onnx")
_model_path = _onnx_path if os.path.exists(_onnx_path) else YOLO_MODEL_PATH
MODEL_PATH  = _model_path
_is_onnx    = _model_path.endswith(".onnx")

print(f"[INFO] Object model : {os.path.basename(_model_path)} | device: cpu")

yolo = YOLO(_model_path, task="detect")

if _is_onnx:
    _dummy = np.zeros((320, 320, 3), dtype=np.uint8)
    try:
        yolo.predict(_dummy, imgsz=320, verbose=False)
        print(f"[INFO] Object model warmed up")
    except Exception as e:
        print(f"[WARN] Object model warmup failed: {e}")