from ultralytics import YOLO
from core.config import POSE_MODEL_PATH

pose_model = YOLO(POSE_MODEL_PATH)