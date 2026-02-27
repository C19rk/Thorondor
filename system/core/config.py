import os
from datetime import datetime

# === DETECTION LOGS ===
# Each session gets its own timestamped .log and .csv inside detection_logs/.
# Format: Feb 23, 2026 07-13-33 AM
_DETECTION_LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "detection_logs")
os.makedirs(_DETECTION_LOGS_DIR, exist_ok=True)

_now = datetime.now()
_ts  = f"{_now.strftime('%b')} {_now.day}, {_now.year} {_now.strftime('%I-%M-%S %p')}"

LOG_FILE = os.path.normpath(os.path.join(_DETECTION_LOGS_DIR, f"detections {_ts}.log"))
CSV_FILE = os.path.normpath(os.path.join(_DETECTION_LOGS_DIR, f"detections {_ts}.csv"))

# === CAMERAS ===
CAMERA_SOURCES = {
    # Format: "rtsp://username:password@ip:554/stream1" (stream1=HD, stream2=low)
    # All cameras must be on the same 2.4 GHz network
    "Camera 1": "rtsp://FINALBOSS:IWILLGRADUATE@192.168.254.110:554/stream1",
    # "Camera 2": "rtsp://FINALBOSS:IWILLGRADUATE@192.168.254.110:554/stream1",
    # "Camera 3": "rtsp://FINALBOSS:IWILLGRADUATE@192.168.254.110:554/stream1",
}

# === RESOLUTION ===
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720

# === MODEL PATHS ===
YOLO_MODEL_PATH      = "../machine_learning/runs/argus_object_detection/weights/best.pt"
YOLO_DESK_MODEL_PATH = "../machine_learning/runs/argus_desk_detection/weights/best.pt"
POSE_MODEL_PATH      = "../machine_learning/runs/pose/argus_pose_estimation/weights/best.pt"

# === DETECTION THRESHOLDS ===
# Lowered from 0.60 → 0.45 for object detection: the model was trained at imgsz=640
# but previously inferred at 320, making far/small objects score lower than they should.
# Now that inference runs at the correct 640, 0.45 gives a good precision/recall balance
# without flooding the screen with false positives.
YOLO_CONF_THRESHOLD      = 0.55

# Desk detection is a large object — keep threshold moderate.
YOLO_DESK_CONF_THRESHOLD = 0.55

# Pose: model is very well trained (mAP50-95=0.926). Keep at 0.45 to catch
# partially visible or far-away persons without sacrificing precision.
POSE_CONF_THRESHOLD      = 0.55

# # === GSM / ALERTS ===
# PHONE_NUMBERS    = ["+639XXXXXXXXX", "+639YYYYYYYYY"]
# ALERT_COOLDOWN   = 10

# SUSPICIOUS_LABELS = ["phone", "smartwatch", "watch", "calculator", "cheating"]