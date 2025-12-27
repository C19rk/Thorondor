# === LOGGING/CSV ===
LOG_FILE = "detections.log"
CSV_FILE = "detections.csv"

# === CAMERAS ===
CAMERA_SOURCES = {
    # "cam1: "rtsp://username:password@tapo_ip_address:554/stream1", (strem 1 for hd 2 for low)"
    # Connect to the same 2.4 GHz network as the camera
    "cam1": "rtsp://camera1:camera1234@192.168.254.112:554/stream2",
    "cam2": "rtsp://camera1:camera1234@192.168.254.112:554/stream2",
    "cam3": "rtsp://camera1:camera1234@192.168.254.112:554/stream2",
}

# --- resolution settings ---
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# === MODELS ===
YOLO_MODEL_PATH = "../App/runs/aidetection7/weights/best.pt"
YOLO_DESK_MODEL_PATH = "../App/runs/aidetectiondesk2/weights/best.pt"
POSE_MODEL_PATH = "../App/yolo11n-pose.pt"

# === THRESHOLDS ===
YOLO_CONF_THRESHOLD = 0.75
YOLO_DESK_CONF_THRESHOLD = 0.75
POSE_CONF_THRESHOLD = 0.25

# == POSE HEAD DETECTION THRESHOLDS ===
H_THRESH = 0.30   # left/right sensitivity
DOWN_THRESH = 0.45
UP_THRESH = -0.35

# === GSM ===
PHONE_NUMBERS = ["+639XXXXXXXXX", "+639YYYYYYYYY"]  # add your numbers here
ALERT_COOLDOWN = 10  # seconds
SUSPICIOUS_LABELS = ["phone","talking"]  # labels that trigger SMS