import cv2, threading
from core.config import CAMERA_SOURCES, FRAME_WIDTH, FRAME_HEIGHT

# === Shared frame storage for low latency ===
frames = {name: None for name in CAMERA_SOURCES.keys()}

# === Background camera capture threads ===
def capture_frames(cam_name, src):
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frames[cam_name] = frame

for name, src in CAMERA_SOURCES.items():
    t = threading.Thread(target=capture_frames, args=(name, src), daemon=True)
    t.start()