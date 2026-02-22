import cv2
import threading
import time
from collections import deque
from core.config import CAMERA_SOURCES, FRAME_WIDTH, FRAME_HEIGHT

# === Shared frame storage — maxlen=1 keeps only the freshest frame ===
frames = {name: deque(maxlen=1) for name in CAMERA_SOURCES.keys()}

# ─────────────────────────────────────────────────────────────────────────────
# TAPO CONCURRENT CONNECTION LIMIT:
#   Tapo cameras allow only 1–2 simultaneous RTSP connections per device.
#   If all 3 CAMERA_SOURCES point to the same IP, the 3rd (and possibly 2nd)
#   connection will fail silently — that camera's frames deque stays empty,
#   latest_annotated never gets a key for it, and the recorder saves 0 frames
#   then deletes the file as "empty". This is what happened to Camera 2.
#
#   Fix: use 3 physically separate Tapo cameras at different IPs in config.py.
#   For single-camera testing, reduce CAMERA_SOURCES to 1 entry.
# ─────────────────────────────────────────────────────────────────────────────

# Extra grab() calls to flush stale RTSP buffer before retrieve().
# 2 works for stream2 (15fps). Use 3 for stream1 (25fps).
DRAIN_GRABS = 3


def _open_rtsp(src: str) -> cv2.VideoCapture:
    """Open RTSP with low-latency FFmpeg settings."""
    cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    return cap


def capture_frames(cam_name, src):
    cap = _open_rtsp(src)

    if not cap.isOpened():
        print(
            f"[ERROR] '{cam_name}' could not open RTSP stream ({src}).\n"
            f"        If Camera 1 works but Camera 2/3 fail on the same IP,\n"
            f"        the Tapo camera has hit its max concurrent RTSP connection limit.\n"
            f"        Use separate physical cameras at different IPs."
        )
        return

    stream_fps = cap.get(cv2.CAP_PROP_FPS)
    if stream_fps <= 0 or stream_fps > 60:
        stream_fps = 15.0
    frame_interval = 1.0 / stream_fps
    print(f"[INFO] Capture started: '{cam_name}' @ {stream_fps:.1f} fps (drain-grab active)")

    consecutive_failures = 0

    while True:
        t0 = time.perf_counter()

        # Drain stale buffered RTSP frames cheaply (no decode) before retrieve
        for _ in range(DRAIN_GRABS):
            cap.grab()

        ret, frame = cap.retrieve()

        if not ret:
            consecutive_failures += 1
            if consecutive_failures > 20:
                print(f"[WARN] Stream lost for '{cam_name}'. Reconnecting...")
                cap.release()
                time.sleep(2)
                cap = _open_rtsp(src)
                if not cap.isOpened():
                    print(f"[ERROR] Reconnect failed for '{cam_name}'. Retrying in 5s...")
                    time.sleep(5)
                    cap = _open_rtsp(src)
                consecutive_failures = 0
            time.sleep(0.05)
            continue

        consecutive_failures = 0
        frames[cam_name].append(frame)  # deque owns frame; AI threads read-only

        elapsed = time.perf_counter() - t0
        sleep_t = frame_interval - elapsed
        if sleep_t > 0.001:
            time.sleep(sleep_t)


# === Background camera capture threads ===
# Stagger each start by 500ms to prevent simultaneous TCP handshakes
# overwhelming the Tapo camera and causing the 2nd/3rd connection to be refused.
for i, (name, src) in enumerate(CAMERA_SOURCES.items()):
    if i > 0:
        time.sleep(0.5)
    t = threading.Thread(target=capture_frames, args=(name, src), daemon=True)
    t.start()
    print(f"[INFO] Capture thread launched for '{name}'")