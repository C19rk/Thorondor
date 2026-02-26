import os
# ── CRITICAL: Set BEFORE any numpy/cv2/onnxruntime imports ───────────────────
os.environ["YOLO_AUTOINSTALL"] = "False"  # Prevent Ultralytics stomping onnxruntime-directml
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS",  "1")
os.environ.setdefault("OMP_WAIT_POLICY",      "PASSIVE")
# ─────────────────────────────────────────────────────────────────────────────

import asyncio
import sys
import threading
import time
import atexit
import secrets
from contextlib import asynccontextmanager
from collections import deque

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from core.vision import generate_frames, latest_annotated, latest_raw
from core.config import FRAME_HEIGHT, FRAME_WIDTH, LOG_FILE, CSV_FILE
from core.recorders import init_recorders
from core.routes import register_routes
from core.auth import init_db

# ─────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────
CAMERA_SOURCES = {}
frames         = {}
recorder       = None
log_recorder   = None
_cam_backend   = 0


# ─────────────────────────────────────────────
# Camera probe
# ─────────────────────────────────────────────
def _find_working_camera():
    backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, 0] if sys.platform == "win32" else [0]
    for idx in range(6):
        for backend in backends:
            try:
                cap = (cv2.VideoCapture(idx, backend)
                       if backend != 0 else cv2.VideoCapture(idx))
                if not cap.isOpened():
                    cap.release(); continue
                ret, _ = cap.read()
                cap.release()
                if ret:
                    print(f"[INFO] Camera found: index={idx}, backend={backend}")
                    return idx, backend
            except Exception:
                pass
    print("[WARN] No working camera found — defaulting index=0")
    return 0, 0

def _open_cap(src, backend):
    if backend != 0:
        cap = cv2.VideoCapture(src, backend)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(src)


# ─────────────────────────────────────────────
# Capture thread
# ─────────────────────────────────────────────
def capture_frames(cam_name, src, backend, fps):
    cap = _open_cap(src, backend)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera resolution: {actual_w}x{actual_h} (requested {FRAME_WIDTH}x{FRAME_HEIGHT})")

    if not cap.isOpened():
        print(f"[ERROR] Could not open webcam: {src}")
        return

    frame_interval = 1.0 / max(fps, 1.0)
    consecutive_failures = 0

    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            consecutive_failures += 1
            if consecutive_failures > 30:
                print(f"[WARN] Webcam {cam_name} failing — re-probing...")
                cap.release()
                time.sleep(1)
                new_idx, new_backend = _find_working_camera()
                cap = _open_cap(new_idx, new_backend)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                consecutive_failures = 0
            time.sleep(0.005)
            continue
        consecutive_failures = 0
        frames[cam_name].append(frame)

        elapsed = time.perf_counter() - t0
        sleep_t = frame_interval - elapsed
        if sleep_t > 0.002:
            time.sleep(sleep_t)


# ─────────────────────────────────────────────
# AI processing loop
# ─────────────────────────────────────────────
def ai_processing_loop(cam_name):
    for _ in generate_frames(cam_name, frames_override=frames, recorder=recorder):
        pass


# ─────────────────────────────────────────────
# MJPEG frame generator
# ─────────────────────────────────────────────
MJPEG_JPEG_QUALITY = 65

def mjpeg_generator(cam_name):
    last_id     = None
    blank       = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    cv2.putText(blank, "Waiting for camera...",
                (30, FRAME_HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)
    _, blank_jpg = cv2.imencode(".jpg", blank, [cv2.IMWRITE_JPEG_QUALITY, MJPEG_JPEG_QUALITY])
    blank_bytes  = blank_jpg.tobytes()

    while True:
        frame = latest_annotated.get(cam_name)

        if frame is None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                blank_bytes +
                b"\r\n"
            )
            time.sleep(0.033)
            continue

        fid = id(frame)
        if fid == last_id:
            time.sleep(0.001)
            continue

        last_id = fid
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, MJPEG_JPEG_QUALITY])
        payload  = jpg.tobytes() if ok else blank_bytes

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            payload +
            b"\r\n"
        )


# ─────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global CAMERA_SOURCES, frames, recorder, log_recorder, _cam_backend

    # Detection logs are preserved — each session writes to its own
    # timestamped file in detection_logs/ (see core/config.py).

    cam_idx, _cam_backend = _find_working_camera()
    CAMERA_SOURCES = {"cam1": cam_idx}
    frames = {"cam1": deque(maxlen=1)}

    cap_init = _open_cap(cam_idx, _cam_backend)
    actual_fps = cap_init.get(cv2.CAP_PROP_FPS)
    if actual_fps <= 0 or actual_fps > 120:
        actual_fps = 30.0
    cap_init.release()
    recorder, log_recorder = init_recorders(fps=actual_fps)

    # Tell the recorder where to find raw webcam frames.
    # This avoids importing core.cameras (which would spin up Tapo RTSP
    # capture threads even in webcam-only mode).
    recorder._frames_source = frames

    # Start capture thread
    threading.Thread(
        target=capture_frames,
        args=("cam1", cam_idx, _cam_backend, actual_fps),
        daemon=True
    ).start()

    # Wait up to 5s for first frame
    for _ in range(50):
        if len(frames.get("cam1", [])) > 0:
            break
        await asyncio.sleep(0.1)

    # Pre-warm AI worker
    warm_done = threading.Event()
    def _warm():
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        try:
            from core.detections.pose   import predict as pp
            from core.detections.object import predict as op
            from core.detections.desk   import predict as dp
            pp(dummy, "cam1"); op(dummy, "cam1"); dp(dummy, "cam1")
        except Exception as e:
            print(f"[WARN] Warmup error: {e}")
        finally:
            warm_done.set()
    threading.Thread(target=_warm, daemon=True).start()
    warm_done.wait(timeout=10)

    # Start AI loop
    threading.Thread(
        target=ai_processing_loop, args=("cam1",), daemon=True
    ).start()

    # Register routes
    register_routes(app, recorder, log_recorder, generate_frames, frames,
                    CAMERA_SOURCES, None, LOG_FILE, follow,
                    templates=templates, template_name="wcapp.html")

    # MJPEG video feed route
    @app.get("/video_feed")
    async def video_feed():
        return StreamingResponse(
            mjpeg_generator("cam1"),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )

    from core.auth import DB_PATH as _db_path
    import os as _os
    print("\n" + "─" * 52)
    print("  Argus Webcam is running!")
    print("  App Link: http://localhost:5001/login")
    print("  Database: http://localhost:5001/admin")
    print("  DB file: " + _os.path.abspath(_db_path))
    print("─" * 52 + "\n")

    yield


# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────
init_db()  # Create users.db if it does not exist

app = FastAPI(lifespan=lifespan)

# Fresh random key every run — all sessions wiped when server stops
app.add_middleware(SessionMiddleware, secret_key=secrets.token_hex(32))

app.mount("/static", StaticFiles(directory="screens/static"), name="static")
templates = Jinja2Templates(directory="screens")


# ─────────────────────────────────────────────
# Log streaming
# ─────────────────────────────────────────────
def follow(logfile):
    logfile.seek(0, 2)
    while True:
        line = logfile.readline()
        if not line:
            time.sleep(0.1)
            continue
        if log_recorder and log_recorder.recording:
            log_recorder.write(line.strip())
        yield f"data: {line}\n\n"


if __name__ == "__main__":
    uvicorn.run(
        "wcapp:app",
        host="0.0.0.0",
        port=5001,
        log_level="warning",
        reload=False
    )