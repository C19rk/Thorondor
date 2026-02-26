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

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from core.vision import generate_frames, latest_annotated
from core.cameras import frames
from core.config import FRAME_HEIGHT, FRAME_WIDTH, LOG_FILE, CSV_FILE, CAMERA_SOURCES
from core.recorders import init_recorders
from core.routes import register_routes
from core.auth import init_db, DB_PATH

recorder     = None
log_recorder = None


def ai_processing_loop(cam_name):
    gen = generate_frames(cam_name, frames_override=frames, recorder=recorder)
    while True:
        try:
            next(gen)
        except StopIteration:
            break
        except Exception as e:
            print(f"AI Loop Error ({cam_name}): {e}")
            time.sleep(1)


MJPEG_JPEG_QUALITY = 65

def mjpeg_generator(cam_name):
    last_id     = None
    blank       = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    cv2.putText(blank, f"Connecting to {cam_name}...",
                (50, FRAME_HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2)
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global recorder, log_recorder

    from core.cameras import detected_fps as _cam_fps
    first_cam = list(CAMERA_SOURCES.keys())[0]
    for _ in range(100):
        if first_cam in _cam_fps:
            break
        await asyncio.sleep(0.1)

    actual_fps = _cam_fps.get(first_cam, 25.0)
    if actual_fps <= 0 or actual_fps > 120:
        actual_fps = 25.0
        print(f"[WARN] Could not read FPS from cameras — defaulting to {actual_fps} fps")
    else:
        print(f"[INFO] Tapo cam FPS (from capture thread): {actual_fps:.2f}")
    recorder, log_recorder = init_recorders(fps=actual_fps)

    warm_events = []
    for cam_name in CAMERA_SOURCES:
        ev = threading.Event()
        warm_events.append(ev)
        def _warm(cn=cam_name, done=ev):
            dummy = np.zeros((320, 320, 3), dtype=np.uint8)
            try:
                from core.detections.pose   import predict as pp
                from core.detections.object import predict as op
                from core.detections.desk   import predict as dp
                pp(dummy, cn); op(dummy, cn); dp(dummy, cn)
            except Exception as e:
                print(f"[WARN] Warmup failed for {cn}: {e}")
            finally:
                done.set()
        threading.Thread(target=_warm, daemon=True).start()

    for ev in warm_events:
        ev.wait(timeout=10)

    for cam_name in CAMERA_SOURCES:
        threading.Thread(
            target=ai_processing_loop, args=(cam_name,), daemon=True
        ).start()

    register_routes(app, recorder, log_recorder, generate_frames, frames,
                    CAMERA_SOURCES, None, LOG_FILE, follow,
                    templates=templates, template_name="app.html")

    @app.get("/video_feed/{cam_name}")
    async def video_feed(cam_name: str):
        return StreamingResponse(
            mjpeg_generator(cam_name),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Cache-Control":     "no-store, no-cache, must-revalidate",
                "Pragma":            "no-cache",
                "X-Accel-Buffering": "no",
            }
        )

    print("\n" + "─" * 52)
    print("  Argus is running!")
    print("  App Link: http://localhost:5000/login")
    print("  Database: http://localhost:5000/admin")
    print("  DB file: " + os.path.abspath(DB_PATH))
    print("─" * 52 + "\n")

    yield

    # Cleanly stop recorders on shutdown
    try:
        if recorder and recorder.recording:
            recorder.stop()
        if log_recorder and log_recorder.recording:
            log_recorder.stop()
    except Exception:
        pass


init_db()

app = FastAPI(lifespan=lifespan)

# Fresh random key every run — sessions invalidated when server stops
app.add_middleware(SessionMiddleware, secret_key=secrets.token_hex(32))

app.mount("/static", StaticFiles(directory="screens/static"), name="static")
templates = Jinja2Templates(directory="screens")


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
        "app:app",
        host="0.0.0.0",
        port=5000,
        log_level="warning",
        reload=False
    )