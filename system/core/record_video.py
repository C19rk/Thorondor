import os
import subprocess
import threading
import platform
import time
from datetime import datetime
from collections import deque

from core.config import FRAME_WIDTH, FRAME_HEIGHT


class VideoRecorder:
    MAX_FPS = 30.0

    def __init__(self, fps=25.0):
        self.fps           = min(float(fps), self.MAX_FPS)
        self.recording     = False
        self.finalizing    = False
        self.directory_set = True
        self.output_dir    = os.path.join(os.getcwd(), "recordings")
        os.makedirs(self.output_dir, exist_ok=True)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        if platform.system() == "Windows":
            self.ffmpeg_exe = os.path.normpath(
                os.path.join(current_dir, "..", "bin", "ffmpeg.exe")
            )
        else:
            local_ffmpeg = os.path.normpath(
                os.path.join(current_dir, "..", "bin", "ffmpeg")
            )
            self.ffmpeg_exe = local_ffmpeg if os.path.isfile(local_ffmpeg) else "ffmpeg"

        self.current_file  = "None"
        self.status_msg    = "Ready"
        self.saved_files   = []   # populated after finalization

        # Per-camera state — keyed by cam_name
        self._cameras      = {}
        self._feed_threads = []

        # Optional external frames dict.
        # wcapp.py sets this to its local webcam frames dict so _feed_raw
        # doesn't have to import core.cameras (which would start Tapo threads).
        # app.py leaves it None and _feed_raw falls back to core.cameras.frames.
        self._frames_source = None

    def _kill_zombies(self):
        if platform.system() == "Windows":
            subprocess.run(
                ["taskkill", "/F", "/IM", "ffmpeg.exe", "/T"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        else:
            subprocess.run(
                ["pkill", "-f", "ffmpeg"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

    def _startupinfo(self):
        """Hide console window on Windows; no-op on Linux/Mac."""
        if platform.system() == "Windows":
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            return si
        return None

    def _spawn_ffmpeg(self, output_path, width, height, input_fps):
        """
        Spawns an ffmpeg process that accepts rawvideo on stdin.
        input_fps: expected input frame rate (used for container header only;
                   actual frame timing comes from -use_wallclock_as_timestamps).
        """
        cmd = [
            self.ffmpeg_exe, "-y",
            "-f",        "rawvideo",
            "-vcodec",   "rawvideo",
            "-pix_fmt",  "bgr24",
            "-s",        f"{width}x{height}",
            "-r",        str(input_fps),
            "-use_wallclock_as_timestamps", "1",
            "-i",        "pipe:0",
            "-c:v",      "libx264",
            "-preset",   "ultrafast",
            "-tune",     "zerolatency",
            "-pix_fmt",  "yuv420p",
            "-vsync",    "vfr",
            "-movflags", "+faststart",
            output_path,
        ]
        return subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            startupinfo=self._startupinfo(),
        )

    def _get_frame_size(self, cam_name):
        """Detect actual frame dimensions from live feed. Wait up to 3s."""
        from core.vision import latest_annotated, latest_raw
        for _ in range(30):
            ann = latest_annotated.get(cam_name)
            raw = latest_raw.get(cam_name)
            frame = ann if ann is not None else raw
            if frame is not None:
                h, w = frame.shape[:2]
                print(f"[INFO] Recorder [{cam_name}] frame size: {w}x{h}")
                return w, h
            time.sleep(0.1)
        print(f"[WARN] Recorder [{cam_name}] no frame found — using config {FRAME_WIDTH}x{FRAME_HEIGHT}")
        return FRAME_WIDTH, FRAME_HEIGHT

    def start(self, cam_name=None, cam_names=None):
        if self.recording or self.finalizing:
            return
        self._kill_zombies()
        self._cameras = {}
        self._feed_threads = []
        self.saved_files = []

        if cam_names:
            targets = cam_names
        elif cam_name:
            targets = [cam_name]
        else:
            from core.config import CAMERA_SOURCES
            targets = list(CAMERA_SOURCES.keys())

        print(f"[INFO] Recorder starting for cameras: {targets}")

        now      = datetime.now()
        ts_label = f"{now.strftime('%b')} {now.day}, {now.year} {now.strftime('%I-%M-%S %p')}"
        self.current_file = os.path.join(
            self.output_dir, f"Argus Surveillance Recording - {ts_label}.mp4"
        )

        try:
            for cn in targets:
                w, h = self._get_frame_size(cn)

                # Webcam sessions use cam_name "cam1" (from wcapp.py).
                # Tapo sessions use "Camera 1", "Camera 2", etc. (from app.py).
                is_webcam = (cn == "cam1")
                if is_webcam:
                    base_name = f"Argus Webcam Surveillance Recording - {ts_label}"
                else:
                    base_name = f"Argus Surveillance Recording - {cn} - {ts_label}"

                ann_path = os.path.join(self.output_dir, f"{base_name}.mp4")
                raw_path = os.path.join(self.output_dir, f"{base_name} (Raw).mp4")

                # Annotated: runs at AI pipeline speed (variable, ≤ camera FPS)
                # Raw:       runs at full camera FPS (self.fps)
                proc_ann = self._spawn_ffmpeg(ann_path, w, h, input_fps=self.fps)
                proc_raw = self._spawn_ffmpeg(raw_path, w, h, input_fps=self.fps)

                time.sleep(0.3)
                if proc_ann.poll() is not None or proc_raw.poll() is not None:
                    print(f"[ERROR] Recorder [{cn}] ffmpeg failed to start")
                    continue

                self._cameras[cn] = {
                    "proc_ann":            proc_ann,
                    "proc_raw":            proc_raw,
                    "ann_path":            ann_path,
                    "raw_path":            raw_path,
                    "fps_window":          deque(maxlen=30),
                    "frames_written":      0,
                    "raw_frames_written":  0,
                    "start_time":          None,
                    "end_time":            None,
                    "raw_start_time":      None,
                    "raw_end_time":        None,
                }
                print(f"[INFO] Recorder [{cn}] ffmpeg started OK — ann+raw at up to {self.fps:.0f} fps")

            if not self._cameras:
                self.status_msg = "Failed to start any camera recorder"
                return

            self.recording  = True
            self.status_msg = f"Recording {len(self._cameras)} camera(s)..."

            for cn in self._cameras:
                # Annotated feed — gated by AI pipeline output
                t_ann = threading.Thread(
                    target=self._feed_annotated, args=(cn,), daemon=True
                )
                # Raw feed — reads directly from camera capture deque,
                # bypassing AI entirely → records at full camera FPS
                t_raw = threading.Thread(
                    target=self._feed_raw, args=(cn,), daemon=True
                )
                t_ann.start()
                t_raw.start()
                self._feed_threads.extend([t_ann, t_raw])

        except Exception as e:
            import traceback
            print(f"[ERROR] Recorder failed to start: {e}")
            traceback.print_exc()
            self.status_msg = f"Error: {e}"

    # ── Annotated feed ────────────────────────────────────────────────────────
    def _feed_annotated(self, cam_name):
        """
        Writes AI-annotated frames to proc_ann.
        Polls latest_annotated as fast as the AI pipeline produces frames —
        no artificial sleep throttle so every AI output frame is captured.
        """
        from core.vision import latest_annotated

        cam            = self._cameras[cam_name]
        proc_ann       = cam["proc_ann"]
        fps_window     = cam["fps_window"]
        last_ann_id    = None
        frames_written = 0

        print(f"[INFO] Recorder [{cam_name}] annotated feed started")

        while self.recording:
            annotated = latest_annotated.get(cam_name)

            if annotated is not None and id(annotated) != last_ann_id:
                last_ann_id = id(annotated)
                t_now = time.perf_counter()
                try:
                    proc_ann.stdin.write(annotated.tobytes())
                    fps_window.append(t_now)
                    if frames_written == 0:
                        cam["start_time"] = t_now
                        print(f"[INFO] Recorder [{cam_name}] first annotated frame: shape={annotated.shape}")
                    frames_written += 1
                    cam["frames_written"] = frames_written
                    cam["end_time"]       = t_now
                except Exception as e:
                    print(f"[ERROR] Recorder [{cam_name}] annotated write: {e}")
                    break
            else:
                # No new frame yet — yield the thread briefly
                time.sleep(0.001)

        print(f"[INFO] Recorder [{cam_name}] annotated feed done. Frames: {frames_written}")

    # ── Raw feed ──────────────────────────────────────────────────────────────
    def _feed_raw(self, cam_name):
        """
        Writes raw camera frames DIRECTLY from the camera capture deque,
        completely bypassing the AI pipeline. This means the raw recording
        runs at the camera's native FPS (e.g. 25 fps for Tapo stream1)
        regardless of how fast pose/object/desk detection runs.
        """
        # Use the injected frames dict if provided (e.g. wcapp.py webcam mode),
        # otherwise import and fall back to core.cameras.frames (app.py Tapo mode).
        # IMPORTANT: the import must stay inside the else-branch so that wcapp.py
        # never triggers cameras.py's module-level RTSP thread launch loop.
        if self._frames_source is not None:
            cam_deque_src = self._frames_source
        else:
            from core.cameras import frames as camera_frames
            cam_deque_src = camera_frames

        cam            = self._cameras[cam_name]
        proc_raw       = cam["proc_raw"]
        interval       = 1.0 / self.fps  # target: camera native FPS
        last_raw_id    = None
        frames_written = 0

        print(f"[INFO] Recorder [{cam_name}] raw feed started — direct from camera deque @ {self.fps:.0f} fps")

        while self.recording:
            t_start   = time.perf_counter()
            cam_deque = cam_deque_src.get(cam_name)

            if cam_deque:
                try:
                    frame = cam_deque[-1]  # latest camera frame
                except IndexError:
                    time.sleep(0.005)
                    continue

                fid = id(frame)
                if fid != last_raw_id:
                    last_raw_id = fid
                    t_now = time.perf_counter()
                    try:
                        proc_raw.stdin.write(frame.tobytes())
                        if frames_written == 0:
                            cam["raw_start_time"] = t_now
                            print(f"[INFO] Recorder [{cam_name}] first raw frame: shape={frame.shape}")
                        frames_written += 1
                        cam["raw_frames_written"] = frames_written
                        cam["raw_end_time"]       = t_now
                    except Exception as e:
                        print(f"[ERROR] Recorder [{cam_name}] raw write: {e}")
                        break

            elapsed = time.perf_counter() - t_start
            sleep_t = interval - elapsed
            if sleep_t > 0.001:
                time.sleep(sleep_t)

        print(f"[INFO] Recorder [{cam_name}] raw feed done. Frames: {frames_written}")

    def stop(self):
        if not self.recording:
            return
        self.recording  = False
        self.finalizing = True
        self.status_msg = "Finalizing..."

        def finalize():
            for t in self._feed_threads:
                if t.is_alive():
                    t.join(timeout=3)
            self._feed_threads.clear()

            for cn, cam in self._cameras.items():
                for proc in (cam["proc_ann"], cam["proc_raw"]):
                    try:
                        proc.stdin.close()
                        proc.wait(timeout=15)
                    except Exception:
                        proc.kill()

            self._cleanup()
            self.finalizing = False

        threading.Thread(target=finalize).start()

    def _actual_fps(self, frames_written, start_time, end_time, fallback):
        """Computes real recorded FPS from measured timing, capped at self.fps."""
        if frames_written > 1 and start_time and end_time:
            elapsed = end_time - start_time
            fps     = round(frames_written / elapsed, 3) if elapsed > 0 else fallback
            return max(1.0, min(fps, self.fps))
        return fallback

    def _cleanup(self):
        time.sleep(0.5)
        saved = []

        for cn, cam in self._cameras.items():
            # Annotated: measured from AI pipeline output
            ann_fps = self._actual_fps(
                cam.get("frames_written", 0),
                cam.get("start_time"),
                cam.get("end_time"),
                fallback=self.fps,
            )
            # Raw: measured from camera deque output (should be ≈ self.fps)
            raw_fps = self._actual_fps(
                cam.get("raw_frames_written", 0),
                cam.get("raw_start_time"),
                cam.get("raw_end_time"),
                fallback=self.fps,
            )

            print(f"[INFO] Recorder [{cn}] annotated FPS: {ann_fps:.2f}  raw FPS: {raw_fps:.2f}")

            for path, fps in ((cam["ann_path"], ann_fps), (cam["raw_path"], raw_fps)):
                if not os.path.exists(path):
                    print(f"[WARN] Recorder [{cn}] file missing: {path}")
                    continue

                size = os.path.getsize(path)
                print(f"[INFO] Recorder [{cn}] {os.path.basename(path)} | {size} bytes")

                if size < 5000:
                    print(f"[WARN] Recorder [{cn}] deleting empty file: {os.path.basename(path)}")
                    os.remove(path)
                    continue

                # Re-stamp PTS with the real measured FPS so playback speed
                # exactly matches what the camera/AI delivered.
                fixed_path = path.replace(".mp4", "_fixed.mp4")
                try:
                    result = subprocess.run(
                        [
                            self.ffmpeg_exe, "-y",
                            "-i",       path,
                            "-vf",      f"setpts=N/{fps}/TB",
                            "-r",       str(fps),
                            "-c:v",     "libx264",
                            "-preset",  "ultrafast",
                            "-pix_fmt", "yuv420p",
                            fixed_path,
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        startupinfo=self._startupinfo(),
                        timeout=120,
                    )
                    if (result.returncode == 0
                            and os.path.exists(fixed_path)
                            and os.path.getsize(fixed_path) > 5000):
                        os.remove(path)
                        os.rename(fixed_path, path)
                        print(f"[INFO] Recorder [{cn}] re-encoded at {fps:.2f} fps: {os.path.basename(path)}")
                    else:
                        if os.path.exists(fixed_path):
                            os.remove(fixed_path)
                        print(f"[WARN] Recorder [{cn}] re-encode failed, keeping original: {os.path.basename(path)}")
                except Exception as e:
                    print(f"[WARN] Recorder [{cn}] re-encode error: {e}")

                saved.append(os.path.basename(path))

        self.saved_files = saved
        self.status_msg = f"Saved {len(saved)} file(s)" if saved else "Recording failed (Empty)"