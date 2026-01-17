import cv2
import os
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
import threading
import queue

class VideoRecorder:
    def __init__(self, fps=15.0, width=640, height=480, max_queue=100):
        self.fps = fps
        self.width = int(width)
        self.height = int(height)
        self.out = None
        self.recording = False       # True while accepting new frames
        self.saving = False          # True while writing queued frames
        self.output_dir = os.path.normpath("recordings")
        self.filename = None
        self.frame_queue = queue.Queue(maxsize=max_queue)
        self.worker = None
        self.lock = threading.Lock()
        self._stop_signal = object()  # sentinel to stop thread

    def set_directory_popup(self):
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected_dir = filedialog.askdirectory(title="Select Save Directory")
        root.destroy()
        if selected_dir:
            self.output_dir = os.path.normpath(selected_dir)
            os.makedirs(self.output_dir, exist_ok=True)
            return self.output_dir
        return None

    def start(self):
        if self.recording or self.saving:
            return
        self.recording = True
        self.out = None
        with self.frame_queue.mutex:
            self.frame_queue.queue.clear()
        self.worker = threading.Thread(target=self._record_worker, daemon=True)
        self.worker.start()
        print("[INFO] Recorder armed. Waiting for first frame...")

    def _record_worker(self):
        self.saving = True
        while self.recording or not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if frame is self._stop_signal:
                continue

            if self.out is None:
                h, w = frame.shape[:2]
                self.width, self.height = w, h
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.filename = os.path.join(self.output_dir, f"recording_{ts}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.out = cv2.VideoWriter(self.filename, fourcc, self.fps, (self.width, self.height))
                if not self.out.isOpened():
                    print("[ERROR] Could not open VideoWriter")
                    self.recording = False
                    self.saving = False
                    return
                print(f"[SUCCESS] Recording started: {self.filename}")

            if (frame.shape[1], frame.shape[0]) != (self.width, self.height):
                frame = cv2.resize(frame, (self.width, self.height))
            self.out.write(frame)

        # Finished writing all queued frames
        if self.out:
            self.out.release()
            self.out = None
        self.saving = False
        print(f"[INFO] Recording fully stopped and saved: {self.filename}")

    def write(self, frame):
        if self.recording:
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass  # drop frames to prevent blocking

    def stop(self):
        if not self.recording:
            return
        self.recording = False
        print("[INFO] Stop requested. Saving remaining frames...")
