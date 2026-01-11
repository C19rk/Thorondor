import cv2
import os
import tkinter as tk
from tkinter import filedialog
from datetime import datetime

class VideoRecorder:
    def __init__(self, fps=15.0, width=640, height=480):
        self.fps = fps
        self.width = int(width)
        self.height = int(height)
        self.out = None
        self.recording = False
        self.output_dir = os.path.normpath("recordings")

    def set_directory_popup(self):
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected_dir = filedialog.askdirectory(title="Select Save Directory")
        root.destroy()
        if selected_dir:
            self.output_dir = os.path.normpath(selected_dir)
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            return self.output_dir
        return None

    def start(self):
        """Prepares the state. Actual file creation happens on the first frame."""
        if self.recording: return
        self.recording = True
        self.out = None # Reset writer to force re-initialization
        print("[INFO] Recorder Armed. Waiting for first frame...")

    def write(self, frame):
        if not self.recording:
            return

        # --- AUTO-INITIALIZE ON FIRST FRAME ---
        if self.out is None:
            # Sync dimensions to the actual incoming frame to prevent size mismatch
            h, w = frame.shape[:2]
            self.width, self.height = w, h
            
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.normpath(os.path.join(self.output_dir, f"argus_{ts}.mp4"))
            
            # Use 'mp4v' for Windows. If it fails, try 'XVID'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(filename, fourcc, self.fps, (self.width, self.height))
            
            if self.out.isOpened():
                print(f"[SUCCESS] File created: {filename} at {w}x{h}")
            else:
                print("[ERROR] Could not open VideoWriter. Check permissions.")
                self.recording = False
                return

        # --- ACTUAL WRITING ---
        try:
            # Double check size matches
            if (frame.shape[1], frame.shape[0]) != (self.width, self.height):
                frame = cv2.resize(frame, (self.width, self.height))
            self.out.write(frame)
        except Exception as e:
            print(f"[ERROR] Write failed: {e}")

    def stop(self):
        self.recording = False
        if self.out is not None:
            self.out.release()
            self.out = None
            print("[INFO] Recording finished and saved.")