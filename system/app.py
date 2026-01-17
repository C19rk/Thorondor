from flask import Flask, Response, render_template, jsonify
from core.vision import generate_frames
from core.cameras import frames
from core.config import FRAME_HEIGHT, FRAME_WIDTH, LOG_FILE, CAMERA_SOURCES
from core.record_video import VideoRecorder
from core.record_logs import LogRecorder
import time, atexit, os

app = Flask(__name__, template_folder='screens')

# -----------------------------
# Recorder
# -----------------------------
recorder = VideoRecorder(fps=15, width=FRAME_WIDTH, height=FRAME_HEIGHT)
recorder.directory_set = False
atexit.register(recorder.stop)

# -----------------------------
# Log Recorder
# -----------------------------
log_recorder = LogRecorder()
atexit.register(log_recorder.stop)

# -----------------------------
# NO CAMERA SETUP - Uses core.cameras.frames instead
# -----------------------------

# -----------------------------
# Log Streaming
# -----------------------------
def follow(logfile):
    logfile.seek(0,2)
    while True:
        line = logfile.readline()
        if not line:
            time.sleep(0.1)
            continue
        if log_recorder.recording:
            log_recorder.write(line.strip())
        yield f"data: {line}\n\n"

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def index():
    default_cam = list(CAMERA_SOURCES.keys())[0]
    return render_template('app.html', cams=CAMERA_SOURCES.keys(), default_cam=default_cam)

@app.route('/set_dir', methods=['POST'])
def set_dir():
    path = recorder.set_directory_popup()
    if path:
        recorder.directory_set = True
        return jsonify({"status": "success", "path": path})
    return jsonify({"status": "cancelled"}), 200

@app.route('/start_record')
def start_record():
    if not hasattr(recorder, 'directory_set') or not recorder.directory_set:
        return jsonify({"status": "error", "message": "Please set directory first"}), 400
    recorder.start()
    return jsonify({"status": "Started"})

@app.route('/stop_record')
def stop_record():
    recorder.stop()
    return jsonify({"status": "Stop requested"})

@app.route('/set_log_dir', methods=['POST'])
def set_log_dir():
    path = log_recorder.set_directory_popup()
    if path:
        return jsonify({"status": "success", "path": path})
    return jsonify({"status": "cancelled"}), 200

@app.route('/start_log_record')
def start_log_record():
    if not log_recorder.directory_set:
        return jsonify({"status": "error", "message": "Please set log directory first"}), 400
    log_recorder.start()
    return jsonify({"status": "Started"})

@app.route('/stop_log_record')
def stop_log_record():
    log_recorder.stop()
    return jsonify({"status": "Stop requested"})

@app.route('/video/<cam_name>')
def video(cam_name):
    if cam_name not in frames:
        return "Camera not found", 404
    return Response(generate_frames(cam_name, frames_override=frames, recorder=recorder),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/log_stream')
def log_stream():
    if not os.path.exists(LOG_FILE):
        open(LOG_FILE,'w').close()
    return Response(follow(open(LOG_FILE,"r")), mimetype="text/event-stream")

@app.route('/record_progress')
def record_progress():
    if recorder.saving:
        qsize = recorder.frame_queue.qsize()
        total = recorder.frame_queue.maxsize
        percent = int((1 - qsize/total)*100)
        return jsonify({"percent": percent, "done": False})
    return jsonify({"percent": 100, "done": True})

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)