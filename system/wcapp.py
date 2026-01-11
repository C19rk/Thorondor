from flask import Flask, Response, render_template_string
from core.vision import generate_frames
from core.config import FRAME_HEIGHT, FRAME_WIDTH, LOG_FILE
import cv2, threading, time

app = Flask(__name__)

# === Webcam setup ===
CAMERA_SOURCES = {"cam1": 0}
frames = {name: None for name in CAMERA_SOURCES.keys()}

def capture_frames(cam_name, src):
    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)  # Use DirectShow on Windows
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 15)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam {src}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        frames[cam_name] = frame

# Start threads
for name, src in CAMERA_SOURCES.items():
    t = threading.Thread(target=capture_frames, args=(name, src), daemon=True)
    t.start()

# === SSE log streaming ===
def follow(logfile):
    logfile.seek(0,2)
    while True:
        line = logfile.readline()
        if not line:
            time.sleep(0.1)
            continue
        yield f"data: {line}\n\n"

# === Flask routes ===
@app.route('/')
def index():
    html = """
    <html>
    <head>
        <title>Argus Webcam</title>
        <style>
            body { font-family: monospace; margin:0; padding:0; }
            .container { display: flex; height: 100vh; }
            .video { flex: 2; padding: 10px; }
            .log { flex: 1; padding: 10px; background-color: #f0f0f0; overflow-y: scroll; border-left: 2px solid #ccc; }
            img { width: 100%; height: auto; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="video">
                <h2>Live Webcam</h2>
                <img id="videoStream" src="/video/cam1">
            </div>
            <div class="log">
                <h2>Live Detection Log</h2>
                <div id="log"></div>
            </div>
        </div>
        <script>
            var evtSource = new EventSource("/log_stream");
            var logDiv = document.getElementById("log");
            evtSource.onmessage = function(e){
                logDiv.innerHTML += e.data + "<br>";
                logDiv.scrollTop = logDiv.scrollHeight;
            }
        </script>
    </body>
    </html>
    """
    return render_template_string(html)

@app.route('/video/<cam_name>')
def video(cam_name):
    # Use the webcam frames from this file
    return Response(generate_frames(cam_name, frames_override=frames),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/log_stream')
def log_stream():
    logfile = open(LOG_FILE,"r")
    return Response(follow(logfile), mimetype="text/event-stream")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
