from flask import Flask, Response, render_template_string
from core.vision import generate_frames
from core.cameras import frames
from core.config import LOG_FILE, CAMERA_SOURCES
import time

app = Flask(__name__)

# === SSE log streaming ===
def follow(logfile):
    logfile.seek(0,2)
    while True:
        line = logfile.readline()
        if not line:
            time.sleep(0.1)
            continue
        yield f"data: {line}\n\n"

# === Flask index page ===
@app.route('/')
def index():
    template = """
    <html>
    <head>
        <title>Argus - Multi-Camera Cheating Detection</title>
        <style>
            body { font-family: monospace; margin:0; padding:0; }
            .container { display: flex; height: 100vh; }
            .video { flex: 2; padding: 10px; }
            .log { flex: 1; padding: 10px; background-color: #f0f0f0; overflow-y: scroll; border-left: 2px solid #ccc; }
            img { width: 100%; height: auto; }
            .buttons { margin-bottom: 10px; }
            button { margin-right: 10px; padding: 8px 16px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="video">
                <h2>Live Video</h2>
                <div class="buttons">
                    {% for cam in cams %}
                    <button onclick="switchCam('{{cam}}')">{{cam}}</button>
                    {% endfor %}
                </div>
                <img id="videoStream" src="/video/{{default_cam}}">
            </div>
            <div class="log">
                <h2>Live Detection Log</h2>
                <div id="log"></div>
            </div>
        </div>
        <script>
            function switchCam(camName){ document.getElementById("videoStream").src="/video/"+camName; }
            var evtSource = new EventSource("/log_stream");
            var logDiv = document.getElementById("log");
            evtSource.onmessage = function(e){ logDiv.innerHTML+=e.data+"<br>"; logDiv.scrollTop=logDiv.scrollHeight; }
        </script>
    </body>
    </html>
    """
    default_cam = list(CAMERA_SOURCES.keys())[0]
    return render_template_string(template, cams=CAMERA_SOURCES.keys(), default_cam=default_cam)

# === Video streaming route ===
@app.route('/video/<cam_name>')
def video(cam_name):
    if cam_name not in frames:
        return "Camera not found", 404
    return Response(generate_frames(cam_name),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# === Log streaming route ===
@app.route('/log_stream')
def log_stream():
    logfile = open(LOG_FILE, "r")
    return Response(follow(logfile), mimetype="text/event-stream")

if __name__=="__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
