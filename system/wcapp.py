from flask import Flask, Response, render_template_string, jsonify, request
from core.vision import generate_frames
from core.config import FRAME_HEIGHT, FRAME_WIDTH, LOG_FILE
from core.record_video import VideoRecorder
import cv2, threading, time, atexit, os

app = Flask(__name__)

recorder = VideoRecorder(fps=15, width=FRAME_WIDTH, height=FRAME_HEIGHT)
atexit.register(recorder.stop)

CAMERA_SOURCES = {"cam1": 0}
frames = {name: None for name in CAMERA_SOURCES.keys()}

def capture_frames(cam_name, src):
    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 15)
    if not cap.isOpened(): return
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        frames[cam_name] = frame

for name, src in CAMERA_SOURCES.items():
    threading.Thread(target=capture_frames, args=(name, src), daemon=True).start()

def follow(logfile):
    logfile.seek(0,2)
    while True:
        line = logfile.readline()
        if not line:
            time.sleep(0.1)
            continue
        yield f"data: {line}\n\n"

@app.route('/set_dir', methods=['POST'])
def set_dir():
    path = recorder.set_directory_popup()
    if path:
        return jsonify({"status": "success", "path": path})
    return jsonify({"status": "cancelled"}), 200

@app.route('/start_record')
def start_record():
    recorder.start()
    return jsonify({"status": "Started"})

@app.route('/stop_record')
def stop_record():
    recorder.stop()
    return jsonify({"status": "Stopped"})

@app.route('/')
def index():
    html = """
    <html>
    <head>
        <title>Argus Webcam</title>
        <style>
            body { font-family: monospace; margin:0; padding:0; }
            .container { display: flex; height: 100vh; }
            .video { flex: 2; padding: 10px; position: relative; }
            .log { flex: 1; padding: 10px; background-color: #f0f0f0; overflow-y: scroll; border-left: 2px solid #ccc; }
            img { width: 100%; height: auto; border: 1px solid #333; }
            
            .overlay {
                position: absolute; top: 20px; right: 20px;
                background: rgba(255,255,255,0.9); padding: 10px;
                border: 1px solid #000; display: flex; gap: 5px; align-items: center;
            }
            .btn { cursor: pointer; border: 1px solid #000; padding: 5px 10px; font-weight: bold; }
            .rec-btn { background: red; color: white; display: block; }
            .stop-btn { background: white; color: black; display: none; }
            
            /* Blinking Recording Label */
            #status { color: red; font-weight: bold; display: none; margin-right: 10px; animation: blinker 1s linear infinite; }
            @keyframes blinker { 50% { opacity: 0; } }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="video">
                <h2>Live Webcam <span id="status">● REC</span></h2>
                <div class="overlay">
                    <button class="btn" onclick="saveDir()" style="background: #1877f2; color: white;">SET DIR</button>
                    <button id="startB" class="btn rec-btn" onclick="doRec('start')">RECORD</button>
                    <button id="stopB" class="btn stop-btn" onclick="doRec('stop')">STOP</button>
                </div>
                <img id="videoStream" src="/video/cam1">
            </div>
            <div class="log">
                <h2>Live Detection Log</h2>
                <div id="log"></div>
            </div>
        </div>
        <script>
            function saveDir() {
                fetch('/set_dir', { method: 'POST' })
                .then(r => r.json())
                .then(data => { if(data.path) alert("Saving to: " + data.path); });
            }

            function doRec(action) {
                // UI updates IMMEDIATELY so you see the change
                let isStart = (action === 'start');
                
                fetch('/' + action + '_record')
                .then(response => {
                    if (response.ok) {
                        document.getElementById('startB').style.display = isStart ? 'none' : 'block';
                        document.getElementById('stopB').style.display = isStart ? 'block' : 'none';
                        document.getElementById('status').style.display = isStart ? 'inline' : 'none';
                        console.log("Recording " + action + "ed");
                    }
                })
                .catch(err => alert("Server error: " + err));
            }

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
    return Response(generate_frames(cam_name, frames_override=frames, recorder=recorder),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/log_stream')
def log_stream():
    if not os.path.exists(LOG_FILE): open(LOG_FILE, 'w').close()
    return Response(follow(open(LOG_FILE,"r")), mimetype="text/event-stream")

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)