from flask import Flask, Response, render_template_string
import cv2, threading, time, logging, csv
from datetime import datetime
from ultralytics import YOLO

app = Flask(__name__)

# === Logging Setup ===
csv_file = "detections.csv"
with open(csv_file, "a", newline="") as f:
    writer = csv.writer(f)
    f.seek(0, 2)
    if f.tell() == 0:
        writer.writerow(["timestamp", "camera", "object", "confidence", "x1", "y1", "x2", "y2"])


log_file = "detections.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
)

# === YOLO model ===
yolo = YOLO("../App/runs/aidetection7/weights/best.pt")

# === Camera RTSP URLs ===
camera_sources = {
    # "cam1: "rtsp://username:password@tapo_ip_address:554/stream1", (strem 1 for hd 2 for low)"
    # Connect to the same 2.4 GHz network as the camera
    "cam1": "rtsp://camera1:camera1234@192.168.0.100:554/stream2",
    "cam2": "rtsp://camera1:camera1234@192.168.0.100:554/stream2",
    "cam3": "rtsp://camera1:camera1234@192.168.0.100:554/stream2",
}

# === Shared frame storage for low latency ===
frames = {name: None for name in camera_sources.keys()}

# === Background camera capture threads ===
def capture_frames(cam_name, src):
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frames[cam_name] = frame

for name, src in camera_sources.items():
    t = threading.Thread(target=capture_frames, args=(name, src), daemon=True)
    t.start()

# === Generate MJPEG frames with YOLO overlay ===
def generate_frames(cam_name):
    while True:
        frame = frames.get(cam_name)
        if frame is None:
            time.sleep(0.01)
            continue

        results = yolo.predict(frame, imgsz=640, conf=0.75)
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0].item())
                label = yolo.names[cls]
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                color = (255,0,0) if label=="person" else (0,255,255)
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                logging.info(f"[{cam_name}] Detected {label} ({conf:.2f}) at [{x1},{y1},{x2},{y2}]")
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                # --- Write to CSV ---
                with open(csv_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp, cam_name, label, conf, x1, y1, x2, y2])

                # --- Write human-readable log for SSE panel ---
                readable_line = f"{timestamp} - [{cam_name}] Detected {label} ({conf:.2f}) at [{x1},{y1},{x2},{y2}]"

                with open("detections.log", "a") as f:
                    f.write(readable_line + "\n")

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# === SSE live log streaming ===
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
                    <button onclick="switchCam('cam1')">Camera 1</button>
                    <button onclick="switchCam('cam2')">Camera 2</button>
                    <button onclick="switchCam('cam3')">Camera 3</button>
                </div>
                <img id="videoStream" src="/video/cam1">
            </div>
            <div class="log">
                <h2>Live Detection Log</h2>
                <div id="log"></div>
            </div>
        </div>
        <script>
            function switchCam(camName) {
                document.getElementById("videoStream").src = "/video/" + camName;
            }
            var evtSource = new EventSource("/log_stream");
            var logDiv = document.getElementById("log");
            evtSource.onmessage = function(e) {
                logDiv.innerHTML += e.data + "<br>";
                logDiv.scrollTop = logDiv.scrollHeight;
            };
        </script>
    </body>
    </html>
    """
    return render_template_string(template)

@app.route('/video/<cam_name>')
def video(cam_name):
    if cam_name not in frames:
        return "Camera not found", 404
    return Response(generate_frames(cam_name),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/log_stream')
def log_stream():
    logfile = open(log_file, "r")
    return Response(follow(logfile), mimetype="text/event-stream")

# === Run server ===
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
