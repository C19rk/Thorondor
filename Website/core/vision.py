import cv2, time, logging, csv
from datetime import datetime
from core.cameras import frames
from core.yolo_models import yolo
from core.yolo_desk_models import yolo_desk
from core.pose_models import pose_model
from core.config import YOLO_CONF_THRESHOLD, YOLO_DESK_CONF_THRESHOLD, POSE_CONF_THRESHOLD, SUSPICIOUS_LABELS, ALERT_COOLDOWN, PHONE_NUMBERS, LOG_FILE, CSV_FILE, H_THRESH, DOWN_THRESH, UP_THRESH
from core.gsm import gsm

last_alert_time = {}

def draw_skeleton(frame, keypoints, conf_threshold=0.3, color=(0,255,0), thickness=2):
    connections = [
        (0,1),(0,2),(1,3),(2,4),(0,5),(0,6),(5,7),(7,9),(6,8),(8,10),
        (5,6),(11,12),(11,13),(13,15),(12,14),(14,16)
    ]
    for i,j in connections:
        if i < len(keypoints) and j < len(keypoints):
            if keypoints.shape[1]==3:
                x1,y1,c1 = keypoints[i]
                x2,y2,c2 = keypoints[j]
                if c1<conf_threshold or c2<conf_threshold: continue
            else:
                x1,y1 = keypoints[i]
                x2,y2 = keypoints[j]
            cv2.line(frame,(int(x1),int(y1)),(int(x2),int(y2)),color,thickness)
    for kp in keypoints:
        if keypoints.shape[1]==3:
            x,y,c = kp
            if c<conf_threshold: continue
        else:
            x,y = kp
        cv2.circle(frame,(int(x),int(y)),3,color,-1)


def generate_frames(cam_name, frames_override=None):
    frames_dict = frames_override if frames_override else frames
    while True:
        frame = frames_dict.get(cam_name)
        if frame is None:
            time.sleep(0.01)
            continue

        # === YOLO Detection ===
        results = yolo.predict(frame, imgsz=640, conf=YOLO_CONF_THRESHOLD)
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0].item())
                label = yolo.names[cls]
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                color = (255,0,0) if label=="person" else (0,255,255)
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                cv2.putText(frame,f"{label} {conf:.2f}",(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                with open(CSV_FILE,"a",newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp, cam_name, label, conf, x1, y1, x2, y2])
                
                logging.info(f"[{cam_name}] Detected {label} ({conf:.2f}) at [{x1},{y1},{x2},{y2}]")

                # GSM alert
                if label in SUSPICIOUS_LABELS:
                    key = f"{cam_name}_{label}"
                    now = time.time()
                    if key not in last_alert_time or now - last_alert_time[key] > ALERT_COOLDOWN:
                        for number in PHONE_NUMBERS:
                            gsm.send_sms(number,f"Alert! Detected {label} on {cam_name}")
                        last_alert_time[key] = now

        # === YOLO Desk Detection ===
        desk_results = yolo_desk.predict(frame, imgsz=640, conf=YOLO_DESK_CONF_THRESHOLD)
        for r in desk_results:
            for box in r.boxes:
                cls = int(box.cls[0].item())
                label = yolo_desk.names[cls]
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                color = (0, 128, 255)
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                cv2.putText(frame,f"Desk:{label} {conf:.2f}",(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
                
                logging.info(f"[{cam_name}] Desk detected {label} ({conf:.2f}) at [{x1},{y1},{x2},{y2}]")

        # === YOLO Pose Detection ===
        pose_results = pose_model.predict(frame, imgsz=640, conf=POSE_CONF_THRESHOLD)
        for r in pose_results:
            if hasattr(r, "keypoints") and r.keypoints is not None:
                for person_kpts in r.keypoints.xy:
                    keypoints_np = person_kpts.cpu().numpy()
                    draw_skeleton(frame,keypoints_np)

        ret, buffer = cv2.imencode('.jpg',frame)
        if not ret: continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'+buffer.tobytes()+b'\r\n')