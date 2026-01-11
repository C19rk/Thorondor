import cv2, time, logging, csv
import numpy as np
from datetime import datetime
from core.cameras import frames
from core.yolo_models import yolo
from core.yolo_desk_models import yolo_desk
from core.pose_models import pose_model
from core.config import YOLO_CONF_THRESHOLD, YOLO_DESK_CONF_THRESHOLD, POSE_CONF_THRESHOLD, SUSPICIOUS_LABELS, ALERT_COOLDOWN, PHONE_NUMBERS, LOG_FILE, CSV_FILE, H_THRESH, DOWN_THRESH, UP_THRESH
from core.gsm import gsm

last_alert_time = {}

def classify_behavior(keypoints):
    try:
        # Index Mapping: 0:Nose, 1:LEye, 2:REye, 3:LEar, 4:REar, 5:LSh, 6:RSh, 9:LWrist, 10:RWrist
        nose = keypoints[0]
        l_eye, r_eye = keypoints[1], keypoints[2]
        l_ear, r_ear = keypoints[3], keypoints[4]
        l_sh, r_sh = keypoints[5], keypoints[6]
        l_wrist, r_wrist = keypoints[9], keypoints[10]

        # 1. TALKING FIX: Increased sensitivity
        # We look for the vertical distance from nose to the eye line.
        eye_mid_y = (l_eye[1] + r_eye[1]) / 2
        eye_dist = np.linalg.norm(l_eye - r_eye)
        face_stretch = abs(nose[1] - eye_mid_y)
        
        # Lowered threshold from 0.9 to 0.75 for higher sensitivity
        if face_stretch > (eye_dist * 0.75): 
            return "talking"

        # 2. PHONE: Hand near head
        if np.linalg.norm(l_wrist - l_ear) < 60 or np.linalg.norm(r_wrist - r_ear) < 60:
            return "phone"

        # 3. LOOKING AWAY: Profile compression
        ear_to_ear = np.linalg.norm(l_ear - r_ear)
        dist_n_l = abs(nose[0] - l_ear[0])
        dist_n_r = abs(nose[0] - r_ear[0])
        # If nose is too close to one ear, you are looking away
        if dist_n_l < (ear_to_ear * 0.25) or dist_n_r < (ear_to_ear * 0.25):
            return "looking_away"

        # 4. HEAD DOWN
        avg_sh_y = (l_sh[1] + r_sh[1]) / 2
        if nose[1] > avg_sh_y - 5:
            return "head_down"

        return "normal"
    except:
        return "normal"

def draw_skeleton(frame, keypoints, color=(0,255,0)):
    connections = [(0,1),(0,2),(1,3),(2,4),(0,5),(0,6),(5,7),(7,9),(6,8),(8,10),(5,6),(11,12),(11,13),(13,15),(12,14),(14,16)]
    behavior = classify_behavior(keypoints)
    
    # Draw Status
    status_color = (0, 0, 255) if behavior != "normal" else (0, 255, 0)
    cv2.putText(frame, behavior.upper(), (int(keypoints[0][0]) - 30, int(keypoints[0][1]) - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    # Simplified Skeleton (no labels)
    for i, j in connections:
        if i < len(keypoints) and j < len(keypoints):
            pt1, pt2 = tuple(keypoints[i].astype(int)), tuple(keypoints[j].astype(int))
            if pt1 != (0,0) and pt2 != (0,0):
                cv2.line(frame, pt1, pt2, color, 2)

def generate_frames(cam_name):
    global frame_count
    while True:
        frame = frames.get(cam_name)
        if frame is None:
            time.sleep(0.01)
            continue
        
        frame_count += 1
        # LAG FIX: Only run heavy AI on every 2nd frame (Skip 50% of processing)
        if frame_count % 2 != 0:
            # Re-encode and send previous frame quickly to maintain stream
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            continue

        # Reduce image size for prediction to increase speed
        # Results are mapped back to original size automatically by YOLO
        results = yolo.predict(frame, imgsz=320, conf=YOLO_CONF_THRESHOLD, verbose=False)
        desk_results = yolo_desk.predict(frame, imgsz=320, conf=YOLO_DESK_CONF_THRESHOLD, verbose=False)
        pose_results = pose_model.predict(frame, imgsz=320, conf=POSE_CONF_THRESHOLD, verbose=False)

        # --- Draw YOLO ---
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                label = yolo.names[int(box.cls[0])]
                cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)

        # --- Draw Pose ---
        for r in pose_results:
            if hasattr(r, "keypoints") and r.keypoints is not None:
                for person_kpts in r.keypoints.xy.cpu().numpy():
                    if len(person_kpts) > 0:
                        draw_skeleton(frame, person_kpts)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

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