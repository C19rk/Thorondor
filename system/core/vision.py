import cv2, time, logging, csv
import numpy as np
from datetime import datetime

from core.cameras import frames
from core.yolo_models import yolo
from core.yolo_desk_models import yolo_desk
from core.pose_models import pose_model
from core.config import (
    YOLO_CONF_THRESHOLD,
    YOLO_DESK_CONF_THRESHOLD,
    POSE_CONF_THRESHOLD,
    SUSPICIOUS_LABELS,
    ALERT_COOLDOWN,
    PHONE_NUMBERS,
    LOG_FILE,
    CSV_FILE
)
from core.gsm import gsm

last_alert_time = {}
frame_count = 0

# --------------------------------------------------
# BEHAVIOR CLASSIFICATION
# --------------------------------------------------
def classify_behavior(keypoints):
    try:
        nose = keypoints[0]
        l_eye, r_eye = keypoints[1], keypoints[2]
        l_ear, r_ear = keypoints[3], keypoints[4]
        l_sh, r_sh = keypoints[5], keypoints[6]
        l_wrist, r_wrist = keypoints[9], keypoints[10]

        # PHONE
        if np.linalg.norm(l_wrist - l_ear) < 55 or np.linalg.norm(r_wrist - r_ear) < 55:
            return "phone"

        # TALKING
        eye_mid_y = (l_eye[1] + r_eye[1]) / 2
        eye_dist = np.linalg.norm(l_eye - r_eye)
        face_stretch = abs(nose[1] - eye_mid_y)
        if face_stretch > eye_dist * 0.75:
            return "talking"

        # LOOKING AWAY
        ear_dist = np.linalg.norm(l_ear - r_ear)
        if abs(nose[0] - l_ear[0]) < ear_dist * 0.2 or abs(nose[0] - r_ear[0]) < ear_dist * 0.2:
            return "looking_away"

        # HEAD DOWN
        avg_sh_y = (l_sh[1] + r_sh[1]) / 2
        if nose[1] > avg_sh_y - 5:
            return "head_down"

        return "normal"
    except:
        return "normal"


# --------------------------------------------------
# DRAW SKELETON
# --------------------------------------------------
def draw_skeleton(frame, keypoints, color=(0, 255, 0)):
    connections = [
        (0,1),(0,2),(1,3),(2,4),
        (0,5),(0,6),(5,7),(7,9),
        (6,8),(8,10),(5,6),
        (11,12),(11,13),(13,15),
        (12,14),(14,16)
    ]

    behavior = classify_behavior(keypoints)
    status_color = (0, 0, 255) if behavior != "normal" else (0, 255, 0)

    cv2.putText(
        frame,
        behavior.upper(),
        (int(keypoints[0][0]) - 30, int(keypoints[0][1]) - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        status_color,
        2,
        cv2.LINE_AA
    )

    for i, j in connections:
        if i < len(keypoints) and j < len(keypoints):
            pt1 = tuple(keypoints[i].astype(int))
            pt2 = tuple(keypoints[j].astype(int))
            if pt1 != (0, 0) and pt2 != (0, 0):
                cv2.line(frame, pt1, pt2, color, 2)


# --------------------------------------------------
# FRAME GENERATOR
# --------------------------------------------------
def generate_frames(cam_name, frames_override=None, recorder=None):
    global frame_count
    frames_dict = frames_override if frames_override else frames

    while True:
        frame = frames_dict.get(cam_name)
        if frame is None:
            time.sleep(0.01)
            continue

        frame_count += 1

        # ------------------------------
        # FRAME SKIP (NO POSE REDRAW)
        # ------------------------------
        if frame_count % 2 != 0:
            if recorder:
                recorder.write(frame)
            ret, buffer = cv2.imencode(".jpg", frame)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                buffer.tobytes() +
                b"\r\n"
            )
            continue

        # ------------------------------
        # AI INFERENCE
        # ------------------------------
        results = yolo.predict(
            frame,
            imgsz=320,
            conf=YOLO_CONF_THRESHOLD,
            verbose=False
        )

        desk_results = yolo_desk.predict(
            frame,
            imgsz=320,
            conf=YOLO_DESK_CONF_THRESHOLD,
            verbose=False
        )

        pose_results = pose_model.predict(
            frame,
            imgsz=320,
            conf=POSE_CONF_THRESHOLD,
            verbose=False
        )

        # ------------------------------
        # YOLO OBJECT DETECTION
        # ------------------------------
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0].item())
                label = yolo.names[cls]
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                color = (255, 0, 0) if label == "person" else (0, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

                timestamp_now = datetime.now().strftime("%H:%M:%S")
                log_msg = f"[{timestamp_now}] {cam_name}: {label.upper()} detected ({conf:.2f})"

                with open(LOG_FILE, "a") as f:
                    f.write(log_msg + "\n")

                with open(CSV_FILE, "a", newline="") as f:
                    csv.writer(f).writerow([datetime.now(), cam_name, label, conf])

                if label in SUSPICIOUS_LABELS:
                    alert_key = f"{cam_name}_{label}"
                    now = time.time()
                    if alert_key not in last_alert_time or now - last_alert_time[alert_key] > ALERT_COOLDOWN:
                        for number in PHONE_NUMBERS:
                            gsm.send_sms(number, f"ALERT: {label} on {cam_name}")
                        last_alert_time[alert_key] = now

        # ------------------------------
        # DESK DETECTION
        # ------------------------------
        for r in desk_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 128, 255), 2)

        # ------------------------------
        # POSE (NO PERSISTENCE, NO GHOSTS)
        # ------------------------------
        for r in pose_results:
            if hasattr(r, "keypoints") and r.keypoints is not None:
                kpts_data = r.keypoints.xy.cpu().numpy()

                for person_kpts in kpts_data:
                    if len(person_kpts) > 0:
                        draw_skeleton(frame, person_kpts)

        # ------------------------------
        # RECORD + STREAM
        # ------------------------------
        if recorder:
            recorder.write(frame)

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buffer.tobytes() +
            b"\r\n"
        )
