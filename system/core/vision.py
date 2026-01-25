import cv2, time, csv
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

# ------------------------------
# STATE MEMORY
# ------------------------------
last_alert_time = {}
last_behavior_state = {}     # per camera
last_object_state = {}       # per camera
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

        if np.linalg.norm(l_wrist - l_ear) < 55 or np.linalg.norm(r_wrist - r_ear) < 55:
            return "Phone Use"

        eye_mid_y = (l_eye[1] + r_eye[1]) / 2
        eye_dist = np.linalg.norm(l_eye - r_eye)
        face_stretch = abs(nose[1] - eye_mid_y)
        if face_stretch > eye_dist * 0.75:
            return "Talking"

        ear_dist = np.linalg.norm(l_ear - r_ear)
        if abs(nose[0] - l_ear[0]) < ear_dist * 0.2 or abs(nose[0] - r_ear[0]) < ear_dist * 0.2:
            return "Looking Away"

        avg_sh_y = (l_sh[1] + r_sh[1]) / 2
        if nose[1] > avg_sh_y - 5:
            return "Head Down"

        return "Normal"
    except:
        return "Normal"


# --------------------------------------------------
# DRAW SKELETON
# --------------------------------------------------
def draw_skeleton(frame, keypoints):
    connections = [
        (0,1),(0,2),(1,3),(2,4),
        (0,5),(0,6),(5,7),(7,9),
        (6,8),(8,10),(5,6),
        (11,12),(11,13),(13,15),
        (12,14),(14,16)
    ]

    behavior = classify_behavior(keypoints)
    color = (0, 0, 255) if behavior != "Normal" else (0, 255, 0)

    cv2.putText(
        frame,
        behavior,
        (int(keypoints[0][0]) - 40, int(keypoints[0][1]) - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

    for i, j in connections:
        if i < len(keypoints) and j < len(keypoints):
            pt1 = tuple(keypoints[i].astype(int))
            pt2 = tuple(keypoints[j].astype(int))
            if pt1 != (0, 0) and pt2 != (0, 0):
                cv2.line(frame, pt1, pt2, color, 2)

    return behavior


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
        # FRAME SKIP
        # ------------------------------
        if frame_count % 2 != 0:
            # Always write to recorder if it's recording, regardless of AI processing
            if recorder and recorder.recording:
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
        # AI INFERENCE (always run, even when recording)
        # ------------------------------
        results = yolo.predict(frame, imgsz=320, conf=YOLO_CONF_THRESHOLD, verbose=False)
        desk_results = yolo_desk.predict(frame, imgsz=320, conf=YOLO_DESK_CONF_THRESHOLD, verbose=False)
        pose_results = pose_model.predict(frame, imgsz=320, conf=POSE_CONF_THRESHOLD, verbose=False)

        # ==================================================
        # OBJECT DETECTION (STATE CHANGE)
        # ==================================================
        current_objects = set()

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0].item())
                label = yolo.names[cls].capitalize()
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                current_objects.add(label)

        prev_objects = last_object_state.get(cam_name, set())

        for obj in current_objects - prev_objects:
            timestamp = datetime.now()

            with open(LOG_FILE, "a") as f:
                f.write(f"[{timestamp.strftime('%H:%M:%S')}] {cam_name}: Object Detected: {obj}\n")

            with open(CSV_FILE, "a", newline="") as f:
                csv.writer(f).writerow([timestamp, cam_name, "Object Detected", obj, ""])

            if obj.lower() in SUSPICIOUS_LABELS:
                alert_key = f"{cam_name}_{obj}"
                now = time.time()
                if alert_key not in last_alert_time or now - last_alert_time[alert_key] > ALERT_COOLDOWN:
                    for number in PHONE_NUMBERS:
                        gsm.send_sms(number, f"ALERT: {obj} detected on {cam_name}")
                    last_alert_time[alert_key] = now

        for obj in prev_objects - current_objects:
            timestamp = datetime.now()

            with open(LOG_FILE, "a") as f:
                f.write(f"[{timestamp.strftime('%H:%M:%S')}] {cam_name}: Object Left: {obj}\n")

            with open(CSV_FILE, "a", newline="") as f:
                csv.writer(f).writerow([timestamp, cam_name, "Object Left", obj, ""])

        last_object_state[cam_name] = current_objects

                # ==================================================
        # DESK DETECTION (STATEFUL, like object detection)
        # ==================================================
        current_desks = set()

        # collect person boxes to avoid detecting humans as desks
        person_boxes = []
        for r in pose_results:
            if hasattr(r, "boxes") and r.boxes is not None:
                for box in r.boxes:
                    px1, py1, px2, py2 = map(int, box.xyxy[0].tolist())
                    person_boxes.append((px1, py1, px2, py2))

        for r in desk_results:
            for box in r.boxes:
                cls = int(box.cls[0].item())
                label = yolo_desk.names.get(cls, "").lower()
                if label != "desk":
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                w = x2 - x1
                h = y2 - y1

                # reject human-shaped boxes
                if h / (w + 1e-6) > 0.85:
                    continue

                # reject boxes overlapping with people
                overlaps_person = False
                for px1, py1, px2, py2 in person_boxes:
                    ix1 = max(x1, px1)
                    iy1 = max(y1, py1)
                    ix2 = min(x2, px2)
                    iy2 = min(y2, py2)
                    if ix1 < ix2 and iy1 < iy2:
                        overlaps_person = True
                        break

                if overlaps_person:
                    continue

                # valid desk
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 128, 255), 2)
                cv2.putText(
                    frame,
                    "Desk",
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 128, 255),
                    2
                )

                current_desks.add(f"{x1}_{y1}_{x2}_{y2}")  # unique desk ID

        # compare with previous state
        prev_desks = last_object_state.get(f"{cam_name}_desk", set())

        # new desks
        for desk in current_desks - prev_desks:
            timestamp = datetime.now()
            with open(LOG_FILE, "a") as f:
                f.write(f"[{timestamp.strftime('%H:%M:%S')}] {cam_name}: Desk Detected: {desk}\n")
            with open(CSV_FILE, "a", newline="") as f:
                csv.writer(f).writerow([timestamp, cam_name, "Desk Detected", desk, ""])

        # desks removed
        for desk in prev_desks - current_desks:
            timestamp = datetime.now()
            with open(LOG_FILE, "a") as f:
                f.write(f"[{timestamp.strftime('%H:%M:%S')}] {cam_name}: Desk Left: {desk}\n")
            with open(CSV_FILE, "a", newline="") as f:
                csv.writer(f).writerow([timestamp, cam_name, "Desk Left", desk, ""])

        # update state
        last_object_state[f"{cam_name}_desk"] = current_desks


        # ==================================================
        # BEHAVIOR (STATE CHANGE)
        # ==================================================
        for r in pose_results:
            if hasattr(r, "keypoints") and r.keypoints is not None:
                for person_kpts in r.keypoints.xy.cpu().numpy():
                    if len(person_kpts) == 0:
                        continue

                    behavior = draw_skeleton(frame, person_kpts)
                    prev_behavior = last_behavior_state.get(cam_name, "Normal")

                    if behavior != prev_behavior:
                        last_behavior_state[cam_name] = behavior
                        timestamp = datetime.now()

                        with open(LOG_FILE, "a") as f:
                            f.write(
                                f"[{timestamp.strftime('%H:%M:%S')}] "
                                f"{cam_name}: Behavior Changed: {behavior}\n"
                            )

                        with open(CSV_FILE, "a", newline="") as f:
                            csv.writer(f).writerow([
                                timestamp,
                                cam_name,
                                "Behavior Changed",
                                behavior,
                                ""
                            ])
                            

        # ------------------------------
        # RECORD + STREAM
        # ------------------------------
        # Write processed frame with bounding boxes to recorder
        if recorder and recorder.recording:
            recorder.write(frame)

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret: continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"