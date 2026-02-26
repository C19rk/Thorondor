import cv2
import csv
import numpy as np
import time
from datetime import datetime

from core.pose_models import pose_model, DEVICE
from core.config import (
    POSE_CONF_THRESHOLD,
    LOG_FILE,
    CSV_FILE,
)
from core.lighting_distance import preprocess_frame, _dynamic_confidence

POSE_LABELS = {
    0: "Cheating",
    1: "Normal",
}

LABEL_COLORS = {
    "Cheating": (0, 0, 255),
    "Normal":   (0, 255, 0),
}

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6), (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

KPT_CONF_THRESHOLD = 0.3
IOU_THRESHOLD      = 0.3

# Inference resolution — 16:9 slice of imgsz=256.
_INFER_W = 256
_INFER_H = 144

_person_instances: dict[str, dict] = {}
_next_instance_id = 0


def _new_id():
    global _next_instance_id
    _next_instance_id += 1
    return _next_instance_id


def _get_label(cls):
    name = pose_model.names.get(cls, str(cls))
    if name == str(cls):
        return POSE_LABELS.get(cls, f"Class{cls}")
    return name.capitalize()


def _iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA);   interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(areaA + areaB - interArea)


def _match_persons(prev_instances, current_detections):
    used_prev = set()
    matched   = {}
    new_dets  = []

    for det in current_detections:
        best_id, best_iou = None, IOU_THRESHOLD
        for inst_id, prev in prev_instances.items():
            if inst_id in used_prev:
                continue
            iou = _iou(det["box"], prev["box"])
            if iou > best_iou:
                best_iou = iou
                best_id  = inst_id

        if best_id is not None:
            matched[best_id] = det
            used_prev.add(best_id)
        else:
            new_dets.append(det)

    lost_ids = [i for i in prev_instances if i not in used_prev]
    return matched, new_dets, lost_ids


def _draw_skeleton(frame, keypoints_xy, keypoints_conf, color):
    for i, j in SKELETON:
        if i >= len(keypoints_xy) or j >= len(keypoints_xy):
            continue
        if keypoints_conf[i] < KPT_CONF_THRESHOLD or keypoints_conf[j] < KPT_CONF_THRESHOLD:
            continue
        pt1 = tuple(keypoints_xy[i].astype(int))
        pt2 = tuple(keypoints_xy[j].astype(int))
        if pt1[0] > 1 and pt1[1] > 1 and pt2[0] > 1 and pt2[1] > 1:
            cv2.line(frame, pt1, pt2, color, 2)

    for kpt, conf in zip(keypoints_xy, keypoints_conf):
        if conf < KPT_CONF_THRESHOLD:
            continue
        x, y = int(kpt[0]), int(kpt[1])
        if x > 1 and y > 1:
            cv2.circle(frame, (x, y), 3, color, -1)


def _log_behavior(cam_name, inst_id, label, timestamp):
    with open(LOG_FILE, "a") as f:
        f.write(
            f"[{timestamp.strftime('%H:%M:%S')}] "
            f"{cam_name}: Person {inst_id} behavior: {label}\n"
        )
    with open(CSV_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            timestamp, cam_name, "Behavior Changed", label, f"person_{inst_id}"
        ])


def predict(frame, cam_name):
    """Run pose inference on a downscaled, lighting-normalised copy.
    Works in bright/dark lighting and at far/close distances.
    Returns raw matched detections dict. Does NOT draw.
    """
    orig_h, orig_w = frame.shape[:2]

    # ── Lighting normalisation (CLAHE + adaptive gamma) ──
    enhanced, brightness = preprocess_frame(frame)
    small = cv2.resize(enhanced, (_INFER_W, _INFER_H), interpolation=cv2.INTER_LINEAR)
    scale_x = orig_w / _INFER_W
    scale_y = orig_h / _INFER_H

    # Adaptive confidence — relax in poor lighting
    conf_threshold = _dynamic_confidence(POSE_CONF_THRESHOLD, brightness)

    infer_area = _INFER_W * _INFER_H

    pose_results = pose_model.predict(
        small,
        imgsz=256,
        conf=conf_threshold,
        verbose=False,
        device=DEVICE if DEVICE != "directml" else "cpu",
    )

    current_detections = []

    for r in pose_results:
        boxes    = r.boxes     if (hasattr(r, "boxes")     and r.boxes     is not None) else []
        kpts_obj = r.keypoints if (hasattr(r, "keypoints") and r.keypoints is not None) else None

        kpts_xy   = kpts_obj.xy.cpu().numpy()   if kpts_obj is not None else []
        kpts_conf = kpts_obj.conf.cpu().numpy() if kpts_obj is not None else []

        for idx, box in enumerate(boxes):
            cls      = int(box.cls[0].item())
            conf_val = float(box.conf[0].item())
            label    = _get_label(cls)

            # Scale from inference space → original frame space
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Distance-aware confidence check (inference-space area ratio)
            infer_box_area = (x2 - x1) * (y2 - y1)
            area_ratio = infer_box_area / infer_area
            eff_conf = _dynamic_confidence(conf_threshold, brightness, area_ratio)
            if conf_val < eff_conf:
                continue

            x1 = int(x1 * scale_x); y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x); y2 = int(y2 * scale_y)

            kp_xy   = kpts_xy[idx]   if idx < len(kpts_xy)   and len(kpts_xy[idx]) > 0   else np.zeros((17, 2))
            kp_conf = kpts_conf[idx] if idx < len(kpts_conf) and len(kpts_conf[idx]) > 0 else np.zeros(17)

            # Scale keypoints back to full-res
            if kp_xy is not None and len(kp_xy):
                kp_xy = kp_xy.copy()
                kp_xy[:, 0] *= scale_x
                kp_xy[:, 1] *= scale_y

            current_detections.append({
                "box":       (x1, y1, x2, y2),
                "label":     label,
                "conf":      conf_val,
                "kpts_xy":   kp_xy,
                "kpts_conf": kp_conf,
            })

    prev_instances = _person_instances.get(cam_name, {})
    matched, new_dets, lost_ids = _match_persons(prev_instances, current_detections)

    for det in new_dets:
        inst_id = _new_id()
        matched[inst_id] = det
        _log_behavior(cam_name, inst_id, det["label"], datetime.now())

    for inst_id, det in matched.items():
        if inst_id in prev_instances:
            prev_label = prev_instances[inst_id]["label"]
            if det["label"] != prev_label:
                _log_behavior(cam_name, inst_id, det["label"], datetime.now())

    _person_instances[cam_name] = {
        inst_id: {"box": det["box"], "label": det["label"]}
        for inst_id, det in matched.items()
    }

    return matched  # raw detections only, no drawing


def draw(frame, matched):
    """Draw cached detections onto any fresh frame. No ghosting."""
    person_boxes = []
    for inst_id, det in matched.items():
        x1, y1, x2, y2 = det["box"]
        label  = det["label"]
        color  = LABEL_COLORS.get(label, (0, 255, 0))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"P{inst_id} {label} {det['conf']:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2
        )
        _draw_skeleton(frame, det["kpts_xy"], det["kpts_conf"], color)
        person_boxes.append((x1, y1, x2, y2))
    return frame, person_boxes


def process(frame, cam_name):
    """Legacy wrapper: predict + draw in one call."""
    matched = predict(frame, cam_name)
    return draw(frame, matched)