import cv2
import csv
import threading
from datetime import datetime

from core.yolo_models import yolo
from core.config import (
    YOLO_CONF_THRESHOLD,
    LOG_FILE,
    CSV_FILE,
)
from core.lighting_distance import preprocess_frame, _dynamic_confidence

OBJECT_LABELS = {
    0: "Phone",
    1: "Calculator",
    2: "Smartwatch",
    3: "Watch",
}
OBJECT_COLORS     = {k: (0, 225, 255) for k in OBJECT_LABELS}
OBJECT_TEXT_COLOR = (0, 140, 255)
_LABEL_TO_CLS     = {v: k for k, v in OBJECT_LABELS.items()}

IOU_THRESHOLD = 0.3

# Inference resolution — 16:9 slice of imgsz=320.
_INFER_W = 320
_INFER_H = 180

_last_object_state = {}
_next_instance_id  = 0

# ── Buffered file I/O ─────────────────────────────────────────────────────────
_log_lock   = threading.Lock()
_log_file   = None
_csv_file   = None
_csv_writer = None

def _init_io():
    global _log_file, _csv_file, _csv_writer
    if _log_file is None:
        _log_file   = open(LOG_FILE, "a", buffering=1)
        _csv_file   = open(CSV_FILE, "a", newline="", buffering=1)
        _csv_writer = csv.writer(_csv_file)

def _write_event(timestamp, cam_name, event, label, inst_id):
    _init_io()
    with _log_lock:
        _log_file.write(f"[{timestamp.strftime('%H:%M:%S')}] {cam_name}: {event}: {label} (id={inst_id})\n")
        _csv_writer.writerow([timestamp, cam_name, event, label, inst_id])
# ─────────────────────────────────────────────────────────────────────────────

def _get_label(cls):
    return OBJECT_LABELS.get(cls, f"Class{cls}")

def _get_color(cls):
    return OBJECT_COLORS.get(cls, (0, 225, 255))

def _new_id():
    global _next_instance_id
    _next_instance_id += 1
    return _next_instance_id

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

def _match_instances(prev_instances, current_detections):
    used_prev         = set()
    matched_instances = {}
    unmatched_current = []
    for label, box, conf in current_detections:
        best_id, best_iou = None, IOU_THRESHOLD
        for inst_id, (prev_label, prev_box) in prev_instances.items():
            if inst_id in used_prev or prev_label != label:
                continue
            iou = _iou(box, prev_box)
            if iou > best_iou:
                best_iou = iou
                best_id  = inst_id
        if best_id is not None:
            matched_instances[best_id] = (label, box)
            used_prev.add(best_id)
        else:
            unmatched_current.append((label, box, conf))
    lost_ids = [i for i in prev_instances if i not in used_prev]
    return matched_instances, unmatched_current, lost_ids


def predict(frame, cam_name):
    """Run YOLO inference on downscaled, lighting-normalised frame.
    Works in bright/dark lighting and at far/close distances.
    Returns matched instances dict. Does NOT draw.
    """
    orig_h, orig_w = frame.shape[:2]

    # ── Lighting normalisation (CLAHE + adaptive gamma) ──
    enhanced, brightness = preprocess_frame(frame)
    small   = cv2.resize(enhanced, (_INFER_W, _INFER_H), interpolation=cv2.INTER_LINEAR)
    scale_x = orig_w / _INFER_W
    scale_y = orig_h / _INFER_H

    # Adaptive confidence — relax in poor lighting
    conf_threshold = _dynamic_confidence(YOLO_CONF_THRESHOLD, brightness)

    infer_area = _INFER_W * _INFER_H

    results = yolo.predict(
        small,
        imgsz=320,
        conf=conf_threshold,
        verbose=False,
        device="cpu",
    )

    current_detections = []
    for r in results:
        for box in r.boxes:
            cls      = int(box.cls[0].item())
            conf_val = float(box.conf[0].item())
            label    = _get_label(cls)
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Compute area ratio at inference scale (for distance-aware conf)
            infer_box_area = (x2 - x1) * (y2 - y1)
            area_ratio = infer_box_area / infer_area

            # Scale back to original resolution
            x1 = int(x1 * scale_x); y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x); y2 = int(y2 * scale_y)

            # Distance-aware confidence filter
            eff_conf = _dynamic_confidence(conf_threshold, brightness, area_ratio)
            if conf_val < eff_conf:
                continue

            current_detections.append((label, (x1, y1, x2, y2), conf_val))

    prev_instances = _last_object_state.get(cam_name, {})
    matched, new_detections, lost_ids = _match_instances(prev_instances, current_detections)

    for label, box, conf in new_detections:
        inst_id = _new_id()
        matched[inst_id] = (label, box)
        _write_event(datetime.now(), cam_name, "Object Detected", label, inst_id)

    for inst_id in lost_ids:
        label, _ = prev_instances[inst_id]
        _write_event(datetime.now(), cam_name, "Object Left", label, inst_id)

    _last_object_state[cam_name] = matched
    return matched


def draw(frame, matched):
    """Draw cached matched instances. Fast — pure OpenCV, no inference."""
    for inst_id, (label, (x1, y1, x2, y2)) in matched.items():
        color = _get_color(_LABEL_TO_CLS.get(label, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, OBJECT_TEXT_COLOR, 2)
    return frame, {label for label, _ in matched.values()}


def process(frame, cam_name):
    """Legacy blocking path. Prefer predict()+draw()."""
    matched = predict(frame, cam_name)
    return draw(frame, matched)