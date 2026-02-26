import cv2
import csv
import threading
from datetime import datetime

from core.yolo_desk_models import yolo_desk, DEVICE
from core.config import (
    YOLO_DESK_CONF_THRESHOLD,
    LOG_FILE,
    CSV_FILE,
)
from core.lighting_distance import preprocess_frame, _dynamic_confidence

DESK_LABELS   = {0: "Desk"}
DESK_COLOR    = (255, 0, 0)
_SNAP_GRID    = 10
IOU_THRESHOLD = 0.3

# Inference resolution — 16:9 slice of imgsz=320.
_INFER_W = 320
_INFER_H = 180

# Far-distance support: accept smaller boxes (was 1000, lowered to 400)
_MIN_BOX_AREA = 400

_last_desk_state: dict[str, dict] = {}
_next_instance_id = 0

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
        _log_file.write(f"[{timestamp.strftime('%H:%M:%S')}] {cam_name}: {event} (id={inst_id})\n")
        _csv_writer.writerow([timestamp, cam_name, event, label, inst_id])
# ─────────────────────────────────────────────────────────────────────────────

def _snap(value, grid=_SNAP_GRID):
    return (value // grid) * grid

def _snap_box(box):
    x1, y1, x2, y2 = box
    return (_snap(x1), _snap(y1), _snap(x2), _snap(y2))

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

def _match_desks(prev_instances, current_boxes):
    used_prev = set()
    matched   = {}
    new_boxes = []
    for box in current_boxes:
        best_id, best_iou = None, IOU_THRESHOLD
        for inst_id, prev_box in prev_instances.items():
            if inst_id in used_prev:
                continue
            iou = _iou(box, prev_box)
            if iou > best_iou:
                best_iou = iou
                best_id  = inst_id
        if best_id is not None:
            matched[best_id] = box
            used_prev.add(best_id)
        else:
            new_boxes.append(box)
    lost_ids = [i for i in prev_instances if i not in used_prev]
    return matched, new_boxes, lost_ids


def predict(frame, cam_name, person_boxes=None):
    """Run YOLO desk inference on downscaled, lighting-normalised frame.
    Works in bright/dark lighting and at far/close distances.
    Returns matched instances dict. Does NOT draw.
    """
    if person_boxes is None:
        person_boxes = []

    orig_h, orig_w = frame.shape[:2]

    # ── Lighting normalisation (CLAHE + adaptive gamma) ──
    enhanced, brightness = preprocess_frame(frame)
    small   = cv2.resize(enhanced, (_INFER_W, _INFER_H), interpolation=cv2.INTER_LINEAR)
    scale_x = orig_w / _INFER_W
    scale_y = orig_h / _INFER_H

    # Adaptive confidence — relax in poor lighting
    conf_threshold = _dynamic_confidence(YOLO_DESK_CONF_THRESHOLD, brightness)

    infer_area = _INFER_W * _INFER_H

    desk_results = yolo_desk.predict(
        small,
        imgsz=320,
        conf=conf_threshold,
        verbose=False,
        device=DEVICE if DEVICE != "directml" else "cpu",
    )

    current_boxes = []

    for r in desk_results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Compute area ratio at inference scale (for distance-aware conf)
            infer_box_area = (x2 - x1) * (y2 - y1)
            area_ratio = infer_box_area / infer_area

            # Scale back to original resolution
            x1 = int(x1 * scale_x); y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x); y2 = int(y2 * scale_y)
            box_area = (x2 - x1) * (y2 - y1)

            # Distance-aware confidence: small (far) objects get looser threshold
            eff_conf = _dynamic_confidence(conf_threshold, brightness, area_ratio)
            raw_conf = float(box.conf[0].item())
            if raw_conf < eff_conf:
                continue
            if box_area < _MIN_BOX_AREA:
                continue

            overlaps_person = False
            for px1, py1, px2, py2 in person_boxes:
                ix1 = max(x1, px1); iy1 = max(y1, py1)
                ix2 = min(x2, px2); iy2 = min(y2, py2)
                if ix1 < ix2 and iy1 < iy2:
                    if (ix2 - ix1) * (iy2 - iy1) / (box_area + 1e-6) > 0.6:
                        overlaps_person = True
                        break

            if not overlaps_person:
                current_boxes.append(_snap_box((x1, y1, x2, y2)))

    prev_instances = _last_desk_state.get(cam_name, {})
    matched, new_boxes, lost_ids = _match_desks(prev_instances, current_boxes)

    for box in new_boxes:
        inst_id = _new_id()
        matched[inst_id] = box
        _write_event(datetime.now(), cam_name, "Desk Detected", "desk", inst_id)

    for inst_id in lost_ids:
        _write_event(datetime.now(), cam_name, "Desk Left", "desk", inst_id)

    _last_desk_state[cam_name] = matched
    return matched


def draw(frame, matched):
    """Draw cached desk boxes. Fast — pure OpenCV, no inference."""
    for inst_id, (x1, y1, x2, y2) in matched.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), DESK_COLOR, 2)
        cv2.putText(frame, "Desk", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, DESK_COLOR, 2)
    return frame


def process(frame, cam_name, person_boxes=None):
    """Legacy blocking path. Prefer predict()+draw()."""
    matched = predict(frame, cam_name, person_boxes)
    return draw(frame, matched)