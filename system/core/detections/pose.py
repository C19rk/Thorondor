import cv2
import csv
import numpy as np
from collections import deque
from datetime import datetime

from core.pose_models import pose_model
from core.config import (
    POSE_CONF_THRESHOLD,
    LOG_FILE,
    CSV_FILE,
)
from core.lighting_distance import (
    preprocess_frame,
    _dynamic_confidence,
    _estimate_sharpness,
    get_tta_frames,
)

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

KPT_CONF_THRESHOLD = 0.2   # lowered from 0.3 — draws more keypoints on partial views
IOU_THRESHOLD      = 0.3

# Keep the original resize that was working
_INFER_W = 256
_INFER_H = 144

# Centre-point fallback: if IOU fails, match by proximity (15% of frame diagonal)
_CENTRE_MATCH_RATIO  = 0.15

# Majority-vote label smoothing over last N frames — stops Cheating/Normal flickering
_LABEL_SMOOTH_WINDOW = 8

# ── Per-camera state ──────────────────────────────────────────────────────────
_person_instances: dict[str, dict] = {}
_label_history:    dict[str, dict] = {}
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


def _box_centre(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


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


def _match_persons(prev_instances, current_detections, frame_w, frame_h):
    """
    Two-stage matching:
      Stage 1 — IOU (works when person barely moves)
      Stage 2 — Centre-point distance fallback (works when person shifts/leans)
    Stops the tracker assigning a new ID every few frames.
    """
    diag             = (frame_w ** 2 + frame_h ** 2) ** 0.5
    centre_threshold = diag * _CENTRE_MATCH_RATIO

    used_prev = set()
    matched   = {}
    new_dets  = []

    for det in current_detections:
        best_id   = None
        best_iou  = IOU_THRESHOLD
        best_dist = centre_threshold
        det_cx, det_cy = _box_centre(det["box"])

        # Stage 1: IOU
        for inst_id, prev in prev_instances.items():
            if inst_id in used_prev:
                continue
            iou = _iou(det["box"], prev["box"])
            if iou > best_iou:
                best_iou = iou
                best_id  = inst_id

        # Stage 2: centre-point fallback
        if best_id is None:
            for inst_id, prev in prev_instances.items():
                if inst_id in used_prev:
                    continue
                px, py = _box_centre(prev["box"])
                dist   = ((det_cx - px) ** 2 + (det_cy - py) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_id   = inst_id

        if best_id is not None:
            matched[best_id] = det
            used_prev.add(best_id)
        else:
            new_dets.append(det)

    lost_ids = [i for i in prev_instances if i not in used_prev]
    return matched, new_dets, lost_ids


def _smoothed_label(cam_name, inst_id, raw_label):
    """Majority vote over last _LABEL_SMOOTH_WINDOW frames per person."""
    if cam_name not in _label_history:
        _label_history[cam_name] = {}
    if inst_id not in _label_history[cam_name]:
        _label_history[cam_name][inst_id] = deque(maxlen=_LABEL_SMOOTH_WINDOW)
    history = _label_history[cam_name][inst_id]
    history.append(raw_label)
    counts = {}
    for lbl in history:
        counts[lbl] = counts.get(lbl, 0) + 1
    return max(counts, key=counts.get)


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


def _run_inference(small_frame, conf_threshold, scale_x, scale_y):
    """Run pose model on one frame variant. Returns list of detection dicts."""
    pose_results = pose_model.predict(
        small_frame,
        imgsz=256,
        conf=conf_threshold,
        verbose=False,
        device="cpu",
    )

    detections = []
    for r in pose_results:
        boxes    = r.boxes     if (hasattr(r, "boxes")     and r.boxes     is not None) else []
        kpts_obj = r.keypoints if (hasattr(r, "keypoints") and r.keypoints is not None) else None

        kpts_xy   = kpts_obj.xy.cpu().numpy()   if kpts_obj is not None else []
        kpts_conf = kpts_obj.conf.cpu().numpy() if kpts_obj is not None else []

        for idx, box in enumerate(boxes):
            cls      = int(box.cls[0].item())
            conf_val = float(box.conf[0].item())
            label    = _get_label(cls)

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Scale back to original resolution
            x1 = int(x1 * scale_x); y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x); y2 = int(y2 * scale_y)

            kp_xy   = kpts_xy[idx].copy()   if idx < len(kpts_xy)   and len(kpts_xy[idx])   > 0 else np.zeros((17, 2))
            kp_conf = kpts_conf[idx].copy() if idx < len(kpts_conf) and len(kpts_conf[idx]) > 0 else np.zeros(17)

            if len(kp_xy):
                kp_xy[:, 0] *= scale_x
                kp_xy[:, 1] *= scale_y

            detections.append({
                "box":       (x1, y1, x2, y2),
                "label":     label,
                "conf":      conf_val,
                "kpts_xy":   kp_xy,
                "kpts_conf": kp_conf,
            })

    return detections


def _merge_pose_tta(all_detections, iou_threshold=0.3):
    """Merge TTA variants — keep highest-confidence box per person."""
    if len(all_detections) == 1:
        return all_detections[0]
    merged = [det for dets in all_detections for det in dets]
    if not merged:
        return []
    merged.sort(key=lambda d: d["conf"], reverse=True)
    kept = []
    for det in merged:
        if not any(_iou(det["box"], k["box"]) > iou_threshold for k in kept):
            kept.append(det)
    return kept


def predict(frame, cam_name):
    """
    Run pose inference using the original 256×144 resize that works well,
    with 4 additional fixes applied on top:
      1. No double confidence penalty  — detects close AND far
      2. TTA for dark/far conditions   — better recall
      3. Centre-point tracking         — stable person IDs
      4. Label smoothing               — no Cheating/Normal flickering
    """
    orig_h, orig_w = frame.shape[:2]

    enhanced, brightness = preprocess_frame(frame)
    sharpness = _estimate_sharpness(frame)
    scale_x   = orig_w / _INFER_W
    scale_y   = orig_h / _INFER_H

    # Single threshold — NOT re-applied per box (was causing over-filtering)
    conf_threshold = _dynamic_confidence(POSE_CONF_THRESHOLD, brightness)

    # TTA: no overhead in normal conditions, 2-3 variants in dark/far
    tta_variants = get_tta_frames(enhanced, brightness, sharpness)
    is_multi     = len(tta_variants) > 1

    all_detections = []
    for i, variant in enumerate(tta_variants):
        small = cv2.resize(variant, (_INFER_W, _INFER_H), interpolation=cv2.INTER_LINEAR)
        dets  = _run_inference(small, conf_threshold, scale_x, scale_y)

        # Un-flip boxes and keypoints from the flipped variant
        if is_multi and i == len(tta_variants) - 1:
            for det in dets:
                x1, y1, x2, y2 = det["box"]
                det["box"] = (orig_w - x2, y1, orig_w - x1, y2)
                if len(det["kpts_xy"]):
                    det["kpts_xy"] = det["kpts_xy"].copy()
                    det["kpts_xy"][:, 0] = orig_w - det["kpts_xy"][:, 0]

        all_detections.append(dets)

    current_detections = _merge_pose_tta(all_detections, iou_threshold=IOU_THRESHOLD)

    prev_instances = _person_instances.get(cam_name, {})
    matched, new_dets, lost_ids = _match_persons(
        prev_instances, current_detections, orig_w, orig_h
    )

    # Apply label smoothing to all matched persons
    for inst_id, det in matched.items():
        det["label"] = _smoothed_label(cam_name, inst_id, det["label"])

    # New persons — seed their label history
    for det in new_dets:
        inst_id = _new_id()
        det["label"] = _smoothed_label(cam_name, inst_id, det["label"])
        matched[inst_id] = det
        _log_behavior(cam_name, inst_id, det["label"], datetime.now())

    # Log only sustained changes (post-smoothing)
    for inst_id, det in matched.items():
        if inst_id in prev_instances:
            if det["label"] != prev_instances[inst_id]["label"]:
                _log_behavior(cam_name, inst_id, det["label"], datetime.now())

    # Clean up label history for lost persons
    for inst_id in lost_ids:
        if cam_name in _label_history and inst_id in _label_history[cam_name]:
            del _label_history[cam_name][inst_id]

    _person_instances[cam_name] = {
        inst_id: {"box": det["box"], "label": det["label"]}
        for inst_id, det in matched.items()
    }

    return matched


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