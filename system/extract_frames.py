"""
extract_frames.py — Argus Object Detection Model Evaluator
Runs object detection, pose estimation, and desk detection on every
extracted frame.  Evaluation metrics (TP/FP/FN) are computed for
object detection only, as that is what the session CSV tracks.
"""

import os
import re
import sys
import csv
import glob
import argparse
import cv2
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("[ERROR] ultralytics not installed. Run: pip install ultralytics")

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.enums import TA_CENTER
    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False
    print("  ⚠  reportlab not installed — PDF export will be skipped.  Run: pip install reportlab")

# ─────────────────────────────────────────────────────────────────────────────
# Terminal UI helpers  (no external deps — pure ANSI)
# ─────────────────────────────────────────────────────────────────────────────
_IS_TTY = sys.stdout.isatty()

class C:
    """ANSI colour palette — gracefully degrades on non-TTY outputs."""
    _on = _IS_TTY
    RESET  = "\033[0m"     if _on else ""
    BOLD   = "\033[1m"     if _on else ""
    DIM    = "\033[2m"     if _on else ""
    # Foreground
    WHITE  = "\033[97m"    if _on else ""
    CYAN   = "\033[96m"    if _on else ""
    GREEN  = "\033[92m"    if _on else ""
    YELLOW = "\033[93m"    if _on else ""
    RED    = "\033[91m"    if _on else ""
    BLUE   = "\033[94m"    if _on else ""
    GRAY   = "\033[90m"    if _on else ""
    # Backgrounds
    BG_BLUE  = "\033[44m"  if _on else ""
    BG_DARK  = "\033[40m"  if _on else ""


def _c(color, text):
    return f"{color}{text}{C.RESET}"


def _header(title, width=66):
    """Print a bold banner header."""
    pad   = max(0, width - len(title) - 4)
    left  = pad // 2
    right = pad - left
    bar   = "─" * width
    print(f"\n{_c(C.BOLD + C.CYAN, bar)}")
    print(f"{_c(C.BOLD + C.CYAN, '│')} {' ' * left}{_c(C.BOLD + C.WHITE, title)}{' ' * right} {_c(C.BOLD + C.CYAN, '│')}")
    print(f"{_c(C.BOLD + C.CYAN, bar)}")


def _section(title):
    """Print a subtle section divider."""
    line = f"  {_c(C.CYAN + C.BOLD, '▸')} {_c(C.BOLD, title)}"
    print(f"\n{line}")
    print(f"  {_c(C.GRAY, '─' * 60)}")


def _ok(msg):
    print(f"  {_c(C.GREEN, '✔')}  {msg}")

def _warn(msg):
    print(f"  {_c(C.YELLOW, '⚠')}  {_c(C.YELLOW, msg)}")

def _err(msg):
    print(f"  {_c(C.RED, '✖')}  {_c(C.RED, msg)}")

def _info(msg):
    print(f"  {_c(C.GRAY, '·')}  {msg}")


def _progress(done, total, start_time, bar_width=36):
    """
    Overwrite the current line with a progress bar + ETA.
    Example:  ████████████░░░░░░░░  512 / 1066  [00:14 < 00:13]
    """
    frac     = done / total if total else 0
    filled   = int(bar_width * frac)
    empty    = bar_width - filled
    bar      = _c(C.CYAN,  "█" * filled) + _c(C.GRAY, "░" * empty)
    pct      = f"{frac * 100:5.1f}%"
    elapsed  = (datetime.now() - start_time).total_seconds()
    eta_s    = (elapsed / frac - elapsed) if frac > 0 else 0
    elapsed_str = f"{int(elapsed)//60:02d}:{int(elapsed)%60:02d}"
    eta_str     = f"{int(eta_s)//60:02d}:{int(eta_s)%60:02d}"
    counter  = _c(C.BOLD, f"{done:>{len(str(total))}}/{total}")
    timing   = _c(C.GRAY, f"[{elapsed_str} < {eta_str}]")
    print(f"\r  {bar}  {counter}  {pct}  {timing}   ", end="", flush=True)


def _score_bar(value, width=20):
    """Return a compact coloured ░▒▓█ bar representing a 0-1 score."""
    filled = int(round(value * width))
    empty  = width - filled
    if value >= 0.75:
        color = C.GREEN
    elif value >= 0.50:
        color = C.YELLOW
    else:
        color = C.RED
    return _c(color, "█" * filled) + _c(C.GRAY, "░" * empty)


# ── Object detection ──────────────────────────────────────────────────────────
OBJECT_LABELS    = {0: "Phone", 1: "Calculator", 2: "Smartwatch", 3: "Watch"}
LABEL_TO_ID      = {v.lower(): k for k, v in OBJECT_LABELS.items()}
OBJ_COLOR        = (0, 225, 255)
OBJ_TEXT_COLOR   = (0, 140, 255)
OBJ_CONF         = 0.55
OBJ_INFER_W      = 640
OBJ_INFER_H      = 360

# ── Pose estimation ───────────────────────────────────────────────────────────
POSE_LABELS      = {0: "Cheating", 1: "Normal"}
POSE_COLORS      = {"Cheating": (0, 0, 255), "Normal": (0, 255, 0)}
POSE_CONF        = 0.55
POSE_KPT_CONF    = 0.3
POSE_INFER_W     = 256
POSE_INFER_H     = 144
SKELETON         = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6), (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

# ── Desk detection ────────────────────────────────────────────────────────────
DESK_COLOR       = (255, 0, 0)
DESK_CONF        = 0.55
DESK_INFER_W     = 640
DESK_INFER_H     = 360
DESK_MIN_AREA    = 1000

# ── Evaluation ────────────────────────────────────────────────────────────────
IOU_THRESHOLD    = 0.50

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE              = os.path.dirname(os.path.abspath(__file__))
_ML_RUNS           = os.path.normpath(os.path.join(_HERE, "..", "machine_learning", "runs"))

DEFAULT_OBJ_MODEL  = os.path.join(_ML_RUNS, "argus_object_detection",  "weights", "best.pt")
DEFAULT_POSE_MODEL = os.path.join(_ML_RUNS, "pose", "argus_pose_estimation", "weights", "best.pt")
DEFAULT_DESK_MODEL = os.path.join(_ML_RUNS, "argus_desk_detection",    "weights", "best.pt")

RECORDINGS_DIR     = os.path.join(_HERE, "recordings")
DETECTION_LOGS_DIR = os.path.join(_HERE, "detection_logs")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _ts_label(dt):
    return f"{dt.strftime('%b')} {dt.day}, {dt.year} {dt.strftime('%I-%M-%S %p')}"


def _parse_video_timestamp(video_path):
    """
    Extract the recording timestamp from an Argus video filename.

    Handles the format produced by record_video.py:
        Argus Surveillance Recording - Camera 1 - Feb 23, 2026 10-27-15 AM.mp4
        Argus Webcam Surveillance Recording - Feb 23, 2026 10-27-15 AM.mp4

    Falls back to the legacy YYYYMMDD_HHMMSS format, then to datetime.now()
    with a loud warning — a wrong start time makes every frame's ground-truth
    query land at the wrong moment, silently corrupting TP/FP/FN counts.
    """
    stem = os.path.splitext(os.path.basename(video_path))[0]

    # Primary: Argus format  "Feb 23, 2026 10-27-15 AM"
    argus_re = re.compile(
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        r"\s+(\d{1,2}),\s+(\d{4})\s+(\d{2}-\d{2}-\d{2})\s+(AM|PM)"
    )
    m = argus_re.search(stem)
    if m:
        try:
            return datetime.strptime(m.group(0), "%b %d, %Y %I-%M-%S %p")
        except ValueError:
            pass

    # Fallback: legacy YYYYMMDD_HHMMSS format
    m2 = re.search(r"(\d{8})_(\d{6})", stem)
    if m2:
        try:
            return datetime.strptime(m2.group(1) + m2.group(2), "%Y%m%d%H%M%S")
        except ValueError:
            pass

    # Last resort — warn loudly; every frame timestamp will be wrong.
    _warn(
        f"Couldn't read a recording timestamp from the filename:\n"
        f"         {os.path.basename(video_path)}\n"
        f"         Frame times will be anchored to right now, which will\n"
        f"         corrupt TP/FP/FN counts. Rename the file to include a\n"
        f"         timestamp like  'Feb 23, 2026 10-27-15 AM'  or  '20260223_102715'."
    )
    return datetime.now()


def _make_output_folder(video_path, base_dir):
    dt          = _parse_video_timestamp(video_path)
    session_dir = os.path.join(base_dir, _ts_label(dt))
    frames_dir  = os.path.join(session_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    return session_dir, frames_dir, dt


# ─────────────────────────────────────────────────────────────────────────────
# Interactive pickers
# ─────────────────────────────────────────────────────────────────────────────
def _pick_video():
    videos = sorted(glob.glob(os.path.join(RECORDINGS_DIR, "*.mp4")))

    _header("ARGUS  ·  Model Evaluator")
    print(f"\n  {_c(C.GRAY, 'Runs object, pose, and desk detection on a recorded session')}")
    print(f"  {_c(C.GRAY, 'and scores the models against the matching detection log.')}\n")

    if not videos:
        _warn(f"No .mp4 files found in  {RECORDINGS_DIR}")
        print()
        path = input(f"  {_c(C.BOLD, 'Paste the full path to your video:')}  ").strip().strip('"\'')
        return path

    raw_videos   = [v for v in videos if "(Raw)" in v or "_raw" in os.path.basename(v).lower()]
    other_videos = [v for v in videos if v not in raw_videos]
    ordered      = raw_videos + other_videos

    _section("Choose a video to evaluate")
    for i, v in enumerate(ordered, 1):
        size_mb  = os.path.getsize(v) / (1024 * 1024)
        is_raw   = v in raw_videos
        tag      = _c(C.GREEN + C.BOLD, " ★ raw") if is_raw else _c(C.GRAY, "      ")
        name     = _c(C.WHITE if is_raw else C.GRAY, os.path.basename(v))
        idx_str  = _c(C.CYAN + C.BOLD, f"[{i}]")
        size_str = _c(C.GRAY, f"({size_mb:.1f} MB)")
        print(f"    {idx_str}{tag}  {name}  {size_str}")

    print(f"\n    {_c(C.CYAN + C.BOLD, '[0]')}  {_c(C.GRAY, 'Enter a custom path')}\n")
    print(f"  {_c(C.DIM, '★ raw = unprocessed recording, best for evaluation')}\n")

    while True:
        try:
            raw = input(f"  {_c(C.BOLD, 'Your choice:')}  ").strip()
            idx = int(raw)
            if idx == 0:
                path = input(f"  {_c(C.BOLD, 'Video path:')}  ").strip().strip('"\'')
                return path
            if 1 <= idx <= len(ordered):
                chosen = ordered[idx - 1]
                _ok(f"Selected  {_c(C.WHITE, os.path.basename(chosen))}")
                return chosen
        except (ValueError, KeyboardInterrupt):
            print(f"\n  {_c(C.GRAY, 'Goodbye.')}"); sys.exit(0)
        print(f"  {_c(C.YELLOW, f'Please enter a number between 0 and {len(ordered)}.')}")


def _pick_csv():
    csvs = sorted(glob.glob(os.path.join(DETECTION_LOGS_DIR, "*.csv")), reverse=True)

    _section("Choose a detection log  (CSV ground truth)")
    print(f"  {_c(C.GRAY, 'Pick the session log that was recorded at the same time as your video.')}")
    print(f"  {_c(C.GRAY, 'The evaluator uses it to decide what was actually in each frame.')}\n")

    if not csvs:
        _warn(f"No CSV files found in  {DETECTION_LOGS_DIR}")
        print()
        print(f"    {_c(C.CYAN + C.BOLD, '[1]')}  Enter the path manually")
        print(f"    {_c(C.CYAN + C.BOLD, '[0]')}  {_c(C.GRAY, 'Skip  (extract frames only, no scoring)')}\n")
        while True:
            try:
                c = input(f"  {_c(C.BOLD, 'Your choice:')}  ").strip()
                if c == "0":
                    _info("Skipping evaluation — frames will be extracted without scoring.")
                    return None
                if c == "1":
                    p = input(f"  {_c(C.BOLD, 'CSV path:')}  ").strip().strip('"\'')
                    if os.path.isfile(p):
                        _ok(f"Using  {_c(C.WHITE, os.path.basename(p))}")
                        return p
                    _warn("File not found. Try again.")
            except KeyboardInterrupt:
                print(f"\n  {_c(C.GRAY, 'Goodbye.')}"); sys.exit(0)

    for i, c in enumerate(csvs, 1):
        rows     = max(sum(1 for _ in open(c, encoding="utf-8", errors="ignore")) - 1, 0)
        idx_str  = _c(C.CYAN + C.BOLD, f"[{i}]")
        name     = _c(C.WHITE, os.path.basename(c))
        ev_str   = _c(C.GRAY, f"{rows} events")
        print(f"    {idx_str}  {name}  {ev_str}")

    print(f"\n    {_c(C.CYAN + C.BOLD, '[0]')}  {_c(C.GRAY, 'Skip  (extract frames only, no scoring)')}\n")

    while True:
        try:
            idx = int(input(f"  {_c(C.BOLD, 'Your choice:')}  ").strip())
            if idx == 0:
                _info("Skipping evaluation — frames will be extracted without scoring.")
                return None
            if 1 <= idx <= len(csvs):
                chosen = csvs[idx - 1]
                _ok(f"Using  {_c(C.WHITE, os.path.basename(chosen))}")
                return chosen
        except (ValueError, KeyboardInterrupt):
            print(f"\n  {_c(C.GRAY, 'Goodbye.')}"); sys.exit(0)
        print(f"  {_c(C.YELLOW, f'Please enter a number between 0 and {len(csvs)}.')}")


# ─────────────────────────────────────────────────────────────────────────────
# CSV ground-truth helpers  (object detection only)
# ─────────────────────────────────────────────────────────────────────────────
# All event types written by the three detection modules:
#   object.py  -> "Object Detected", "Object Left"      label=Phone/Calculator/...
#   desk.py    -> "Desk Detected",   "Desk Left"         label="desk"
#   pose.py    -> "Behavior Changed"                     label=Cheating/Normal  col4=person_id
_ALL_EVENTS = {
    "Object Detected", "Object Left",
    "Desk Detected",   "Desk Left",
    "Behavior Changed",
}

def _load_csv_events(csv_path):
    """Load every detection event from the session CSV.
    Returns list of (timestamp, event_type, label, extra) sorted by time.
    extra = person_id for pose events, inst_id for desk/object (or empty string).
    """
    events = []
    with open(csv_path, newline="", encoding="utf-8", errors="ignore") as f:
        for row in csv.reader(f):
            if len(row) < 4: continue
            try:
                ts    = datetime.fromisoformat(row[0].strip())
                event = row[2].strip()
                label = row[3].strip()
                extra = row[4].strip() if len(row) > 4 else ""
                if event in _ALL_EVENTS:
                    events.append((ts, event, label, extra))
            except (ValueError, IndexError):
                continue
    events.sort(key=lambda x: x[0])
    return events


def _active_labels_at(events, query_time, tp_window=5.0, fn_window=2.0):
    """Object detection ground truth for a single frame.

    Returns (tp_labels, fn_labels) — two separate sets with intentionally
    different window sizes:

    tp_labels  (wide ±tp_window)
        Used to validate model detections.  If the model fires and a GT event
        exists anywhere within ±tp_window seconds, it is a TRUE POSITIVE.
        Wide window absorbs tracker jitter so real detections aren't called FP.

    fn_labels  (tight backward fn_window)
        Used to count misses.  A label is only considered "should have been
        detected here" if a GT Detected event fired within the last fn_window
        seconds.  This prevents a single Detected event from inflating FN
        across hundreds of frames where the object genuinely wasn't visible.
    """
    tp_start = query_time - timedelta(seconds=tp_window)
    tp_end   = query_time + timedelta(seconds=tp_window)
    fn_start = query_time - timedelta(seconds=fn_window)

    tp_labels = set()
    fn_labels = set()

    for ts, event, label, *_ in events:
        if ts > tp_end:
            break
        if event != "Object Detected":
            continue
        key = label.lower()
        if tp_start <= ts <= tp_end:
            tp_labels.add(key)
        if fn_start <= ts <= query_time:
            fn_labels.add(key)

    return tp_labels, fn_labels


def _desk_active_at(events, query_time, tp_window=5.0, fn_window=2.0):
    """Desk ground truth: returns (gt_for_tp, gt_for_fn).

    gt_for_tp — True if any Desk Detected within ±tp_window  (avoid FP)
    gt_for_fn — True if any Desk Detected within last fn_window  (count misses)
    """
    tp_start = query_time - timedelta(seconds=tp_window)
    tp_end   = query_time + timedelta(seconds=tp_window)
    fn_start = query_time - timedelta(seconds=fn_window)

    gt_tp = False
    gt_fn = False

    for ts, event, label, *_ in events:
        if ts > tp_end:
            break
        if event != "Desk Detected":
            continue
        if tp_start <= ts <= tp_end:
            gt_tp = True
        if fn_start <= ts <= query_time:
            gt_fn = True

    return gt_tp, gt_fn


def _cheating_active_at(events, query_time, tp_window=5.0, fn_window=2.0):
    """Pose ground truth: returns (gt_for_tp, gt_for_fn).

    gt_for_tp — True if any Cheating Behavior Changed within ±tp_window
    gt_for_fn — True if any Cheating Behavior Changed within last fn_window
    """
    tp_start = query_time - timedelta(seconds=tp_window)
    tp_end   = query_time + timedelta(seconds=tp_window)
    fn_start = query_time - timedelta(seconds=fn_window)

    gt_tp = False
    gt_fn = False

    for ts, event, label, extra in events:
        if ts > tp_end:
            break
        if event != "Behavior Changed" or label.lower() != "cheating":
            continue
        if tp_start <= ts <= tp_end:
            gt_tp = True
        if fn_start <= ts <= query_time:
            gt_fn = True

    return gt_tp, gt_fn


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame inference
# ─────────────────────────────────────────────────────────────────────────────
def _run_pose(model, frame):
    """Returns (detections list, person_boxes list).
    person_boxes is passed to desk detection to filter overlapping boxes."""
    if model is None:
        return [], []

    orig_h, orig_w = frame.shape[:2]
    small   = cv2.resize(frame, (POSE_INFER_W, POSE_INFER_H))
    scale_x = orig_w / POSE_INFER_W
    scale_y = orig_h / POSE_INFER_H

    results = model.predict(small, imgsz=256, conf=POSE_CONF, verbose=False)

    detections   = []
    person_boxes = []

    for r in results:
        boxes    = r.boxes     if (hasattr(r, "boxes")     and r.boxes     is not None) else []
        kpts_obj = r.keypoints if (hasattr(r, "keypoints") and r.keypoints is not None) else None
        kpts_xy   = kpts_obj.xy.cpu().numpy()   if kpts_obj is not None else []
        kpts_conf = kpts_obj.conf.cpu().numpy() if kpts_obj is not None else []

        for idx, box in enumerate(boxes):
            cls      = int(box.cls[0].item())
            conf_val = float(box.conf[0].item())
            label    = POSE_LABELS.get(cls, f"Class{cls}")

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1 = int(x1 * scale_x); y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x); y2 = int(y2 * scale_y)

            kp_xy   = kpts_xy[idx].copy()   if idx < len(kpts_xy)   else np.zeros((17, 2))
            kp_conf = kpts_conf[idx].copy() if idx < len(kpts_conf) else np.zeros(17)
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
            person_boxes.append((x1, y1, x2, y2))

    return detections, person_boxes


def _draw_pose(canvas, detections):
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label  = det["label"]
        color  = POSE_COLORS.get(label, (0, 255, 0))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, f"{label} {det['conf']:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        kp_xy, kp_conf = det["kpts_xy"], det["kpts_conf"]
        for i, j in SKELETON:
            if i >= len(kp_xy) or j >= len(kp_xy): continue
            if kp_conf[i] < POSE_KPT_CONF or kp_conf[j] < POSE_KPT_CONF: continue
            pt1 = tuple(kp_xy[i].astype(int))
            pt2 = tuple(kp_xy[j].astype(int))
            if pt1[0] > 1 and pt1[1] > 1 and pt2[0] > 1 and pt2[1] > 1:
                cv2.line(canvas, pt1, pt2, color, 2)
        for kpt, conf in zip(kp_xy, kp_conf):
            if conf < POSE_KPT_CONF: continue
            x, y = int(kpt[0]), int(kpt[1])
            if x > 1 and y > 1:
                cv2.circle(canvas, (x, y), 3, color, -1)


def _run_desk(model, frame, person_boxes):
    """Returns list of (x1,y1,x2,y2) desk boxes, filtered against person_boxes."""
    if model is None:
        return []

    orig_h, orig_w = frame.shape[:2]
    small   = cv2.resize(frame, (DESK_INFER_W, DESK_INFER_H))
    scale_x = orig_w / DESK_INFER_W
    scale_y = orig_h / DESK_INFER_H

    results = model.predict(small, imgsz=640, conf=DESK_CONF, verbose=False)

    boxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1 = int(x1 * scale_x); y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x); y2 = int(y2 * scale_y)
            area = (x2 - x1) * (y2 - y1)
            if area < DESK_MIN_AREA:
                continue
            overlaps = False
            for px1, py1, px2, py2 in person_boxes:
                ix1 = max(x1, px1); iy1 = max(y1, py1)
                ix2 = min(x2, px2); iy2 = min(y2, py2)
                if ix1 < ix2 and iy1 < iy2:
                    if (ix2 - ix1) * (iy2 - iy1) / (area + 1e-6) > 0.6:
                        overlaps = True
                        break
            if not overlaps:
                boxes.append((x1, y1, x2, y2))
    return boxes


def _draw_desk(canvas, boxes):
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), DESK_COLOR, 2)
        cv2.putText(canvas, "Desk", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, DESK_COLOR, 2)


def _run_object(model, frame, conf):
    """Returns list of (label, (x1,y1,x2,y2), conf_val)."""
    if model is None:
        return []

    orig_h, orig_w = frame.shape[:2]
    small   = cv2.resize(frame, (OBJ_INFER_W, OBJ_INFER_H))
    scale_x = orig_w / OBJ_INFER_W
    scale_y = orig_h / OBJ_INFER_H

    results = model.predict(small, imgsz=640, conf=conf, verbose=False)

    detections = []
    for r in results:
        for box in r.boxes:
            cls      = int(box.cls[0].item())
            conf_val = float(box.conf[0].item())
            label    = OBJECT_LABELS.get(cls, f"Class{cls}")
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1 = int(x1 * scale_x); y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x); y2 = int(y2 * scale_y)
            detections.append((label, (x1, y1, x2, y2), conf_val))
    return detections


def _draw_object(canvas, detections):
    for label, (x1, y1, x2, y2), conf in detections:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), OBJ_COLOR, 2)
        cv2.putText(canvas, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, OBJ_TEXT_COLOR, 2)


# ─────────────────────────────────────────────────────────────────────────────
# PDF report
# ─────────────────────────────────────────────────────────────────────────────
def _generate_pdf(pdf_path, video_name, csv_name, args,
                  csv_rows, all_tp, all_fp, all_fn,
                  prec_all, rec_all, f1_all,
                  frames_evaluated, eval_start, eval_end):
    if not REPORTLAB_OK:
        _warn("Skipping PDF — reportlab not installed.")
        return

    doc       = SimpleDocTemplate(pdf_path, pagesize=letter,
                                  rightMargin=72, leftMargin=72,
                                  topMargin=72, bottomMargin=36)
    styles    = getSampleStyleSheet()
    DARK_BLUE = HexColor("#00008B")
    MID_BLUE  = HexColor("#003580")

    title_style = ParagraphStyle("ArgusTitle", parent=styles["Heading1"],
        fontSize=22, textColor=DARK_BLUE, spaceAfter=6,
        alignment=TA_CENTER, fontName="Helvetica-Bold")
    sub_style = ParagraphStyle("ArgusSub", parent=styles["Normal"],
        fontSize=10, textColor=MID_BLUE, spaceAfter=24,
        alignment=TA_CENTER, fontName="Helvetica")
    heading_style = ParagraphStyle("ArgusHeading", parent=styles["Heading2"],
        fontSize=13, textColor=DARK_BLUE, spaceAfter=8,
        spaceBefore=14, fontName="Helvetica-Bold")
    body_style = ParagraphStyle("ArgusBody", parent=styles["BodyText"],
        fontSize=10, fontName="Helvetica", spaceAfter=4)

    duration = eval_end - eval_start
    elements = []

    elements.append(Paragraph("ARGUS DETECTION SYSTEM", title_style))
    elements.append(Paragraph("Object Detection Model Evaluation Report", sub_style))
    elements.append(Spacer(1, 0.1 * inch))

    elements.append(Paragraph("Session Information", heading_style))
    meta_data = [
        ["Report Generated",   datetime.now().strftime("%B %d, %Y at %I:%M:%S %p")],
        ["Evaluation Started", eval_start.strftime("%B %d, %Y at %I:%M:%S %p")],
        ["Evaluation Ended",   eval_end.strftime("%B %d, %Y at %I:%M:%S %p")],
        ["Duration",           str(duration).split(".")[0]],
        ["Video File",         video_name],
        ["CSV Ground Truth",   csv_name],
        ["Frames Evaluated",   str(frames_evaluated)],
        ["Conf Threshold",     str(args.conf)],
        ["IoU Threshold",      str(args.iou_threshold)],
        ["GT Debounce Window", f"{args.gt_window}s  (presence window for CSV ground truth)"],
    ]
    meta_table = Table(meta_data, colWidths=[2.2*inch, 4.0*inch])
    meta_table.setStyle(TableStyle([
        ("FONTNAME",       (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",       (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE",       (0, 0), (-1, -1), 10),
        ("TEXTCOLOR",      (0, 0), (0, -1), DARK_BLUE),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [HexColor("#F0F4FF"), HexColor("#FFFFFF")]),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
        ("TOPPADDING",     (0, 0), (-1, -1), 5),
        ("GRID",           (0, 0), (-1, -1), 0.3, HexColor("#C0C8E0")),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 0.25 * inch))

    elements.append(Paragraph("Evaluation Results", heading_style))
    header = ["Class", "TP", "FP", "FN", "Precision", "Recall", "F1-Score"]
    rows   = [header]
    for label, tp, fp, fn, precision, recall, f1 in csv_rows:
        rows.append([label, str(tp), str(fp), str(fn),
                     f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}"])
    rows.append(["OVERALL", str(all_tp), str(all_fp), str(all_fn),
                 f"{prec_all:.4f}", f"{rec_all:.4f}", f"{f1_all:.4f}"])

    col_w = [1.6*inch, 0.6*inch, 0.6*inch, 0.6*inch, 1.0*inch, 0.9*inch, 0.9*inch]
    tbl   = Table(rows, colWidths=col_w)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0),  DARK_BLUE),
        ("TEXTCOLOR",      (0, 0), (-1, 0),  HexColor("#FFFFFF")),
        ("FONTNAME",       (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, 0),  10),
        ("ALIGN",          (0, 0), (-1, 0),  "CENTER"),
        ("FONTNAME",       (0, 1), (-1, -2), "Helvetica"),
        ("FONTSIZE",       (0, 1), (-1, -1), 9),
        ("ALIGN",          (1, 1), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -2), [HexColor("#F0F4FF"), HexColor("#FFFFFF")]),
        ("BACKGROUND",     (0, -1), (-1, -1), HexColor("#D0D8F0")),
        ("FONTNAME",       (0, -1), (-1, -1), "Helvetica-Bold"),
        ("GRID",           (0, 0), (-1, -1), 0.4, HexColor("#A0A8C0")),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
        ("TOPPADDING",     (0, 0), (-1, -1), 5),
    ]))
    elements.append(tbl)
    elements.append(Spacer(1, 0.25 * inch))

    elements.append(Paragraph("Metric Definitions", heading_style))
    for d in [
        "<b>True Positive (TP)</b>: Model correctly detected an object confirmed by the session log.",
        "<b>False Positive (FP)</b>: Model detected an object not present in the session log.",
        "<b>False Negative (FN)</b>: Session log confirmed an object the model failed to detect.",
        "<b>Precision</b>: TP / (TP + FP) — of all detections made, how many were correct.",
        "<b>Recall</b>: TP / (TP + FN) — of all actual objects, how many were detected.",
        "<b>F1-Score</b>: Harmonic mean of Precision and Recall. >= 0.75 is good, >= 0.85 is excellent.",
    ]:
        elements.append(Paragraph(d, body_style))
    elements.append(Spacer(1, 0.15 * inch))

    elements.append(Paragraph("Ground Truth Method", heading_style))
    elements.append(Paragraph(
        "Ground truth was derived from the session detection log CSV. "
        "For each extracted frame, its timestamp was estimated from the video "
        "start time and frame index. The set of objects actively detected "
        "(Object Detected with no subsequent Object Left) at that moment "
        "was used as the ground truth label set for presence-based TP/FP/FN matching. "
        "Pose estimation and desk detection were also run on each frame and "
        "drawn onto annotated output images, but are not included in the metrics "
        "above as the CSV does not provide structured ground truth for them.",
        body_style,
    ))

    doc.build(elements)
    _ok(f"PDF saved  {_c(C.GRAY, pdf_path)}\n")


def _div(n, d):
    return n / d if d else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────
def run_evaluation(args):
    if not os.path.isfile(args.video):
        _err(f"Video not found:  {args.video}"); sys.exit(1)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        _err(f"OpenCV couldn't open:  {args.video}"); sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_fps      = cap.get(cv2.CAP_PROP_FPS) or 15.0
    vid_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_dir, frames_dir, vid_start_dt = _make_output_folder(args.video, args.output_dir)

    # ── Load models ───────────────────────────────────────────────────────────
    obj_model  = None
    pose_model = None
    desk_model = None

    if not args.extract_only:
        _section("Loading models")
        if not os.path.isfile(args.model):
            _err(f"Object model not found:\n        {args.model}"); sys.exit(1)

        print(f"  {_c(C.GRAY, 'Object model')}   ", end="", flush=True)
        obj_model = YOLO(args.model)
        _ok(f"Object detection ready  {_c(C.GRAY, f'conf={args.conf}')}")

        if os.path.isfile(args.pose_model):
            print(f"  {_c(C.GRAY, 'Pose model')}     ", end="", flush=True)
            pose_model = YOLO(args.pose_model)
            _ok("Pose estimation ready")
        else:
            _warn(f"Pose model not found — pose estimation will be skipped\n"
                  f"         Expected:  {args.pose_model}")

        if os.path.isfile(args.desk_model):
            print(f"  {_c(C.GRAY, 'Desk model')}     ", end="", flush=True)
            desk_model = YOLO(args.desk_model)
            _ok("Desk detection ready")
        else:
            _warn(f"Desk model not found — desk detection will be skipped\n"
                  f"         Expected:  {args.desk_model}")

    _section("Session details")
    frames_to_process = (total_frames + args.step - 1) // args.step
    dur_s  = total_frames / vid_fps
    dur_str = f"{int(dur_s)//60}m {int(dur_s)%60:02d}s"
    _info(f"Video       {_c(C.WHITE, os.path.basename(args.video))}")
    _info(f"Recorded    {_c(C.WHITE, vid_start_dt.strftime('%b %d, %Y  %I:%M:%S %p'))}  {_c(C.GRAY, '(used as t=0 for ground truth)')}")
    _info(f"Resolution  {_c(C.WHITE, f'{vid_w}×{vid_h}')}  @ {vid_fps:.1f} fps  ·  {dur_str}  ·  {total_frames} frames")
    _info(f"Frames out  {_c(C.WHITE, str(frames_to_process))}  {_c(C.GRAY, f'(every {args.step} frame)' if args.step > 1 else '')}")
    _info(f"Output dir  {_c(C.GRAY, frames_dir)}")
    if args.csv:
        _info(f"Ground truth  {_c(C.WHITE, os.path.basename(args.csv))}")
    models_str = "  ".join([
        _c(C.GREEN, "obj") if obj_model else _c(C.GRAY, "obj"),
        _c(C.GREEN, "pose") if pose_model else _c(C.GRAY, "pose"),
        _c(C.GREEN, "desk") if desk_model else _c(C.GRAY, "desk"),
    ])
    _info(f"Models      {models_str}")
    if not args.extract_only and args.csv:
        _info(f"GT window   {_c(C.WHITE, f'{args.gt_window}s')}  {_c(C.GRAY, '± for TP/FP  |  fn-window')}  {_c(C.WHITE, f'{args.fn_window}s')}  {_c(C.GRAY, 'backward for FN')}")
    print()

    events     = []
    has_labels = bool(args.csv) and not args.extract_only
    if has_labels:
        events = _load_csv_events(args.csv)
        _ok(f"Loaded {_c(C.WHITE, str(len(events)))} events from CSV\n")

    # ── Frame loop ────────────────────────────────────────────────────────────
    total_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    eval_start   = datetime.now()
    frame_idx    = 0
    saved_idx    = 0

    _section("Processing frames")
    frames_expected = (total_frames + args.step - 1) // args.step

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % args.step == 0:
            name = f"{saved_idx:04d}"
            cv2.imwrite(os.path.join(frames_dir, f"{name}.jpg"), frame)

            if not args.extract_only:
                # Run in same order as vision.py:
                # 1. Pose → person_boxes
                pose_dets, person_boxes = _run_pose(pose_model, frame)
                # 2. Object detection
                obj_dets = _run_object(obj_model, frame, args.conf)
                # 3. Desk, filtered by person_boxes
                desk_boxes = _run_desk(desk_model, frame, person_boxes)

                # Evaluate all three models against CSV ground truth
                if has_labels:
                    frame_time = vid_start_dt + timedelta(seconds=frame_idx / vid_fps)

                    # ── Object detection ──────────────────────────────────
                    pred_labels = {det[0].lower() for det in obj_dets}
                    gt_tp_labels, gt_fn_labels = _active_labels_at(
                        events, frame_time, args.gt_window, args.fn_window)

                    for lbl in pred_labels:
                        cls_id = LABEL_TO_ID.get(lbl, -1)
                        if cls_id < 0: continue
                        if lbl in gt_tp_labels:
                            total_counts[cls_id]["tp"] += 1
                        else:
                            total_counts[cls_id]["fp"] += 1
                    for lbl in gt_fn_labels:
                        cls_id = LABEL_TO_ID.get(lbl, -1)
                        if cls_id < 0: continue
                        if lbl not in pred_labels:
                            total_counts[cls_id]["fn"] += 1

                    # ── Desk detection ────────────────────────────────────
                    if desk_model is not None:
                        pred_desk        = len(desk_boxes) > 0
                        gt_desk_tp, gt_desk_fn = _desk_active_at(
                            events, frame_time, args.gt_window, args.fn_window)
                        if pred_desk and gt_desk_tp:
                            total_counts["desk"]["tp"] += 1
                        elif pred_desk and not gt_desk_tp:
                            total_counts["desk"]["fp"] += 1
                        elif not pred_desk and gt_desk_fn:
                            total_counts["desk"]["fn"] += 1

                    # ── Pose / Cheating detection ─────────────────────────
                    if pose_model is not None:
                        pred_cheating = any(d["label"] == "Cheating" for d in pose_dets)
                        gt_cheat_tp, gt_cheat_fn = _cheating_active_at(
                            events, frame_time, args.gt_window, args.fn_window)
                        if pred_cheating and gt_cheat_tp:
                            total_counts["cheating"]["tp"] += 1
                        elif pred_cheating and not gt_cheat_tp:
                            total_counts["cheating"]["fp"] += 1
                        elif not pred_cheating and gt_cheat_fn:
                            total_counts["cheating"]["fn"] += 1

                # Draw all three models onto annotated frame
                if args.save_annotated:
                    ann = frame.copy()
                    _draw_pose(ann, pose_dets)
                    _draw_object(ann, obj_dets)
                    _draw_desk(ann, desk_boxes)
                    cv2.imwrite(os.path.join(frames_dir, f"{name}_annotated.jpg"), ann)

            saved_idx += 1
            _progress(saved_idx, frames_expected, eval_start)

        frame_idx += 1

    cap.release()
    eval_end = datetime.now()
    elapsed  = (eval_end - eval_start).total_seconds()
    print()  # newline after progress bar
    _ok(f"{saved_idx} frames saved  {_c(C.GRAY, f'({elapsed:.1f}s  ·  {frames_dir})')}\n")

    if not has_labels:
        if args.extract_only:
            _info("Extract-only mode — no evaluation run.")
        else:
            _info("No CSV selected — evaluation skipped.")
        return

    # ── Compute all metrics first, then interpret ─────────────────────────────
    all_tp = all_fp = all_fn = 0
    csv_rows   = []
    row_data   = {}   # label -> (tp, fp, fn, precision, recall, f1)

    def _compute(label, key, include_in_overall=True):
        nonlocal all_tp, all_fp, all_fn
        c          = total_counts[key]
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        if include_in_overall:
            all_tp += tp; all_fp += fp; all_fn += fn
        precision  = _div(tp, tp + fp)
        recall     = _div(tp, tp + fn)
        f1         = _div(2 * precision * recall, precision + recall)
        csv_rows.append((label, tp, fp, fn, precision, recall, f1))
        row_data[label] = (tp, fp, fn, precision, recall, f1)

    for cls_id in sorted(OBJECT_LABELS):
        _compute(OBJECT_LABELS[cls_id], cls_id)
    if desk_model is not None:
        _compute("Desk", "desk")
    _compute("Cheating", "cheating")

    prec_all = _div(all_tp, all_tp + all_fp)
    rec_all  = _div(all_tp, all_tp + all_fn)
    f1_all   = _div(2 * prec_all * rec_all, prec_all + rec_all)

    # ── Plain-English interpretation ──────────────────────────────────────────
    _section("What the results mean")

    def _interpret_class(label, tp, fp, fn, precision, recall, f1):
        """Emit a short paragraph explaining this class's numbers in plain English."""
        total_pred  = tp + fp
        total_real  = tp + fn
        f1_col      = C.GREEN if f1 >= 0.75 else (C.YELLOW if f1 >= 0.50 else C.RED)
        header      = f"  {_c(C.BOLD + C.WHITE, label)}  {_c(f1_col + C.BOLD, f'F1 {f1:.2f}')}"
        print(header)

        # Nothing was detected and nothing was there
        if total_pred == 0 and total_real == 0:
            print(f"  {_c(C.GRAY, 'Not present in this session — no detections and no ground-truth events.')}")
            print()
            return

        # Nothing predicted at all but should have been
        if total_pred == 0 and total_real > 0:
            print(f"  {_c(C.YELLOW, f'The model never detected a {label} in this entire session,')}")
            print(f"  {_c(C.YELLOW, f'even though the session log recorded {total_real} instance(s).')}")
            print(f"  {_c(C.YELLOW, 'It missed everything — the model may not generalise to this footage.')}")
            print()
            return

        # Build the narrative
        lines = []

        # TP — what went right
        if tp > 0:
            tp_pct = tp / max(total_pred, 1) * 100
            lines.append(
                f"Out of {total_pred} time{'s' if total_pred != 1 else ''} the model "
                f"said it saw a {label}, {tp} of those ({tp_pct:.0f}%) were correct."
            )
        else:
            lines.append(f"The model never made a correct {label} detection in this session.")

        # FP — over-detection
        if fp > 0:
            if fp > tp * 3:
                severity = "extremely trigger-happy"
            elif fp > tp:
                severity = "over-detecting"
            else:
                severity = "occasionally over-detecting"
            lines.append(
                f"It was {severity} though — {fp} false alarm{'s' if fp != 1 else ''} "
                f"where it called something a {label} when it wasn't. "
                f"{'That is more false alarms than correct hits, which drags precision down to ' + f'{precision:.2f}.' if fp >= tp else f'Precision sits at {precision:.2f}.'}"
            )
        else:
            lines.append(
                f"It had zero false alarms — every {label} it reported was confirmed. "
                f"Precision is perfect at 1.00."
            )

        # FN — missed detections
        if fn > 0:
            if fn > tp:
                lines.append(
                    f"On the other side, it missed {fn} real {label} instance{'s' if fn != 1 else ''} "
                    f"— more than it caught. Recall is only {recall:.2f}, meaning it's regularly "
                    f"failing to spot what's actually there."
                )
            else:
                lines.append(
                    f"It also missed {fn} real instance{'s' if fn != 1 else ''} "
                    f"(recall {recall:.2f}), though it caught more than it missed."
                )
        else:
            lines.append(
                f"Impressively, it didn't miss a single real {label} — recall is 1.00."
            )

        # Overall F1 verdict
        if f1 >= 0.85:
            lines.append(f"Overall, the {label} detector is working very well.")
        elif f1 >= 0.75:
            lines.append(f"The {label} detector is solid but has some room left to improve.")
        elif f1 >= 0.50:
            lines.append(
                f"Performance is mediocre. "
                + (f"Reducing false alarms would help most." if fp > fn else
                   f"Getting it to catch more real instances would help most.")
            )
        else:
            lines.append(
                f"This is poor performance. "
                + ("There are far more false alarms than real detections — consider raising the confidence threshold."
                   if fp > fn * 2 else
                   "The model is missing most real detections — it may not be trained well enough on this class.")
            )

        # Print wrapped at ~72 chars
        import textwrap
        for line in lines:
            for wrapped in textwrap.wrap(line, width=70):
                print(f"  {_c(C.GRAY, wrapped)}" if wrapped == line else f"  {_c(C.GRAY, wrapped)}")
        print()

    for cls_id in sorted(OBJECT_LABELS):
        label = OBJECT_LABELS[cls_id]
        _interpret_class(label, *row_data[label])

    if desk_model is not None:
        _interpret_class("Desk", *row_data["Desk"])

    _interpret_class("Cheating", *row_data["Cheating"])

    # ── Overall summary ───────────────────────────────────────────────────────
    print(f"  {_c(C.GRAY, '─' * 60)}")
    print(f"  {_c(C.BOLD, 'Overall')}  —  "
          f"{all_tp} correct  /  {all_fp} false alarms  /  {all_fn} missed")
    print(f"  Precision {prec_all:.2f}   Recall {rec_all:.2f}   "
          f"{_c(C.BOLD, 'F1')} {_c((C.GREEN if f1_all >= 0.75 else C.YELLOW if f1_all >= 0.50 else C.RED) + C.BOLD, f'{f1_all:.2f}')}")
    print()

    import textwrap
    if f1_all >= 0.85:
        verdict = ("The suite is performing excellently across the board. "
                   "You can trust these results in a real session.")
    elif f1_all >= 0.75:
        verdict = ("The suite is performing well overall. There are a few classes "
                   "pulling the score down — the per-class notes above show where to focus.")
    elif f1_all >= 0.50:
        verdict = (
            "Performance is moderate. "
            + (f"The biggest problem right now is false alarms ({all_fp} FP vs {all_fn} FN) — "
               f"try raising the confidence threshold or tightening the GT window."
               if all_fp > all_fn else
               f"The biggest problem right now is missed detections ({all_fn} FN vs {all_fp} FP) — "
               f"the models aren't catching enough of what's really there.")
        )
    else:
        if all_fp > all_fn * 2:
            verdict = (
                f"The results are poor, driven mainly by {all_fp} false alarms. "
                f"Something in the footage is being misidentified repeatedly. "
                f"Try a higher confidence threshold, check that the ground-truth CSV "
                f"matches this video, or increase --gt-window if flicker is the cause."
            )
        else:
            verdict = (
                f"The results are poor, with {all_fn} missed detections dominating. "
                f"The models may not be recognising objects in this lighting or camera angle. "
                f"Consider retraining on footage closer to these conditions."
            )

    for line in textwrap.wrap(verdict, width=68):
        print(f"  {_c(C.WHITE, line)}")
    print()

    csv_out = os.path.join(out_dir, "evaluation_results.csv")
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "tp", "fp", "fn", "precision", "recall", "f1_score"])
        for row in csv_rows:
            w.writerow([row[0], row[1], row[2], row[3],
                        f"{row[4]:.4f}", f"{row[5]:.4f}", f"{row[6]:.4f}"])
        w.writerow(["OVERALL", all_tp, all_fp, all_fn,
                    f"{prec_all:.4f}", f"{rec_all:.4f}", f"{f1_all:.4f}"])
    print()
    _ok(f"CSV saved  {_c(C.GRAY, csv_out)}")

    folder_ts = os.path.basename(out_dir)
    pdf_path  = os.path.join(out_dir, f"Argus Evaluation Report - {folder_ts}.pdf")
    _generate_pdf(
        pdf_path,
        video_name       = os.path.basename(args.video),
        csv_name         = os.path.basename(args.csv),
        args             = args,
        csv_rows         = csv_rows,
        all_tp=all_tp, all_fp=all_fp, all_fn=all_fn,
        prec_all=prec_all, rec_all=rec_all, f1_all=f1_all,
        frames_evaluated = saved_idx,
        eval_start       = eval_start,
        eval_end         = eval_end,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(prog="extract_frames.py",
        description="Argus - Object / Pose / Desk Model Evaluator")
    parser.add_argument("--video",          default=None)
    parser.add_argument("--csv",            default=None)
    parser.add_argument("--model",          default=DEFAULT_OBJ_MODEL,
                        help="Path to object detection weights (best.pt)")
    parser.add_argument("--pose-model",     default=DEFAULT_POSE_MODEL,
                        help="Path to pose estimation weights (best.pt)")
    parser.add_argument("--desk-model",     default=DEFAULT_DESK_MODEL,
                        help="Path to desk detection weights (best.pt)")
    parser.add_argument("--output-dir",     default="extracted_frames")
    parser.add_argument("--step",           type=int,   default=1)
    parser.add_argument("--conf",           type=float, default=OBJ_CONF,
                        help="Confidence threshold for object detection")
    parser.add_argument("--iou-threshold",  type=float, default=IOU_THRESHOLD)
    parser.add_argument("--gt-window",      type=float, default=5.0,
                        help="TP/FP validation window in seconds (default 5). "
                             "A detection is TP if a GT event exists within ±gt-window.")
    parser.add_argument("--fn-window",      type=float, default=2.0,
                        help="FN counting window in seconds (default 2). "
                             "A frame only counts as a miss if a GT event fired "
                             "within the last fn-window seconds.")
    parser.add_argument("--extract-only",   action="store_true",
                        help="Save raw frames only, skip all model inference")
    parser.add_argument("--save-annotated", action="store_true",
                        help="Also save annotated frames with all model overlays")

    args = parser.parse_args()

    if args.video is None:
        args.video = _pick_video()
        print()

    if args.csv is None and not args.extract_only:
        args.csv = _pick_csv()
        print()

    run_evaluation(args)


if __name__ == "__main__":
    main()