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
    print("[WARN] reportlab not installed — PDF export skipped. Run: pip install reportlab")

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
    print(
        f"\n[WARN] Could not parse a recording timestamp from:\n"
        f"       {os.path.basename(video_path)}\n"
        f"       Ground-truth frame times will be relative to RIGHT NOW.\n"
        f"       TP/FP/FN counts will be incorrect. Rename the file to include\n"
        f"       a timestamp like \'Feb 23, 2026 10-27-15 AM\' or \'20260223_102715\'.\n"
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
    print("\n" + "=" * 64)
    print("  ARGUS - Object / Pose / Desk Model Evaluator")
    print("=" * 64)
    if not videos:
        print(f"\n  [!] No .mp4 files found in: {RECORDINGS_DIR}")
        return input("  Enter full path to video file: ").strip().strip('"\'')
    raw_videos   = [v for v in videos if "(Raw)" in v or "_raw" in os.path.basename(v).lower()]
    other_videos = [v for v in videos if v not in raw_videos]
    ordered      = raw_videos + other_videos
    print(f"\n  Videos in recordings/  (* = raw, recommended for evaluation)\n")
    for i, v in enumerate(ordered, 1):
        size_mb = os.path.getsize(v) / (1024 * 1024)
        star    = "* " if v in raw_videos else "  "
        print(f"    [{i}]{star}{os.path.basename(v)}  ({size_mb:.1f} MB)")
    print(f"\n    [0] Enter a custom path\n")
    while True:
        try:
            idx = int(input(f"  Select video [1-{len(ordered)}]: ").strip())
            if idx == 0:
                return input("  Path: ").strip().strip('"\'')
            if 1 <= idx <= len(ordered):
                return ordered[idx - 1]
        except (ValueError, KeyboardInterrupt):
            print("\n  Cancelled."); sys.exit(0)
        print(f"  Enter a number between 0 and {len(ordered)}.")


def _pick_csv():
    csvs = sorted(glob.glob(os.path.join(DETECTION_LOGS_DIR, "*.csv")), reverse=True)
    print("\n  -- Detection Log (CSV) ----------------------------------------------")
    print("  Pick the session CSV that matches your video recording time.")
    print("  This is used as ground truth for object detection evaluation.\n")
    if not csvs:
        print(f"  [!] No CSV files found in: {DETECTION_LOGS_DIR}")
        print("\n    [1] Enter path manually\n    [0] Skip\n")
        while True:
            try:
                c = input("  Choice [1/0]: ").strip()
                if c == "0": return None
                if c == "1":
                    p = input("  CSV path: ").strip().strip('"\'')
                    return p if os.path.isfile(p) else None
            except KeyboardInterrupt:
                sys.exit(0)
    print(f"  Session logs found (newest first):\n")
    for i, c in enumerate(csvs, 1):
        rows = max(sum(1 for _ in open(c, encoding="utf-8", errors="ignore")) - 1, 0)
        print(f"    [{i}] {os.path.basename(c)}  ({rows} events)")
    print(f"\n    [0] Skip - extract frames only, no evaluation\n")
    while True:
        try:
            idx = int(input(f"  Select CSV [1-{len(csvs)}]: ").strip())
            if idx == 0: return None
            if 1 <= idx <= len(csvs): return csvs[idx - 1]
        except (ValueError, KeyboardInterrupt):
            print("\n  Cancelled."); sys.exit(0)
        print(f"  Enter a number between 0 and {len(csvs)}.")


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


def _active_labels_at(events, query_time):
    """Object detection ground truth: set of lower-case labels currently active."""
    active = {}
    for ts, event, label, *_ in events:
        if ts > query_time: break
        key = label.lower()
        if event == "Object Detected":
            active[key] = active.get(key, 0) + 1
        elif event == "Object Left":
            active[key] = max(0, active.get(key, 0) - 1)
    return {lbl for lbl, cnt in active.items() if cnt > 0}


def _desk_active_at(events, query_time):
    """Desk ground truth: True if any desk instance is currently active."""
    active = 0
    for ts, event, label, *_ in events:
        if ts > query_time: break
        if event == "Desk Detected":
            active += 1
        elif event == "Desk Left":
            active = max(0, active - 1)
    return active > 0


def _cheating_active_at(events, query_time):
    """Pose ground truth: True if any tracked person is currently labeled Cheating.
    pose.py logs 'Behavior Changed' every time a person's label changes, so the
    current label for person P is their most recent 'Behavior Changed' event.
    """
    # Track most recent label per person_id
    person_labels = {}
    for ts, event, label, extra in events:
        if ts > query_time: break
        if event == "Behavior Changed":
            person_labels[extra] = label.lower()   # extra = "person_N"
    return any(lbl == "cheating" for lbl in person_labels.values())


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
        print("[WARN] Skipping PDF - reportlab not installed.")
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
    print(f"[INFO] PDF report saved to:\n       {pdf_path}\n")


def _div(n, d):
    return n / d if d else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────
def run_evaluation(args):
    if not os.path.isfile(args.video):
        sys.exit(f"\n[ERROR] Video not found: {args.video}")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"\n[ERROR] Could not open: {args.video}")

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
        if not os.path.isfile(args.model):
            sys.exit(f"[ERROR] Object model not found:\n        {args.model}")
        print(f"[INFO] Loading object model  -> {args.model}")
        obj_model = YOLO(args.model)
        print(f"[INFO] Object model ready.  Conf={args.conf}  IoU={args.iou_threshold}")

        if os.path.isfile(args.pose_model):
            print(f"[INFO] Loading pose model    -> {args.pose_model}")
            pose_model = YOLO(args.pose_model)
            print(f"[INFO] Pose model ready.")
        else:
            print(f"[WARN] Pose model not found — skipping pose estimation.\n"
                  f"       Expected: {args.pose_model}")

        if os.path.isfile(args.desk_model):
            print(f"[INFO] Loading desk model    -> {args.desk_model}")
            desk_model = YOLO(args.desk_model)
            print(f"[INFO] Desk model ready.")
        else:
            print(f"[WARN] Desk model not found — skipping desk detection.\n"
                  f"       Expected: {args.desk_model}")

    print(f"\n{'='*64}")
    print(f"  Video      : {os.path.basename(args.video)}")
    print(f"  Video start: {vid_start_dt.strftime('%b %d, %Y %I:%M:%S %p')}  "
          f"(used as t=0 for CSV ground-truth lookup)")
    print(f"  Resolution : {vid_w}x{vid_h}  |  FPS: {vid_fps:.2f}")
    print(f"  Frames     : {total_frames}  |  Step: every {args.step} frame(s)")
    print(f"  Frames dir : {frames_dir}")
    print(f"  Session dir: {out_dir}")
    if args.csv:
        print(f"  CSV log    : {os.path.basename(args.csv)}")
    print(f"  Models     : obj={'yes' if obj_model else 'skip'}  "
          f"pose={'yes' if pose_model else 'skip'}  "
          f"desk={'yes' if desk_model else 'skip'}")
    print(f"{'='*64}\n")

    events     = []
    has_labels = bool(args.csv) and not args.extract_only
    if has_labels:
        events = _load_csv_events(args.csv)
        print(f"[INFO] Loaded {len(events)} events from CSV.\n")

    # ── Frame loop ────────────────────────────────────────────────────────────
    total_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    eval_start   = datetime.now()
    frame_idx    = 0
    saved_idx    = 0

    print("[INFO] Extracting and evaluating frames ...\n")

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
                    gt_labels   = _active_labels_at(events, frame_time)

                    for lbl in pred_labels:
                        cls_id = LABEL_TO_ID.get(lbl, -1)
                        if cls_id < 0: continue
                        if lbl in gt_labels:
                            total_counts[cls_id]["tp"] += 1
                        else:
                            total_counts[cls_id]["fp"] += 1
                    for lbl in gt_labels:
                        cls_id = LABEL_TO_ID.get(lbl, -1)
                        if cls_id < 0: continue
                        if lbl not in pred_labels:
                            total_counts[cls_id]["fn"] += 1

                    # ── Desk detection ────────────────────────────────────
                    if desk_model is not None:
                        pred_desk = len(desk_boxes) > 0
                        gt_desk   = _desk_active_at(events, frame_time)
                        if pred_desk and gt_desk:
                            total_counts["desk"]["tp"] += 1
                        elif pred_desk and not gt_desk:
                            total_counts["desk"]["fp"] += 1
                        elif not pred_desk and gt_desk:
                            total_counts["desk"]["fn"] += 1

                    # ── Pose / Cheating detection ─────────────────────────
                    if pose_model is not None:
                        pred_cheating = any(d["label"] == "Cheating" for d in pose_dets)
                        gt_cheating   = _cheating_active_at(events, frame_time)
                        if pred_cheating and gt_cheating:
                            total_counts["cheating"]["tp"] += 1
                        elif pred_cheating and not gt_cheating:
                            total_counts["cheating"]["fp"] += 1
                        elif not pred_cheating and gt_cheating:
                            total_counts["cheating"]["fn"] += 1

                # Draw all three models onto annotated frame
                if args.save_annotated:
                    ann = frame.copy()
                    _draw_pose(ann, pose_dets)
                    _draw_object(ann, obj_dets)
                    _draw_desk(ann, desk_boxes)
                    cv2.imwrite(os.path.join(frames_dir, f"{name}_annotated.jpg"), ann)

            saved_idx += 1
            if saved_idx % 100 == 0:
                print(f"  ... {saved_idx} frames processed", end="\r")

        frame_idx += 1

    cap.release()
    eval_end = datetime.now()
    print(f"\n[INFO] Done - {saved_idx} frames saved to:\n       {frames_dir}\n")

    if not has_labels:
        if args.extract_only:
            print("[INFO] Extract-only mode — evaluation skipped.")
        else:
            print("[INFO] No CSV log selected — evaluation skipped.")
        return

    # ── Print evaluation ──────────────────────────────────────────────────────
    print(f"{'='*66}")
    print("  EVALUATION RESULTS - Argus Detection Suite")
    print(f"{'='*66}")
    print(f"  IoU Threshold   : {args.iou_threshold}")
    print(f"  Conf Threshold  : {args.conf}")
    print(f"  Frames Evaluated: {saved_idx}")
    print(f"  CSV Ground Truth: {os.path.basename(args.csv)}")
    print(f"{'='*66}")
    print(f"  {'Class':<18} {'TP':>5} {'FP':>5} {'FN':>5}  "
          f"{'Precision':>10} {'Recall':>8} {'F1-Score':>9}")
    print(f"  {'-'*67}")

    all_tp = all_fp = all_fn = 0
    csv_rows = []

    def _print_and_collect(label, key, include_in_overall=True):
        nonlocal all_tp, all_fp, all_fn
        c          = total_counts[key]
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        if include_in_overall:
            all_tp += tp; all_fp += fp; all_fn += fn
        precision  = _div(tp, tp + fp)
        recall     = _div(tp, tp + fn)
        f1         = _div(2 * precision * recall, precision + recall)
        print(f"  {label:<18} {tp:>5} {fp:>5} {fn:>5}  "
              f"{precision:>10.4f} {recall:>8.4f} {f1:>9.4f}")
        csv_rows.append((label, tp, fp, fn, precision, recall, f1))

    # Object detection rows
    print(f"  -- Object Detection {'─'*47}")
    for cls_id in sorted(OBJECT_LABELS):
        _print_and_collect(OBJECT_LABELS[cls_id], cls_id)

    # Desk detection row (only if model was loaded)
    if desk_model is not None:
        print(f"  -- Desk Detection {'─'*49}")
        _print_and_collect("Desk", "desk")

    # Pose / Cheating row (always shown)
    print(f"  -- Pose Estimation {'─'*48}")
    _print_and_collect("Cheating", "cheating")

    prec_all = _div(all_tp, all_tp + all_fp)
    rec_all  = _div(all_tp, all_tp + all_fn)
    f1_all   = _div(2 * prec_all * rec_all, prec_all + rec_all)

    print(f"  {'-'*67}")
    print(f"  {'OVERALL':>18} {all_tp:>5} {all_fp:>5} {all_fn:>5}  "
          f"{prec_all:>10.4f} {rec_all:>8.4f} {f1_all:>9.4f}")
    print(f"{'='*66}\n")

    csv_out = os.path.join(out_dir, "evaluation_results.csv")
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "tp", "fp", "fn", "precision", "recall", "f1_score"])
        for row in csv_rows:
            w.writerow([row[0], row[1], row[2], row[3],
                        f"{row[4]:.4f}", f"{row[5]:.4f}", f"{row[6]:.4f}"])
        w.writerow(["OVERALL", all_tp, all_fp, all_fn,
                    f"{prec_all:.4f}", f"{rec_all:.4f}", f"{f1_all:.4f}"])
    print(f"[INFO] CSV saved to:\n       {csv_out}\n")

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