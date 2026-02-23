"""
extract_frames.py — Argus Object Detection Model Evaluator
"""

import os
import re
import sys
import csv
import glob
import argparse
import cv2
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

OBJECT_LABELS    = {0: "Phone", 1: "Calculator", 2: "Smartwatch", 3: "Watch"}
LABEL_TO_ID      = {v.lower(): k for k, v in OBJECT_LABELS.items()}
CONF_THRESHOLD   = 0.75
IOU_THRESHOLD    = 0.50
INFER_W, INFER_H = 320, 180

_HERE              = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL      = os.path.normpath(os.path.join(
    _HERE, "..", "machine_learning", "runs",
    "argus_object_detection", "weights", "best.pt",
))
RECORDINGS_DIR     = os.path.join(_HERE, "recordings")
DETECTION_LOGS_DIR = os.path.join(_HERE, "detection_logs")


def _ts_label(dt):
    return f"{dt.strftime('%b')} {dt.day}, {dt.year} {dt.strftime('%I-%M-%S %p')}"


def _make_output_folder(video_path, base_dir):
    stem  = os.path.splitext(os.path.basename(video_path))[0]
    match = re.search(r"(\d{8})_(\d{6})", stem)
    dt    = (datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")
             if match else datetime.now())
    path  = os.path.join(base_dir, _ts_label(dt))
    os.makedirs(path, exist_ok=True)
    return path, dt


def _pick_video():
    videos = sorted(glob.glob(os.path.join(RECORDINGS_DIR, "*.mp4")))
    print("\n" + "=" * 64)
    print("  ARGUS - Object Detection Model Evaluator")
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
    print("  This is used as ground truth - no manual annotation needed.\n")
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


def _load_csv_events(csv_path):
    events = []
    with open(csv_path, newline="", encoding="utf-8", errors="ignore") as f:
        for row in csv.reader(f):
            if len(row) < 4: continue
            try:
                ts    = datetime.fromisoformat(row[0].strip())
                event = row[2].strip()
                label = row[3].strip()
                if event in ("Object Detected", "Object Left"):
                    events.append((ts, event, label))
            except (ValueError, IndexError):
                continue
    events.sort(key=lambda x: x[0])
    return events


def _active_labels_at(events, query_time):
    active = {}
    for ts, event, label in events:
        if ts > query_time: break
        key = label.lower()
        if event == "Object Detected":
            active[key] = active.get(key, 0) + 1
        elif event == "Object Left":
            active[key] = max(0, active.get(key, 0) - 1)
    return {lbl for lbl, cnt in active.items() if cnt > 0}


def _generate_pdf(pdf_path, video_name, csv_name, args,
                  csv_rows, all_tp, all_fp, all_fn,
                  prec_all, rec_all, f1_all,
                  frames_evaluated, eval_start, eval_end):
    if not REPORTLAB_OK:
        print("[WARN] Skipping PDF - reportlab not installed.")
        return

    doc      = SimpleDocTemplate(pdf_path, pagesize=letter,
                                 rightMargin=72, leftMargin=72,
                                 topMargin=72, bottomMargin=36)
    styles   = getSampleStyleSheet()
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
    rows.append(["OVERALL (micro)", str(all_tp), str(all_fp), str(all_fn),
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
        "was used as the ground truth label set for presence-based TP/FP/FN matching.",
        body_style,
    ))

    doc.build(elements)
    print(f"[INFO] PDF report saved to:\n       {pdf_path}\n")


def _div(n, d):
    return n / d if d else 0.0


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

    out_dir, vid_start_dt = _make_output_folder(args.video, args.output_dir)

    print(f"\n{'='*64}")
    print(f"  Video      : {os.path.basename(args.video)}")
    print(f"  Resolution : {vid_w}x{vid_h}  |  FPS: {vid_fps:.2f}")
    print(f"  Frames     : {total_frames}  |  Step: every {args.step} frame(s)")
    print(f"  Output     : {out_dir}")
    if args.csv:
        print(f"  CSV log    : {os.path.basename(args.csv)}")
    print(f"{'='*64}\n")

    events     = []
    has_labels = bool(args.csv) and not args.extract_only
    if has_labels:
        events = _load_csv_events(args.csv)
        print(f"[INFO] Loaded {len(events)} events from CSV.\n")

    model = None
    if not args.extract_only:
        if not os.path.isfile(args.model):
            sys.exit(f"[ERROR] Model weights not found:\n        {args.model}")
        print(f"[INFO] Loading model -> {args.model}")
        model = YOLO(args.model)
        print(f"[INFO] Model ready.  Conf={args.conf}  IoU={args.iou_threshold}\n")

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
            cv2.imwrite(os.path.join(out_dir, f"{name}.jpg"), frame)

            if not args.extract_only:
                small   = cv2.resize(frame, (INFER_W, INFER_H))
                scale_x = vid_w / INFER_W
                scale_y = vid_h / INFER_H
                results = model.predict(small, imgsz=320, conf=args.conf, verbose=False)

                pred_labels = set()
                pred_boxes  = []
                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls[0].item())
                        lbl = OBJECT_LABELS.get(cls, "").lower()
                        pred_labels.add(lbl)
                        if args.save_annotated:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            pred_boxes.append((cls, [int(x1*scale_x), int(y1*scale_y),
                                                     int(x2*scale_x), int(y2*scale_y)]))

                if has_labels:
                    frame_time = vid_start_dt + timedelta(seconds=frame_idx / vid_fps)
                    gt_labels  = _active_labels_at(events, frame_time)

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

                if args.save_annotated:
                    ann = frame.copy()
                    for cls, (x1, y1, x2, y2) in pred_boxes:
                        lbl = OBJECT_LABELS.get(cls, f"Class{cls}")
                        cv2.rectangle(ann, (x1, y1), (x2, y2), (0, 225, 255), 2)
                        cv2.putText(ann, lbl, (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2)
                    cv2.imwrite(os.path.join(out_dir, f"{name}_annotated.jpg"), ann)

            saved_idx += 1
            if saved_idx % 100 == 0:
                print(f"  ... {saved_idx} frames processed", end="\r")

        frame_idx += 1

    cap.release()
    eval_end = datetime.now()
    print(f"\n[INFO] Done - {saved_idx} frames saved to:\n       {out_dir}\n")

    if not has_labels:
        if args.extract_only:
            print("[INFO] Extract-only mode - evaluation skipped.")
        else:
            print("[INFO] No CSV log selected - evaluation skipped.")
        return

    print(f"{'='*66}")
    print("  EVALUATION RESULTS - Argus Object Detection")
    print(f"{'='*66}")
    print(f"  IoU Threshold   : {args.iou_threshold}")
    print(f"  Conf Threshold  : {args.conf}")
    print(f"  Frames Evaluated: {saved_idx}")
    print(f"  CSV Ground Truth: {os.path.basename(args.csv)}")
    print(f"{'='*66}")
    print(f"  {'Class':<15} {'TP':>5} {'FP':>5} {'FN':>5}  "
          f"{'Precision':>10} {'Recall':>8} {'F1-Score':>9}")
    print(f"  {'-'*64}")

    all_tp = all_fp = all_fn = 0
    csv_rows = []

    for cls_id in sorted(OBJECT_LABELS):
        c          = total_counts[cls_id]
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        all_tp    += tp;  all_fp += fp;  all_fn += fn
        precision  = _div(tp, tp + fp)
        recall     = _div(tp, tp + fn)
        f1         = _div(2 * precision * recall, precision + recall)
        label      = OBJECT_LABELS[cls_id]
        print(f"  {label:<15} {tp:>5} {fp:>5} {fn:>5}  "
              f"{precision:>10.4f} {recall:>8.4f} {f1:>9.4f}")
        csv_rows.append((label, tp, fp, fn, precision, recall, f1))

    prec_all = _div(all_tp, all_tp + all_fp)
    rec_all  = _div(all_tp, all_tp + all_fn)
    f1_all   = _div(2 * prec_all * rec_all, prec_all + rec_all)

    print(f"  {'-'*64}")
    print(f"  {'OVERALL (micro)':<15} {all_tp:>5} {all_fp:>5} {all_fn:>5}  "
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


def main():
    parser = argparse.ArgumentParser(prog="extract_frames.py",
        description="Argus - YOLO Object Detection Model Evaluator")
    parser.add_argument("--video",          default=None)
    parser.add_argument("--csv",            default=None)
    parser.add_argument("--model",          default=DEFAULT_MODEL)
    parser.add_argument("--output-dir",     default="extracted_frames")
    parser.add_argument("--step",           type=int,   default=1)
    parser.add_argument("--conf",           type=float, default=CONF_THRESHOLD)
    parser.add_argument("--iou-threshold",  type=float, default=IOU_THRESHOLD)
    parser.add_argument("--extract-only",   action="store_true")
    parser.add_argument("--save-annotated", action="store_true")

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