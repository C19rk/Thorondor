"""
re_export_onnx.py  —  Re-exports all three Argus models to ONNX at imgsz=640.

Run from the system/ directory:
    python re_export_onnx.py

This replaces the old ONNX files (exported at 320/256) with new ones at 640,
which matches the training resolution and restores far-distance detection.
The .pt files are NOT modified.
"""

import os
import sys

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("[ERROR] ultralytics not installed. Run: pip install ultralytics")

# ── Model paths (relative to system/) ────────────────────────────────────────
MODELS = [
    {
        "name":  "Object Detection",
        "pt":    "../machine_learning/runs/argus_object_detection/weights/best.pt",
        "imgsz": 640,
    },
    {
        "name":  "Desk Detection",
        "pt":    "../machine_learning/runs/argus_desk_detection/weights/best.pt",
        "imgsz": 640,
    },
    {
        "name":  "Pose Estimation",
        "pt":    "../machine_learning/runs/pose/argus_pose_estimation/weights/best.pt",
        "imgsz": 256,
    },
]

# ─────────────────────────────────────────────────────────────────────────────

for m in MODELS:
    pt_path   = os.path.normpath(os.path.join(os.path.dirname(__file__), m["pt"]))
    onnx_path = pt_path.replace(".pt", ".onnx")

    if not os.path.isfile(pt_path):
        print(f"[SKIP] {m['name']}: .pt not found at {pt_path}")
        continue

    print(f"\n[EXPORT] {m['name']}")
    print(f"         .pt   → {pt_path}")
    print(f"         .onnx → {onnx_path}")
    print(f"         imgsz = {m['imgsz']}")

    model = YOLO(pt_path)
    model.export(
        format="onnx",
        imgsz=m["imgsz"],
        simplify=True,
        dynamic=False,   # fixed input shape — faster inference
        opset=12,
    )

    if os.path.isfile(onnx_path):
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"         ✓ Done — {size_mb:.1f} MB")
    else:
        print(f"         ✗ Export may have saved to a different path — check above.")

print("\n[DONE] All models re-exported. Restart your system to use the new ONNX files.\n")