"""
Preprocessing utilities that improve detection robustness under:
  - Bright / dark / mixed lighting conditions
  - Far / close subject distances (small / large objects in frame)

How it works
------------
1. preprocess_frame(frame)
   ├─ Estimates brightness via HSV-V channel of the centre crop.
   ├─ Estimates sharpness via Laplacian variance → "close / medium / far".
   ├─ Applies bilateral denoising for very dark frames (noise → false edges).
   ├─ Applies CLAHE in LAB space for local contrast normalisation.
   ├─ Applies adaptive gamma: lifts dark frames, compresses blown highlights.
   └─ Applies unsharp masking when subject appears far / blurry.
   Returns (enhanced_frame, brightness).  API is drop-in compatible.

2. _dynamic_confidence(base_conf, brightness, box_area_ratio=None)
   Lowers the YOLO confidence threshold when lighting is poor or subject is far.

3. get_tta_frames(frame, brightness, sharpness)
   Test-Time Augmentation — returns a list of frame variants to run inference on
   when conditions are difficult (dark, far, or both). Detections from all
   variants are merged, dramatically improving recall without retraining.

   TTA is ONLY activated in difficult conditions to avoid slowing down
   normal well-lit close-range frames.

Drop-in usage (unchanged):
    from core.lighting_distance import preprocess_frame, _dynamic_confidence
    enhanced, brightness = preprocess_frame(frame)

TTA usage (in predict functions):
    from core.lighting_distance import preprocess_frame, _dynamic_confidence, get_tta_frames, merge_tta_detections
    enhanced, brightness = preprocess_frame(frame)
    frames_to_run = get_tta_frames(enhanced, brightness)
    # run model on each, collect all boxes, then:
    final_detections = merge_tta_detections(all_detections, iou_threshold=0.3)
"""

import threading
import cv2
import numpy as np

# ── CLAHE parameters ──────────────────────────────────────────────────────────
_CLAHE_CLIP = 2.5
_CLAHE_GRID = (8, 8)

_clahe_local = threading.local()


def _get_clahe() -> cv2.CLAHE:
    if not hasattr(_clahe_local, "clahe"):
        _clahe_local.clahe = cv2.createCLAHE(
            clipLimit=_CLAHE_CLIP, tileGridSize=_CLAHE_GRID
        )
    return _clahe_local.clahe


# ── Brightness estimation ─────────────────────────────────────────────────────

def _estimate_brightness(frame_bgr: np.ndarray) -> float:
    """Return mean V channel (0–255) of the centre 50% crop."""
    h, w = frame_bgr.shape[:2]
    cy, cx = h // 2, w // 2
    crop = frame_bgr[cy - h // 4 : cy + h // 4, cx - w // 4 : cx + w // 4]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 2]))


# ── Sharpness / distance estimation ──────────────────────────────────────────

def _estimate_sharpness(frame_bgr: np.ndarray) -> float:
    """
    Return Laplacian variance as a proxy for image sharpness.
    Low  → blurry / far subject.
    High → sharp / close subject.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _distance_category(sharpness: float) -> str:
    if sharpness >= 300:
        return "close"
    elif sharpness >= 100:
        return "medium"
    else:
        return "far"


# ── LUT helper ────────────────────────────────────────────────────────────────

def _build_lut(gamma: float, invert: bool = False) -> np.ndarray:
    exp = gamma if invert else (1.0 / gamma)
    return np.array(
        [((i / 255.0) ** exp) * 255 for i in range(256)], dtype=np.uint8
    )


# ── Per-frame noise reduction ─────────────────────────────────────────────────

def _denoise_dark(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(frame_bgr, d=5, sigmaColor=40, sigmaSpace=40)


# ── Lighting enhancement ──────────────────────────────────────────────────────

def _enhance_lighting(frame_bgr: np.ndarray, brightness: float) -> np.ndarray:
    if brightness < 60:
        frame_bgr = _denoise_dark(frame_bgr)

    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b_ch = cv2.split(lab)
    l_eq = _get_clahe().apply(l)

    if brightness < 80:
        gamma = max(0.40, brightness / 100.0)
        l_eq = cv2.LUT(l_eq, _build_lut(gamma, invert=False))
    elif brightness > 185:
        gamma = min(1.6, brightness / 160.0)
        l_eq = cv2.LUT(l_eq, _build_lut(gamma, invert=True))

    merged = cv2.merge([l_eq, a, b_ch])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


# ── Sharpening ────────────────────────────────────────────────────────────────

def _sharpen_frame(frame_bgr: np.ndarray, strength: float = 1.0) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    k = max(3, (w // 160) | 1)
    blurred = cv2.GaussianBlur(frame_bgr, (k, k), 0)
    return cv2.addWeighted(frame_bgr, 1.0 + strength, blurred, -strength, 0)


# ── Public preprocessing entry-point ─────────────────────────────────────────

def preprocess_frame(frame_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Apply lighting normalisation and distance-aware sharpening.
    Returns (enhanced_frame, brightness).
    """
    brightness = _estimate_brightness(frame_bgr)
    sharpness  = _estimate_sharpness(frame_bgr)
    dist_cat   = _distance_category(sharpness)

    enhanced = _enhance_lighting(frame_bgr, brightness)

    if dist_cat == "far":
        enhanced = _sharpen_frame(enhanced, strength=1.0)
    elif dist_cat == "medium":
        enhanced = _sharpen_frame(enhanced, strength=0.5)

    return enhanced, brightness


# ── Test-Time Augmentation ────────────────────────────────────────────────────

def _is_difficult(brightness: float, sharpness: float) -> bool:
    """
    Returns True when conditions are difficult enough to warrant TTA.
    TTA is skipped for normal well-lit close-range frames to avoid
    unnecessary CPU cost.

    Difficult = any of:
      - Very dark  (brightness < 80)
      - Very bright/overexposed (brightness > 200)
      - Far/blurry subject (sharpness < 150)
    """
    return brightness < 80 or brightness > 200 or sharpness < 150


def get_tta_frames(frame_bgr: np.ndarray, brightness: float, sharpness: float) -> list[np.ndarray]:
    """
    Return a list of augmented frame variants for test-time augmentation.

    In normal conditions → returns [frame_bgr] (single frame, no overhead).
    In difficult conditions → returns up to 3 variants:
      1. Base enhanced frame (already preprocessed)
      2. Slightly brighter variant  — helps catch objects in darker regions
      3. Horizontally flipped       — catches objects the model is biased
                                      against detecting on one side

    The caller runs inference on each variant and merges results with
    merge_tta_detections(). Boxes from flipped variants are un-flipped
    before merging so all coordinates are in original frame space.
    """
    if not _is_difficult(brightness, sharpness):
        return [frame_bgr]

    variants = [frame_bgr]

    # Variant 2: brighter version — helps in dark rooms
    if brightness < 80:
        brighter = cv2.convertScaleAbs(frame_bgr, alpha=1.3, beta=20)
        variants.append(brighter)

    # Variant 3: horizontal flip — model may be biased toward one side
    # (flip flag stored alongside so caller can un-flip boxes)
    flipped = cv2.flip(frame_bgr, 1)
    variants.append(flipped)

    return variants


def unflip_boxes(boxes: list, frame_width: int) -> list:
    """
    Mirror bounding box x-coordinates back to original space after
    running inference on a horizontally flipped frame.

    boxes: list of (label, (x1,y1,x2,y2), conf)  OR  list of (x1,y1,x2,y2)
    """
    unflipped = []
    for item in boxes:
        if len(item) == 3:
            label, (x1, y1, x2, y2), conf = item
            new_x1 = frame_width - x2
            new_x2 = frame_width - x1
            unflipped.append((label, (new_x1, y1, new_x2, y2), conf))
        else:
            x1, y1, x2, y2 = item
            unflipped.append((frame_width - x2, y1, frame_width - x1, y2))
    return unflipped


def merge_tta_detections(
    all_detections: list[list],
    iou_threshold: float = 0.3,
) -> list:
    """
    Merge detections from multiple TTA variants using Non-Maximum Suppression.

    all_detections: list of detection lists, one per TTA variant.
      Each detection is (label, (x1,y1,x2,y2), conf).

    Returns a deduplicated list keeping the highest-confidence box
    when multiple variants detected the same object.
    """
    if len(all_detections) == 1:
        return all_detections[0]

    # Flatten all detections
    merged = []
    for det_list in all_detections:
        merged.extend(det_list)

    if not merged:
        return []

    # Group by label, then NMS within each group
    from collections import defaultdict
    by_label = defaultdict(list)
    for label, box, conf in merged:
        by_label[label].append((box, conf))

    final = []
    for label, items in by_label.items():
        items.sort(key=lambda x: x[1], reverse=True)  # sort by conf desc
        kept = []
        for box, conf in items:
            # Suppress if heavily overlapping with an already-kept box
            suppress = False
            for kept_box, _ in kept:
                if _iou(box, kept_box) > iou_threshold:
                    suppress = True
                    break
            if not suppress:
                kept.append((box, conf))
        for box, conf in kept:
            final.append((label, box, conf))

    return final


def _iou(boxA, boxB) -> float:
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA);   interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(areaA + areaB - interArea)


# ── Dynamic confidence adjustment ────────────────────────────────────────────

def _dynamic_confidence(
    base_conf: float,
    brightness: float,
    box_area_ratio: float | None = None,
) -> float:
    """
    Lower the YOLO confidence threshold when lighting is poor or subject is far.

    Lighting penalties  (mean V channel, 0–255)
    ─────────────────────────────────────────────
    < 70  or > 210   very dark / blown out   → −0.12
    < 100 or > 185   dim / moderately bright → −0.06

    Distance penalties  (inference-space bounding-box area ÷ frame area)
    ─────────────────────────────────────────────────────────────────────
    area_ratio < 0.005   very far             → −0.16
    area_ratio < 0.020   medium distance      → −0.08

    Result is clamped to [0.15, base_conf].
    """
    adj = base_conf

    if brightness < 70 or brightness > 210:
        adj -= 0.12
    elif brightness < 100 or brightness > 185:
        adj -= 0.06

    if box_area_ratio is not None:
        if box_area_ratio < 0.005:
            adj -= 0.16
        elif box_area_ratio < 0.02:
            adj -= 0.08

    return float(np.clip(adj, 0.15, base_conf))