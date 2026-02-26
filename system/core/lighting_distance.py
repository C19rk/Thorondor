"""
Preprocessing utilities that improve detection robustness under:
  - Bright / dark / mixed lighting conditions
  - Far / close subject distances (small / large objects)

Drop-in usage:
    from core.lighting_distance import preprocess_frame, _dynamic_confidence
    enhanced, brightness = preprocess_frame(frame)
"""

import threading
import cv2
import numpy as np

# ── CLAHE parameters ──────────────────────────────────────────────────────────
_CLAHE_CLIP = 2.5
_CLAHE_GRID = (8, 8)

# One CLAHE instance per thread — cv2.CLAHE.apply() is NOT thread-safe when
# the same object is shared across threads. pose/object/desk workers all call
# preprocess_frame() concurrently, so each needs its own instance.
_clahe_local = threading.local()

def _get_clahe():
    if not hasattr(_clahe_local, "clahe"):
        _clahe_local.clahe = cv2.createCLAHE(clipLimit=_CLAHE_CLIP, tileGridSize=_CLAHE_GRID)
    return _clahe_local.clahe


def _estimate_brightness(frame_bgr: np.ndarray) -> float:
    """Return mean V channel (0–255) of the centre crop."""
    h, w = frame_bgr.shape[:2]
    cy, cx = h // 2, w // 2
    crop = frame_bgr[cy - h//4: cy + h//4, cx - w//4: cx + w//4]
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 2]))


def _enhance_lighting(frame_bgr: np.ndarray, brightness: float) -> np.ndarray:
    """
    Adaptive CLAHE in LAB space + gamma correction.
    Dark frames:   gamma lift to pull up shadows.
    Bright frames: gentle compression to recover blown highlights.
    """
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = _get_clahe().apply(l)

    if brightness < 80:
        # Dark room — lift shadows aggressively
        gamma = max(0.40, brightness / 100.0)   # 0.40 … 0.80
        lut   = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                          for i in range(256)], dtype=np.uint8)
        l_eq  = cv2.LUT(l_eq, lut)

    elif brightness > 185:
        # Bright / overexposed — compress highlights
        gamma = min(1.6, brightness / 160.0)
        lut   = np.array([((i / 255.0) ** gamma) * 255
                          for i in range(256)], dtype=np.uint8)
        l_eq  = cv2.LUT(l_eq, lut)

    merged = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _dynamic_confidence(base_conf: float, brightness: float,
                         box_area_ratio: float | None = None) -> float:
    """
    Lower confidence threshold when lighting is bad or subject is far away.

    Distance thresholds (inference-space area ratio):
      < 0.02  — medium distance  → -0.08
      < 0.005 — far away         → additional -0.08  (total -0.16)

    Lighting thresholds (mean V, 0-255):
      < 70 or > 210 — poor lighting  → -0.12
      < 100 or > 185 — dim / bright  → -0.06

    Result is clamped to [0.15, base_conf].
    """
    adj = base_conf

    # Lighting penalty
    if brightness < 70 or brightness > 210:
        adj -= 0.12
    elif brightness < 100 or brightness > 185:
        adj -= 0.06

    # Distance penalty — relaxed to catch medium-distance subjects
    if box_area_ratio is not None:
        if box_area_ratio < 0.005:
            adj -= 0.16   # very far
        elif box_area_ratio < 0.02:
            adj -= 0.08   # medium distance

    return float(np.clip(adj, 0.15, base_conf))


def preprocess_frame(frame_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Return (enhanced_frame, brightness).
    enhanced_frame — same size/dtype, lighting normalised.
    brightness     — float 0–255 (mean V of centre crop).
    """
    brightness = _estimate_brightness(frame_bgr)
    enhanced   = _enhance_lighting(frame_bgr, brightness)
    return enhanced, brightness