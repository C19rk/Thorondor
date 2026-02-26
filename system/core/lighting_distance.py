"""
Preprocessing utilities that improve detection robustness under:
  - Bright / dark / mixed lighting conditions
  - Far / close subject distances (small / large objects)

Drop-in usage:
    from core.lighting_distance import preprocess_frame
    enhanced = preprocess_frame(frame)   # returns BGR uint8, same size
"""

import cv2
import numpy as np


# ── CLAHE parameters ──────────────────────────────────────────────────────────
_CLAHE_CLIP  = 2.5   # contrast limit — higher = more aggressive
_CLAHE_GRID  = (8, 8)  # tile grid size
_clahe = cv2.createCLAHE(clipLimit=_CLAHE_CLIP, tileGridSize=_CLAHE_GRID)


def _estimate_brightness(frame_bgr: np.ndarray) -> float:
    """Return mean V channel (0–255) of the centre crop."""
    h, w = frame_bgr.shape[:2]
    cy, cx = h // 2, w // 2
    crop   = frame_bgr[cy - h//4: cy + h//4, cx - w//4: cx + w//4]
    hsv    = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 2]))


def _enhance_lighting(frame_bgr: np.ndarray, brightness: float) -> np.ndarray:
    """
    Apply adaptive CLAHE in LAB space.
    Dark frames: also do a mild gamma-lift.
    Bright frames: mild histogram stretch to pull down blown highlights.
    """
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = _clahe.apply(l)

    # --- Dark frame: lift shadows with gamma < 1 ---
    if brightness < 60:
        gamma   = max(0.45, brightness / 90.0)  # 0.45 … 0.67
        inv_g   = 1.0 / gamma
        lut     = np.array([((i / 255.0) ** inv_g) * 255
                            for i in range(256)], dtype=np.uint8)
        l_eq    = cv2.LUT(l_eq, lut)

    # --- Bright frame: compress highlights ---
    elif brightness > 200:
        # Gentle gamma > 1 to pull back blown-out regions
        gamma   = min(1.6, brightness / 160.0)
        lut     = np.array([((i / 255.0) ** gamma) * 255
                            for i in range(256)], dtype=np.uint8)
        l_eq    = cv2.LUT(l_eq, lut)

    merged = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _dynamic_confidence(base_conf: float, brightness: float,
                         box_area_ratio: float | None = None) -> float:
    """
    Lower confidence threshold when:
      - lighting is bad (very dark or very bright)
      - objects are small (far away)
    Returns adjusted conf, clamped to [0.20, base_conf].
    """
    adj = base_conf

    # Lighting penalty
    if brightness < 50 or brightness > 210:
        adj -= 0.12
    elif brightness < 80 or brightness > 185:
        adj -= 0.06

    # Distance penalty — small box = far away
    if box_area_ratio is not None and box_area_ratio < 0.005:
        adj -= 0.08

    return float(np.clip(adj, 0.20, base_conf))


def preprocess_frame(frame_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Return (enhanced_frame, brightness).
    enhanced_frame — same size / dtype as input, lighting normalised.
    brightness     — float 0-255 (mean V of centre crop), for callers that
                     want to adapt confidence thresholds dynamically.
    """
    brightness = _estimate_brightness(frame_bgr)
    enhanced   = _enhance_lighting(frame_bgr, brightness)
    return enhanced, brightness