"""
lighting_distance.py - Lighting and Distance Detection Module
Analyzes video frames for lighting conditions and object distance estimation
Compatible with macOS Sonoma 14.8.2+
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class LightingLevel(Enum):
    """Lighting intensity classification"""
    VERY_DARK = "Very Dark"      # < 20
    DARK = "Dark"                 # 20-50
    DIM = "Dim"                   # 50-100
    NORMAL = "Normal"             # 100-150
    BRIGHT = "Bright"             # 150-200
    VERY_BRIGHT = "Very Bright"   # > 200


class LightingCondition(Enum):
    """Lighting quality/type classification"""
    UNEVEN = "Uneven"
    BACKLIT = "Backlit"
    FRONTAL = "Frontal"
    SIDE = "Side Lit"
    UNIFORM = "Uniform"


@dataclass
class LightingAnalysis:
    """Result of lighting analysis"""
    brightness_level: float  # 0-255
    lighting_quality: LightingCondition
    lighting_intensity: LightingLevel
    histogram_mean: float
    histogram_std: float
    brightness_variance: float
    contrast_ratio: float
    has_shadows: bool
    shadow_percentage: float


@dataclass
class DistanceAnalysis:
    """Result of distance analysis"""
    estimated_distance: float  # in arbitrary units
    focal_length: float
    object_width: float
    perceived_size: float
    reliability_score: float  # 0-1, higher is more reliable
    distance_category: str  # "Close", "Medium", "Far"


class LightingDetector:
    """Detects and analyzes lighting conditions in video frames"""

    def __init__(self, hist_bins: int = 256):
        self.hist_bins = hist_bins

    def analyze_brightness(self, frame: np.ndarray) -> Tuple[float, float, float]:
        """
        Analyze overall brightness of the frame.
        
        Returns:
            brightness_mean: Average brightness (0-255)
            brightness_std: Standard deviation of brightness
            brightness_variance: Variance of brightness distribution
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        brightness_mean = np.mean(gray)
        brightness_std = np.std(gray)
        brightness_variance = np.var(gray)

        return brightness_mean, brightness_std, brightness_variance

    def analyze_contrast(self, frame: np.ndarray) -> float:
        """
        Analyze contrast using Michelson contrast formula.
        
        Returns:
            contrast_ratio: L_max - L_min / L_max + L_min
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        l_max = np.max(gray)
        l_min = np.min(gray)

        if (l_max + l_min) == 0:
            return 0.0

        contrast = (l_max - l_min) / (l_max + l_min)
        return float(contrast)

    def detect_shadows(self, frame: np.ndarray, threshold: int = 50) -> Tuple[bool, float]:
        """
        Detect presence and percentage of shadows in frame.
        
        Args:
            frame: Input frame (BGR or grayscale)
            threshold: Brightness threshold for shadow detection
            
        Returns:
            has_shadows: Boolean indicating presence of shadows
            shadow_percentage: Percentage of frame that is shadowed (0-100)
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        shadow_mask = gray < threshold
        shadow_percentage = (np.sum(shadow_mask) / shadow_mask.size) * 100

        return shadow_percentage > 5, shadow_percentage

    def detect_lighting_condition(self, frame: np.ndarray) -> LightingCondition:
        """
        Detect type of lighting condition.
        
        Returns:
            LightingCondition enum indicating the type of lighting
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
        else:
            gray = frame
            h, w = gray.shape

        # Divide frame into regions
        mid_h, mid_w = h // 2, w // 2
        top_half = gray[:mid_h, :]
        bottom_half = gray[mid_h:, :]
        left_half = gray[:, :mid_w]
        right_half = gray[:, mid_w:]
        center_region = gray[h//4:3*h//4, w//4:3*w//4]

        top_brightness = np.mean(top_half)
        bottom_brightness = np.mean(bottom_half)
        left_brightness = np.mean(left_half)
        right_brightness = np.mean(right_half)
        center_brightness = np.mean(center_region)

        # Analyze brightness distribution
        brightness_values = [top_brightness, bottom_brightness, left_brightness, right_brightness]
        brightness_std = np.std(brightness_values)

        # Check for backlit condition (bright top, dark bottom)
        if top_brightness > center_brightness + 30 and bottom_brightness < center_brightness - 30:
            return LightingCondition.BACKLIT

        # Check for side lighting
        if abs(left_brightness - right_brightness) > 40:
            return LightingCondition.SIDE

        # Check for uneven lighting
        if brightness_std > 25:
            return LightingCondition.UNEVEN

        # Check for frontal lighting
        if center_brightness > 120:
            return LightingCondition.FRONTAL

        # Default to uniform
        return LightingCondition.UNIFORM

    def classify_brightness_level(self, brightness: float) -> LightingLevel:
        """Classify brightness into predefined levels"""
        if brightness < 20:
            return LightingLevel.VERY_DARK
        elif brightness < 50:
            return LightingLevel.DARK
        elif brightness < 100:
            return LightingLevel.DIM
        elif brightness < 150:
            return LightingLevel.NORMAL
        elif brightness < 200:
            return LightingLevel.BRIGHT
        else:
            return LightingLevel.VERY_BRIGHT

    def analyze(self, frame: np.ndarray) -> LightingAnalysis:
        """
        Perform complete lighting analysis on frame.
        
        Args:
            frame: Input frame (BGR or grayscale)
            
        Returns:
            LightingAnalysis object with detailed lighting metrics
        """
        brightness_mean, brightness_std, brightness_variance = self.analyze_brightness(frame)
        contrast_ratio = self.analyze_contrast(frame)
        has_shadows, shadow_percentage = self.detect_shadows(frame)
        lighting_condition = self.detect_lighting_condition(frame)
        lighting_intensity = self.classify_brightness_level(brightness_mean)

        return LightingAnalysis(
            brightness_level=brightness_mean,
            lighting_quality=lighting_condition,
            lighting_intensity=lighting_intensity,
            histogram_mean=brightness_mean,
            histogram_std=brightness_std,
            brightness_variance=brightness_variance,
            contrast_ratio=contrast_ratio,
            has_shadows=has_shadows,
            shadow_percentage=shadow_percentage,
        )


class DistanceDetector:
    """Estimates object distance using size-based analysis"""

    def __init__(self, focal_length: float = 800.0):
        """
        Initialize distance detector.
        
        Args:
            focal_length: Camera focal length (in pixels)
                         Adjust based on your camera calibration
        """
        self.focal_length = focal_length
        self.reference_sizes = {
            "Phone": 70,      # mm
            "Calculator": 100,  # mm
            "Smartwatch": 45,   # mm
            "Watch": 50,        # mm
        }

    def estimate_distance(
        self,
        bbox_width: float,
        bbox_height: float,
        object_label: str = None,
        known_width: Optional[float] = None,
    ) -> DistanceAnalysis:
        """
        Estimate distance to object using perspective analysis.
        
        Args:
            bbox_width: Bounding box width in pixels
            bbox_height: Bounding box height in pixels
            object_label: Label of detected object
            known_width: Known real-world width in mm (optional, overrides reference)
            
        Returns:
            DistanceAnalysis with estimated distance and reliability
        """
        # Determine object width
        if known_width:
            object_width = known_width
        elif object_label and object_label in self.reference_sizes:
            object_width = self.reference_sizes[object_label]
        else:
            object_width = 70  # Default fallback (phone width)

        # Use average dimension for more stable estimation
        perceived_size = (bbox_width + bbox_height) / 2

        # Avoid division by zero
        if perceived_size < 1:
            perceived_size = 1

        # Calculate distance using pinhole camera model
        # distance = (object_width * focal_length) / perceived_size
        estimated_distance = (object_width * self.focal_length) / perceived_size

        # Calculate reliability based on bbox size
        # Larger detected objects = more reliable distance estimate
        reliability_score = min(perceived_size / 100, 1.0)

        # Classify distance
        if estimated_distance < 300:
            distance_category = "Close"
        elif estimated_distance < 700:
            distance_category = "Medium"
        else:
            distance_category = "Far"

        return DistanceAnalysis(
            estimated_distance=estimated_distance,
            focal_length=self.focal_length,
            object_width=object_width,
            perceived_size=perceived_size,
            reliability_score=reliability_score,
            distance_category=distance_category,
        )

    def calibrate_focal_length(self, bbox_width: float, known_distance: float, object_width: float) -> None:
        """
        Calibrate focal length using a known distance measurement.
        
        Args:
            bbox_width: Detected bounding box width in pixels
            known_distance: Known distance to object in mm
            object_width: Known object width in mm
        """
        if bbox_width > 0 and known_distance > 0:
            self.focal_length = (bbox_width * known_distance) / object_width

    def estimate_distance_from_multiple_detections(
        self, detections: list
    ) -> Dict[str, DistanceAnalysis]:
        """
        Estimate distances for multiple detected objects.
        
        Args:
            detections: List of detection dicts with 'bbox', 'label' keys
                       bbox format: [x1, y1, x2, y2]
                       
        Returns:
            Dictionary mapping detection to DistanceAnalysis
        """
        results = {}

        for i, det in enumerate(detections):
            if "bbox" not in det:
                continue

            x1, y1, x2, y2 = det["bbox"]
            width = x2 - x1
            height = y2 - y1
            label = det.get("label", "Phone")

            distance_analysis = self.estimate_distance(width, height, label)
            results[f"detection_{i}"] = distance_analysis

        return results


class LightingAndDistanceAnalyzer:
    """Combined analyzer for lighting and distance detection"""

    def __init__(self, focal_length: float = 800.0):
        self.lighting_detector = LightingDetector()
        self.distance_detector = DistanceDetector(focal_length=focal_length)

    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """
        Perform complete analysis on frame.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Dictionary containing lighting and distance analysis
        """
        lighting_analysis = self.lighting_detector.analyze(frame)

        return {
            "lighting": {
                "brightness_level": round(lighting_analysis.brightness_level, 2),
                "brightness_level_category": lighting_analysis.lighting_intensity.value,
                "lighting_condition": lighting_analysis.lighting_quality.value,
                "contrast_ratio": round(lighting_analysis.contrast_ratio, 3),
                "brightness_variance": round(lighting_analysis.brightness_variance, 2),
                "has_shadows": lighting_analysis.has_shadows,
                "shadow_percentage": round(lighting_analysis.shadow_percentage, 2),
            },
            "timestamp": None,  # Will be set by caller
        }

    def analyze_frame_with_detections(
        self, frame: np.ndarray, detections: list
    ) -> Dict:
        """
        Perform complete analysis with object detections.
        
        Args:
            frame: Input frame (BGR)
            detections: List of detected objects
            
        Returns:
            Dictionary containing lighting and distance analysis
        """
        base_analysis = self.analyze_frame(frame)
        distance_analyses = self.distance_detector.estimate_distance_from_multiple_detections(
            detections
        )

        base_analysis["detections"] = {}
        for det_key, distance_analysis in distance_analyses.items():
            base_analysis["detections"][det_key] = {
                "estimated_distance": round(distance_analysis.estimated_distance, 2),
                "distance_category": distance_analysis.distance_category,
                "reliability_score": round(distance_analysis.reliability_score, 3),
                "object_width_mm": distance_analysis.object_width,
            }

        return base_analysis


# Export main classes
__all__ = [
    "LightingDetector",
    "DistanceDetector",
    "LightingAndDistanceAnalyzer",
    "LightingAnalysis",
    "DistanceAnalysis",
    "LightingLevel",
    "LightingCondition",
]