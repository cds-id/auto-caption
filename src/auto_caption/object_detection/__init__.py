"""
Object detection module for smart caption positioning.

This module provides functionality to detect faces, people, and other important
objects in videos to ensure captions don't obstruct key visual elements.
"""

from .object_detector import (
    ObjectDetector,
    DetectedObject,
    ObjectType,
    DetectionResult,
    SafeZone
)
from .position_optimizer import (
    PositionOptimizer,
    PositionConstraint,
    OptimizationResult,
    PositioningStrategy
)
from .face_tracker import (
    FaceTracker,
    TrackedFace,
    FaceTrackingResult
)

__all__ = [
    "ObjectDetector",
    "DetectedObject",
    "ObjectType",
    "DetectionResult",
    "SafeZone",
    "PositionOptimizer",
    "PositionConstraint",
    "OptimizationResult",
    "PositioningStrategy",
    "FaceTracker",
    "TrackedFace",
    "FaceTrackingResult",
]