"""
Auto-Caption: Emotion-aware video captioning for short-form content.

An AI-powered tool that generates emotionally-intelligent captions for videos
by combining speech recognition with emotion detection and style adaptation.
"""

__version__ = "0.1.0"
__author__ = "Auto Caption Team"
__email__ = "info@ciptadusa.com"

from .caption_generator import CaptionGenerator
from .utils import format_timestamp, parse_time
from .models import WhisperModel, get_available_models
from .emotion_detector import EmotionDetector, EmotionCategory, EmotionScore, EmotionDetectionResult
from .caption_styler import CaptionStyler, StyleIntensity, Platform, EmotionStyleMap
from .video_merger import VideoMerger
from .subtitle import ASSGenerator, ASSStyle

__all__ = [
    "CaptionGenerator",
    "WhisperModel",
    "get_available_models",
    "format_timestamp",
    "parse_time",
    "EmotionDetector",
    "EmotionCategory",
    "EmotionScore",
    "EmotionDetectionResult",
    "CaptionStyler",
    "StyleIntensity",
    "Platform",
    "EmotionStyleMap",
    "VideoMerger",
    "ASSGenerator",
    "ASSStyle",
    "__version__",
    "__author__",
    "__email__",
]
