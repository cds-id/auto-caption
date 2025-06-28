"""
Auto-Caption: Automatic video captioning using OpenAI Whisper.

A command-line tool for generating accurate captions/subtitles from video files
using state-of-the-art speech recognition.
"""

__version__ = "0.1.0"
__author__ = "Auto Caption Team"
__email__ = "your.email@example.com"

from .caption_generator import CaptionGenerator
from .utils import format_timestamp, parse_time
from .models import WhisperModel, get_available_models

__all__ = [
    "CaptionGenerator",
    "WhisperModel",
    "get_available_models",
    "format_timestamp",
    "parse_time",
    "__version__",
    "__author__",
    "__email__",
]