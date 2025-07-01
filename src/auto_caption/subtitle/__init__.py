"""
Subtitle generation module for Auto-Caption.

This module provides functionality for generating styled subtitle files
in various formats (ASS, SRT) with emotion-based formatting.
"""

from .ass_generator import ASSGenerator, ASSStyle

__all__ = [
    "ASSGenerator",
    "ASSStyle",
]