"""
Font configuration module for ASS subtitle generation.

This module manages custom fonts for emotion-based subtitle styling,
including font detection, fallback management, and platform-specific adjustments.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import platform

from ..emotion_detector import EmotionCategory


@dataclass
class FontInfo:
    """Information about a font."""
    name: str
    file_path: Optional[str] = None
    fallback_names: List[str] = None
    style_attributes: Dict[str, any] = None
    
    def __post_init__(self):
        if self.fallback_names is None:
            self.fallback_names = []
        if self.style_attributes is None:
            self.style_attributes = {}


class FontStyle(Enum):
    """Font style categories."""
    MODERN = "modern"          # MADE AVENUE - Clean, modern, approachable
    DRAMATIC = "dramatic"      # CINEMATOGRAFICA - Bold, cinematic, impactful
    EXPRESSIVE = "expressive"  # ALMOST TEXTUAL - Emotional, handwritten feel


class FontConfig:
    """
    Configuration manager for custom fonts in subtitle generation.
    """
    
    # Default font mappings
    DEFAULT_FONTS = {
        "MADE AVENUE": FontInfo(
            name="MADE AVENUE",
            fallback_names=["Helvetica Neue", "Arial Rounded MT Bold", "Arial", "sans-serif"],
            style_attributes={
                "weight": "medium",
                "style": "modern",
                "readability": "high",
                "emotion_fit": ["happy", "excited", "neutral"],
                "characteristics": "Clean, modern, approachable, friendly",
                "best_for": "Positive emotions, uplifting content, energetic delivery"
            }
        ),
        "CINEMATOGRAFICA": FontInfo(
            name="CINEMATOGRAFICA",
            fallback_names=["Impact", "Bebas Neue", "Anton", "Arial Black", "sans-serif"],
            style_attributes={
                "weight": "bold",
                "style": "dramatic",
                "readability": "medium",
                "emotion_fit": ["angry", "sarcastic", "dramatic", "surprised", "disgusted"],
                "characteristics": "Bold, cinematic, impactful, attention-grabbing",
                "best_for": "Intense emotions, dramatic moments, strong reactions"
            }
        ),
        "ALMOST TEXTUAL": FontInfo(
            name="ALMOST TEXTUAL",
            fallback_names=["Amatic SC", "Kalam", "Caveat", "Comic Sans MS", "cursive"],
            style_attributes={
                "weight": "light",
                "style": "expressive",
                "readability": "medium",
                "emotion_fit": ["sad", "anxious", "contemplative", "fearful"],
                "characteristics": "Soft, handwritten, emotional, personal",
                "best_for": "Vulnerable emotions, introspective moments, gentle delivery"
            }
        )
    }
    
    # Emotion to font style mapping
    EMOTION_FONT_MAPPING = {
        # MADE AVENUE - Clean, modern, friendly emotions
        EmotionCategory.HAPPY: "MADE AVENUE",
        EmotionCategory.EXCITED: "MADE AVENUE",
        EmotionCategory.NEUTRAL: "MADE AVENUE",
        EmotionCategory.HUMOROUS: "MADE AVENUE",
        EmotionCategory.CASUAL: "MADE AVENUE",
        EmotionCategory.PROFESSIONAL: "MADE AVENUE",
        
        # CINEMATOGRAFICA - Bold, dramatic, intense emotions
        EmotionCategory.ANGRY: "CINEMATOGRAFICA",
        EmotionCategory.SARCASTIC: "CINEMATOGRAFICA",
        EmotionCategory.SURPRISED: "CINEMATOGRAFICA",
        EmotionCategory.DISGUSTED: "CINEMATOGRAFICA",
        EmotionCategory.MOTIVATIONAL: "CINEMATOGRAFICA",
        EmotionCategory.DRAMATIC: "CINEMATOGRAFICA",
        
        # ALMOST TEXTUAL - Soft, emotional, introspective emotions
        EmotionCategory.SAD: "ALMOST TEXTUAL",
        EmotionCategory.FEARFUL: "ALMOST TEXTUAL",
        EmotionCategory.ANXIOUS: "ALMOST TEXTUAL",
        EmotionCategory.CONTEMPLATIVE: "ALMOST TEXTUAL",
        EmotionCategory.ROMANTIC: "ALMOST TEXTUAL",
        EmotionCategory.NOSTALGIC: "ALMOST TEXTUAL",
        EmotionCategory.MELANCHOLIC: "ALMOST TEXTUAL",
        EmotionCategory.CONFUSED: "ALMOST TEXTUAL"
    }
    
    def __init__(
        self,
        custom_font_dir: Optional[str] = None,
        font_config_file: Optional[str] = None,
        auto_detect: bool = True
    ):
        """
        Initialize font configuration.
        
        Args:
            custom_font_dir: Directory containing custom font files
            font_config_file: Path to JSON configuration file
            auto_detect: Automatically detect system fonts
        """
        self.custom_font_dir = Path(custom_font_dir) if custom_font_dir else None
        self.font_config_file = font_config_file
        self.auto_detect = auto_detect
        
        # Font cache
        self._font_cache = {}
        self._system_fonts = {}
        
        # Load configuration
        self._load_configuration()
        
        # Detect system fonts if enabled
        if self.auto_detect:
            self._detect_system_fonts()
    
    def _load_configuration(self):
        """Load font configuration from file if provided."""
        if self.font_config_file and os.path.exists(self.font_config_file):
            try:
                with open(self.font_config_file, 'r') as f:
                    config = json.load(f)
                    
                # Update font paths
                for font_name, font_path in config.get("font_paths", {}).items():
                    if font_name in self.DEFAULT_FONTS:
                        self.DEFAULT_FONTS[font_name].file_path = font_path
                        
                # Add custom fonts
                for font_name, font_info in config.get("custom_fonts", {}).items():
                    self.DEFAULT_FONTS[font_name] = FontInfo(
                        name=font_name,
                        file_path=font_info.get("path"),
                        fallback_names=font_info.get("fallbacks", []),
                        style_attributes=font_info.get("attributes", {})
                    )
            except Exception as e:
                print(f"Warning: Could not load font config: {e}")
    
    def _detect_system_fonts(self):
        """Detect available system fonts."""
        # Platform-specific font directories
        font_dirs = self._get_system_font_directories()
        
        for font_dir in font_dirs:
            if os.path.exists(font_dir):
                for font_file in Path(font_dir).rglob("*.ttf"):
                    font_name = font_file.stem
                    self._system_fonts[font_name.lower()] = str(font_file)
                    
                for font_file in Path(font_dir).rglob("*.otf"):
                    font_name = font_file.stem
                    self._system_fonts[font_name.lower()] = str(font_file)
    
    def _get_system_font_directories(self) -> List[str]:
        """Get system font directories based on platform."""
        system = platform.system()
        
        if system == "Windows":
            return [
                "C:/Windows/Fonts",
                os.path.expanduser("~/AppData/Local/Microsoft/Windows/Fonts")
            ]
        elif system == "Darwin":  # macOS
            return [
                "/System/Library/Fonts",
                "/Library/Fonts",
                os.path.expanduser("~/Library/Fonts")
            ]
        else:  # Linux/Unix
            return [
                "/usr/share/fonts",
                "/usr/local/share/fonts",
                os.path.expanduser("~/.fonts"),
                os.path.expanduser("~/.local/share/fonts")
            ]
    
    def get_font_for_emotion(self, emotion: EmotionCategory) -> str:
        """
        Get the appropriate font name for an emotion.
        
        Args:
            emotion: The emotion category
            
        Returns:
            Font name suitable for the emotion
        """
        return self.EMOTION_FONT_MAPPING.get(emotion, "MADE AVENUE")
    
    def get_font_path(self, font_name: str) -> Optional[str]:
        """
        Get the file path for a font.
        
        Args:
            font_name: Name of the font
            
        Returns:
            Path to font file or None if not found
        """
        # Check cache first
        if font_name in self._font_cache:
            return self._font_cache[font_name]
        
        # Check custom font directory
        if self.custom_font_dir and self.custom_font_dir.exists():
            for ext in ['.ttf', '.otf', '.ttc']:
                font_path = self.custom_font_dir / f"{font_name}{ext}"
                if font_path.exists():
                    self._font_cache[font_name] = str(font_path)
                    return str(font_path)
        
        # Check configured path
        font_info = self.DEFAULT_FONTS.get(font_name)
        if font_info and font_info.file_path and os.path.exists(font_info.file_path):
            self._font_cache[font_name] = font_info.file_path
            return font_info.file_path
        
        # Check system fonts
        font_lower = font_name.lower().replace(" ", "")
        if font_lower in self._system_fonts:
            self._font_cache[font_name] = self._system_fonts[font_lower]
            return self._system_fonts[font_lower]
        
        return None
    
    def get_font_with_fallback(self, font_name: str) -> str:
        """
        Get font name with fallback if original is not available.
        
        Args:
            font_name: Preferred font name
            
        Returns:
            Available font name (original or fallback)
        """
        # Check if font exists
        if self.get_font_path(font_name):
            return font_name
        
        # Try fallbacks
        font_info = self.DEFAULT_FONTS.get(font_name)
        if font_info:
            for fallback in font_info.fallback_names:
                if self.get_font_path(fallback):
                    return fallback
        
        # Default fallback
        return "Arial"
    
    def get_font_attributes(self, font_name: str) -> Dict[str, any]:
        """Get style attributes for a font."""
        font_info = self.DEFAULT_FONTS.get(font_name)
        if font_info:
            return font_info.style_attributes
        return {}
    
    def install_fonts(self, font_package_path: str):
        """
        Install fonts from a package directory.
        
        Args:
            font_package_path: Path to directory containing font files
        """
        if not self.custom_font_dir:
            # Create default custom font directory
            self.custom_font_dir = Path.home() / ".auto-caption" / "fonts"
        
        self.custom_font_dir.mkdir(parents=True, exist_ok=True)
        
        package_path = Path(font_package_path)
        if not package_path.exists():
            raise ValueError(f"Font package not found: {font_package_path}")
        
        installed_fonts = []
        
        # Copy font files
        for font_file in package_path.glob("*.[to]tf"):
            dest = self.custom_font_dir / font_file.name
            if not dest.exists():
                import shutil
                shutil.copy2(font_file, dest)
                installed_fonts.append(font_file.stem)
        
        # Clear cache to detect new fonts
        self._font_cache.clear()
        if self.auto_detect:
            self._detect_system_fonts()
        
        return installed_fonts
    
    def generate_font_css(self) -> str:
        """
        Generate CSS @font-face declarations for web use.
        
        Returns:
            CSS string with font-face declarations
        """
        css_parts = []
        
        for font_name, font_info in self.DEFAULT_FONTS.items():
            font_path = self.get_font_path(font_name)
            if font_path:
                css_parts.append(f"""
@font-face {{
    font-family: '{font_name}';
    src: url('{font_path}') format('truetype');
    font-weight: normal;
    font-style: normal;
}}""")
        
        return "\n".join(css_parts)
    
    def validate_fonts(self) -> Dict[str, bool]:
        """
        Validate availability of all configured fonts.
        
        Returns:
            Dictionary of font names and their availability status
        """
        validation_results = {}
        
        for font_name in self.DEFAULT_FONTS:
            validation_results[font_name] = self.get_font_path(font_name) is not None
        
        return validation_results
    
    def get_emotion_font_config(self, emotion: EmotionCategory) -> Dict[str, any]:
        """
        Get complete font configuration for an emotion.
        
        Args:
            emotion: The emotion category
            
        Returns:
            Dictionary with font name, path, and attributes
        """
        font_name = self.get_font_for_emotion(emotion)
        font_with_fallback = self.get_font_with_fallback(font_name)
        
        return {
            "name": font_with_fallback,
            "original_name": font_name,
            "path": self.get_font_path(font_with_fallback),
            "attributes": self.get_font_attributes(font_name),
            "is_fallback": font_with_fallback != font_name
        }