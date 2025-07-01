"""
Caption styling module for emotion-aware text formatting.

This module transforms plain transcriptions into emotionally-formatted captions
that convey emotion through punctuation, capitalization, and text emphasis.
"""

import re
import random
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import json

from .emotion_detector import EmotionCategory, EmotionDetectionResult


class StyleIntensity(Enum):
    """Intensity levels for text formatting."""
    SUBTLE = "subtle"
    MEDIUM = "medium"
    INTENSE = "intense"


class Platform(Enum):
    """Supported social media platforms."""
    TIKTOK = "tiktok"
    INSTAGRAM = "instagram"
    YOUTUBE_SHORTS = "youtube_shorts"
    GENERAL = "general"


class EmotionStyleMap:
    """Mapping of emotions to text formatting styles."""

    STYLE_TEMPLATES = {
        EmotionCategory.HAPPY: {
            "subtle": {
                "capitalization": "normal",
                "punctuation_style": "normal",
                "punctuation_marks": ["!", "."],
                "text_emphasis": "none"
            },
            "medium": {
                "capitalization": "emphasis",
                "punctuation_style": "enthusiastic",
                "punctuation_marks": ["!", "!!"],
                "text_emphasis": "moderate"
            },
            "intense": {
                "capitalization": "emphasis",
                "punctuation_style": "very_enthusiastic",
                "punctuation_marks": ["!!", "!!!"],
                "text_emphasis": "strong"
            }
        },
        EmotionCategory.SAD: {
            "subtle": {
                "capitalization": "lower",
                "punctuation_style": "minimal",
                "punctuation_marks": [".", "..."],
                "text_emphasis": "none"
            },
            "medium": {
                "capitalization": "lower",
                "punctuation_style": "trailing",
                "punctuation_marks": ["...", "..."],
                "text_emphasis": "minimal"
            },
            "intense": {
                "capitalization": "lower",
                "punctuation_style": "heavy_trailing",
                "punctuation_marks": ["...", "...."],
                "text_emphasis": "none"
            }
        },
        EmotionCategory.ANGRY: {
            "subtle": {
                "capitalization": "emphasis",
                "punctuation_style": "firm",
                "punctuation_marks": [".", "!"],
                "text_emphasis": "moderate"
            },
            "medium": {
                "capitalization": "upper",
                "punctuation_style": "forceful",
                "punctuation_marks": ["!", "!!"],
                "text_emphasis": "strong"
            },
            "intense": {
                "capitalization": "upper",
                "punctuation_style": "very_forceful",
                "punctuation_marks": ["!!", "!!!"],
                "text_emphasis": "very_strong"
            }
        },
        EmotionCategory.SARCASTIC: {
            "subtle": {
                "capitalization": "mixed",
                "punctuation_style": "ironic",
                "punctuation_marks": [".", "..."],
                "text_emphasis": "subtle_ironic"
            },
            "medium": {
                "capitalization": "mixed",
                "punctuation_style": "very_ironic",
                "punctuation_marks": ["...", "?"],
                "text_emphasis": "ironic"
            },
            "intense": {
                "capitalization": "mixed",
                "punctuation_style": "heavily_ironic",
                "punctuation_marks": ["...", "...?"],
                "text_emphasis": "strong_ironic"
            }
        },
        EmotionCategory.ANXIOUS: {
            "subtle": {
                "capitalization": "normal",
                "punctuation_style": "uncertain",
                "punctuation_marks": ["?", "..."],
                "text_emphasis": "nervous"
            },
            "medium": {
                "capitalization": "lower",
                "punctuation_style": "questioning",
                "punctuation_marks": ["?", "...?"],
                "text_emphasis": "nervous"
            },
            "intense": {
                "capitalization": "mixed",
                "punctuation_style": "very_uncertain",
                "punctuation_marks": ["??", "?!"],
                "text_emphasis": "very_nervous"
            }
        },
        EmotionCategory.NEUTRAL: {
            "subtle": {
                "capitalization": "normal",
                "punctuation_style": "normal",
                "punctuation_marks": ["."],
                "text_emphasis": "none"
            },
            "medium": {
                "capitalization": "normal",
                "punctuation_style": "normal",
                "punctuation_marks": [".", "!"],
                "text_emphasis": "none"
            },
            "intense": {
                "capitalization": "normal",
                "punctuation_style": "slight",
                "punctuation_marks": ["!", "."],
                "text_emphasis": "minimal"
            }
        },
        EmotionCategory.EXCITED: {
            "subtle": {
                "capitalization": "emphasis",
                "punctuation_style": "enthusiastic",
                "punctuation_marks": ["!", "!!"],
                "text_emphasis": "moderate"
            },
            "medium": {
                "capitalization": "emphasis",
                "punctuation_style": "very_enthusiastic",
                "punctuation_marks": ["!!", "!!!"],
                "text_emphasis": "strong"
            },
            "intense": {
                "capitalization": "upper",
                "punctuation_style": "extremely_enthusiastic",
                "punctuation_marks": ["!!!", "!!!!"],
                "text_emphasis": "very_strong"
            }
        },
        EmotionCategory.CONTEMPLATIVE: {
            "subtle": {
                "capitalization": "normal",
                "punctuation_style": "thoughtful",
                "punctuation_marks": [".", "..."],
                "text_emphasis": "none"
            },
            "medium": {
                "capitalization": "normal",
                "punctuation_style": "reflective",
                "punctuation_marks": ["...", "?"],
                "text_emphasis": "minimal"
            },
            "intense": {
                "capitalization": "normal",
                "punctuation_style": "deeply_reflective",
                "punctuation_marks": ["...", "...?"],
                "text_emphasis": "minimal"
            }
        }
    }

    @classmethod
    def get_style_template(
        cls,
        emotion: EmotionCategory,
        intensity: StyleIntensity
    ) -> Dict[str, Any]:
        """Get style template for emotion and intensity."""
        # Use neutral as fallback
        emotion_styles = cls.STYLE_TEMPLATES.get(
            emotion,
            cls.STYLE_TEMPLATES[EmotionCategory.NEUTRAL]
        )

        return emotion_styles.get(
            intensity.value,
            emotion_styles["medium"]
        )


class CaptionStyler:
    """
    Transforms plain captions into emotionally-formatted text.
    """

    def __init__(
        self,
        default_intensity: StyleIntensity = StyleIntensity.MEDIUM,
        default_platform: Platform = Platform.GENERAL,
        custom_style_map: Optional[Dict] = None,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the caption styler.

        Args:
            default_intensity: Default formatting intensity
            default_platform: Default target platform
            custom_style_map: Custom emotion-to-style mappings
            random_seed: Random seed for consistent formatting
        """
        self.default_intensity = default_intensity
        self.default_platform = default_platform
        self.custom_style_map = custom_style_map or {}

        if random_seed:
            random.seed(random_seed)

    def style_caption(
        self,
        text: str,
        emotion: EmotionCategory,
        confidence: float,
        intensity: Optional[StyleIntensity] = None,
        platform: Optional[Platform] = None
    ) -> Dict[str, Any]:
        """
        Apply emotional formatting to caption text.

        Args:
            text: Original caption text
            emotion: Detected emotion
            confidence: Emotion detection confidence
            intensity: Formatting intensity (uses default if None)
            platform: Target platform (uses default if None)

        Returns:
            Dictionary with formatted text and metadata
        """
        intensity = intensity or self.default_intensity
        platform = platform or self.default_platform

        # Get style template
        style_template = EmotionStyleMap.get_style_template(emotion, intensity)

        # Apply text transformations
        formatted_text = self._apply_text_transformations(
            text,
            style_template,
            confidence
        )

        # Apply platform-specific adjustments
        formatted_text = self._apply_platform_style(
            formatted_text,
            platform,
            emotion
        )

        return {
            "original_text": text,
            "formatted_text": formatted_text,
            "emotion": emotion.value,
            "confidence": confidence,
            "intensity": intensity.value,
            "platform": platform.value,
            "formatting_metadata": {
                "capitalization": style_template["capitalization"],
                "punctuation_style": style_template["punctuation_style"],
                "text_emphasis": style_template.get("text_emphasis", "none")
            }
        }

    def _apply_text_transformations(
        self,
        text: str,
        style_template: Dict[str, Any],
        confidence: float
    ) -> str:
        """Apply text transformations based on style template."""
        # Apply capitalization
        text = self._apply_capitalization(
            text,
            style_template["capitalization"],
            confidence
        )

        # Apply punctuation style
        text = self._apply_punctuation(
            text,
            style_template["punctuation_style"],
            style_template["punctuation_marks"],
            confidence
        )

        # Apply text emphasis
        text_emphasis = style_template.get("text_emphasis", "none")
        if text_emphasis != "none":
            text = self._apply_emphasis(
                text,
                text_emphasis,
                confidence
            )

        return text

    def _apply_capitalization(
        self,
        text: str,
        style: str,
        confidence: float
    ) -> str:
        """Apply capitalization style to text."""
        if style == "normal":
            return text
        elif style == "upper":
            return text.upper()
        elif style == "lower":
            return text.lower()
        elif style == "emphasis":
            # Capitalize important words
            words = text.split()
            emphasized = []

            for word in words:
                # Emphasize words longer than 4 characters
                if len(word) > 4 and confidence > 0.7:
                    emphasized.append(word.upper())
                else:
                    emphasized.append(word)

            return " ".join(emphasized)
        elif style == "mixed":
            # Random mixed case for sarcasm
            result = ""
            for i, char in enumerate(text):
                if char.isalpha():
                    if random.random() < 0.5:
                        result += char.upper()
                    else:
                        result += char.lower()
                else:
                    result += char
            return result

        return text

    def _apply_punctuation(
        self,
        text: str,
        style: str,
        punctuation_marks: List[str],
        confidence: float
    ) -> str:
        """Apply punctuation style to text."""
        if not punctuation_marks:
            return text

        # Remove existing ending punctuation
        text = re.sub(r'[.!?]+$', '', text.strip())

        if style == "minimal":
            # Use minimal punctuation
            if confidence > 0.8:
                return text + punctuation_marks[0]
            else:
                return text
        elif style == "normal":
            # Standard punctuation
            return text + punctuation_marks[0]
        elif style == "dramatic":
            # Use dramatic punctuation based on confidence
            if confidence > 0.8:
                mark = punctuation_marks[-1] if len(punctuation_marks) > 1 else punctuation_marks[0]
            else:
                mark = punctuation_marks[0]
            return text + mark
        elif style == "excessive":
            # Use most dramatic punctuation
            return text + punctuation_marks[-1]

        return text

    def _apply_emphasis(
        self,
        text: str,
        emphasis_type: str,
        confidence: float
    ) -> str:
        """Apply text emphasis through capitalization patterns."""
        if confidence < 0.6:
            return text

        words = text.split()
        emphasized_words = []

        if emphasis_type == "moderate":
            # Capitalize important words (longer than 4 chars)
            for word in words:
                if len(word) > 4 and not word.startswith("'"):
                    emphasized_words.append(word.upper())
                else:
                    emphasized_words.append(word)

        elif emphasis_type == "strong":
            # Capitalize most words
            for word in words:
                if len(word) > 2 and not word.startswith("'"):
                    emphasized_words.append(word.upper())
                else:
                    emphasized_words.append(word)

        elif emphasis_type == "very_strong":
            # Capitalize everything
            return text.upper()

        elif emphasis_type in ["subtle_ironic", "ironic", "strong_ironic"]:
            # Apply sarcastic capitalization patterns
            for i, word in enumerate(words):
                if emphasis_type == "subtle_ironic" and i % 3 == 0:
                    emphasized_words.append(word.upper())
                elif emphasis_type == "ironic" and random.random() < 0.5:
                    emphasized_words.append(word.upper())
                elif emphasis_type == "strong_ironic":
                    # Alternate casing within words
                    new_word = ""
                    for j, char in enumerate(word):
                        if char.isalpha() and j % 2 == 0:
                            new_word += char.upper()
                        else:
                            new_word += char.lower()
                    emphasized_words.append(new_word)
                else:
                    emphasized_words.append(word)

        elif emphasis_type == "nervous":
            # Subtle stuttering effect on some words
            for word in words:
                if len(word) > 3 and random.random() < 0.2:
                    # Repeat first letter
                    emphasized_words.append(f"{word[0]}-{word}")
                else:
                    emphasized_words.append(word)

        else:
            return text

        return " ".join(emphasized_words)



    def _apply_platform_style(
        self,
        text: str,
        platform: Platform,
        emotion: EmotionCategory
    ) -> str:
        """Apply platform-specific styling adjustments."""
        if platform == Platform.TIKTOK:
            # TikTok prefers shorter, punchier text
            if len(text) > 100:
                # Truncate and add ellipsis
                text = text[:97] + "..."

            # Add trending indicators for certain emotions
            if emotion in [EmotionCategory.EXCITED, EmotionCategory.HAPPY]:
                if random.random() < 0.3:
                    text = "POV: " + text

        elif platform == Platform.INSTAGRAM:
            # Instagram allows slightly longer captions
            if len(text) > 125:
                text = text[:122] + "..."

        elif platform == Platform.YOUTUBE_SHORTS:
            # YouTube Shorts can have longer captions
            if len(text) > 150:
                text = text[:147] + "..."

        return text


    def style_segment_batch(
        self,
        segments: List[Dict[str, Any]],
        emotion_result: EmotionDetectionResult,
        intensity: Optional[StyleIntensity] = None,
        platform: Optional[Platform] = None
    ) -> List[Dict[str, Any]]:
        """
        Style multiple caption segments based on temporal emotions.

        Args:
            segments: List of caption segments with text and timing
            emotion_result: Complete emotion detection result
            intensity: Style intensity override
            platform: Target platform override

        Returns:
            List of styled segments
        """
        styled_segments = []

        for segment in segments:
            # Find emotion at this timestamp
            timestamp = segment.get("start", 0)
            segment_emotion = self._find_emotion_at_timestamp(
                timestamp,
                emotion_result.temporal_emotions
            )

            # Format the segment text
            formatted = self.style_caption(
                segment["text"],
                segment_emotion["emotion"],
                segment_emotion["confidence"],
                intensity,
                platform
            )

            # Merge with original segment data
            formatted_segment = {
                **segment,
                "original_text": segment["text"],
                "text": formatted["formatted_text"],
                "emotion": formatted["emotion"],
                "confidence": formatted["confidence"],
                "timestamp": timestamp
            }
            
            styled_segments.append(formatted_segment)

        return styled_segments

    def _find_emotion_at_timestamp(
        self,
        timestamp: float,
        temporal_emotions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Find the emotion at a specific timestamp."""
        for temporal in temporal_emotions:
            if temporal["start"] <= timestamp < temporal["end"]:
                # temporal["dominant_emotion"] is already a string value
                emotion_str = temporal["dominant_emotion"]
                # Find the corresponding EmotionCategory enum
                for emotion in EmotionCategory:
                    if emotion.value == emotion_str:
                        return {
                            "emotion": emotion,
                            "confidence": temporal["confidence"]
                        }
                # If not found, default to neutral
                return {
                    "emotion": EmotionCategory.NEUTRAL,
                    "confidence": temporal["confidence"]
                }

        # Default to neutral if not found
        return {
            "emotion": EmotionCategory.NEUTRAL,
            "confidence": 0.5
        }

    def generate_style_variations(
        self,
        text: str,
        emotion: EmotionCategory,
        confidence: float,
        num_variations: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple style variations for A/B testing.

        Args:
            text: Original caption text
            emotion: Detected emotion
            confidence: Emotion confidence
            num_variations: Number of variations to generate

        Returns:
            List of style variations
        """
        variations = []
        intensities = list(StyleIntensity)

        for i in range(num_variations):
            # Vary intensity
            intensity = intensities[i % len(intensities)]

            # Vary platform
            platforms = list(Platform)
            platform = platforms[(i + 1) % len(platforms)]

            # Generate variation
            variation = self.style_caption(
                text,
                emotion,
                confidence,
                intensity,
                platform
            )

            variation["variation_id"] = i + 1
            variations.append(variation)

        return variations
