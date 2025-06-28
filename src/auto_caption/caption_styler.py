"""
Caption styling module for emotion-aware text generation.

This module transforms plain transcriptions into emotionally-styled captions
that match the detected emotional context of the video.
"""

import re
import random
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import json

from .emotion_detector import EmotionCategory, EmotionDetectionResult


class StyleIntensity(Enum):
    """Intensity levels for caption styling."""
    SUBTLE = "subtle"
    MEDIUM = "medium"
    INTENSE = "intense"


class Platform(Enum):
    """Supported social media platforms."""
    TIKTOK = "tiktok"
    INSTAGRAM = "instagram"
    YOUTUBE_SHORTS = "youtube_shorts"
    GENERAL = "general"


@dataclass
class CaptionStyle:
    """Style configuration for captions."""
    emotion: EmotionCategory
    intensity: StyleIntensity
    platform: Platform
    
    # Text transformations
    capitalization: str  # 'normal', 'upper', 'lower', 'mixed', 'emphasis'
    punctuation_style: str  # 'normal', 'minimal', 'excessive', 'dramatic'
    emoji_usage: str  # 'none', 'subtle', 'moderate', 'heavy'
    
    # Timing adjustments
    timing_offset: float  # Seconds to adjust timing
    duration_multiplier: float  # Multiply segment duration
    
    # Visual suggestions
    suggested_effects: List[str]
    suggested_font_style: str
    suggested_color_scheme: Dict[str, str]


class EmotionStyleMap:
    """Mapping of emotions to caption styles."""
    
    STYLE_TEMPLATES = {
        EmotionCategory.HAPPY: {
            "subtle": {
                "capitalization": "normal",
                "punctuation_style": "normal",
                "emoji_usage": "subtle",
                "emoji_set": ["😊", "🙂", "✨"],
                "punctuation_marks": ["!", "!!"],
                "text_modifiers": []
            },
            "medium": {
                "capitalization": "emphasis",
                "punctuation_style": "dramatic",
                "emoji_usage": "moderate",
                "emoji_set": ["😄", "🎉", "💫", "⭐", "🌟"],
                "punctuation_marks": ["!!", "!!!", "~"],
                "text_modifiers": ["*", "~"]
            },
            "intense": {
                "capitalization": "upper",
                "punctuation_style": "excessive",
                "emoji_usage": "heavy",
                "emoji_set": ["🤩", "🎊", "🎈", "🌈", "💖", "🔥"],
                "punctuation_marks": ["!!!", "!!!!"],
                "text_modifiers": ["***", "~~~"]
            }
        },
        EmotionCategory.SAD: {
            "subtle": {
                "capitalization": "lower",
                "punctuation_style": "minimal",
                "emoji_usage": "subtle",
                "emoji_set": ["💭", "🌧️"],
                "punctuation_marks": [".", "..."],
                "text_modifiers": []
            },
            "medium": {
                "capitalization": "lower",
                "punctuation_style": "dramatic",
                "emoji_usage": "moderate",
                "emoji_set": ["😢", "💔", "🥺", "☔"],
                "punctuation_marks": ["...", "...."],
                "text_modifiers": ["*"]
            },
            "intense": {
                "capitalization": "lower",
                "punctuation_style": "minimal",
                "emoji_usage": "heavy",
                "emoji_set": ["😭", "💔", "😔", "🌧️", "⛈️"],
                "punctuation_marks": ["....", "....."],
                "text_modifiers": []
            }
        },
        EmotionCategory.ANGRY: {
            "subtle": {
                "capitalization": "emphasis",
                "punctuation_style": "normal",
                "emoji_usage": "subtle",
                "emoji_set": ["😤", "💢"],
                "punctuation_marks": [".", "!"],
                "text_modifiers": []
            },
            "medium": {
                "capitalization": "upper",
                "punctuation_style": "dramatic",
                "emoji_usage": "moderate",
                "emoji_set": ["😠", "😡", "🔥", "💥"],
                "punctuation_marks": ["!", "!!"],
                "text_modifiers": ["**"]
            },
            "intense": {
                "capitalization": "upper",
                "punctuation_style": "excessive",
                "emoji_usage": "heavy",
                "emoji_set": ["🤬", "😡", "👿", "🔥", "⚡", "💣"],
                "punctuation_marks": ["!!!", "!!!!"],
                "text_modifiers": ["***"]
            }
        },
        EmotionCategory.SARCASTIC: {
            "subtle": {
                "capitalization": "mixed",
                "punctuation_style": "normal",
                "emoji_usage": "subtle",
                "emoji_set": ["🙃", "😏"],
                "punctuation_marks": [".", "~"],
                "text_modifiers": ["*"]
            },
            "medium": {
                "capitalization": "mixed",
                "punctuation_style": "dramatic",
                "emoji_usage": "moderate",
                "emoji_set": ["🙄", "😏", "🤔", "💅"],
                "punctuation_marks": ["~", "~~"],
                "text_modifiers": ["*", "~"]
            },
            "intense": {
                "capitalization": "mixed",
                "punctuation_style": "excessive",
                "emoji_usage": "heavy",
                "emoji_set": ["🙄", "😤", "🤡", "💀", "☠️"],
                "punctuation_marks": ["~~~", "~!~"],
                "text_modifiers": ["***", "~~~"]
            }
        },
        EmotionCategory.ANXIOUS: {
            "subtle": {
                "capitalization": "normal",
                "punctuation_style": "minimal",
                "emoji_usage": "subtle",
                "emoji_set": ["😰", "😟"],
                "punctuation_marks": ["?", "..."],
                "text_modifiers": []
            },
            "medium": {
                "capitalization": "lower",
                "punctuation_style": "dramatic",
                "emoji_usage": "moderate",
                "emoji_set": ["😰", "😨", "🫨", "💭"],
                "punctuation_marks": ["??", "...?"],
                "text_modifiers": ["*"]
            },
            "intense": {
                "capitalization": "mixed",
                "punctuation_style": "excessive",
                "emoji_usage": "heavy",
                "emoji_set": ["😱", "🫨", "😰", "💀", "🆘"],
                "punctuation_marks": ["???", "?!?"],
                "text_modifiers": ["**"]
            }
        },
        EmotionCategory.NEUTRAL: {
            "subtle": {
                "capitalization": "normal",
                "punctuation_style": "normal",
                "emoji_usage": "none",
                "emoji_set": [],
                "punctuation_marks": ["."],
                "text_modifiers": []
            },
            "medium": {
                "capitalization": "normal",
                "punctuation_style": "normal",
                "emoji_usage": "subtle",
                "emoji_set": ["👍", "✨"],
                "punctuation_marks": [".", "!"],
                "text_modifiers": []
            },
            "intense": {
                "capitalization": "normal",
                "punctuation_style": "normal",
                "emoji_usage": "moderate",
                "emoji_set": ["💫", "⭐", "🌟"],
                "punctuation_marks": ["!", "!!"],
                "text_modifiers": ["*"]
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
    Transforms plain captions into emotionally-styled text.
    """
    
    def __init__(
        self,
        default_intensity: StyleIntensity = StyleIntensity.MEDIUM,
        default_platform: Platform = Platform.GENERAL,
        custom_style_map: Optional[Dict] = None,
        enable_emoji: bool = True,
        enable_effects: bool = True,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the caption styler.
        
        Args:
            default_intensity: Default styling intensity
            default_platform: Default target platform
            custom_style_map: Custom emotion-to-style mappings
            enable_emoji: Whether to include emoji in styling
            enable_effects: Whether to suggest visual effects
            random_seed: Random seed for consistent styling
        """
        self.default_intensity = default_intensity
        self.default_platform = default_platform
        self.custom_style_map = custom_style_map or {}
        self.enable_emoji = enable_emoji
        self.enable_effects = enable_effects
        
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
        Apply emotional styling to caption text.
        
        Args:
            text: Original caption text
            emotion: Detected emotion
            confidence: Emotion detection confidence
            intensity: Style intensity (uses default if None)
            platform: Target platform (uses default if None)
            
        Returns:
            Dictionary with styled text and metadata
        """
        intensity = intensity or self.default_intensity
        platform = platform or self.default_platform
        
        # Get style template
        style_template = EmotionStyleMap.get_style_template(emotion, intensity)
        
        # Apply text transformations
        styled_text = self._apply_text_transformations(
            text,
            style_template,
            confidence
        )
        
        # Add emoji if enabled
        if self.enable_emoji and style_template["emoji_usage"] != "none":
            styled_text = self._add_emoji(
                styled_text,
                style_template,
                confidence
            )
        
        # Apply platform-specific adjustments
        styled_text = self._apply_platform_style(
            styled_text,
            platform,
            emotion
        )
        
        # Generate visual suggestions
        visual_suggestions = {}
        if self.enable_effects:
            visual_suggestions = self._generate_visual_suggestions(
                emotion,
                intensity,
                platform
            )
        
        return {
            "original_text": text,
            "styled_text": styled_text,
            "emotion": emotion.value,
            "confidence": confidence,
            "intensity": intensity.value,
            "platform": platform.value,
            "visual_suggestions": visual_suggestions,
            "style_metadata": {
                "capitalization": style_template["capitalization"],
                "punctuation_style": style_template["punctuation_style"],
                "emoji_usage": style_template["emoji_usage"]
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
        
        # Apply text modifiers (asterisks, tildes, etc.)
        if style_template["text_modifiers"]:
            text = self._apply_modifiers(
                text,
                style_template["text_modifiers"],
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
    
    def _apply_modifiers(
        self,
        text: str,
        modifiers: List[str],
        confidence: float
    ) -> str:
        """Apply text modifiers like asterisks or tildes."""
        if not modifiers or confidence < 0.6:
            return text
        
        modifier = modifiers[0]
        
        # Apply modifier to emphasized words
        words = text.split()
        modified_words = []
        
        for word in words:
            # Emphasize longer or important words
            if len(word) > 5 or word.isupper():
                modified_words.append(f"{modifier}{word}{modifier}")
            else:
                modified_words.append(word)
        
        return " ".join(modified_words)
    
    def _add_emoji(
        self,
        text: str,
        style_template: Dict[str, Any],
        confidence: float
    ) -> str:
        """Add appropriate emoji to styled text."""
        emoji_set = style_template.get("emoji_set", [])
        if not emoji_set:
            return text
        
        emoji_usage = style_template["emoji_usage"]
        
        if emoji_usage == "subtle":
            # Add one emoji at the end
            if confidence > 0.7 and random.random() < 0.8:
                emoji = random.choice(emoji_set)
                return f"{text} {emoji}"
        elif emoji_usage == "moderate":
            # Add emoji at beginning and/or end
            if confidence > 0.6:
                if random.random() < 0.5:
                    emoji = random.choice(emoji_set)
                    return f"{emoji} {text}"
                else:
                    emoji = random.choice(emoji_set)
                    return f"{text} {emoji}"
        elif emoji_usage == "heavy":
            # Add multiple emoji
            if confidence > 0.5:
                emoji1 = random.choice(emoji_set)
                emoji2 = random.choice(emoji_set)
                return f"{emoji1} {text} {emoji2}"
        
        return text
    
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
    
    def _generate_visual_suggestions(
        self,
        emotion: EmotionCategory,
        intensity: StyleIntensity,
        platform: Platform
    ) -> Dict[str, Any]:
        """Generate visual effect suggestions based on emotion."""
        suggestions = {
            "text_animation": [],
            "color_scheme": {},
            "font_style": "",
            "effects": [],
            "timing_adjustments": {}
        }
        
        # Define visual mappings
        emotion_visuals = {
            EmotionCategory.HAPPY: {
                "text_animation": ["bounce", "pop", "sparkle"],
                "color_scheme": {
                    "primary": "#FFD700",  # Gold
                    "secondary": "#FF69B4",  # Hot pink
                    "background": "#FFF8DC"  # Cornsilk
                },
                "font_style": "rounded, playful",
                "effects": ["confetti", "stars", "rainbow"],
                "timing_adjustments": {
                    "appear_speed": "fast",
                    "emphasis_delay": 0.2
                }
            },
            EmotionCategory.SAD: {
                "text_animation": ["fade", "drip", "fall"],
                "color_scheme": {
                    "primary": "#4682B4",  # Steel blue
                    "secondary": "#708090",  # Slate gray
                    "background": "#F0F8FF"  # Alice blue
                },
                "font_style": "thin, delicate",
                "effects": ["rain", "blur", "desaturate"],
                "timing_adjustments": {
                    "appear_speed": "slow",
                    "emphasis_delay": 0.5
                }
            },
            EmotionCategory.ANGRY: {
                "text_animation": ["shake", "slam", "fire"],
                "color_scheme": {
                    "primary": "#DC143C",  # Crimson
                    "secondary": "#8B0000",  # Dark red
                    "background": "#2F0000"  # Very dark red
                },
                "font_style": "bold, aggressive",
                "effects": ["shake", "fire", "lightning"],
                "timing_adjustments": {
                    "appear_speed": "instant",
                    "emphasis_delay": 0.1
                }
            },
            EmotionCategory.SARCASTIC: {
                "text_animation": ["tilt", "wave", "flip"],
                "color_scheme": {
                    "primary": "#9370DB",  # Medium purple
                    "secondary": "#FF1493",  # Deep pink
                    "background": "#E6E6FA"  # Lavender
                },
                "font_style": "italic, quirky",
                "effects": ["wink", "eye_roll", "air_quotes"],
                "timing_adjustments": {
                    "appear_speed": "medium",
                    "emphasis_delay": 0.3
                }
            }
        }
        
        # Get emotion-specific visuals
        visuals = emotion_visuals.get(emotion, emotion_visuals[EmotionCategory.NEUTRAL])
        
        # Adjust based on intensity
        if intensity == StyleIntensity.SUBTLE:
            # Use fewer effects
            suggestions["text_animation"] = [visuals["text_animation"][0]]
            suggestions["effects"] = visuals["effects"][:1]
        elif intensity == StyleIntensity.MEDIUM:
            # Use moderate effects
            suggestions["text_animation"] = visuals["text_animation"][:2]
            suggestions["effects"] = visuals["effects"][:2]
        elif intensity == StyleIntensity.INTENSE:
            # Use all effects
            suggestions["text_animation"] = visuals["text_animation"]
            suggestions["effects"] = visuals["effects"]
        
        # Copy other suggestions
        suggestions["color_scheme"] = visuals["color_scheme"]
        suggestions["font_style"] = visuals["font_style"]
        suggestions["timing_adjustments"] = visuals["timing_adjustments"]
        
        return suggestions
    
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
            
            # Style the segment text
            styled = self.style_caption(
                segment["text"],
                segment_emotion["emotion"],
                segment_emotion["confidence"],
                intensity,
                platform
            )
            
            # Merge with original segment data
            styled_segment = {
                **segment,
                **styled,
                "timestamp": timestamp
            }
            
            styled_segments.append(styled_segment)
        
        return styled_segments
    
    def _find_emotion_at_timestamp(
        self,
        timestamp: float,
        temporal_emotions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Find the emotion at a specific timestamp."""
        for temporal in temporal_emotions:
            if temporal["start"] <= timestamp < temporal["end"]:
                return {
                    "emotion": EmotionCategory(temporal["dominant_emotion"]),
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