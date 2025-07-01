"""
ASS (Advanced SubStation Alpha) subtitle generator for emotion-styled captions.

This module generates ASS format subtitles with rich styling based on detected emotions,
providing better performance and compatibility than direct video rendering.
"""

import os
import json
import colorsys
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
from datetime import timedelta

from ..emotion_detector import EmotionCategory
from ..caption_styler import StyleIntensity, Platform


@dataclass
class ASSStyle:
    """ASS subtitle style definition."""
    name: str
    fontname: str = "Arial"
    fontsize: int = 48
    primary_color: str = "&H00FFFFFF"  # AABBGGRR format
    secondary_color: str = "&H00FFFFFF"
    outline_color: str = "&H00000000"
    back_color: str = "&H80000000"
    bold: int = 0
    italic: int = 0
    underline: int = 0
    strikeout: int = 0
    scale_x: int = 100
    scale_y: int = 100
    spacing: int = 0
    angle: float = 0.0
    border_style: int = 1  # 1 = outline + shadow, 3 = opaque box
    outline: float = 2.0
    shadow: float = 1.0
    alignment: int = 2  # 1-9 (numpad style)
    margin_l: int = 10
    margin_r: int = 10
    margin_v: int = 10
    encoding: int = 1
    blur: float = 0.0


class ASSGenerator:
    """
    Generates ASS format subtitles with emotion-based styling.
    """

    # Platform-specific settings
    PLATFORM_SETTINGS = {
        Platform.TIKTOK: {
            "base_fontsize": 56,
            "margin_v": 150,  # Higher margin for TikTok UI
            "alignment": 2,   # Bottom center
            "outline": 3.0,
            "shadow": 2.0,
            "blur": 0.5
        },
        Platform.INSTAGRAM: {
            "base_fontsize": 52,
            "margin_v": 120,
            "alignment": 2,
            "outline": 2.5,
            "shadow": 1.5,
            "blur": 0.3
        },
        Platform.YOUTUBE_SHORTS: {
            "base_fontsize": 54,
            "margin_v": 100,
            "alignment": 2,
            "outline": 2.5,
            "shadow": 2.0,
            "blur": 0.4
        },
        Platform.GENERAL: {
            "base_fontsize": 48,
            "margin_v": 80,
            "alignment": 2,
            "outline": 2.0,
            "shadow": 1.0,
            "blur": 0.0
        }
    }

    # Emotion color schemes (in ASS AABBGGRR format)
    EMOTION_COLORS = {
        EmotionCategory.HAPPY: {
            "primary": "&H00FFE033",    # Bright yellow
            "secondary": "&H00FFB300",  # Orange
            "outline": "&H00000000",    # Black
            "shadow": "&H80000000"      # Semi-transparent black
        },
        EmotionCategory.SAD: {
            "primary": "&H00C4A484",    # Muted blue-gray
            "secondary": "&H00998877",  # Darker gray
            "outline": "&H00333333",    # Dark gray
            "shadow": "&H80000000"
        },
        EmotionCategory.ANGRY: {
            "primary": "&H002020FF",    # Bright red
            "secondary": "&H001515CC",  # Darker red
            "outline": "&H00000033",    # Very dark red
            "shadow": "&H80000000"
        },
        EmotionCategory.EXCITED: {
            "primary": "&H00FF00FF",    # Magenta
            "secondary": "&H00FF66FF",  # Pink
            "outline": "&H00330033",    # Dark purple
            "shadow": "&H80000000"
        },
        EmotionCategory.FEARFUL: {
            "primary": "&H00AA88CC",    # Pale purple
            "secondary": "&H00886699",  # Muted purple
            "outline": "&H00222222",    # Dark gray
            "shadow": "&H80000000"
        },
        EmotionCategory.SARCASTIC: {
            "primary": "&H0099FFFF",    # Cyan-yellow
            "secondary": "&H0066CCCC",  # Teal
            "outline": "&H00003333",    # Dark teal
            "shadow": "&H80000000"
        },
        EmotionCategory.ANXIOUS: {
            "primary": "&H00CCCC99",    # Pale yellow-green
            "secondary": "&H00999966",  # Muted green
            "outline": "&H00333333",
            "shadow": "&H80000000"
        },
        EmotionCategory.NEUTRAL: {
            "primary": "&H00FFFFFF",    # White
            "secondary": "&H00E0E0E0",  # Light gray
            "outline": "&H00000000",    # Black
            "shadow": "&H80000000"
        },
        EmotionCategory.CONTEMPLATIVE: {
            "primary": "&H00E6D4B3",    # Soft beige
            "secondary": "&H00C0A080",  # Warm gray
            "outline": "&H00333333",
            "shadow": "&H80000000"
        }
    }

    def __init__(
        self,
        platform: Platform = Platform.GENERAL,
        style_intensity: StyleIntensity = StyleIntensity.MEDIUM,
        video_resolution: Tuple[int, int] = (1920, 1080),
        custom_fonts: Optional[Dict[str, str]] = None,
        verbose: bool = False
    ):
        """
        Initialize ASS generator.

        Args:
            platform: Target platform
            style_intensity: Styling intensity level
            video_resolution: Video resolution for scaling
            custom_fonts: Custom font mappings by emotion
            verbose: Enable verbose output
        """
        self.platform = platform
        self.style_intensity = style_intensity
        self.video_resolution = video_resolution
        self.custom_fonts = custom_fonts or {}
        self.verbose = verbose

        # Get platform settings
        self.platform_config = self.PLATFORM_SETTINGS[platform]

        # Calculate font scaling based on resolution
        self.font_scale = min(video_resolution) / 1080.0

    def generate_ass_file(
        self,
        caption_data: Dict[str, Any],
        output_path: str,
        title: str = "Auto-Caption Emotion Subtitles"
    ) -> str:
        """
        Generate ASS subtitle file from caption data.

        Args:
            caption_data: Caption data with emotion metadata
            output_path: Output file path
            title: Subtitle title

        Returns:
            Path to generated ASS file
        """
        # Create ASS content
        ass_content = []

        # Add header
        ass_content.extend(self._generate_header(title))

        # Add styles for each emotion
        ass_content.extend(self._generate_styles())

        # Add events (subtitles)
        ass_content.extend(self._generate_events(caption_data))

        # Write to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(ass_content))

        if self.verbose:
            print(f"Generated ASS file: {output_path}")

        return str(output_path)

    def _generate_header(self, title: str) -> List[str]:
        """Generate ASS header section."""
        return [
            "[Script Info]",
            f"Title: {title}",
            "ScriptType: v4.00+",
            "WrapStyle: 0",
            f"ScaledBorderAndShadow: yes",
            f"YCbCr Matrix: TV.601",
            f"PlayResX: {self.video_resolution[0]}",
            f"PlayResY: {self.video_resolution[1]}",
            "",
            "[Aegisub Project Garbage]",
            "Export Encoding: UTF-8",
            ""
        ]

    def _generate_styles(self) -> List[str]:
        """Generate style definitions for each emotion."""
        styles = ["[V4+ Styles]"]
        styles.append("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
                     "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
                     "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
                     "Alignment, MarginL, MarginR, MarginV, Encoding")

        # Create style for each emotion
        for emotion in EmotionCategory:
            style = self._create_emotion_style(emotion)
            styles.append(self._format_style(style))

        # Add default style
        default_style = self._create_emotion_style(EmotionCategory.NEUTRAL)
        default_style.name = "Default"
        styles.append(self._format_style(default_style))

        styles.append("")
        return styles

    def _create_emotion_style(self, emotion: EmotionCategory) -> ASSStyle:
        """Create ASS style for specific emotion."""
        colors = self.EMOTION_COLORS.get(emotion, self.EMOTION_COLORS[EmotionCategory.NEUTRAL])
        
        # Base style from platform settings
        base_fontsize = int(self.platform_config["base_fontsize"] * self.font_scale)
        
        # Adjust for style intensity
        intensity_multipliers = {
            StyleIntensity.SUBTLE: 0.9,
            StyleIntensity.MEDIUM: 1.0,
            StyleIntensity.INTENSE: 1.2
        }
        fontsize = int(base_fontsize * intensity_multipliers[self.style_intensity])

        # Get font for emotion
        fontname = self.custom_fonts.get(emotion.value, self._get_emotion_font(emotion))

        # Emotion-specific styling
        style_params = self._get_emotion_style_params(emotion)

        return ASSStyle(
            name=f"Emotion_{emotion.value}",
            fontname=fontname,
            fontsize=fontsize,
            primary_color=colors["primary"],
            secondary_color=colors["secondary"],
            outline_color=colors["outline"],
            back_color=colors["shadow"],
            bold=style_params["bold"],
            italic=style_params["italic"],
            scale_x=style_params["scale_x"],
            scale_y=style_params["scale_y"],
            outline=self.platform_config["outline"] * style_params["outline_mult"],
            shadow=self.platform_config["shadow"] * style_params["shadow_mult"],
            alignment=self.platform_config["alignment"],
            margin_v=self.platform_config["margin_v"],
            blur=self.platform_config["blur"] * style_params["blur_mult"]
        )

    def _get_emotion_font(self, emotion: EmotionCategory) -> str:
        """Get appropriate font for emotion."""
        font_mapping = {
            EmotionCategory.HAPPY: "Arial Rounded MT Bold",
            EmotionCategory.SAD: "Georgia",
            EmotionCategory.ANGRY: "Impact",
            EmotionCategory.EXCITED: "Comic Sans MS",
            EmotionCategory.FEARFUL: "Trebuchet MS",
            EmotionCategory.SARCASTIC: "Courier New",
            EmotionCategory.ANXIOUS: "Calibri",
            EmotionCategory.NEUTRAL: "Arial",
            EmotionCategory.CONTEMPLATIVE: "Times New Roman"
        }

        # Fallback to Arial if font doesn't exist
        return font_mapping.get(emotion, "Arial")

    def _get_emotion_style_params(self, emotion: EmotionCategory) -> Dict[str, Any]:
        """Get emotion-specific style parameters."""
        params = {
            EmotionCategory.HAPPY: {
                "bold": 1,
                "italic": 0,
                "scale_x": 105,
                "scale_y": 105,
                "outline_mult": 1.2,
                "shadow_mult": 1.3,
                "blur_mult": 1.2
            },
            EmotionCategory.SAD: {
                "bold": 0,
                "italic": 1,
                "scale_x": 95,
                "scale_y": 95,
                "outline_mult": 0.8,
                "shadow_mult": 0.7,
                "blur_mult": 1.5
            },
            EmotionCategory.ANGRY: {
                "bold": 1,
                "italic": 0,
                "scale_x": 110,
                "scale_y": 110,
                "outline_mult": 1.5,
                "shadow_mult": 1.5,
                "blur_mult": 0.8
            },
            EmotionCategory.EXCITED: {
                "bold": 1,
                "italic": 0,
                "scale_x": 108,
                "scale_y": 108,
                "outline_mult": 1.3,
                "shadow_mult": 1.4,
                "blur_mult": 1.0
            },
            EmotionCategory.SARCASTIC: {
                "bold": 0,
                "italic": 1,
                "scale_x": 100,
                "scale_y": 100,
                "outline_mult": 1.0,
                "shadow_mult": 1.0,
                "blur_mult": 0.5
            }
        }

        # Default parameters
        default = {
            "bold": 0,
            "italic": 0,
            "scale_x": 100,
            "scale_y": 100,
            "outline_mult": 1.0,
            "shadow_mult": 1.0,
            "blur_mult": 1.0
        }

        return params.get(emotion, default)

    def _format_style(self, style: ASSStyle) -> str:
        """Format style object as ASS style line."""
        return (f"Style: {style.name},{style.fontname},{style.fontsize},"
                f"{style.primary_color},{style.secondary_color},{style.outline_color},"
                f"{style.back_color},{style.bold},{style.italic},{style.underline},"
                f"{style.strikeout},{style.scale_x},{style.scale_y},{style.spacing},"
                f"{style.angle},{style.border_style},{style.outline},{style.shadow},"
                f"{style.alignment},{style.margin_l},{style.margin_r},{style.margin_v},"
                f"{style.encoding}")

    def _generate_events(self, caption_data: Dict[str, Any]) -> List[str]:
        """Generate subtitle events from caption segments."""
        events = ["[Events]"]
        events.append("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text")

        segments = caption_data.get("segments", [])

        for segment in segments:
            # Get timing
            start_time = self._format_time(segment.get("start", 0))
            end_time = self._format_time(segment.get("end", 0))

            # Get emotion and style
            emotion_meta = segment.get("emotion_metadata", {})
            emotion_str = emotion_meta.get("emotion", "neutral")
            
            # Find matching emotion enum
            emotion = EmotionCategory.NEUTRAL
            for e in EmotionCategory:
                if e.value == emotion_str:
                    emotion = e
                    break

            style_name = f"Emotion_{emotion.value}"

            # Get text with any additional formatting
            text = segment.get("text", "")
            formatted_text = self._apply_emotion_effects(text, emotion, emotion_meta)

            # Create event line
            event = f"Dialogue: 0,{start_time},{end_time},{style_name},,0,0,0,,{formatted_text}"
            events.append(event)

        return events

    def _format_time(self, seconds: float) -> str:
        """Format time in ASS format (h:mm:ss.cc)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        
        return f"{hours}:{minutes:02d}:{secs:05.2f}"

    def _apply_emotion_effects(
        self,
        text: str,
        emotion: EmotionCategory,
        metadata: Dict[str, Any]
    ) -> str:
        """Apply emotion-specific text effects using ASS override tags."""
        effects = []

        # Intensity-based effects
        if self.style_intensity == StyleIntensity.INTENSE:
            if emotion == EmotionCategory.HAPPY:
                # Add bounce effect
                effects.append(r"{\move(0,-10,0,0,0,200)}")
            elif emotion == EmotionCategory.ANGRY:
                # Add shake effect
                effects.append(r"{\fscx120\fscy120}")
            elif emotion == EmotionCategory.SAD:
                # Add fade effect
                effects.append(r"{\alpha&H40&}")
            elif emotion == EmotionCategory.EXCITED:
                # Add rotation
                effects.append(r"{\frz5}")
            elif emotion == EmotionCategory.ANXIOUS:
                # Add subtle shake
                effects.append(r"{\fsp2}")

        # Add karaoke-style effects for certain emotions
        if emotion in [EmotionCategory.HAPPY, EmotionCategory.EXCITED] and self.style_intensity != StyleIntensity.SUBTLE:
            # Highlight important words
            words = text.split()
            formatted_words = []
            for word in words:
                if len(word) > 4:  # Emphasize longer words
                    formatted_words.append(f"{{\\fscx110\\fscy110}}{word}{{\\r}}")
                else:
                    formatted_words.append(word)
            text = " ".join(formatted_words)

        # Combine effects and text
        if effects:
            return "".join(effects) + text
        return text

    def create_styled_srt(
        self,
        caption_data: Dict[str, Any],
        output_path: str
    ) -> str:
        """
        Create a simple SRT file with emotion markers for compatibility.

        Args:
            caption_data: Caption data with emotions
            output_path: Output SRT file path

        Returns:
            Path to generated SRT file
        """
        import srt

        subtitles = []
        segments = caption_data.get("segments", [])

        for idx, segment in enumerate(segments, 1):
            # Get emotion for reference
            emotion_meta = segment.get("emotion_metadata", {})
            emotion = emotion_meta.get("emotion", "neutral")

            # Create subtitle
            subtitle = srt.Subtitle(
                index=idx,
                start=timedelta(seconds=segment.get("start", 0)),
                end=timedelta(seconds=segment.get("end", 0)),
                content=segment.get("text", "").strip()
            )
            subtitles.append(subtitle)

        # Write SRT file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(srt.compose(subtitles))

        return str(output_path)