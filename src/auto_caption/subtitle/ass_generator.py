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

    # Emotion-specific position and size adjustments
    EMOTION_ADJUSTMENTS = {
        EmotionCategory.HAPPY: {
            "size_multiplier": 1.0,  # Consistent size across all emotions
            "position_adjustment": -60,  # Much higher on screen - jumping with joy
            "alignment": 8  # Top center - happiness rises
        },
        EmotionCategory.SAD: {
            "size_multiplier": 1.0,  # Consistent size across all emotions
            "position_adjustment": 80,  # Much lower on screen - weighted down
            "alignment": 2  # Bottom center
        },
        EmotionCategory.ANGRY: {
            "size_multiplier": 1.0,  # Consistent size across all emotions
            "position_adjustment": -40,  # Higher, dominating presence
            "alignment": 5  # Center of screen - confrontational
        },
        EmotionCategory.EXCITED: {
            "size_multiplier": 1.0,  # Consistent size across all emotions
            "position_adjustment": -70,  # Very high - bouncing with excitement
            "alignment": 8  # Top center
        },
        EmotionCategory.FEARFUL: {
            "size_multiplier": 1.0,  # Consistent size across all emotions
            "position_adjustment": 50,  # Lower, hiding
            "alignment": 1  # Bottom left - cornered
        },
        EmotionCategory.SARCASTIC: {
            "size_multiplier": 1.0,  # Consistent size across all emotions
            "position_adjustment": -10,  # Slightly off-center
            "alignment": 6  # Middle right - sideways delivery
        },
        EmotionCategory.ANXIOUS: {
            "size_multiplier": 1.0,  # Consistent size across all emotions
            "position_adjustment": 30,  # Lower middle
            "alignment": 5  # Center - frozen in place
        },
        EmotionCategory.NEUTRAL: {
            "size_multiplier": 1.0,  # Standard size
            "position_adjustment": 0,  # Standard position
            "alignment": 2  # Bottom center - default
        },
        EmotionCategory.CONTEMPLATIVE: {
            "size_multiplier": 1.0,  # Consistent size across all emotions
            "position_adjustment": -30,  # Slightly higher - thoughtful
            "alignment": 8  # Top center - looking up/thinking
        },
        EmotionCategory.SURPRISED: {
            "size_multiplier": 1.0,  # Consistent size across all emotions
            "position_adjustment": -50,  # Higher - jumping in surprise
            "alignment": 8  # Top center
        },
        EmotionCategory.DISGUSTED: {
            "size_multiplier": 1.0,  # Consistent size across all emotions
            "position_adjustment": 20,  # Lower middle - recoiling
            "alignment": 4  # Middle left - turning away
        },
        EmotionCategory.IRONIC: {
            "size_multiplier": 1.0,  # Consistent size across all emotions
            "position_adjustment": -15,  # Slightly off-center
            "alignment": 6  # Middle right - sideways glance
        },
        EmotionCategory.MELANCHOLIC: {
            "size_multiplier": 1.0,  # Consistent size across all emotions
            "position_adjustment": 60,  # Lower - heavy feeling
            "alignment": 2  # Bottom center
        },
        EmotionCategory.CONFIDENT: {
            "size_multiplier": 1.0,  # Consistent size across all emotions
            "position_adjustment": -20,  # Slightly elevated - standing tall
            "alignment": 5  # Center - direct and bold
        },
        EmotionCategory.CONFUSED: {
            "size_multiplier": 1.0,  # Consistent size across all emotions
            "position_adjustment": 10,  # Slightly lower - uncertain
            "alignment": 5  # Center - questioning
        },
        EmotionCategory.MOTIVATIONAL: {
            "size_multiplier": 1.0,  # Consistent size across all emotions
            "position_adjustment": -45,  # High - uplifting
            "alignment": 8  # Top center - inspiring
        },
        EmotionCategory.HUMOROUS: {
            "size_multiplier": 1.0,  # Consistent size across all emotions
            "position_adjustment": -25,  # Slightly high - light-hearted
            "alignment": 5  # Center
        },
        EmotionCategory.DRAMATIC: {
            "size_multiplier": 1.0,  # Consistent size across all emotions
            "position_adjustment": -35,  # Higher - theatrical presence
            "alignment": 5  # Center - commanding attention
        },
        EmotionCategory.CASUAL: {
            "size_multiplier": 1.0,  # Consistent size across all emotions
            "position_adjustment": 0,  # Standard position - relaxed
            "alignment": 2  # Bottom center
        },
        EmotionCategory.PROFESSIONAL: {
            "size_multiplier": 1.0,  # Consistent size across all emotions
            "position_adjustment": -10,  # Slightly elevated - formal
            "alignment": 2  # Bottom center - standard
        },
        EmotionCategory.ROMANTIC: {
            "size_multiplier": 1.0,  # Consistent size across all emotions
            "position_adjustment": -40,  # Higher - dreamy
            "alignment": 8  # Top center - floating
        },
        EmotionCategory.NOSTALGIC: {
            "size_multiplier": 1.0,  # Consistent size across all emotions
            "position_adjustment": -20,  # Slightly higher - reminiscent
            "alignment": 8  # Top center - looking back
        }
    }

    # Emotion color schemes (in ASS AABBGGRR format)
    EMOTION_COLORS = {
        EmotionCategory.HAPPY: {
            "primary": "&H0000FFFF",    # Bright sunny yellow
            "secondary": "&H0000D4FF",  # Golden yellow
            "outline": "&H00202020",    # Standard outline
            "shadow": "&H80000000"      # Standard shadow
        },
        EmotionCategory.SAD: {
            "primary": "&H00FF9966",    # Deep ocean blue
            "secondary": "&H00CC7755",  # Muted steel blue
            "outline": "&H00663333",    # Dark blue-gray
            "shadow": "&HA0664422"      # Heavy blue shadow
        },
        EmotionCategory.ANGRY: {
            "primary": "&H000000FF",    # Pure intense red
            "secondary": "&H000033CC",  # Deep crimson
            "outline": "&H00000099",    # Dark blood red
            "shadow": "&HFF000066"      # Red glow shadow
        },
        EmotionCategory.EXCITED: {
            "primary": "&H00FF00FF",    # Electric magenta
            "secondary": "&H00FF33CC",  # Hot pink
            "outline": "&H00CC0099",    # Vibrant purple
            "shadow": "&H80FF00AA"      # Pink glow
        },
        EmotionCategory.FEARFUL: {
            "primary": "&H00EECCAA",    # Pale ghostly blue
            "secondary": "&H00CCAA88",  # Faded purple-gray
            "outline": "&H00554433",    # Dark shadow
            "shadow": "&HC0443322"      # Deep shadow
        },
        EmotionCategory.SARCASTIC: {
            "primary": "&H0033FFCC",    # Sharp lime green
            "secondary": "&H0000CCAA",  # Acid yellow-green
            "outline": "&H00006655",    # Dark green edge
            "shadow": "&H8000AA88"      # Green shadow
        },
        EmotionCategory.ANXIOUS: {
            "primary": "&H00BBDDCC",    # Nervous pale green
            "secondary": "&H0099BBAA",  # Shaky mint
            "outline": "&H00445544",    # Uncertain edge
            "shadow": "&HA0667766"      # Blurred shadow
        },
        EmotionCategory.NEUTRAL: {
            "primary": "&H00FFFFFF",    # Pure white
            "secondary": "&H00F0F0F0",  # Soft white
            "outline": "&H00202020",    # Charcoal gray
            "shadow": "&H80000000"      # Standard shadow
        },
        EmotionCategory.CONTEMPLATIVE: {
            "primary": "&H00DDB896",    # Thoughtful lavender
            "secondary": "&H00C4A685",  # Wise purple-gray
            "outline": "&H00665544",    # Deep thought edge
            "shadow": "&H90554433"      # Soft shadow
        },
        EmotionCategory.SURPRISED: {
            "primary": "&H00FFCCFF",    # Bright pink-white shock
            "secondary": "&H00FFAAFF",  # Electric pink
            "outline": "&H00AA66AA",    # Purple edge
            "shadow": "&H80CC88CC"      # Pink glow
        },
        EmotionCategory.DISGUSTED: {
            "primary": "&H0066AA88",    # Sickly green
            "secondary": "&H00558877",  # Murky green
            "outline": "&H00334433",    # Dark swamp
            "shadow": "&HA0445544"      # Heavy shadow
        },
        EmotionCategory.IRONIC: {
            "primary": "&H0099FFDD",    # Cyan-green twist
            "secondary": "&H0077DDBB",  # Teal irony
            "outline": "&H00446655",    # Dark teal edge
            "shadow": "&H80668877"      # Twisted shadow
        },
        EmotionCategory.MELANCHOLIC: {
            "primary": "&H00CC9988",    # Dusty blue-gray
            "secondary": "&H00AA8877",  # Faded memories
            "outline": "&H00554444",    # Dark gray edge
            "shadow": "&HA0665555"      # Heavy melancholy
        },
        EmotionCategory.CONFIDENT: {
            "primary": "&H00FFFFFF",    # Pure white - strong and clear
            "secondary": "&H00F0F0F0",  # Bright white
            "outline": "&H00303030",    # Bold outline
            "shadow": "&H60000000"      # Sharp shadow
        },
        EmotionCategory.CONFUSED: {
            "primary": "&H00CCCCCC",    # Gray uncertainty
            "secondary": "&H00AAAAAA",  # Lighter gray
            "outline": "&H00666666",    # Foggy edge
            "shadow": "&H90888888"      # Blurred shadow
        },
        EmotionCategory.MOTIVATIONAL: {
            "primary": "&H0000CCFF",    # Bright orange-gold
            "secondary": "&H0000AADD",  # Energizing orange
            "outline": "&H00005588",    # Strong edge
            "shadow": "&H60006699"      # Uplifting glow
        },
        EmotionCategory.HUMOROUS: {
            "primary": "&H0099FFFF",    # Light yellow-green
            "secondary": "&H0077DDDD",  # Playful lime
            "outline": "&H00448844",    # Fun green edge
            "shadow": "&H80559955"      # Light shadow
        },
        EmotionCategory.DRAMATIC: {
            "primary": "&H00CC00FF",    # Deep purple-red
            "secondary": "&H00AA00DD",  # Royal purple
            "outline": "&H00660099",    # Dark dramatic edge
            "shadow": "&HFF8800BB"      # Dramatic glow
        },
        EmotionCategory.CASUAL: {
            "primary": "&H00FFFFFF",    # Plain white - common subtitle
            "secondary": "&H00F0F0F0",  # Soft white
            "outline": "&H00202020",    # Standard gray outline
            "shadow": "&H80000000"      # Standard shadow
        },
        EmotionCategory.PROFESSIONAL: {
            "primary": "&H00FFFFFF",    # Clean white - formal
            "secondary": "&H00F5F5F5",  # Pure white
            "outline": "&H00181818",    # Sharp outline
            "shadow": "&H80000000"      # Professional shadow
        },
        EmotionCategory.ROMANTIC: {
            "primary": "&H00FFCCEE",    # Soft pink-rose
            "secondary": "&H00FFAADD",  # Warm pink
            "outline": "&H00AA6688",    # Rose edge
            "shadow": "&H80CC8899"      # Soft romantic glow
        },
        EmotionCategory.NOSTALGIC: {
            "primary": "&H00DDCCBB",    # Sepia-tinted cream
            "secondary": "&H00CCBBAA",  # Vintage beige
            "outline": "&H00776655",    # Old photo edge
            "shadow": "&H90887766"      # Faded shadow
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
        
        # Get emotion adjustments
        emotion_adj = self.EMOTION_ADJUSTMENTS.get(emotion, self.EMOTION_ADJUSTMENTS[EmotionCategory.NEUTRAL])
        
        # Adjust for style intensity
        intensity_multipliers = {
            StyleIntensity.SUBTLE: 0.85,  # More subtle differences
            StyleIntensity.MEDIUM: 1.0,
            StyleIntensity.INTENSE: 1.35  # More dramatic differences
        }
        
        # Apply emotion size multiplier and intensity
        fontsize = int(base_fontsize * intensity_multipliers[self.style_intensity] * emotion_adj["size_multiplier"])

        # Get font for emotion
        fontname = self.custom_fonts.get(emotion.value, self._get_emotion_font(emotion))

        # Emotion-specific styling
        style_params = self._get_emotion_style_params(emotion)

        # Calculate adjusted margin based on emotion
        margin_v = self.platform_config["margin_v"] + emotion_adj["position_adjustment"]
        margin_v = max(20, margin_v)  # Ensure minimum margin

        # Use emotion-specific alignment
        alignment = emotion_adj["alignment"]

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
            alignment=alignment,
            margin_v=margin_v,
            blur=self.platform_config["blur"] * style_params["blur_mult"]
        )

    def _get_emotion_font(self, emotion: EmotionCategory) -> str:
        """Get appropriate font for emotion."""
        font_mapping = {
            # MADE AVENUE - Clean, modern, friendly emotions
            EmotionCategory.HAPPY: "MADE AVENUE",  # Clean and uplifting
            EmotionCategory.EXCITED: "MADE AVENUE",  # Energetic and positive
            EmotionCategory.NEUTRAL: "MADE AVENUE",  # Default clean look
            
            # CINEMATOGRAFICA - Bold, dramatic, intense emotions
            EmotionCategory.ANGRY: "CINEMATOGRAFICA",  # Bold and aggressive
            EmotionCategory.SARCASTIC: "CINEMATOGRAFICA",  # Sharp and cutting
            EmotionCategory.SURPRISED: "CINEMATOGRAFICA",  # Dramatic impact
            EmotionCategory.DISGUSTED: "CINEMATOGRAFICA",  # Strong reaction
            
            # ALMOST TEXTUAL - Soft, emotional, introspective emotions
            EmotionCategory.SAD: "ALMOST TEXTUAL",  # Gentle and emotional
            EmotionCategory.FEARFUL: "ALMOST TEXTUAL",  # Vulnerable
            EmotionCategory.ANXIOUS: "ALMOST TEXTUAL",  # Uncertain
            EmotionCategory.CONTEMPLATIVE: "ALMOST TEXTUAL",  # Thoughtful
        }

        # Fallback to MADE AVENUE if font doesn't exist
        return font_mapping.get(emotion, "MADE AVENUE")

    def _get_emotion_style_params(self, emotion: EmotionCategory) -> Dict[str, Any]:
        """Get emotion-specific style parameters."""
        params = {
            EmotionCategory.HAPPY: {
                "bold": 1,  # Bold for emphasis
                "italic": 0,
                "scale_x": 115,  # Wider for joy
                "scale_y": 120,  # Taller for uplift
                "outline_mult": 1.3,
                "shadow_mult": 1.5,  # Strong shadow for pop
                "blur_mult": 0.5  # Sharp and clear
            },
            EmotionCategory.SAD: {
                "bold": 0,
                "italic": 1,  # Slanted, drooping
                "scale_x": 85,  # Compressed
                "scale_y": 90,  # Smaller overall
                "outline_mult": 0.6,  # Thin outline
                "shadow_mult": 0.5,  # Faint shadow
                "blur_mult": 2.0  # Blurry, tearful
            },
            EmotionCategory.ANGRY: {
                "bold": 1,  # Very bold
                "italic": 0,
                "scale_x": 130,  # Wide and imposing
                "scale_y": 125,  # Tall and strong
                "outline_mult": 2.0,  # Thick outline
                "shadow_mult": 2.0,  # Heavy shadow
                "blur_mult": 0.3  # Very sharp
            },
            EmotionCategory.EXCITED: {
                "bold": 1,
                "italic": 0,
                "scale_x": 120,  # Wide for energy
                "scale_y": 115,  # Bouncy
                "outline_mult": 1.5,
                "shadow_mult": 1.8,  # Dynamic shadow
                "blur_mult": 0.7
            },
            EmotionCategory.FEARFUL: {
                "bold": 0,
                "italic": 0,
                "scale_x": 90,  # Shrinking
                "scale_y": 85,  # Small
                "outline_mult": 0.7,
                "shadow_mult": 0.6,
                "blur_mult": 1.8  # Shaky, unclear
            },
            EmotionCategory.SARCASTIC: {
                "bold": 1,  # Bold for emphasis
                "italic": 1,  # Slanted for attitude
                "scale_x": 110,  # Stretched for effect
                "scale_y": 100,
                "outline_mult": 1.2,
                "shadow_mult": 1.3,
                "blur_mult": 0.4  # Sharp wit
            },
            EmotionCategory.ANXIOUS: {
                "bold": 0,
                "italic": 0,
                "scale_x": 92,  # Slightly compressed
                "scale_y": 88,  # Smaller
                "outline_mult": 0.8,
                "shadow_mult": 0.7,
                "blur_mult": 1.5  # Slightly unclear
            },
            EmotionCategory.CONTEMPLATIVE: {
                "bold": 0,
                "italic": 1,  # Thoughtful slant
                "scale_x": 95,
                "scale_y": 100,
                "outline_mult": 0.9,
                "shadow_mult": 1.0,
                "blur_mult": 1.2  # Soft focus
            },
            EmotionCategory.NEUTRAL: {
                "bold": 0,
                "italic": 0,
                "scale_x": 100,  # Standard size
                "scale_y": 100,  # Standard size
                "outline_mult": 1.0,  # Standard outline
                "shadow_mult": 1.0,  # Standard shadow
                "blur_mult": 0.0  # Clear like common subtitles
            },
            EmotionCategory.SURPRISED: {
                "bold": 1,  # Bold for impact
                "italic": 0,
                "scale_x": 118,  # Wide eyes effect
                "scale_y": 122,  # Tall for shock
                "outline_mult": 1.6,
                "shadow_mult": 1.7,  # Strong shadow
                "blur_mult": 0.4  # Sharp surprise
            },
            EmotionCategory.DISGUSTED: {
                "bold": 0,
                "italic": 1,  # Recoiling
                "scale_x": 88,  # Pulling back
                "scale_y": 92,  # Shrinking away
                "outline_mult": 0.8,
                "shadow_mult": 0.9,
                "blur_mult": 1.3  # Slightly nauseous blur
            },
            EmotionCategory.IRONIC: {
                "bold": 0,
                "italic": 1,  # Sideways delivery
                "scale_x": 108,  # Slightly stretched
                "scale_y": 98,  # Slightly compressed
                "outline_mult": 1.1,
                "shadow_mult": 1.2,
                "blur_mult": 0.5  # Clear irony
            },
            EmotionCategory.MELANCHOLIC: {
                "bold": 0,
                "italic": 1,  # Drooping
                "scale_x": 90,  # Compressed
                "scale_y": 94,  # Weighted down
                "outline_mult": 0.7,
                "shadow_mult": 0.8,
                "blur_mult": 1.6  # Hazy memories
            },
            EmotionCategory.CONFIDENT: {
                "bold": 1,  # Strong presence
                "italic": 0,
                "scale_x": 105,  # Slightly wider
                "scale_y": 105,  # Standing tall
                "outline_mult": 1.3,
                "shadow_mult": 1.2,
                "blur_mult": 0.0  # Crystal clear
            },
            EmotionCategory.CONFUSED: {
                "bold": 0,
                "italic": 0,
                "scale_x": 96,  # Slightly uncertain
                "scale_y": 98,  # Questioning
                "outline_mult": 0.9,
                "shadow_mult": 1.0,
                "blur_mult": 1.4  # Foggy confusion
            },
            EmotionCategory.MOTIVATIONAL: {
                "bold": 1,  # Strong and inspiring
                "italic": 0,
                "scale_x": 112,  # Expanding energy
                "scale_y": 110,  # Uplifting
                "outline_mult": 1.4,
                "shadow_mult": 1.5,
                "blur_mult": 0.2  # Sharp focus
            },
            EmotionCategory.HUMOROUS: {
                "bold": 0,
                "italic": 0,
                "scale_x": 103,  # Slightly playful
                "scale_y": 102,  # Light bounce
                "outline_mult": 1.1,
                "shadow_mult": 1.2,
                "blur_mult": 0.6  # Clear humor
            },
            EmotionCategory.DRAMATIC: {
                "bold": 1,  # Theatrical presence
                "italic": 0,
                "scale_x": 125,  # Wide dramatic effect
                "scale_y": 118,  # Commanding height
                "outline_mult": 1.8,
                "shadow_mult": 2.0,  # Deep theatrical shadow
                "blur_mult": 0.3  # Sharp drama
            },
            EmotionCategory.CASUAL: {
                "bold": 0,
                "italic": 0,
                "scale_x": 100,  # Standard size
                "scale_y": 100,  # Standard size
                "outline_mult": 1.0,  # Standard outline
                "shadow_mult": 1.0,  # Standard shadow
                "blur_mult": 0.0  # Clear like common subtitles
            },
            EmotionCategory.PROFESSIONAL: {
                "bold": 0,
                "italic": 0,
                "scale_x": 100,  # Standard size
                "scale_y": 100,  # Standard size
                "outline_mult": 1.0,  # Standard outline
                "shadow_mult": 1.0,  # Standard shadow
                "blur_mult": 0.0  # Clear like common subtitles
            },
            EmotionCategory.ROMANTIC: {
                "bold": 0,
                "italic": 1,  # Soft romantic slant
                "scale_x": 98,  # Gentle
                "scale_y": 103,  # Slightly dreamy
                "outline_mult": 0.8,
                "shadow_mult": 1.1,  # Soft glow
                "blur_mult": 0.8  # Soft romantic focus
            },
            EmotionCategory.NOSTALGIC: {
                "bold": 0,
                "italic": 1,  # Looking back
                "scale_x": 94,  # Fading memories
                "scale_y": 97,  # Slightly smaller
                "outline_mult": 0.7,
                "shadow_mult": 0.9,
                "blur_mult": 1.5  # Hazy memories
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
        is_word_by_word = caption_data.get("metadata", {}).get("word_by_word", False)

        # Track position for word-by-word mode
        word_position_x = 0
        prev_segment_index = -1
        
        # Screen center position
        screen_center_x = self.video_resolution[0] // 2
        screen_center_y = self.video_resolution[1] // 2

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
            
            # Handle word-by-word mode
            if is_word_by_word:
                word_index = emotion_meta.get("word_index", 0)
                animation_style = emotion_meta.get("animation_style", "typewriter")
                is_emphasized = emotion_meta.get("is_emphasized", False)
                segment_index = emotion_meta.get("segment_index", 0)
                
                # Reset position for new segment
                if segment_index != prev_segment_index:
                    word_position_x = screen_center_x
                    prev_segment_index = segment_index
                
                # Use position data from word timing metadata if available
                position_offset = emotion_meta.get("position_offset", (0, 0))
                x_offset, y_offset = position_offset
                
                # Convert offset to absolute position
                word_position_x = screen_center_x + x_offset
                
                # Ensure position stays within video bounds
                margin = 50  # Minimum margin from edges
                word_position_x = max(margin, min(self.video_resolution[0] - margin, word_position_x))
                
                # Apply word-specific formatting with pre-calculated positioning
                formatted_text = self._apply_word_effects(
                    text, emotion, emotion_meta, 
                    word_position_x, word_index, 
                    animation_style, is_emphasized
                )
                
                # Update position for next word based on emotion
                if emotion in [EmotionCategory.HAPPY, EmotionCategory.EXCITED]:
                    word_position_x += len(text) * 25  # Wider spacing for energetic emotions
                elif emotion in [EmotionCategory.SAD, EmotionCategory.FEARFUL]:
                    word_position_x += len(text) * 15  # Tighter spacing for subdued emotions
                else:
                    word_position_x += len(text) * 20  # Normal spacing
            else:
                formatted_text = self._apply_emotion_effects(text, emotion, emotion_meta)

            # Create event line with safe zone margins
            margin_override = 0  # Margins are now handled by position offsets
            
            event = f"Dialogue: 0,{start_time},{end_time},{style_name},,0,0,{margin_override},,{formatted_text}"
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
                # Add bounce effect with vertical movement
                effects.append(r"{\move(0,-15,0,0,0,300)}")
            elif emotion == EmotionCategory.ANGRY:
                # Add aggressive shake effect
                effects.append(r"{\t(0,100,\frz3)\t(100,200,\frz-3)\t(200,300,\frz0)}")
            elif emotion == EmotionCategory.SAD:
                # Add fade and droop effect
                effects.append(r"{\fade(255,0,255,0,500,2000)\t(0,1000,\frz-2)}")
            elif emotion == EmotionCategory.EXCITED:
                # Add energetic rotation and scale
                effects.append(r"{\t(0,200,\fscx120\fscy120)\t(200,400,\fscx100\fscy100)}")
            elif emotion == EmotionCategory.ANXIOUS:
                # Add nervous jitter
                effects.append(r"{\t(0,50,\fsp1)\t(50,100,\fsp-1)\t(100,150,\fsp0)}")
            elif emotion == EmotionCategory.SARCASTIC:
                # Add eye-roll effect
                effects.append(r"{\t(0,300,\frz10)\t(300,600,\frz0)}")
        elif self.style_intensity == StyleIntensity.MEDIUM:
            if emotion == EmotionCategory.HAPPY:
                # Subtle bounce
                effects.append(r"{\move(0,-5,0,0,0,200)}")
            elif emotion == EmotionCategory.ANGRY:
                # Slight emphasis
                effects.append(r"{\fscx110\fscy110}")
            elif emotion == EmotionCategory.SAD:
                # Gentle fade
                effects.append(r"{\alpha&H20&}")
            elif emotion == EmotionCategory.EXCITED:
                # Small scale pulse
                effects.append(r"{\t(0,150,\fscx105\fscy105)}")

        # Add karaoke-style effects for certain emotions
        if emotion in [EmotionCategory.HAPPY, EmotionCategory.EXCITED] and self.style_intensity != StyleIntensity.SUBTLE:
            # Highlight important words
            words = text.split()
            formatted_words = []
            for word in words:
                if len(word) > 4:  # Emphasize longer words
                    formatted_words.append(f"{{\\fscx115\\fscy115\\b1}}{word}{{\\r}}")
                else:
                    formatted_words.append(word)
            text = " ".join(formatted_words)

        # Add position-based movement for dynamic emotions
        if emotion in [EmotionCategory.EXCITED, EmotionCategory.HAPPY] and self.style_intensity == StyleIntensity.INTENSE:
            # Add slight horizontal movement
            effects.insert(0, r"{\move(-5,0,5,0,0,500)}")

        # Combine effects and text
        if effects:
            return "".join(effects) + text
        return text
    
    def _apply_word_effects(
        self,
        text: str,
        emotion: EmotionCategory,
        metadata: Dict[str, Any],
        position_x: int,
        word_index: int,
        animation_style: str,
        is_emphasized: bool
    ) -> str:
        """Apply word-by-word specific effects using ASS override tags."""
        effects = []
        
        # Get size multiplier from metadata or emotion adjustments
        size_multiplier = metadata.get("size_multiplier", 1.0)
        emotion_adj = self.EMOTION_ADJUSTMENTS.get(emotion, self.EMOTION_ADJUSTMENTS[EmotionCategory.NEUTRAL])
        
        # Apply emotion-based size scaling
        base_scale_x = size_multiplier * 100
        base_scale_y = size_multiplier * 100
        
        # Apply intensity multiplier
        intensity_multipliers = {
            StyleIntensity.SUBTLE: 0.85,
            StyleIntensity.MEDIUM: 1.0,
            StyleIntensity.INTENSE: 1.35
        }
        intensity_mult = intensity_multipliers[self.style_intensity]
        
        emotion_scale_x = int(base_scale_x * intensity_mult)
        emotion_scale_y = int(base_scale_y * intensity_mult)
        
        # Get position offset from metadata
        position_offset = metadata.get("position_offset", (0, 0))
        x_offset, y_offset = position_offset
        
        # Convert to screen coordinates
        center_x = self.video_resolution[0] // 2
        center_y = self.video_resolution[1] // 2
        absolute_x = center_x + x_offset
        absolute_y = center_y + y_offset
        
        # Word animation based on style with safe zone positioning
        if animation_style == "fade_in":
            # Fade in effect with calculated positioning
            effects.append(rf"{{\fscx{emotion_scale_x}\fscy{emotion_scale_y}\fade(255,0,0,0,300)\pos({position_x},{absolute_y})}}")
        
        elif animation_style == "pop_in":
            # Scale from 0 to emotion-based size
            effects.append(rf"{{\pos({position_x},{absolute_y})\t(0,200,\fscx0\fscy0)\t(200,300,\fscx{emotion_scale_x}\fscy{emotion_scale_y})}}")
        
        elif animation_style == "slide_in":
            # Slide from left with calculated positioning
            start_x = position_x - 50
            effects.append(rf"{{\fscx{emotion_scale_x}\fscy{emotion_scale_y}\move({start_x},{absolute_y},{position_x},{absolute_y},0,300)}}")
        
        elif animation_style == "bounce_in":
            # Bounce effect with emotion-aware heights
            bounce_height = metadata.get("bounce_height", 0.03)
            bounce_pixels = int(self.video_resolution[1] * bounce_height)
            
            if word_index % 2 == 0:
                bounce_offset = -bounce_pixels
            else:
                bounce_offset = bounce_pixels
            
            effects.append(rf"{{\fscx{emotion_scale_x}\fscy{emotion_scale_y}\move({position_x},{absolute_y + bounce_offset},{position_x},{absolute_y},0,400)}}")
        
        elif animation_style == "wave":
            # Wave pattern is already calculated in position offset
            effects.append(rf"{{\fscx{emotion_scale_x}\fscy{emotion_scale_y}\pos({position_x},{absolute_y})}}")
        
        elif animation_style == "karaoke":
            # Karaoke-style highlight with calculated positioning
            effects.append(rf"{{\fscx{emotion_scale_x}\fscy{emotion_scale_y}\pos({position_x},{absolute_y})}}")
            if word_index == 0:
                effects.append(r"{\k30}")  # Karaoke timing
        
        elif animation_style == "typewriter":
            # Typewriter effect with calculated positioning
            effects.append(rf"{{\fscx{emotion_scale_x}\fscy{emotion_scale_y}\pos({position_x},{absolute_y})}}")
        
        elif animation_style == "emphasis":
            # Emphasis animation based on emotion
            if emotion == EmotionCategory.ANGRY:
                # Shake effect for angry
                import random
                shake_x = position_x + random.randint(-5, 5)
                shake_y = absolute_y + random.randint(-5, 5)
                effects.append(rf"{{\fscx{emotion_scale_x * 1.2}\fscy{emotion_scale_y * 1.2}\pos({shake_x},{shake_y})}}")
            elif emotion == EmotionCategory.HAPPY:
                # Jump effect for happy
                jump_offset = int(self.video_resolution[1] * 0.02)  # 2% of height
                jump_y = absolute_y - jump_offset
                effects.append(rf"{{\fscx{emotion_scale_x}\fscy{emotion_scale_y}\move({position_x},{jump_y},{position_x},{absolute_y},0,200)}}")
            else:
                effects.append(rf"{{\fscx{emotion_scale_x}\fscy{emotion_scale_y}\pos({position_x},{absolute_y})}}")
        
        else:
            # Default positioning with calculated coordinates
            effects.append(rf"{{\fscx{emotion_scale_x}\fscy{emotion_scale_y}\pos({position_x},{absolute_y})}}")
        
        # Emphasis effects with enhanced emotion-based scaling
        if is_emphasized:
            # Extra size boost for emphasized words
            emphasis_boost = 1.3
            new_scale_x = int(emotion_scale_x * emphasis_boost)
            new_scale_y = int(emotion_scale_y * emphasis_boost)
            
            # Override or add to existing scale
            effects.append(rf"{{\fscx{new_scale_x}\fscy{new_scale_y}\b1}}")
            
            # Add emotion-specific emphasis with stronger colors
            emotion_colors = self.EMOTION_COLORS.get(emotion, self.EMOTION_COLORS[EmotionCategory.NEUTRAL])
            effects.append(rf"{{\c{emotion_colors['primary']}}}")
            
            # Add emotion-specific effects for emphasis
            if emotion == EmotionCategory.HAPPY:
                # Add glow effect for happy
                effects.append(r"{\blur2}")
            elif emotion == EmotionCategory.ANGRY:
                # Add shadow for angry
                effects.append(r"{\shad3}")
            elif emotion == EmotionCategory.SAD:
                # Add fade for sad
                effects.append(r"{\alpha&H40&}")
        
        # Apply emotion-specific alignment based on position
        alignment = emotion_adj["alignment"]
        effects.append(rf"{{\an{alignment}}}")
        
        # Add rotation from metadata
        rotation_angle = metadata.get("rotation_angle", 0.0)
        if rotation_angle != 0:
            effects.append(rf"{{\frz{rotation_angle}}}")
        
        # Combine effects and text
        if effects:
            return "".join(effects) + text + r"{\r}"
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