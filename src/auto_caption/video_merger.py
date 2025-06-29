"""
Video merger module for applying emotion-styled captions to videos.

This module handles the rendering of emotionally-styled captions onto videos,
including text animations, color schemes, and visual effects based on detected emotions.
"""

import os
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2

from moviepy.editor import (
    VideoFileClip,
    TextClip,
    CompositeVideoClip,
    ImageClip,
    ColorClip,
    concatenate_videoclips
)
from moviepy.video.fx import resize, loop
from moviepy.video.tools.drawing import circle
import moviepy.video.fx.all as vfx

from .emotion_detector import EmotionCategory
from .caption_styler import StyleIntensity, Platform


class TextAnimation(Enum):
    """Available text animation styles."""
    NONE = "none"
    FADE = "fade"
    SLIDE = "slide"
    BOUNCE = "bounce"
    SHAKE = "shake"
    POP = "pop"
    WAVE = "wave"
    TYPEWRITER = "typewriter"
    GLOW = "glow"
    SPARKLE = "sparkle"
    FIRE = "fire"
    DRIP = "drip"
    SLAM = "slam"
    TILT = "tilt"
    FLIP = "flip"
    EXPAND = "expand"
    VIBRATE = "vibrate"
    PULSE = "pulse"
    SWIRL = "swirl"
    GLITCH = "glitch"


class EffectType(Enum):
    """Available visual effects."""
    NONE = "none"
    CONFETTI = "confetti"
    STARS = "stars"
    RAIN = "rain"
    SNOW = "snow"
    FIRE = "fire"
    LIGHTNING = "lightning"
    HEARTS = "hearts"
    SPARKLES = "sparkles"
    SMOKE = "smoke"
    BLUR_BACKGROUND = "blur_background"
    VIGNETTE = "vignette"
    SHAKE_FRAME = "shake_frame"
    COLOR_OVERLAY = "color_overlay"
    PARTICLE_BURST = "particle_burst"


@dataclass
class CaptionStyle:
    """Style configuration for a caption segment."""
    text: str
    start_time: float
    end_time: float
    emotion: EmotionCategory
    color_primary: str
    color_secondary: str
    color_background: str
    font_style: str
    text_animation: List[str]
    effects: List[str]
    position: Tuple[str, str]  # (horizontal, vertical) alignment
    font_size: int
    stroke_width: int
    stroke_color: str
    opacity: float
    animation_speed: float
    emphasis_delay: float


class VideoMerger:
    """
    Merges emotion-styled captions with video files.
    
    Handles rendering of styled text, animations, and visual effects
    based on emotion detection results.
    """
    
    def __init__(
        self,
        platform: Platform = Platform.GENERAL,
        quality: str = "high",
        font_path: Optional[str] = None,
        output_format: str = "mp4",
        fps: Optional[int] = None,
        verbose: bool = False
    ):
        """
        Initialize the video merger.
        
        Args:
            platform: Target platform for optimization
            quality: Output quality ('low', 'medium', 'high')
            font_path: Path to custom font file
            output_format: Output video format
            fps: Output frames per second (None to match input)
            verbose: Enable verbose output
        """
        self.platform = platform
        self.quality = quality
        self.font_path = font_path or self._get_default_font()
        self.output_format = output_format
        self.fps = fps
        self.verbose = verbose
        
        # Platform-specific settings
        self.platform_settings = self._get_platform_settings()
        
        # Quality presets
        self.quality_presets = {
            "low": {"bitrate": "1M", "preset": "fast"},
            "medium": {"bitrate": "3M", "preset": "medium"},
            "high": {"bitrate": "8M", "preset": "slow"}
        }
    
    def _get_default_font(self) -> str:
        """Get default font path based on system."""
        # Try to find a good default font
        font_candidates = [
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "C:/Windows/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf"
        ]
        
        for font in font_candidates:
            if os.path.exists(font):
                return font
        
        # Fallback to moviepy's default
        return "Arial"
    
    def _get_platform_settings(self) -> Dict[str, Any]:
        """Get platform-specific video settings."""
        settings = {
            Platform.TIKTOK: {
                "resolution": (1080, 1920),  # 9:16 vertical
                "max_duration": 180,  # 3 minutes
                "safe_zone": (0.9, 0.8),  # 90% width, 80% height
                "caption_position": ("center", "bottom"),
                "font_size_multiplier": 1.2
            },
            Platform.INSTAGRAM: {
                "resolution": (1080, 1350),  # 4:5 vertical
                "max_duration": 90,  # 90 seconds for reels
                "safe_zone": (0.85, 0.75),
                "caption_position": ("center", "bottom"),
                "font_size_multiplier": 1.1
            },
            Platform.YOUTUBE_SHORTS: {
                "resolution": (1080, 1920),  # 9:16 vertical
                "max_duration": 60,  # 60 seconds
                "safe_zone": (0.9, 0.85),
                "caption_position": ("center", "bottom"),
                "font_size_multiplier": 1.15
            },
            Platform.GENERAL: {
                "resolution": None,  # Keep original
                "max_duration": None,
                "safe_zone": (0.9, 0.9),
                "caption_position": ("center", "bottom"),
                "font_size_multiplier": 1.0
            }
        }
        
        return settings.get(self.platform, settings[Platform.GENERAL])
    
    def merge_with_video(
        self,
        video_path: str,
        caption_data: Union[str, Dict[str, Any]],
        output_path: str,
        preview_mode: bool = False,
        custom_style_overrides: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Merge styled captions with video.
        
        Args:
            video_path: Path to input video
            caption_data: Path to JSON caption file or caption dictionary
            output_path: Path for output video
            preview_mode: Generate low-quality preview
            custom_style_overrides: Custom style overrides
            
        Returns:
            Path to output video file
        """
        # Load caption data
        if isinstance(caption_data, str):
            with open(caption_data, 'r', encoding='utf-8') as f:
                captions = json.load(f)
        else:
            captions = caption_data
        
        # Load video
        video = VideoFileClip(video_path)
        
        # Apply platform-specific transformations
        video = self._apply_platform_settings(video)
        
        # Process each caption segment
        caption_clips = []
        effect_clips = []
        
        for segment in captions.get("segments", []):
            # Create caption style from segment data
            caption_style = self._create_caption_style(
                segment,
                video.size,
                custom_style_overrides
            )
            
            # Create text clip with styling
            text_clip = self._create_styled_text_clip(
                caption_style,
                video.fps
            )
            
            if text_clip:
                caption_clips.append(text_clip)
            
            # Create effect clips
            effects = self._create_effect_clips(
                caption_style,
                video.size,
                video.fps
            )
            effect_clips.extend(effects)
        
        # Composite all elements
        final_video = self._composite_video(
            video,
            caption_clips,
            effect_clips,
            preview_mode
        )
        
        # Write output video
        output_path = self._write_video(
            final_video,
            output_path,
            preview_mode
        )
        
        # Clean up
        video.close()
        final_video.close()
        
        return output_path
    
    def _apply_platform_settings(self, video: VideoFileClip) -> VideoFileClip:
        """Apply platform-specific video settings."""
        settings = self.platform_settings
        
        # Resize if needed
        if settings["resolution"]:
            target_w, target_h = settings["resolution"]
            current_w, current_h = video.size
            
            # Calculate scaling to fit
            scale_w = target_w / current_w
            scale_h = target_h / current_h
            scale = min(scale_w, scale_h)
            
            if scale != 1.0:
                new_size = (int(current_w * scale), int(current_h * scale))
                video = video.resize(new_size)
                
                # Add padding if needed
                if new_size[0] < target_w or new_size[1] < target_h:
                    # Create black background
                    bg = ColorClip(size=(target_w, target_h), color=(0, 0, 0))
                    bg = bg.set_duration(video.duration)
                    
                    # Center video on background
                    video = CompositeVideoClip([
                        bg,
                        video.set_position("center")
                    ])
        
        # Trim duration if needed
        if settings["max_duration"] and video.duration > settings["max_duration"]:
            video = video.subclip(0, settings["max_duration"])
        
        return video
    
    def _create_caption_style(
        self,
        segment: Dict[str, Any],
        video_size: Tuple[int, int],
        custom_overrides: Optional[Dict[str, Any]]
    ) -> CaptionStyle:
        """Create caption style from segment data."""
        # Extract emotion metadata
        emotion_meta = segment.get("emotion_metadata", {})
        visual_suggestions = emotion_meta.get("visual_suggestions", {})
        color_scheme = visual_suggestions.get("color_scheme", {})
        
        # Get emotion
        emotion_str = emotion_meta.get("emotion", "neutral")
        emotion = EmotionCategory(emotion_str)
        
        # Calculate font size based on video size and platform
        base_font_size = int(video_size[1] * 0.05)  # 5% of height
        font_size = int(base_font_size * self.platform_settings["font_size_multiplier"])
        
        # Apply custom overrides if provided
        if custom_overrides:
            color_scheme.update(custom_overrides.get("colors", {}))
            font_size = custom_overrides.get("font_size", font_size)
        
        # Get position
        h_pos, v_pos = self.platform_settings["caption_position"]
        
        return CaptionStyle(
            text=segment.get("text", ""),
            start_time=segment.get("start", 0),
            end_time=segment.get("end", 0),
            emotion=emotion,
            color_primary=color_scheme.get("primary", "#FFFFFF"),
            color_secondary=color_scheme.get("secondary", "#CCCCCC"),
            color_background=color_scheme.get("background", "#000000"),
            font_style=visual_suggestions.get("font_style", "normal"),
            text_animation=visual_suggestions.get("text_animation", ["fade"]),
            effects=visual_suggestions.get("effects", []),
            position=(h_pos, v_pos),
            font_size=font_size,
            stroke_width=max(2, int(font_size * 0.1)),
            stroke_color="#000000",
            opacity=0.95,
            animation_speed=visual_suggestions.get("timing_adjustments", {}).get("appear_speed", "medium"),
            emphasis_delay=visual_suggestions.get("timing_adjustments", {}).get("emphasis_delay", 0.3)
        )
    
    def _create_styled_text_clip(
        self,
        style: CaptionStyle,
        fps: float
    ) -> Optional[CompositeVideoClip]:
        """Create styled text clip with animations."""
        # Create base text clip
        try:
            text_clip = TextClip(
                style.text,
                fontsize=style.font_size,
                color=style.color_primary,
                font=self.font_path,
                stroke_color=style.stroke_color,
                stroke_width=style.stroke_width,
                method='caption',
                size=(int(self.platform_settings["safe_zone"][0] * 1080), None),
                align='center'
            )
        except Exception as e:
            if self.verbose:
                print(f"Error creating text clip: {e}")
            return None
        
        # Set duration and timing
        duration = style.end_time - style.start_time
        text_clip = text_clip.set_duration(duration)
        text_clip = text_clip.set_start(style.start_time)
        
        # Apply position
        text_clip = text_clip.set_position(style.position)
        
        # Apply animations
        for animation in style.text_animation[:1]:  # Use primary animation
            text_clip = self._apply_text_animation(
                text_clip,
                animation,
                style.animation_speed,
                fps
            )
        
        # Add background if needed
        if style.opacity < 1.0 or style.color_background != "#000000":
            # Create semi-transparent background
            bg_clip = self._create_text_background(
                text_clip,
                style.color_background,
                style.opacity
            )
            
            if bg_clip:
                text_clip = CompositeVideoClip([bg_clip, text_clip])
        
        return text_clip
    
    def _apply_text_animation(
        self,
        clip: TextClip,
        animation: str,
        speed: str,
        fps: float
    ) -> TextClip:
        """Apply animation to text clip."""
        duration = clip.duration
        
        # Speed multipliers
        speed_mult = {"fast": 0.5, "medium": 1.0, "slow": 2.0}.get(speed, 1.0)
        
        if animation == "fade":
            # Fade in and out
            fade_duration = 0.3 * speed_mult
            clip = clip.crossfadein(fade_duration).crossfadeout(fade_duration)
            
        elif animation == "slide":
            # Slide in from bottom
            def slide_pos(t):
                progress = min(1.0, t / (0.5 * speed_mult))
                y_offset = int((1 - progress) * 100)
                return ('center', 'bottom', 0, y_offset)
            clip = clip.set_position(slide_pos)
            
        elif animation == "bounce":
            # Bounce effect
            def bounce_pos(t):
                progress = t / duration
                bounce_height = 20 * abs(np.sin(progress * np.pi * 4))
                return ('center', 'bottom', 0, int(bounce_height))
            clip = clip.set_position(bounce_pos)
            
        elif animation == "shake":
            # Shake effect
            def shake_pos(t):
                if t < 0.5 * speed_mult:
                    shake_x = np.random.randint(-5, 5)
                    shake_y = np.random.randint(-5, 5)
                else:
                    shake_x = shake_y = 0
                return ('center', 'bottom', shake_x, shake_y)
            clip = clip.set_position(shake_pos)
            
        elif animation == "pop":
            # Pop/scale effect
            def pop_scale(t):
                progress = min(1.0, t / (0.3 * speed_mult))
                if progress < 0.5:
                    scale = 0.5 + progress
                else:
                    scale = 1.0 + (1.0 - progress) * 0.2
                return scale
            clip = clip.resize(pop_scale)
            
        elif animation == "typewriter":
            # Typewriter effect
            def typewriter_mask(t):
                progress = t / (duration * 0.7)
                char_count = int(progress * len(clip.txt))
                return clip.txt[:char_count]
            # This would need custom implementation
            
        elif animation == "glow":
            # Add glow effect (simplified)
            clip = clip.fx(vfx.colorx, 1.2)
            
        elif animation == "pulse":
            # Pulse effect
            def pulse_scale(t):
                pulse = 1.0 + 0.1 * np.sin(t * np.pi * 4 / speed_mult)
                return pulse
            clip = clip.resize(pulse_scale)
        
        return clip
    
    def _create_text_background(
        self,
        text_clip: TextClip,
        bg_color: str,
        opacity: float
    ) -> Optional[ColorClip]:
        """Create background for text clip."""
        try:
            # Get text dimensions
            w, h = text_clip.size
            
            # Add padding
            padding = 20
            bg_size = (w + 2 * padding, h + 2 * padding)
            
            # Convert hex color to RGB
            bg_color_rgb = tuple(int(bg_color[i:i+2], 16) for i in (1, 3, 5))
            
            # Create background clip
            bg_clip = ColorClip(
                size=bg_size,
                color=bg_color_rgb
            )
            
            # Set properties
            bg_clip = bg_clip.set_duration(text_clip.duration)
            bg_clip = bg_clip.set_start(text_clip.start)
            bg_clip = bg_clip.set_opacity(opacity * 0.7)
            
            # Position behind text
            pos = text_clip.pos
            if callable(pos):
                def bg_pos(t):
                    text_p = pos(t) if callable(pos) else pos
                    return (text_p[0], text_p[1] - padding)
                bg_clip = bg_clip.set_position(bg_pos)
            else:
                bg_clip = bg_clip.set_position((pos[0], pos[1] - padding))
            
            return bg_clip
            
        except Exception as e:
            if self.verbose:
                print(f"Error creating text background: {e}")
            return None
    
    def _create_effect_clips(
        self,
        style: CaptionStyle,
        video_size: Tuple[int, int],
        fps: float
    ) -> List[ImageClip]:
        """Create visual effect clips."""
        effect_clips = []
        
        for effect in style.effects[:2]:  # Limit to 2 effects
            if effect == "confetti":
                clips = self._create_confetti_effect(
                    style.start_time,
                    style.end_time,
                    video_size,
                    style.color_primary
                )
                effect_clips.extend(clips)
                
            elif effect == "stars":
                clips = self._create_stars_effect(
                    style.start_time,
                    style.end_time,
                    video_size,
                    style.color_secondary
                )
                effect_clips.extend(clips)
                
            elif effect == "hearts":
                clips = self._create_hearts_effect(
                    style.start_time,
                    style.end_time,
                    video_size,
                    style.color_primary
                )
                effect_clips.extend(clips)
                
            elif effect == "fire":
                # Fire effect would need particle system
                pass
                
            elif effect == "rain":
                clips = self._create_rain_effect(
                    style.start_time,
                    style.end_time,
                    video_size
                )
                effect_clips.extend(clips)
        
        return effect_clips
    
    def _create_confetti_effect(
        self,
        start_time: float,
        end_time: float,
        video_size: Tuple[int, int],
        color: str
    ) -> List[ImageClip]:
        """Create confetti particle effect."""
        clips = []
        duration = end_time - start_time
        
        # Create 10-20 confetti pieces
        for i in range(15):
            # Create small colored rectangle
            size = (10, 20)
            confetti = self._create_particle_image(size, color)
            
            # Random starting position at top
            start_x = np.random.randint(0, video_size[0])
            start_y = -size[1]
            
            # Create movement function
            def move_func(t, sx=start_x):
                progress = t / duration
                x = sx + np.sin(t * 2) * 50  # Sway effect
                y = progress * (video_size[1] + 100)
                return (int(x), int(y))
            
            # Create clip
            clip = ImageClip(confetti, duration=duration)
            clip = clip.set_position(move_func)
            clip = clip.set_start(start_time + i * 0.1)  # Stagger start
            
            clips.append(clip)
        
        return clips
    
    def _create_stars_effect(
        self,
        start_time: float,
        end_time: float,
        video_size: Tuple[int, int],
        color: str
    ) -> List[ImageClip]:
        """Create star particle effect."""
        clips = []
        duration = end_time - start_time
        
        for i in range(8):
            # Create star image
            star = self._create_star_image(20, color)
            
            # Random position
            x = np.random.randint(50, video_size[0] - 50)
            y = np.random.randint(50, video_size[1] // 2)
            
            # Create pulsing effect
            def pulse_func(t):
                scale = 1.0 + 0.3 * np.sin(t * np.pi * 2)
                return scale
            
            # Create clip
            clip = ImageClip(star, duration=duration)
            clip = clip.set_position((x, y))
            clip = clip.resize(pulse_func)
            clip = clip.set_start(start_time)
            clip = clip.set_opacity(0.8)
            
            clips.append(clip)
        
        return clips
    
    def _create_hearts_effect(
        self,
        start_time: float,
        end_time: float,
        video_size: Tuple[int, int],
        color: str
    ) -> List[ImageClip]:
        """Create floating hearts effect."""
        clips = []
        duration = end_time - start_time
        
        for i in range(6):
            # Create heart image
            heart = self._create_heart_image(30, color)
            
            # Random starting position at bottom
            start_x = np.random.randint(100, video_size[0] - 100)
            start_y = video_size[1]
            
            # Float upward with sway
            def float_func(t, sx=start_x):
                progress = t / duration
                x = sx + np.sin(t * 1.5) * 30
                y = start_y - progress * (video_size[1] + 100)
                return (int(x), int(y))
            
            # Create clip
            clip = ImageClip(heart, duration=duration)
            clip = clip.set_position(float_func)
            clip = clip.set_start(start_time + i * 0.2)
            clip = clip.set_opacity(0.7)
            
            clips.append(clip)
        
        return clips
    
    def _create_rain_effect(
        self,
        start_time: float,
        end_time: float,
        video_size: Tuple[int, int]
    ) -> List[ImageClip]:
        """Create rain effect."""
        clips = []
        duration = end_time - start_time
        
        for i in range(20):
            # Create raindrop
            drop = self._create_raindrop_image()
            
            # Random starting position
            start_x = np.random.randint(0, video_size[0])
            start_y = -50
            
            # Fall speed
            speed = np.random.randint(400, 800)
            
            def fall_func(t, sx=start_x, sp=speed):
                x = sx
                y = start_y + t * sp
                return (int(x), int(y))
            
            # Create clip
            clip = ImageClip(drop, duration=duration)
            clip = clip.set_position(fall_func)
            clip = clip.set_start(start_time + i * 0.05)
            clip = clip.set_opacity(0.5)
            
            clips.append(clip)
        
        return clips
    
    def _create_particle_image(
        self,
        size: Tuple[int, int],
        color: str
    ) -> np.ndarray:
        """Create a simple colored particle."""
        # Create RGBA image
        img = Image.new('RGBA', size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Convert hex to RGB
        rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
        
        # Draw rectangle
        draw.rectangle([0, 0, size[0]-1, size[1]-1], fill=(*rgb, 255))
        
        return np.array(img)
    
    def _create_star_image(self, size: int, color: str) -> np.ndarray:
        """Create a star shape."""
        img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Convert hex to RGB
        rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
        
        # Draw star (simplified as asterisk)
        center = size // 2
        draw.line([center, 0, center, size], fill=(*rgb, 255), width=2)
        draw.line([0, center, size, center], fill=(*rgb, 255), width=2)
        draw.line([0, 0, size, size], fill=(*rgb, 255), width=2)
        draw.line([size, 0, 0, size], fill=(*rgb, 255), width=2)
        
        return np.array(img)
    
    def _create_heart_image(self, size: int, color: str) -> np.ndarray:
        """Create a heart shape."""
        img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Convert hex to RGB
        rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
        
        # Draw heart (simplified)
        points = [
            (size//2, size-5),
            (5, size//3),
            (5, size//4),
            (size//4, 5),
            (size//2, size//3),
            (3*size//4, 5),
            (size-5, size//4),
            (size-5, size//3),
            (size//2, size-5)
        ]
        draw.polygon(points, fill=(*rgb, 255))
        
        return np.array(img)
    
    def _create_raindrop_image(self) -> np.ndarray:
        """Create a raindrop shape."""
        img = Image.new('RGBA', (4, 20), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Light blue raindrop
        draw.rectangle([1, 0, 2, 19], fill=(173, 216, 230, 180))
        
        return np.array(img)
    
    def _composite_video(
        self,
        base_video: VideoFileClip,
        caption_clips: List[CompositeVideoClip],
        effect_clips: List[ImageClip],
        preview_mode: bool
    ) -> CompositeVideoClip:
        """Composite all elements into final video."""
        # Start with base video
        clips = [base_video]
        
        # Add effects (behind captions)
        clips.extend(effect_clips)
        
        # Add captions (on top)
        clips.extend(caption_clips)
        
        # Create composite
        final = CompositeVideoClip(clips)
        
        # Apply preview mode settings
        if preview_mode:
            final = final.resize(0.5)  # Half resolution
            final = final.set_fps(15)  # Lower framerate
        
        return final
    
    def _write_video(
        self,
        video: CompositeVideoClip,
        output_path: str,
        preview_mode: bool
    ) -> str:
        """Write final video to file."""
        # Get quality settings
        quality = "low" if preview_mode else self.quality
        preset = self.quality_presets[quality]
        
        # Set codec based on format
        if self.output_format == "mp4":
            codec = "libx264"
        elif self.output_format == "webm":
            codec = "libvpx"
        else:
            codec = None
        
        # Write video
        video.write_videofile(
            output_path,
            fps=self.fps or video.fps,
            codec=codec,
            bitrate=preset["bitrate"],
            preset=preset["preset"],
            audio_codec="aac",
            threads=4,
            logger=None if not self.verbose else "bar"
        )
        
        return output_path
    
    def create_preview_grid(
        self,
        video_path: str,
        caption_data: Union[str, Dict[str, Any]],
        output_path: str,
        grid_size: Tuple[int, int] = (2, 2),
        segment_duration: float = 2.0
    ) -> str:
        """
        Create a preview grid showing different emotion styles.
        
        Args:
            video_path: Path to input video
            caption_data: Caption data with emotions
            output_path: Path for output preview
            grid_size: Grid dimensions (rows, cols)
            segment_duration: Duration of each preview segment
            
        Returns:
            Path to preview video
        """
        # Load caption data
        if isinstance(caption_data, str):
            with open(caption_data, 'r', encoding='utf-8') as f:
                captions = json.load(f)
        else:
            captions = caption_data
        
        # Load video
        video = VideoFileClip(video_path)
        
        # Get unique emotions from segments
        emotions = []
        for segment in captions.get("segments", []):
            emotion = segment.get("emotion_metadata", {}).get("emotion", "neutral")
            if emotion not in [e for e, _ in emotions]:
                emotions.append((emotion, segment))
        
        # Limit to grid size
        max_previews = grid_size[0] * grid_size[1]
        emotions = emotions[:max_previews]
        
        # Create preview clips
        preview_clips = []
        cell_width = video.size[0] // grid_size[1]
        cell_height = video.size[1] // grid_size[0]
        
        for idx, (emotion, segment) in enumerate(emotions):
            # Get position in grid
            row = idx // grid_size[1]
            col = idx % grid_size[1]
            
            # Extract video segment
            start = segment.get("start", 0)
            end = min(start + segment_duration, video.duration)
            clip = video.subclip(start, end)
            
            # Resize to fit grid cell
            clip = clip.resize((cell_width, cell_height))
            
            # Add emotion label
            label = TextClip(
                f"{emotion.upper()}",
                fontsize=20,
                color='white',
                font=self.font_path,
                stroke_color='black',
                stroke_width=2
            )
            label = label.set_duration(clip.duration)
            label = label.set_position(('center', 'top')).set_margin(10)
            
            # Composite label on clip
            clip = CompositeVideoClip([clip, label])
            
            # Position in grid
            x = col * cell_width
            y = row * cell_height
            clip = clip.set_position((x, y))
            
            preview_clips.append(clip)
        
        # Create grid composite
        grid_size_pixels = (grid_size[1] * cell_width, grid_size[0] * cell_height)
        background = ColorClip(size=grid_size_pixels, color=(0, 0, 0))
        background = background.set_duration(segment_duration)
        
        final = CompositeVideoClip([background] + preview_clips)
        
        # Write preview
        final.write_videofile(
            output_path,
            fps=15,  # Lower FPS for preview
            codec="libx264",
            preset="fast",
            logger=None if not self.verbose else "bar"
        )
        
        # Clean up
        video.close()
        final.close()
        
        return output_path
    
    def batch_merge(
        self,
        video_caption_pairs: List[Tuple[str, Union[str, Dict]]],
        output_dir: str,
        name_pattern: str = "{name}_captioned.{ext}",
        parallel: bool = False,
        progress_callback: Optional[callable] = None
    ) -> List[str]:
        """
        Batch merge multiple videos with captions.
        
        Args:
            video_caption_pairs: List of (video_path, caption_data) tuples
            output_dir: Directory for output videos
            name_pattern: Output naming pattern
            parallel: Process videos in parallel
            progress_callback: Progress callback function
            
        Returns:
            List of output video paths
        """
        output_paths = []
        total = len(video_caption_pairs)
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for idx, (video_path, caption_data) in enumerate(video_caption_pairs):
            if progress_callback:
                progress_callback(idx, total, os.path.basename(video_path))
            
            try:
                # Generate output filename
                video_name = Path(video_path).stem
                ext = self.output_format
                output_name = name_pattern.format(name=video_name, ext=ext)
                output_path = os.path.join(output_dir, output_name)
                
                # Merge video with captions
                result_path = self.merge_with_video(
                    video_path,
                    caption_data,
                    output_path
                )
                
                output_paths.append(result_path)
                
                if self.verbose:
                    print(f"✓ Processed: {video_name}")
                    
            except Exception as e:
                if self.verbose:
                    print(f"✗ Failed {video_name}: {str(e)}")
                output_paths.append(None)
        
        if progress_callback:
            progress_callback(total, total, "Complete")
        
        return output_paths