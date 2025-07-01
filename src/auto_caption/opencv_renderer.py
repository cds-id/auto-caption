"""
OpenCV-based caption renderer for emotion-styled video captions.

This module provides a fallback renderer when ffmpeg or ASS subtitle support
is not available. It renders captions directly onto video frames using OpenCV.

Note: For production use, the ASS subtitle approach with ffmpeg is recommended
for better performance and styling capabilities.
"""

import os
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import numpy as np
import cv2
from tqdm import tqdm

from .emotion_detector import EmotionCategory
from .caption_styler import StyleIntensity, Platform


@dataclass
class RenderedCaption:
    """Rendered caption with positioning and styling info."""
    text: str
    start_frame: int
    end_frame: int
    position: Tuple[int, int]
    font_scale: float
    color: Tuple[int, int, int]
    bg_color: Tuple[int, int, int]
    thickness: int
    animation: str
    animation_params: Dict[str, Any]


class OpenCVRenderer:
    """
    Fallback renderer for emotion-styled captions using OpenCV.
    
    This renderer directly draws text onto video frames and is useful when:
    - ffmpeg is not available
    - ASS subtitle support is not available
    - Quick preview generation is needed
    
    For production use, prefer the ASS subtitle approach with VideoMerger.
    """

    # OpenCV font mappings
    FONT_STYLES = {
        "normal": cv2.FONT_HERSHEY_SIMPLEX,
        "bold": cv2.FONT_HERSHEY_DUPLEX,
        "italic": cv2.FONT_HERSHEY_COMPLEX,
        "playful": cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        "dramatic": cv2.FONT_HERSHEY_TRIPLEX
    }

    def __init__(
        self,
        platform: Platform = Platform.GENERAL,
        font_scale_base: float = 1.0,
        line_spacing: float = 1.5,
        padding: int = 20,
        bg_opacity: float = 0.7,
        verbose: bool = False
    ):
        """
        Initialize the OpenCV renderer.

        Args:
            platform: Target platform for optimization
            font_scale_base: Base font scale multiplier
            line_spacing: Line spacing multiplier
            padding: Padding around text
            bg_opacity: Background opacity (0-1)
            verbose: Enable verbose output
        """
        self.platform = platform
        self.font_scale_base = font_scale_base
        self.line_spacing = line_spacing
        self.padding = padding
        self.bg_opacity = bg_opacity
        self.verbose = verbose

        # Platform-specific adjustments
        self.platform_settings = self._get_platform_settings()

    def _get_platform_settings(self) -> Dict[str, Any]:
        """Get platform-specific rendering settings."""
        settings = {
            Platform.TIKTOK: {
                "font_scale_multiplier": 1.3,
                "position_y_offset": 0.8,  # 80% down from top
                "max_width_ratio": 0.9,
                "shadow_offset": 3
            },
            Platform.INSTAGRAM: {
                "font_scale_multiplier": 1.2,
                "position_y_offset": 0.75,
                "max_width_ratio": 0.85,
                "shadow_offset": 2
            },
            Platform.YOUTUBE_SHORTS: {
                "font_scale_multiplier": 1.25,
                "position_y_offset": 0.85,
                "max_width_ratio": 0.9,
                "shadow_offset": 3
            },
            Platform.GENERAL: {
                "font_scale_multiplier": 1.0,
                "position_y_offset": 0.9,
                "max_width_ratio": 0.9,
                "shadow_offset": 2
            }
        }

        return settings.get(self.platform, settings[Platform.GENERAL])

    def render_captions_on_video(
        self,
        video_path: str,
        caption_data: Union[str, Dict[str, Any]],
        output_path: str,
        codec: str = "mp4v",
        preview_mode: bool = False
    ) -> str:
        """
        Render captions directly onto video frames (fallback method).

        This method is a fallback when ffmpeg/ASS subtitle burning is not available.
        It processes video frame by frame, which is slower but more compatible.

        Args:
            video_path: Path to input video
            caption_data: Caption data (path or dict)
            output_path: Path for output video
            codec: Video codec (mp4v, XVID, etc.)
            preview_mode: Low quality preview mode

        Returns:
            Path to output video
        
        Note:
            This method does not preserve audio. Use add_audio_from_original()
            to copy audio from the source video.
        """
        # Load caption data
        if isinstance(caption_data, str):
            with open(caption_data, 'r', encoding='utf-8') as f:
                captions = json.load(f)
        else:
            captions = caption_data

        # Open video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Adjust for preview mode
        if preview_mode:
            width = width // 2
            height = height // 2
            fps = min(fps, 15)

        # Prepare captions for rendering
        rendered_captions = self._prepare_captions(
            captions.get("segments", []),
            fps,
            width,
            height
        )
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        frame_idx = 0
        with tqdm(total=total_frames, desc="Rendering captions", disable=not self.verbose) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize for preview if needed
                if preview_mode:
                    frame = cv2.resize(frame, (width, height))
                
                # Find active captions for this frame
                active_captions = [
                    c for c in rendered_captions
                    if c.start_frame <= frame_idx <= c.end_frame
                ]
                
                # Render each active caption
                for caption in active_captions:
                    frame = self._render_caption_on_frame(
                        frame,
                        caption,
                        frame_idx
                    )
                
                # Write frame
                out.write(frame)
                frame_idx += 1
                pbar.update(1)
        
        # Clean up
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        if self.verbose:
            print(f"✓ Video saved to: {output_path}")
        
        return output_path
    
    def _prepare_captions(
        self,
        segments: List[Dict[str, Any]],
        fps: float,
        width: int,
        height: int
    ) -> List[RenderedCaption]:
        """Prepare caption segments for rendering."""
        rendered_captions = []
        
        for segment in segments:
            # Extract timing
            start_time = segment.get("start", 0)
            end_time = segment.get("end", 0)
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            
            # Extract styling from emotion metadata
            emotion_meta = segment.get("emotion_metadata", {})
            visual_suggestions = emotion_meta.get("visual_suggestions", {})
            color_scheme = visual_suggestions.get("color_scheme", {})
            
            # Parse colors (hex to BGR)
            primary_color = self._hex_to_bgr(color_scheme.get("primary", "#FFFFFF"))
            bg_color = self._hex_to_bgr(color_scheme.get("background", "#000000"))
            
            # Calculate font scale based on video size
            base_scale = min(width, height) / 1000
            font_scale = base_scale * self.font_scale_base * self.platform_settings["font_scale_multiplier"]
            
            # Determine position
            y_offset = self.platform_settings["position_y_offset"]
            position = (width // 2, int(height * y_offset))
            
            # Get animation (simplified for fallback renderer)
            animations = visual_suggestions.get("text_animation", ["fade"])
            primary_animation = animations[0] if animations else "fade"
            # Limit animations for performance in fallback mode
            if primary_animation not in ["fade", "slide", "none"]:
                primary_animation = "fade"
            
            # Split text into lines if too long
            text = segment.get("text", "")
            lines = self._wrap_text(text, width, font_scale)
            
            for line in lines:
                rendered_captions.append(RenderedCaption(
                    text=line,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    position=position,
                    font_scale=font_scale,
                    color=primary_color,
                    bg_color=bg_color,
                    thickness=max(2, int(font_scale * 2)),
                    animation=primary_animation,
                    animation_params={
                        "duration": end_time - start_time,
                        "fps": fps
                    }
                ))
        
        return rendered_captions
    
    def _hex_to_bgr(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to BGR tuple for OpenCV."""
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return (rgb[2], rgb[1], rgb[0])  # Convert RGB to BGR
    
    def _wrap_text(
        self,
        text: str,
        max_width: int,
        font_scale: float
    ) -> List[str]:
        """Wrap text to fit within max width."""
        font = self.FONT_STYLES["normal"]
        max_width = int(max_width * self.platform_settings["max_width_ratio"])
        
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, 2)
            
            if text_width > max_width and current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
            else:
                current_line.append(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines
    
    def _render_caption_on_frame(
        self,
        frame: np.ndarray,
        caption: RenderedCaption,
        frame_idx: int
    ) -> np.ndarray:
        """Render a single caption on a frame."""
        # Calculate animation progress
        progress = (frame_idx - caption.start_frame) / max(1, caption.end_frame - caption.start_frame)
        
        # Apply animation transformations
        position, opacity, scale = self._apply_animation(
            caption.position,
            progress,
            caption.animation,
            caption.animation_params
        )
        
        # Adjust font scale with animation
        font_scale = caption.font_scale * scale
        
        # Get text size
        font = self.FONT_STYLES.get("normal")
        (text_width, text_height), baseline = cv2.getTextSize(
            caption.text,
            font,
            font_scale,
            caption.thickness
        )
        
        # Calculate actual position (centered)
        x = position[0] - text_width // 2
        y = position[1]
        
        # Draw background
        if self.bg_opacity > 0:
            bg_padding = self.padding
            bg_x1 = x - bg_padding
            bg_y1 = y - text_height - bg_padding
            bg_x2 = x + text_width + bg_padding
            bg_y2 = y + baseline + bg_padding
            
            # Create overlay for background
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (bg_x1, bg_y1),
                (bg_x2, bg_y2),
                caption.bg_color,
                -1
            )
            
            # Blend with original frame
            alpha = self.bg_opacity * opacity
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Draw shadow
        shadow_offset = self.platform_settings["shadow_offset"]
        if shadow_offset > 0:
            cv2.putText(
                frame,
                caption.text,
                (x + shadow_offset, y + shadow_offset),
                font,
                font_scale,
                (0, 0, 0),  # Black shadow
                caption.thickness + 1,
                cv2.LINE_AA
            )
        
        # Draw main text
        text_color = tuple(int(c * opacity) for c in caption.color)
        cv2.putText(
            frame,
            caption.text,
            (x, y),
            font,
            font_scale,
            text_color,
            caption.thickness,
            cv2.LINE_AA
        )
        
        return frame
    
    def _apply_animation(
        self,
        base_position: Tuple[int, int],
        progress: float,
        animation: str,
        params: Dict[str, Any]
    ) -> Tuple[Tuple[int, int], float, float]:
        """Apply animation to get position, opacity, and scale."""
        x, y = base_position
        opacity = 1.0
        scale = 1.0
        
        if animation == "fade":
            # Fade in and out
            if progress < 0.1:
                opacity = progress * 10
            elif progress > 0.9:
                opacity = (1 - progress) * 10
            
        elif animation == "slide":
            # Slide up from bottom
            if progress < 0.2:
                slide_progress = progress * 5
                y_offset = int((1 - slide_progress) * 50)
                y = y + y_offset
                opacity = slide_progress
            
        elif animation == "bounce":
            # Bounce effect
            bounce_height = 20 * abs(np.sin(progress * np.pi * 4))
            y = y - int(bounce_height)
            
        elif animation == "pop":
            # Scale pop effect
            if progress < 0.1:
                scale = 0.5 + progress * 5
            elif progress < 0.2:
                scale = 1.0 + (0.2 - progress) * 2
            
        elif animation == "shake":
            # Shake effect
            if progress < 0.3:
                shake_amount = 5
                x += np.random.randint(-shake_amount, shake_amount)
                y += np.random.randint(-shake_amount, shake_amount)
        
        elif animation == "pulse":
            # Pulsing scale
            scale = 1.0 + 0.1 * np.sin(progress * np.pi * 8)
        
        return (x, y), opacity, scale
    
    def add_audio_from_original(
        self,
        original_video: str,
        rendered_video: str,
        output_path: str
    ) -> str:
        """
        Copy audio from original video to rendered video.
        
        This is essential when using the OpenCV fallback renderer since
        it doesn't preserve audio during frame processing.
        
        Args:
            original_video: Path to original video with audio
            rendered_video: Path to rendered video without audio
            output_path: Path for final output with audio
            
        Returns:
            Path to output video (rendered_video if audio copy fails)
        """
        import subprocess
        
        # Check if ffmpeg is available
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            if self.verbose:
                print("Warning: ffmpeg not available, output will have no audio")
            return rendered_video
        
        # Use ffmpeg to copy audio
        cmd = [
            'ffmpeg',
            '-i', rendered_video,
            '-i', original_video,
            '-c:v', 'copy',
            '-c:a', 'copy',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-shortest',
            '-y',
            output_path
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            # Remove temp file
            if rendered_video != output_path:
                os.remove(rendered_video)
            return output_path
        except subprocess.CalledProcessError as e:
            if self.verbose:
                print(f"Error copying audio: {e}")
                print("Output video will not have audio")
            return rendered_video