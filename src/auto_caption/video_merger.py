"""
Video merger module for applying emotion-formatted captions to videos.

This module handles the rendering of emotionally-formatted captions onto videos
using ASS subtitles for better reliability and performance.
"""

import os
import json
import tempfile
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

from moviepy.editor import VideoFileClip
import ffmpeg

from .emotion_detector import EmotionCategory
from .caption_styler import StyleIntensity, Platform
from .subtitle.ass_generator import ASSGenerator








class VideoMerger:
    """
    Merges emotion-formatted captions with video files using ASS subtitles.

    Uses ffmpeg with ASS subtitle burning for reliable and high-quality results.
    """

    def __init__(
        self,
        platform: Platform = Platform.GENERAL,
        quality: str = "high",
        style_intensity: StyleIntensity = StyleIntensity.MEDIUM,
        output_format: str = "mp4",
        fps: Optional[int] = None,
        verbose: bool = False,
        use_hardware_accel: bool = False
    ):
        """
        Initialize the video merger.

        Args:
            platform: Target platform for optimization
            quality: Output quality ('low', 'medium', 'high')
            style_intensity: Caption styling intensity
            output_format: Output video format
            fps: Output frames per second (None to match input)
            verbose: Enable verbose output
            use_hardware_accel: Use hardware acceleration if available
        """
        self.platform = platform
        self.quality = quality
        self.style_intensity = style_intensity
        self.output_format = output_format
        self.fps = fps
        self.verbose = verbose
        self.use_hardware_accel = use_hardware_accel

        # Platform-specific settings
        self.platform_settings = self._get_platform_settings()

        # Quality presets for ffmpeg
        self.quality_presets = {
            "low": {
                "crf": 28,
                "preset": "faster",
                "bitrate": "1M"
            },
            "medium": {
                "crf": 23,
                "preset": "medium",
                "bitrate": "3M"
            },
            "high": {
                "crf": 18,
                "preset": "slow",
                "bitrate": "8M"
            }
        }



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
        subtitle_format: str = "ass"
    ) -> str:
        """
        Merge styled captions with video using ASS subtitles.

        Args:
            video_path: Path to input video
            caption_data: Path to JSON caption file or caption dictionary
            output_path: Path for output video
            preview_mode: Generate low-quality preview
            subtitle_format: Subtitle format ('ass' or 'srt')

        Returns:
            Path to output video file
        """
        # Load caption data
        if isinstance(caption_data, str):
            with open(caption_data, 'r', encoding='utf-8') as f:
                captions = json.load(f)
        else:
            captions = caption_data

        # Get video info
        probe = ffmpeg.probe(video_path)
        video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)

        if not video_stream:
            raise ValueError("No video stream found in input file")

        width = int(video_stream['width'])
        height = int(video_stream['height'])
        video_fps = eval(video_stream.get('r_frame_rate', '25/1'))
        if isinstance(video_fps, float):
            fps_value = video_fps
        else:
            fps_value = 25.0

        # Create temporary subtitle file
        with tempfile.TemporaryDirectory() as temp_dir:
            if subtitle_format == "ass":
                # Generate ASS subtitle file
                subtitle_path = os.path.join(temp_dir, "subtitles.ass")
                ass_generator = ASSGenerator(
                    platform=self.platform,
                    style_intensity=self.style_intensity,
                    video_resolution=(width, height),
                    verbose=self.verbose
                )
                ass_generator.generate_ass_file(
                    captions,
                    subtitle_path,
                    title=f"Auto-Caption - {os.path.basename(video_path)}"
                )
            else:
                # Generate SRT as fallback
                subtitle_path = os.path.join(temp_dir, "subtitles.srt")
                self._generate_srt_file(captions, subtitle_path)

            # Apply platform-specific video transformations if needed
            processed_video = self._preprocess_video_if_needed(
                video_path,
                temp_dir,
                preview_mode
            )

            # Burn subtitles using ffmpeg
            output_path = self._burn_subtitles_with_ffmpeg(
                processed_video,
                subtitle_path,
                output_path,
                preview_mode
            )

        if self.verbose:
            print(f"✓ Video with styled captions saved to: {output_path}")

        return output_path

    def _preprocess_video_if_needed(
        self,
        video_path: str,
        temp_dir: str,
        preview_mode: bool
    ) -> str:
        """Preprocess video for platform requirements if needed."""
        settings = self.platform_settings

        # If no preprocessing needed, return original path
        if not settings["resolution"] and not settings["max_duration"] and not preview_mode:
            return video_path

        # Output path for preprocessed video
        processed_path = os.path.join(temp_dir, "preprocessed.mp4")

        # Build ffmpeg command
        input_stream = ffmpeg.input(video_path)

        # Apply duration limit if needed
        if settings["max_duration"]:
            input_stream = input_stream.trim(duration=settings["max_duration"])

        # Build filter chain
        filters = []

        # Scale for platform or preview
        if preview_mode:
            filters.append("scale=iw/2:ih/2")  # Half resolution for preview
        elif settings["resolution"]:
            target_w, target_h = settings["resolution"]
            # Scale to fit within target resolution while maintaining aspect ratio
            filters.append(f"scale='min({target_w},iw*min({target_w}/iw,{target_h}/ih)):"
                         f"min({target_h},ih*min({target_w}/iw,{target_h}/ih))'")
            # Add padding to reach exact resolution
            filters.append(f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black")

        # Apply filters if any
        if filters:
            input_stream = input_stream.filter('scale', 'iw/2', 'ih/2') if preview_mode else input_stream
            for f in filters[1:] if preview_mode else filters:
                parts = f.split('=', 1)
                if len(parts) == 2:
                    input_stream = input_stream.filter(parts[0], parts[1])

        # Output with appropriate quality
        quality_preset = self.quality_presets["low" if preview_mode else self.quality]

        output = ffmpeg.output(
            input_stream,
            processed_path,
            vcodec='libx264',
            crf=quality_preset["crf"],
            preset=quality_preset["preset"],
            acodec='copy'
        )

        # Run ffmpeg
        try:
            ffmpeg.run(output, overwrite_output=True, capture_stdout=True, capture_stderr=True)
            return processed_path
        except ffmpeg.Error as e:
            if self.verbose:
                print(f"FFmpeg preprocessing error: {e.stderr.decode()}")
            return video_path  # Fall back to original

    def _burn_subtitles_with_ffmpeg(
        self,
        video_path: str,
        subtitle_path: str,
        output_path: str,
        preview_mode: bool
    ) -> str:
        """Burn subtitles into video using ffmpeg."""
        # Get quality settings
        quality = "low" if preview_mode else self.quality
        quality_preset = self.quality_presets[quality]

        # Build ffmpeg command
        input_video = ffmpeg.input(video_path)

        # Prepare subtitle filter
        # For ASS subtitles, we use the subtitles filter
        if subtitle_path.endswith('.ass'):
            # Escape special characters in path for ffmpeg filter
            subtitle_path_escaped = subtitle_path.replace('\\', '/').replace(':', '\\:')
            subtitle_filter = f"subtitles='{subtitle_path_escaped}'"
        else:
            # For SRT, we can also use subtitles filter
            subtitle_path_escaped = subtitle_path.replace('\\', '/').replace(':', '\\:')
            subtitle_filter = f"subtitles='{subtitle_path_escaped}'"

        # Apply video filters
        stream = input_video.video.filter('subtitles', subtitle_path)

        # Add audio stream
        stream = ffmpeg.output(
            stream,
            input_video.audio,
            output_path,
            vcodec='libx264',
            acodec='copy',
            crf=quality_preset["crf"],
            preset=quality_preset["preset"],
            **{'max_muxing_queue_size': '1024'}  # Prevent muxing issues
        )

        # Add hardware acceleration if requested and available
        if self.use_hardware_accel:
            stream = stream.global_args('-hwaccel', 'auto')

        # Run ffmpeg
        try:
            if self.verbose:
                print(f"Running ffmpeg to burn subtitles...")
                ffmpeg.run(stream, overwrite_output=True)
            else:
                ffmpeg.run(stream, overwrite_output=True, quiet=True)
        except ffmpeg.Error as e:
            error_message = e.stderr.decode() if e.stderr else str(e)
            raise RuntimeError(f"Failed to burn subtitles: {error_message}")

        return output_path

    def _generate_srt_file(self, caption_data: Dict[str, Any], output_path: str):
        """Generate a simple SRT file as fallback."""
        import srt
        from datetime import timedelta

        subtitles = []
        segments = caption_data.get("segments", [])

        for idx, segment in enumerate(segments, 1):
            subtitle = srt.Subtitle(
                index=idx,
                start=timedelta(seconds=segment.get("start", 0)),
                end=timedelta(seconds=segment.get("end", 0)),
                content=segment.get("text", "").strip()
            )
            subtitles.append(subtitle)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(srt.compose(subtitles))



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

        # Get unique emotions from segments
        emotions = []
        emotion_segments = {}
        
        for segment in captions.get("segments", []):
            emotion = segment.get("emotion_metadata", {}).get("emotion", "neutral")
            if emotion not in emotion_segments:
                emotion_segments[emotion] = segment
                emotions.append(emotion)

        # Limit to grid size
        max_previews = grid_size[0] * grid_size[1]
        emotions = emotions[:max_previews]

        # Get video info
        probe = ffmpeg.probe(video_path)
        video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        width = int(video_stream['width'])
        height = int(video_stream['height'])

        # Calculate cell dimensions
        cell_width = width // grid_size[1]
        cell_height = height // grid_size[0]

        # Create temporary files for each emotion segment
        with tempfile.TemporaryDirectory() as temp_dir:
            segment_videos = []
            
            for idx, emotion in enumerate(emotions):
                segment = emotion_segments[emotion]
                
                # Create segment-specific caption data
                segment_caption_data = {
                    "segments": [segment]
                }
                
                # Generate video segment with this emotion's style
                segment_output = os.path.join(temp_dir, f"segment_{idx}.mp4")
                
                # Extract video segment
                start = segment.get("start", 0)
                duration = min(segment_duration, segment.get("end", start + segment_duration) - start)
                
                # Create trimmed video with subtitle
                self._create_emotion_preview_segment(
                    video_path,
                    segment_caption_data,
                    segment_output,
                    start,
                    duration,
                    (cell_width, cell_height),
                    emotion
                )
                
                segment_videos.append({
                    'path': segment_output,
                    'position': (idx % grid_size[1], idx // grid_size[1]),
                    'emotion': emotion
                })
            
            # Combine segments into grid using ffmpeg
            self._create_grid_from_segments(
                segment_videos,
                output_path,
                grid_size,
                (cell_width, cell_height),
                segment_duration
            )

        return output_path

    def _create_emotion_preview_segment(
        self,
        video_path: str,
        caption_data: Dict[str, Any],
        output_path: str,
        start_time: float,
        duration: float,
        size: Tuple[int, int],
        emotion: str
    ):
        """Create a single emotion preview segment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate ASS subtitle for this segment
            subtitle_path = os.path.join(temp_dir, f"{emotion}.ass")
            ass_generator = ASSGenerator(
                platform=self.platform,
                style_intensity=self.style_intensity,
                video_resolution=size,
                verbose=self.verbose
            )
            ass_generator.generate_ass_file(caption_data, subtitle_path)
            
            # Extract and process video segment with subtitle
            input_video = ffmpeg.input(video_path, ss=start_time, t=duration)
            
            # Scale to cell size
            stream = input_video.video.filter('scale', size[0], size[1])
            
            # Add subtitle
            stream = stream.filter('subtitles', subtitle_path)
            
            # Add emotion label overlay
            label_text = f"{emotion.upper()}"
            stream = stream.drawtext(
                text=label_text,
                x=10,
                y=10,
                fontsize=24,
                fontcolor='white',
                box=1,
                boxcolor='black@0.5'
            )
            
            # Output
            stream = ffmpeg.output(
                stream,
                input_video.audio,
                output_path,
                vcodec='libx264',
                acodec='aac',
                crf=23,
                preset='fast'
            )
            
            ffmpeg.run(stream, overwrite_output=True, quiet=not self.verbose)

    def _create_grid_from_segments(
        self,
        segments: List[Dict[str, Any]],
        output_path: str,
        grid_size: Tuple[int, int],
        cell_size: Tuple[int, int],
        duration: float
    ):
        """Create a grid video from individual segments using ffmpeg."""
        if not segments:
            return
        
        # Build complex filter for grid layout
        inputs = []
        filter_complex = []
        
        # Add all video inputs
        for seg in segments:
            inputs.extend(['-i', seg['path']])
        
        # Create filter complex for grid
        # First, ensure all videos have the same duration
        for i in range(len(segments)):
            filter_complex.append(f"[{i}:v]setpts=PTS-STARTPTS,scale={cell_size[0]}:{cell_size[1]},setsar=1[v{i}]")
        
        # Create the grid layout
        grid_filter = ""
        for idx, seg in enumerate(segments):
            x, y = seg['position']
            x_pos = x * cell_size[0]
            y_pos = y * cell_size[1]
            
            if idx == 0:
                grid_filter = f"[v0]pad={grid_size[1]*cell_size[0]}:{grid_size[0]*cell_size[1]}:0:0[base]"
                filter_complex.append(grid_filter)
                last_output = "base"
            else:
                overlay_filter = f"[{last_output}][v{idx}]overlay={x_pos}:{y_pos}"
                if idx < len(segments) - 1:
                    overlay_filter += f"[tmp{idx}]"
                    last_output = f"tmp{idx}"
                else:
                    overlay_filter += "[out]"
                filter_complex.append(overlay_filter)
        
        # Combine all filters
        filter_complex_str = ";".join(filter_complex)
        
        # Build ffmpeg command
        cmd = ['ffmpeg', '-y']
        cmd.extend(inputs)
        cmd.extend([
            '-filter_complex', filter_complex_str,
            '-map', '[out]',
            '-c:v', 'libx264',
            '-crf', '23',
            '-preset', 'fast',
            '-t', str(duration),
            output_path
        ])
        
        # Run ffmpeg
        try:
            subprocess.run(cmd, check=True, capture_output=not self.verbose)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to create grid: {e}")

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
