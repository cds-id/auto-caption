"""
Utility functions for the Auto-Caption tool.
"""

import os
import json
import re
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import timedelta

from moviepy.editor import VideoFileClip
import ffmpeg


# Supported video formats
VIDEO_EXTENSIONS = {
    '.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm',
    '.m4v', '.mpg', '.mpeg', '.3gp', '.ogg', '.ogv', '.vob',
    '.m2ts', '.mts', '.ts', '.qt', '.asf', '.rm', '.rmvb',
    '.m2v', '.f4v', '.f4p', '.f4a', '.f4b'
}


def validate_video_file(file_path: str) -> bool:
    """
    Validate if a file is a supported video file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if valid video file, False otherwise
    """
    path = Path(file_path)
    
    # Check if file exists
    if not path.exists() or not path.is_file():
        return False
    
    # Check extension
    return path.suffix.lower() in VIDEO_EXTENSIONS


def get_output_filename(input_path: str, format: str) -> str:
    """
    Generate output filename based on input video path and format.
    
    Args:
        input_path: Path to input video file
        format: Output format (srt, vtt, txt, json)
        
    Returns:
        Output file path
    """
    path = Path(input_path)
    return str(path.with_suffix(f'.{format}'))


def seconds_to_time(seconds: float) -> timedelta:
    """
    Convert seconds to timedelta object.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        timedelta object
    """
    return timedelta(seconds=seconds)


def format_timestamp(seconds: float) -> str:
    """
    Format seconds as timestamp string (HH:MM:SS,mmm).
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted timestamp string for SRT
    """
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((seconds % 1) * 1000)
    
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{milliseconds:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """
    Format seconds as WebVTT timestamp (HH:MM:SS.mmm).
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted timestamp string for VTT
    """
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((seconds % 1) * 1000)
    
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.{milliseconds:03d}"


def parse_time(time_str: str) -> float:
    """
    Parse timestamp string to seconds.
    
    Args:
        time_str: Timestamp string (various formats supported)
        
    Returns:
        Time in seconds
    """
    # Try different timestamp formats
    patterns = [
        # HH:MM:SS,mmm or HH:MM:SS.mmm
        re.compile(r'(\d+):(\d+):(\d+)[,.](\d+)'),
        # HH:MM:SS
        re.compile(r'(\d+):(\d+):(\d+)'),
        # MM:SS
        re.compile(r'(\d+):(\d+)'),
        # Seconds only
        re.compile(r'(\d+(?:\.\d+)?)')
    ]
    
    for i, pattern in enumerate(patterns):
        match = pattern.match(time_str.strip())
        if match:
            groups = match.groups()
            if i == 0:  # HH:MM:SS,mmm
                h, m, s, ms = groups
                return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
            elif i == 1:  # HH:MM:SS
                h, m, s = groups
                return int(h) * 3600 + int(m) * 60 + int(s)
            elif i == 2:  # MM:SS
                m, s = groups
                return int(m) * 60 + int(s)
            elif i == 3:  # Seconds
                return float(groups[0])
    
    raise ValueError(f"Invalid time format: {time_str}")


def extract_audio_from_video(video_path: str, audio_path: str, 
                           sample_rate: int = 16000) -> float:
    """
    Extract audio from video file and save as WAV.
    
    Args:
        video_path: Path to video file
        audio_path: Path to save audio file
        sample_rate: Audio sample rate (default: 16000 Hz for Whisper)
        
    Returns:
        Duration of the video in seconds
        
    Raises:
        RuntimeError: If audio extraction fails
    """
    try:
        # Get video duration first
        probe = ffmpeg.probe(video_path)
        duration = float(probe['streams'][0]['duration'])
        
        # Extract audio using ffmpeg
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.output(stream, audio_path,
                             acodec='pcm_s16le',
                             ac=1,  # Mono
                             ar=sample_rate,
                             loglevel='error')
        ffmpeg.run(stream, overwrite_output=True)
        
        return duration
        
    except ffmpeg.Error as e:
        error_msg = e.stderr.decode() if e.stderr else str(e)
        raise RuntimeError(f"Failed to extract audio: {error_msg}")
    except Exception as e:
        raise RuntimeError(f"Failed to extract audio: {str(e)}")


def clean_text(text: str) -> str:
    """
    Clean and normalize transcribed text.
    
    Args:
        text: Raw transcribed text
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common transcription artifacts
    text = text.strip()
    
    # Remove leading/trailing punctuation artifacts
    text = re.sub(r'^[.,;:!?]+|[.,;:!?]+$', '', text)
    
    # Ensure proper spacing after punctuation
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
    
    # Remove repeated punctuation
    text = re.sub(r'([.!?,])\1+', r'\1', text)
    
    return text.strip()


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load config: {str(e)}")


def save_config(config_path: str, config: Dict[str, Any]):
    """
    Save configuration to JSON file.
    
    Args:
        config_path: Path to save configuration
        config: Configuration dictionary
    """
    try:
        # Create directory if needed
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        raise ValueError(f"Failed to save config: {str(e)}")


def get_video_info(video_path: str) -> Dict[str, Any]:
    """
    Get detailed information about a video file.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video information
    """
    try:
        probe = ffmpeg.probe(video_path)
        
        # Extract video stream info
        video_stream = next((stream for stream in probe['streams'] 
                           if stream['codec_type'] == 'video'), None)
        
        # Extract audio stream info
        audio_stream = next((stream for stream in probe['streams'] 
                           if stream['codec_type'] == 'audio'), None)
        
        info = {
            'duration': float(probe['format'].get('duration', 0)),
            'size': int(probe['format'].get('size', 0)),
            'bit_rate': int(probe['format'].get('bit_rate', 0)),
            'format_name': probe['format'].get('format_name', ''),
        }
        
        if video_stream:
            info['video'] = {
                'codec': video_stream.get('codec_name', ''),
                'width': int(video_stream.get('width', 0)),
                'height': int(video_stream.get('height', 0)),
                'fps': eval(video_stream.get('r_frame_rate', '0/1')),
                'bit_rate': int(video_stream.get('bit_rate', 0))
            }
        
        if audio_stream:
            info['audio'] = {
                'codec': audio_stream.get('codec_name', ''),
                'sample_rate': int(audio_stream.get('sample_rate', 0)),
                'channels': int(audio_stream.get('channels', 0)),
                'bit_rate': int(audio_stream.get('bit_rate', 0))
            }
        
        return info
        
    except Exception as e:
        raise RuntimeError(f"Failed to get video info: {str(e)}")


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string (e.g., "1h 23m 45s")
    """
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 or not parts:
        parts.append(f"{seconds}s")
    
    return " ".join(parts)


def format_size(bytes: int) -> str:
    """
    Format file size in bytes to human-readable string.
    
    Args:
        bytes: Size in bytes
        
    Returns:
        Formatted size string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024.0:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024.0
    
    return f"{bytes:.1f} PB"


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove control characters
    filename = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', filename)
    
    # Limit length
    name, ext = os.path.splitext(filename)
    if len(name) > 200:
        name = name[:200]
    
    return name + ext


def merge_subtitles(subtitle_files: list, output_file: str, format: str = 'srt'):
    """
    Merge multiple subtitle files into one.
    
    Args:
        subtitle_files: List of subtitle file paths
        output_file: Output file path
        format: Output format (srt or vtt)
    """
    import srt
    
    all_subtitles = []
    
    for file_path in subtitle_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Parse based on format
            if file_path.endswith('.srt'):
                subtitles = list(srt.parse(content))
            elif file_path.endswith('.vtt'):
                # Simple VTT parser (you might want to use a proper library)
                # Skip WEBVTT header
                content = re.sub(r'^WEBVTT.*?\n\n', '', content, flags=re.DOTALL)
                # Convert VTT timestamps to SRT format
                content = content.replace('.', ',')
                subtitles = list(srt.parse(content))
            else:
                continue
            
            all_subtitles.extend(subtitles)
    
    # Sort by start time
    all_subtitles.sort(key=lambda x: x.start)
    
    # Re-index
    for i, subtitle in enumerate(all_subtitles, 1):
        subtitle.index = i
    
    # Save merged subtitles
    if format == 'srt':
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(srt.compose(all_subtitles))
    elif format == 'vtt':
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")
            for sub in all_subtitles:
                start = format_timestamp_vtt(sub.start.total_seconds())
                end = format_timestamp_vtt(sub.end.total_seconds())
                f.write(f"{start} --> {end}\n{sub.content}\n\n")