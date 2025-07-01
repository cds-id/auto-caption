"""
Caption generator module for processing video files and generating captions.
"""

import os
import tempfile
import json
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path

import whisper
import numpy as np
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import srt
from tqdm import tqdm

from .utils import (
    format_timestamp,
    format_timestamp_vtt,
    seconds_to_time,
    extract_audio_from_video,
    clean_text
)
from .emotion_detector import EmotionDetector, EmotionDetectionResult
from .caption_styler import CaptionStyler, StyleIntensity, Platform


class CaptionGenerator:
    """
    Main class for generating captions from video files using Whisper.
    """
    
    def __init__(
        self,
        model_name: str = "base",
        language: Optional[str] = None,
        task: str = "transcribe",
        verbose: bool = False,
        threads: int = 4,
        device: Optional[str] = None,
        enable_emotion_detection: bool = False,
        emotion_detector: Optional[EmotionDetector] = None,
        caption_styler: Optional[CaptionStyler] = None
    ):
        """
        Initialize the caption generator.
        
        Args:
            model_name: Whisper model to use (tiny, base, small, medium, large)
            language: Language code (e.g., 'en', 'es') or None for auto-detection
            task: Task to perform ('transcribe' or 'translate')
            verbose: Enable verbose output
            threads: Number of threads to use
            device: Device to use ('cuda' or 'cpu', None for auto)
        """
        self.model_name = model_name
        self.language = language
        self.task = task
        self.verbose = verbose
        self.threads = threads
        self.device = device
        self.enable_emotion_detection = enable_emotion_detection
        self.emotion_detector = emotion_detector
        self.caption_styler = caption_styler
        
        # Load Whisper model
        self._load_model()
    
    def _load_model(self):
        """Load the Whisper model."""
        if self.verbose:
            print(f"Loading Whisper model: {self.model_name}")
        
        # Set number of threads
        os.environ["OMP_NUM_THREADS"] = str(self.threads)
        
        # Load model
        self.model = whisper.load_model(
            self.model_name,
            device=self.device
        )
        
        if self.verbose:
            print(f"Model loaded successfully on device: {self.model.device}")
    
    def generate(
        self,
        video_path: str,
        temperature: float = 0.0,
        progress_callback: Optional[Callable[[float], None]] = None,
        detect_emotions: bool = False,
        style_captions: bool = False,
        style_intensity: Optional[StyleIntensity] = None,
        platform: Optional[Platform] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate captions for a video file.
        
        Args:
            video_path: Path to the video file
            temperature: Temperature for sampling
            progress_callback: Callback function for progress updates
            detect_emotions: Whether to detect emotions in the video
            style_captions: Whether to apply emotion-aware styling
            style_intensity: Styling intensity (if style_captions is True)
            platform: Target platform for styling
            **kwargs: Additional arguments for Whisper
            
        Returns:
            Dictionary containing transcription results and optional emotion data
        """
        # Update progress
        if progress_callback:
            progress_callback(0)
        
        # Extract audio from video
        if self.verbose:
            print(f"Extracting audio from: {video_path}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = os.path.join(temp_dir, "audio.wav")
            duration = extract_audio_from_video(video_path, audio_path)
            
            if progress_callback:
                progress_callback(20)
            
            # Transcribe audio
            if self.verbose:
                print("Transcribing audio...")
            
            # Prepare options
            options = {
                "language": self.language,
                "task": self.task,
                "temperature": temperature,
                "verbose": self.verbose,
                **kwargs
            }
            
            # Remove None values
            options = {k: v for k, v in options.items() if v is not None}
            
            # Transcribe with progress tracking
            if progress_callback:
                # Create a custom progress hook
                def progress_hook(progress):
                    # Map Whisper progress (0-100) to our range (20-90)
                    mapped_progress = 20 + (progress * 0.7)
                    progress_callback(mapped_progress)
                
                # Note: Whisper doesn't have built-in progress callbacks,
                # so we'll simulate progress based on audio duration
                result = self.model.transcribe(audio_path, **options)
                progress_callback(90)
            else:
                result = self.model.transcribe(audio_path, **options)
            
            # Post-process results
            result = self._post_process_result(result, duration)
            
            # Detect emotions if requested
            if detect_emotions or (self.enable_emotion_detection and style_captions):
                if progress_callback:
                    progress_callback(92)
                
                if self.emotion_detector is None:
                    self.emotion_detector = EmotionDetector(verbose=self.verbose)
                
                emotion_result = self.emotion_detector.detect_emotions(video_path)
                result["emotion_data"] = emotion_result.to_dict()
                
                # Apply emotion-aware styling if requested
                if style_captions:
                    if self.caption_styler is None:
                        self.caption_styler = CaptionStyler(
                            default_intensity=style_intensity or StyleIntensity.MEDIUM,
                            default_platform=platform or Platform.GENERAL
                        )
                    
                    # Style each segment
                    for segment in result["segments"]:
                        # Find emotion at this timestamp
                        timestamp = segment.get("start", 0)
                        segment_emotion = self._find_emotion_at_timestamp(
                            timestamp,
                            emotion_result.temporal_emotions
                        )
                        
                        # Apply formatting
                        formatted = self.caption_styler.style_caption(
                            segment["text"],
                            segment_emotion["emotion"],
                            segment_emotion["confidence"],
                            style_intensity,
                            platform
                        )
                        
                        # Update segment
                        segment["original_text"] = segment["text"]
                        segment["text"] = formatted["formatted_text"]
                        segment["emotion_metadata"] = {
                            "emotion": formatted["emotion"],
                            "confidence": formatted["confidence"],
                            "formatting": formatted.get("formatting_metadata", {})
                        }
            
            if progress_callback:
                progress_callback(100)
            
            return result
    
    def _post_process_result(self, result: Dict[str, Any], duration: float) -> Dict[str, Any]:
        """
        Post-process Whisper results.
        
        Args:
            result: Raw Whisper transcription result
            duration: Video duration in seconds
            
        Returns:
            Processed result dictionary
        """
        # Clean text in segments
        for segment in result.get("segments", []):
            segment["text"] = clean_text(segment["text"])
        
        # Add duration and other metadata
        result["duration"] = duration
        result["model"] = self.model_name
        result["task"] = self.task
        
        # If language was auto-detected, ensure it's set
        if not self.language and "language" in result:
            result["detected_language"] = result["language"]
        
        return result
    
    def _find_emotion_at_timestamp(
        self,
        timestamp: float,
        temporal_emotions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Find the emotion at a specific timestamp."""
        from .emotion_detector import EmotionCategory
        
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
    
    def save_output(self, result: Dict[str, Any], output_path: str, format: str):
        """
        Save transcription results to file.
        
        Args:
            result: Transcription results
            output_path: Path to save the output
            format: Output format (srt, vtt, txt, json)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "srt":
            self._save_srt(result, output_path)
        elif format == "vtt":
            self._save_vtt(result, output_path)
        elif format == "txt":
            self._save_txt(result, output_path)
        elif format == "json":
            self._save_json(result, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _save_srt(self, result: Dict[str, Any], output_path: Path):
        """Save as SRT format."""
        subtitles = []
        
        for i, segment in enumerate(result["segments"], 1):
            subtitle = srt.Subtitle(
                index=i,
                start=seconds_to_time(segment["start"]),
                end=seconds_to_time(segment["end"]),
                content=segment["text"].strip()
            )
            subtitles.append(subtitle)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(srt.compose(subtitles))
    
    def _save_vtt(self, result: Dict[str, Any], output_path: Path):
        """Save as WebVTT format."""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            
            for segment in result["segments"]:
                start = format_timestamp_vtt(segment["start"])
                end = format_timestamp_vtt(segment["end"])
                text = segment["text"].strip()
                
                f.write(f"{start} --> {end}\n")
                f.write(f"{text}\n\n")
    
    def _save_txt(self, result: Dict[str, Any], output_path: Path):
        """Save as plain text with timestamps."""
        with open(output_path, "w", encoding="utf-8") as f:
            for segment in result["segments"]:
                start = format_timestamp(segment["start"])
                end = format_timestamp(segment["end"])
                text = segment["text"].strip()
                
                f.write(f"[{start} --> {end}] {text}\n")
    
    def _save_json(self, result: Dict[str, Any], output_path: Path):
        """Save as JSON format."""
        # Create a clean copy for JSON output
        output = {
            "text": result.get("text", ""),
            "segments": [],
            "language": result.get("language", ""),
            "duration": result.get("duration", 0),
            "model": result.get("model", self.model_name),
            "task": result.get("task", self.task)
        }
        
        # Include emotion data if available
        if "emotion_data" in result:
            output["emotion_data"] = result["emotion_data"]
        
        # Include only essential segment information
        for segment in result.get("segments", []):
            output["segments"].append({
                "id": segment.get("id", 0),
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip(),
                "original_text": segment.get("original_text", segment["text"]).strip(),
                "emotion_metadata": segment.get("emotion_metadata", None)
            })
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
    
    def batch_generate(
        self,
        video_paths: List[str],
        output_dir: Optional[str] = None,
        formats: List[str] = ["srt"],
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate captions for multiple video files.
        
        Args:
            video_paths: List of video file paths
            output_dir: Directory to save outputs (None for same as video)
            formats: List of output formats
            progress_callback: Callback for batch progress (current, total, filename)
            **kwargs: Additional arguments for generate()
            
        Returns:
            List of results for each video
        """
        results = []
        
        for i, video_path in enumerate(video_paths):
            if progress_callback:
                progress_callback(i, len(video_paths), os.path.basename(video_path))
            
            try:
                # Generate captions
                result = self.generate(video_path, **kwargs)
                
                # Save outputs
                for format in formats:
                    if output_dir:
                        base_name = Path(video_path).stem
                        output_path = Path(output_dir) / f"{base_name}.{format}"
                    else:
                        output_path = Path(video_path).with_suffix(f".{format}")
                    
                    self.save_output(result, str(output_path), format)
                
                results.append({
                    "video": video_path,
                    "success": True,
                    "result": result
                })
                
            except Exception as e:
                results.append({
                    "video": video_path,
                    "success": False,
                    "error": str(e)
                })
        
        return results