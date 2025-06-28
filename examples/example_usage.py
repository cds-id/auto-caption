#!/usr/bin/env python3
"""
Example usage of the Auto-Caption library.

This script demonstrates various ways to use the auto-caption tool
both programmatically and through the command line.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from auto_caption import (
    CaptionGenerator, WhisperModel, get_available_models,
    EmotionDetector, EmotionCategory,
    CaptionStyler, StyleIntensity, Platform
)
from auto_caption.utils import get_video_info, format_duration, format_size


def example_basic_usage():
    """Basic example: Generate captions for a single video."""
    print("=== Basic Usage Example ===\n")
    
    # Initialize the caption generator with default settings
    generator = CaptionGenerator(model_name="base")
    
    # Example video path (replace with your actual video)
    video_path = "sample_video.mp4"
    
    if not os.path.exists(video_path):
        print(f"Video file '{video_path}' not found. Please provide a valid video file.")
        return
    
    try:
        # Generate captions
        print(f"Generating captions for: {video_path}")
        result = generator.generate(video_path)
        
        # Save in SRT format
        output_path = "output_captions.srt"
        generator.save_output(result, output_path, "srt")
        
        print(f"✓ Captions saved to: {output_path}")
        print(f"  Language detected: {result.get('language', 'unknown')}")
        print(f"  Duration: {format_duration(result['duration'])}")
        print(f"  Number of segments: {len(result['segments'])}")
        
    except Exception as e:
        print(f"Error: {e}")


def example_multiple_formats():
    """Generate captions in multiple formats."""
    print("\n=== Multiple Formats Example ===\n")
    
    generator = CaptionGenerator(model_name="base")
    video_path = "sample_video.mp4"
    
    if not os.path.exists(video_path):
        print(f"Video file '{video_path}' not found.")
        return
    
    try:
        # Generate captions once
        result = generator.generate(video_path)
        
        # Save in multiple formats
        formats = ["srt", "vtt", "txt", "json"]
        for fmt in formats:
            output_path = f"captions.{fmt}"
            generator.save_output(result, output_path, fmt)
            print(f"✓ Saved {fmt.upper()} format: {output_path}")
            
    except Exception as e:
        print(f"Error: {e}")


def example_different_models():
    """Compare results from different Whisper models."""
    print("\n=== Different Models Example ===\n")
    
    video_path = "sample_video.mp4"
    if not os.path.exists(video_path):
        print(f"Video file '{video_path}' not found.")
        return
    
    # Test with different models
    models = ["tiny", "base", "small"]
    
    for model_name in models:
        print(f"\nTesting with model: {model_name}")
        
        try:
            generator = CaptionGenerator(model_name=model_name)
            
            # Get model info
            model_info = WhisperModel(model_name).model_info
            print(f"  Parameters: {model_info['parameters']}")
            print(f"  Speed: {model_info['speed']}")
            
            # Generate captions
            result = generator.generate(video_path)
            
            # Save with model name in filename
            output_path = f"captions_{model_name}.srt"
            generator.save_output(result, output_path, "srt")
            
            print(f"  ✓ Saved to: {output_path}")
            
        except Exception as e:
            print(f"  Error with {model_name}: {e}")


def example_language_specific():
    """Generate captions with specific language settings."""
    print("\n=== Language-Specific Example ===\n")
    
    video_path = "sample_video.mp4"
    if not os.path.exists(video_path):
        print(f"Video file '{video_path}' not found.")
        return
    
    # Example: Force Spanish transcription
    generator = CaptionGenerator(
        model_name="small",
        language="es",  # Spanish
        verbose=True
    )
    
    try:
        result = generator.generate(video_path)
        generator.save_output(result, "captions_spanish.srt", "srt")
        print("✓ Spanish captions generated")
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Example: Translate to English
    generator_translate = CaptionGenerator(
        model_name="small",
        task="translate",  # Translate to English
        verbose=True
    )
    
    try:
        result = generator_translate.generate(video_path)
        generator_translate.save_output(result, "captions_translated.srt", "srt")
        print("✓ Translated captions generated")
        
    except Exception as e:
        print(f"Error: {e}")


def example_batch_processing():
    """Process multiple videos in a directory."""
    print("\n=== Batch Processing Example ===\n")
    
    # Create a test directory with videos
    video_dir = "test_videos"
    if not os.path.exists(video_dir):
        print(f"Directory '{video_dir}' not found. Creating example structure...")
        os.makedirs(video_dir, exist_ok=True)
        print(f"Please add video files to '{video_dir}' directory")
        return
    
    # Find all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(Path(video_dir).glob(f"*{ext}"))
    
    if not video_files:
        print(f"No video files found in '{video_dir}'")
        return
    
    print(f"Found {len(video_files)} video files")
    
    # Initialize generator once for all files
    generator = CaptionGenerator(model_name="base")
    
    # Process each video
    for video_path in video_files:
        print(f"\nProcessing: {video_path.name}")
        
        try:
            result = generator.generate(str(video_path))
            
            # Save captions next to video file
            output_path = video_path.with_suffix('.srt')
            generator.save_output(result, str(output_path), "srt")
            
            print(f"  ✓ Saved to: {output_path}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")


def example_video_info():
    """Get detailed information about a video file."""
    print("\n=== Video Information Example ===\n")
    
    video_path = "sample_video.mp4"
    if not os.path.exists(video_path):
        print(f"Video file '{video_path}' not found.")
        return
    
    try:
        info = get_video_info(video_path)
        
        print(f"Video: {video_path}")
        print(f"  Duration: {format_duration(info['duration'])}")
        print(f"  Size: {format_size(info['size'])}")
        print(f"  Format: {info['format_name']}")
        
        if 'video' in info:
            v = info['video']
            print(f"  Video codec: {v['codec']}")
            print(f"  Resolution: {v['width']}x{v['height']}")
            print(f"  FPS: {v['fps']:.2f}")
        
        if 'audio' in info:
            a = info['audio']
            print(f"  Audio codec: {a['codec']}")
            print(f"  Sample rate: {a['sample_rate']} Hz")
            print(f"  Channels: {a['channels']}")
            
    except Exception as e:
        print(f"Error getting video info: {e}")


def example_custom_settings():
    """Generate captions with custom settings."""
    print("\n=== Custom Settings Example ===\n")
    
    video_path = "sample_video.mp4"
    if not os.path.exists(video_path):
        print(f"Video file '{video_path}' not found.")
        return
    
    # Custom settings
    generator = CaptionGenerator(
        model_name="small",
        language=None,  # Auto-detect
        verbose=True,
        threads=8  # Use more threads
    )
    
    try:
        # Generate with custom parameters
        result = generator.generate(
            video_path,
            temperature=0.2  # Slightly more creative
        )
        
        # Save with custom processing
        output_path = "captions_custom.srt"
        generator.save_output(result, output_path, "srt")
        
        print(f"✓ Custom captions saved to: {output_path}")
        
        # Print first few segments
        print("\nFirst 3 segments:")
        for i, segment in enumerate(result['segments'][:3], 1):
            print(f"{i}. [{segment['start']:.1f}s - {segment['end']:.1f}s]: {segment['text']}")
            
    except Exception as e:
        print(f"Error: {e}")


def example_emotion_detection():
    """Detect emotions in a video."""
    print("\n=== Emotion Detection Example ===\n")
    
    video_path = "sample_video.mp4"
    if not os.path.exists(video_path):
        print(f"Video file '{video_path}' not found.")
        return
    
    try:
        # Initialize emotion detector
        detector = EmotionDetector(verbose=True)
        
        print("Analyzing emotions in video...")
        result = detector.detect_emotions(video_path)
        
        print(f"\nDominant emotion: {result.dominant_emotion.value}")
        print(f"Confidence: {result.emotion_scores[0].confidence:.2%}")
        
        print("\nTop 3 detected emotions:")
        for i, score in enumerate(result.emotion_scores[:3], 1):
            print(f"  {i}. {score.emotion.value}: {score.confidence:.2%}")
        
        # Show temporal emotions
        print("\nEmotion timeline:")
        for segment in result.temporal_emotions[:5]:
            print(f"  {segment['start']:.1f}s - {segment['end']:.1f}s: {segment['dominant_emotion']}")
            
    except Exception as e:
        print(f"Error: {e}")


def example_emotion_aware_captions():
    """Generate emotion-aware styled captions."""
    print("\n=== Emotion-Aware Caption Generation ===\n")
    
    video_path = "sample_video.mp4"
    if not os.path.exists(video_path):
        print(f"Video file '{video_path}' not found.")
        return
    
    try:
        # First, detect emotions
        detector = EmotionDetector()
        emotion_result = detector.detect_emotions(video_path)
        
        # Generate base captions
        generator = CaptionGenerator(model_name="base")
        caption_result = generator.generate(video_path)
        
        # Apply emotion-aware styling
        styler = CaptionStyler(
            default_intensity=StyleIntensity.MEDIUM,
            default_platform=Platform.TIKTOK
        )
        
        print(f"Detected emotion: {emotion_result.dominant_emotion.value}")
        print("\nOriginal vs Styled Captions:")
        
        for segment in caption_result['segments'][:3]:
            original_text = segment['text']
            
            # Style the caption based on emotion
            styled = styler.style_caption(
                original_text,
                emotion_result.dominant_emotion,
                emotion_result.emotion_scores[0].confidence
            )
            
            print(f"\nOriginal: {original_text}")
            print(f"Styled:   {styled['styled_text']}")
            
            # Show visual suggestions
            if 'visual_suggestions' in styled:
                suggestions = styled['visual_suggestions']
                if suggestions.get('text_animation'):
                    print(f"  Suggested animation: {suggestions['text_animation'][0]}")
                if suggestions.get('color_scheme'):
                    print(f"  Primary color: {suggestions['color_scheme'].get('primary')}")
        
        # Save styled result
        output_path = "captions_emotion_styled.srt"
        generator.save_output(caption_result, output_path, "srt")
        print(f"\n✓ Styled captions saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")


def example_platform_specific():
    """Generate platform-specific emotion-aware captions."""
    print("\n=== Platform-Specific Caption Styling ===\n")
    
    video_path = "sample_video.mp4"
    if not os.path.exists(video_path):
        print(f"Video file '{video_path}' not found.")
        return
    
    try:
        # Detect emotions
        detector = EmotionDetector()
        emotion_result = detector.detect_emotions(video_path)
        
        # Generate base captions
        generator = CaptionGenerator(model_name="base")
        caption_result = generator.generate(video_path)
        
        # Test different platforms
        platforms = [Platform.TIKTOK, Platform.INSTAGRAM, Platform.YOUTUBE_SHORTS]
        sample_text = caption_result['segments'][0]['text'] if caption_result['segments'] else "Sample caption text"
        
        for platform in platforms:
            styler = CaptionStyler(default_platform=platform)
            styled = styler.style_caption(
                sample_text,
                emotion_result.dominant_emotion,
                emotion_result.emotion_scores[0].confidence
            )
            
            print(f"\n{platform.value.upper()}:")
            print(f"  {styled['styled_text']}")
            
    except Exception as e:
        print(f"Error: {e}")


def example_cli_commands():
    """Show example CLI commands."""
    print("\n=== CLI Command Examples ===\n")
    
    commands = [
        ("Basic usage", "auto-caption generate video.mp4"),
        ("Emotion-aware generation", "auto-caption generate video.mp4 --emotion-mode auto"),
        ("Manual emotion override", "auto-caption generate video.mp4 --emotion-mode manual --emotion happy"),
        ("Platform-specific", "auto-caption generate video.mp4 --platform tiktok --emotion-mode auto"),
        ("Analyze emotions only", "auto-caption analyze-emotion video.mp4 --output emotions.json"),
        ("Batch with emotions", "auto-caption batch /path/to/videos --emotion-mode auto --platform instagram"),
        ("Download emotion models", "auto-caption download-models --type emotion"),
        ("Specify output format", "auto-caption generate video.mp4 --format srt --format vtt"),
        ("Use specific model", "auto-caption generate video.mp4 --model medium"),
        ("Specify language", "auto-caption generate video.mp4 --language en"),
        ("List available models", "auto-caption list-models"),
        ("Show version", "auto-caption version"),
        ("Configure defaults", "auto-caption config --set default_model small"),
    ]
    
    for description, command in commands:
        print(f"{description}:")
        print(f"  $ {command}\n")


def main():
    """Run all examples."""
    print("Auto-Caption Library Examples")
    print("=" * 50)
    
    # Check if sample video exists
    sample_video = "sample_video.mp4"
    if not os.path.exists(sample_video):
        print(f"\nNote: Please provide a sample video file named '{sample_video}'")
        print("      to run the examples with actual video processing.\n")
    
    # Run examples
    examples = [
        example_basic_usage,
        example_multiple_formats,
        example_different_models,
        example_language_specific,
        example_emotion_detection,
        example_emotion_aware_captions,
        example_platform_specific,
        example_batch_processing,
        example_video_info,
        example_custom_settings,
        example_cli_commands
    ]
    
    for example_func in examples:
        try:
            example_func()
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError in {example_func.__name__}: {e}")
            continue
        
        # Pause between examples
        input("\nPress Enter to continue to next example...")
    
    print("\n" + "=" * 50)
    print("Examples completed!")


if __name__ == "__main__":
    main()