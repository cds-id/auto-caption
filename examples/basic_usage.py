#!/usr/bin/env python3
"""
Basic usage example for Auto-Caption library.

This script demonstrates how to generate captions from a video file
with optional emotion detection and styling.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path if running from examples folder
if __name__ == "__main__":
    examples_dir = Path(__file__).parent
    if examples_dir.name == 'examples':
        project_root = examples_dir.parent
        src_path = project_root / 'src'
        if src_path.exists() and str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

from auto_caption.caption_generator import CaptionGenerator
from auto_caption.models import WhisperModel, is_model_downloaded
from auto_caption.emotion_detector import EmotionDetector
from auto_caption.caption_styler import CaptionStyler, Platform, StyleIntensity


def generate_basic_captions(video_path: str, output_dir: str = "output"):
    """Generate basic captions without emotion detection."""

    print(f"Processing video: {video_path}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize caption generator
    caption_gen = CaptionGenerator(
        model_name="base",  # Options: tiny, base, small, medium, large
        language="en",      # Set to None for auto-detection
        verbose=True
    )

    # Generate captions
    print("\nGenerating captions...")
    result = caption_gen.generate(
        video_path=video_path,
        temperature=0.0,  # Lower temperature for more deterministic results
        word_timestamps=True  # Enable word-level timing
    )

    # Save in multiple formats
    formats = ["srt", "vtt", "txt", "json"]
    for fmt in formats:
        output_path = os.path.join(output_dir, f"captions.{fmt}")
        caption_gen.save_output(result, output_path, fmt)
        print(f"✅ Saved {fmt.upper()} format to: {output_path}")

    # Print summary
    print(f"\n📊 Summary:")
    print(f"  • Language: {result['language']}")
    print(f"  • Duration: {result.get('duration', 0):.1f} seconds")
    print(f"  • Segments: {len(result['segments'])}")
    print(f"  • Text preview: {result['text'][:100]}...")

    return result


def generate_emotion_aware_captions(video_path: str, output_dir: str = "output_emotion"):
    """Generate captions with emotion detection and styling."""

    print(f"Processing video with emotion detection: {video_path}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize components
    emotion_detector = EmotionDetector()
    caption_styler = CaptionStyler(
        default_platform=Platform.TIKTOK,
        default_intensity=StyleIntensity.MEDIUM
    )

    # Initialize caption generator with emotion support
    caption_gen = CaptionGenerator(
        model_name="base",
        language="en",
        enable_emotion_detection=True,
        emotion_detector=emotion_detector,
        caption_styler=caption_styler,
        verbose=True
    )

    # Generate captions with emotion detection
    print("\nGenerating emotion-aware captions...")
    result = caption_gen.generate(
        video_path=video_path,
        detect_emotions=True,
        style_captions=True,
        word_timestamps=True,
        enable_smart_positioning=True
    )

    # Save styled captions
    caption_gen.save_output(result, os.path.join(output_dir, "captions.srt"), "srt")
    caption_gen.save_output(result, os.path.join(output_dir, "captions.ass"), "ass")  # ASS supports styling

    # Display emotion results if available
    if 'emotion_results' in result:
        emotion_data = result['emotion_results']
        print(f"\n🎭 Emotion Analysis:")
        print(f"  • Dominant emotion: {emotion_data.dominant_emotion.value}")
        print(f"  • Confidence: {emotion_data.average_confidence:.2f}")

        # Show emotion distribution
        print("\n📈 Emotion Distribution:")
        for emotion, score in sorted(emotion_data.emotion_distribution.items(),
                                   key=lambda x: x[1], reverse=True)[:5]:
            print(f"  • {emotion}: {score:.1%}")

    return result


def batch_process_videos(video_folder: str, output_base: str = "output_batch"):
    """Process multiple videos in a folder."""

    # Find all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    video_files = []

    folder_path = Path(video_folder)
    for ext in video_extensions:
        video_files.extend(folder_path.glob(f'*{ext}'))

    if not video_files:
        print(f"No video files found in {video_folder}")
        return

    print(f"Found {len(video_files)} videos to process\n")

    # Initialize caption generator once for efficiency
    caption_gen = CaptionGenerator(model_name="base", verbose=False)

    # Process each video
    results = []
    for i, video_path in enumerate(video_files, 1):
        print(f"[{i}/{len(video_files)}] Processing: {video_path.name}")

        output_dir = os.path.join(output_base, video_path.stem)
        os.makedirs(output_dir, exist_ok=True)

        try:
            # Generate captions
            result = caption_gen.generate(
                video_path=str(video_path),
                temperature=0.0
            )

            # Save as SRT
            output_path = os.path.join(output_dir, "captions.srt")
            caption_gen.save_output(result, output_path, "srt")

            results.append({
                'video': video_path.name,
                'status': 'success',
                'language': result['language'],
                'segments': len(result['segments'])
            })
            print(f"   ✅ Success! Language: {result['language']}")

        except Exception as e:
            results.append({
                'video': video_path.name,
                'status': 'error',
                'error': str(e)
            })
            print(f"   ❌ Error: {str(e)}")

    # Summary
    print(f"\n🎉 Batch processing complete!")
    success_count = sum(1 for r in results if r['status'] == 'success')
    print(f"   • Successful: {success_count}/{len(video_files)}")
    print(f"   • Output directory: {output_base}")

    return results


def check_and_download_model(model_name: str = "base"):
    """Check if model is downloaded and download if needed."""

    if is_model_downloaded(model_name):
        print(f"✅ Whisper '{model_name}' model is ready")
    else:
        print(f"📥 Downloading Whisper '{model_name}' model...")
        model = WhisperModel(model_name)
        model.download()
        print(f"✅ Model downloaded successfully")


def main():
    """Main function demonstrating different usage patterns."""

    # Check if model is downloaded
    check_and_download_model("base")

    # Example video path (replace with your actual video)
    video_path = "path/to/your/video.mp4"

    # Check if video exists
    if not os.path.exists(video_path):
        print(f"\n❌ Error: Video file not found: {video_path}")
        print("Please update the video_path variable with a valid video file path.")
        return

    print("\n" + "="*60)
    print("AUTO-CAPTION EXAMPLES")
    print("="*60)

    # Example 1: Basic caption generation
    print("\n1. BASIC CAPTION GENERATION")
    print("-" * 30)
    basic_result = generate_basic_captions(video_path, "output/basic")

    # Example 2: Emotion-aware captions (requires emotion models)
    try:
        print("\n\n2. EMOTION-AWARE CAPTION GENERATION")
        print("-" * 30)
        emotion_result = generate_emotion_aware_captions(video_path, "output/emotion")
    except Exception as e:
        print(f"⚠️  Emotion detection skipped: {str(e)}")
        print("   (This requires additional emotion detection models)")
    
    # Example 3: Batch processing (if you have a folder of videos)
    # Uncomment and modify the path below to test batch processing
    """
    print("\n\n3. BATCH PROCESSING")
    print("-" * 30)
    video_folder = "path/to/video/folder"
    if os.path.exists(video_folder):
        batch_results = batch_process_videos(video_folder, "output/batch")
    else:
        print(f"⚠️  Batch processing skipped: Folder not found: {video_folder}")
    """
    
    print("\n\n✨ All examples completed! Check the output folders for results.")


if __name__ == "__main__":
    main()