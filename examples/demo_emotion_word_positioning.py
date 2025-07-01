#!/usr/bin/env python3
"""
Demo script for emotion-aware word positioning in Auto-Caption.

This script demonstrates how word-by-word captions dynamically change
position and size based on detected emotions in the video.
"""

import os
import sys
import json
from pathlib import Path

# Add the parent directory to the path to import auto_caption
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from auto_caption import (
    CaptionGenerator,
    EmotionDetector,
    EmotionCategory,
    CaptionStyler,
    StyleIntensity,
    Platform,
    WordTimingProcessor,
    WordAnimationStyle,
    VideoMerger,
    ASSGenerator
)


def demonstrate_emotion_word_positioning(video_path: str, output_dir: str):
    """
    Demonstrate emotion-aware word positioning with different emotions.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save output files
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("🎬 Auto-Caption Emotion-Aware Word Positioning Demo")
    print("=" * 50)
    
    # Step 1: Initialize components
    print("\n1. Initializing components...")
    
    # Caption generator with word timestamps enabled
    generator = CaptionGenerator(
        model_name="base",
        language="en",
        verbose=True,
        enable_emotion_detection=True
    )
    
    # Emotion detector
    emotion_detector = EmotionDetector(verbose=True)
    
    # Caption styler
    caption_styler = CaptionStyler(
        default_intensity=StyleIntensity.INTENSE,  # Use intense for more dramatic effects
        default_platform=Platform.GENERAL
    )
    
    # Step 2: Generate captions with emotion detection
    print("\n2. Generating captions with emotion detection...")
    
    result = generator.generate(
        video_path,
        detect_emotions=True,
        style_captions=True,
        style_intensity=StyleIntensity.INTENSE,
        platform=Platform.GENERAL,
        word_timestamps=True  # Enable word-level timestamps
    )
    
    # Step 3: Process word timings for different animation styles
    print("\n3. Processing word timings with emotion-based positioning...")
    
    animation_styles = [
        WordAnimationStyle.TYPEWRITER,
        WordAnimationStyle.BOUNCE_IN,
        WordAnimationStyle.WAVE,
        WordAnimationStyle.POP_IN,
        WordAnimationStyle.EMPHASIS
    ]
    
    for style in animation_styles:
        print(f"\n   Processing {style.value} animation...")
        
        # Create word timing processor
        word_processor = WordTimingProcessor(
            animation_style=style,
            words_per_second=3.0,
            enable_word_timestamps=True
        )
        
        # Process segments into word timings
        word_segments = word_processor.process_segments(
            result['segments'],
            result.get('emotion_data')
        )
        
        # Convert to caption data format
        word_caption_data = {
            "segments": [],
            "metadata": {
                "word_by_word": True,
                "animation_style": style.value
            }
        }
        
        # Convert word segments to caption segments
        for ws in word_segments:
            for w in ws.words:
                segment = {
                    "start": w.start_time,
                    "end": w.end_time,
                    "text": w.word,
                    "emotion_metadata": {
                        "emotion": w.emotion.value,
                        "confidence": w.confidence,
                        "is_emphasized": w.is_emphasized,
                        "word_index": w.word_index,
                        "segment_index": w.segment_index,
                        "animation_style": style.value,
                        "position_offset": w.position_offset,
                        "size_multiplier": w.size_multiplier,
                        "rotation_angle": w.rotation_angle
                    }
                }
                word_caption_data["segments"].append(segment)
        
        # Save caption data
        caption_file = Path(output_dir) / f"captions_{style.value}.json"
        with open(caption_file, 'w', encoding='utf-8') as f:
            json.dump(word_caption_data, f, indent=2)
        
        # Generate ASS subtitle file
        ass_file = Path(output_dir) / f"subtitles_{style.value}.ass"
        ass_generator = ASSGenerator(
            platform=Platform.GENERAL,
            style_intensity=StyleIntensity.INTENSE,
            video_resolution=(1920, 1080),  # Adjust based on your video
            verbose=True
        )
        
        ass_generator.generate_ass_file(
            word_caption_data,
            str(ass_file),
            title=f"Emotion Word Positioning - {style.value}"
        )
        
        # Create video with captions
        print(f"   Creating video with {style.value} animation...")
        
        merger = VideoMerger(
            platform=Platform.GENERAL,
            quality="high",
            style_intensity=StyleIntensity.INTENSE,
            verbose=True
        )
        
        output_video = Path(output_dir) / f"output_{style.value}.mp4"
        merger.merge_with_video(
            video_path,
            word_caption_data,
            str(output_video),
            subtitle_format="ass"
        )
        
        print(f"   ✓ Created: {output_video}")
    
    # Step 4: Create comparison grid
    print("\n4. Creating emotion comparison grid...")
    
    # Create a special caption file with different emotions for demonstration
    demo_segments = create_emotion_demo_segments()
    
    demo_caption_file = Path(output_dir) / "demo_emotions.json"
    with open(demo_caption_file, 'w', encoding='utf-8') as f:
        json.dump(demo_segments, f, indent=2)
    
    # Create comparison grid
    grid_output = Path(output_dir) / "emotion_comparison_grid.mp4"
    merger = VideoMerger(verbose=True)
    
    merger.create_preview_grid(
        video_path,
        demo_segments,
        str(grid_output),
        grid_size=(3, 3),
        segment_duration=3.0
    )
    
    print(f"\n✓ Emotion comparison grid created: {grid_output}")
    
    # Step 5: Generate summary report
    print("\n5. Generating summary report...")
    
    summary = {
        "video": os.path.basename(video_path),
        "detected_emotions": [],
        "word_positioning_examples": {},
        "files_created": []
    }
    
    # Extract emotion distribution
    if 'emotion_data' in result:
        emotion_counts = {}
        for temporal in result['emotion_data'].get('temporal_emotions', []):
            emotion = temporal['dominant_emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        total = sum(emotion_counts.values())
        for emotion, count in emotion_counts.items():
            summary["detected_emotions"].append({
                "emotion": emotion,
                "percentage": round((count / total) * 100, 1)
            })
    
    # Add positioning examples
    for emotion in EmotionCategory:
        pos_style = WordTimingProcessor.EMOTION_POSITION_STYLES.get(
            emotion, 
            WordTimingProcessor.EMOTION_POSITION_STYLES[EmotionCategory.NEUTRAL]
        )
        summary["word_positioning_examples"][emotion.value] = {
            "vertical_offset": pos_style["y_offset_range"],
            "horizontal_spread": pos_style["x_spread"],
            "size_multiplier": pos_style["size_base"],
            "rotation_range": pos_style["rotation_range"]
        }
    
    # List created files
    for file in Path(output_dir).glob("*"):
        summary["files_created"].append(file.name)
    
    # Save summary
    summary_file = Path(output_dir) / "summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 50)
    print("✅ Demo completed successfully!")
    print(f"\nFiles created in: {output_dir}")
    print("\nEmotion-based positioning effects:")
    print("  • HAPPY: Words appear higher, larger, with wider spacing")
    print("  • SAD: Words appear lower, smaller, with tight spacing")
    print("  • ANGRY: Words are large, aggressive positioning with chaotic rotation")
    print("  • EXCITED: Very high positioning with dynamic movement")
    print("  • FEARFUL: Lower positioning, small size, clustered together")
    print("  • ANXIOUS: Scattered positioning with slight trembling")
    print("  • SARCASTIC: Slight tilt with moderate spacing")
    print("\nView the output videos to see the effects in action!")


def create_emotion_demo_segments():
    """Create demo segments showcasing different emotions."""
    emotions = [
        (EmotionCategory.HAPPY, "This is absolutely amazing and wonderful!", 0.9),
        (EmotionCategory.SAD, "I feel so lost and alone...", 0.85),
        (EmotionCategory.ANGRY, "This is completely UNACCEPTABLE!", 0.95),
        (EmotionCategory.EXCITED, "OMG this is incredible! I can't believe it!", 0.9),
        (EmotionCategory.FEARFUL, "I'm scared... what if something goes wrong?", 0.8),
        (EmotionCategory.ANXIOUS, "I don't know... maybe... possibly?", 0.75),
        (EmotionCategory.SARCASTIC, "Oh sure, that's totally going to work...", 0.85),
        (EmotionCategory.NEUTRAL, "Here is some information for you.", 0.9),
        (EmotionCategory.CONTEMPLATIVE, "Perhaps we should think about this more deeply.", 0.8)
    ]
    
    segments = []
    current_time = 0.0
    segment_duration = 3.0
    
    for emotion, text, confidence in emotions:
        # Create word-by-word segments
        words = text.split()
        word_duration = segment_duration / len(words)
        
        for i, word in enumerate(words):
            segment = {
                "start": current_time + (i * word_duration),
                "end": current_time + ((i + 1) * word_duration),
                "text": word,
                "emotion_metadata": {
                    "emotion": emotion.value,
                    "confidence": confidence,
                    "is_emphasized": len(word) > 4 and word.isupper(),
                    "word_index": i,
                    "segment_index": emotions.index((emotion, text, confidence)),
                    "animation_style": "bounce_in"
                }
            }
            segments.append(segment)
        
        current_time += segment_duration + 0.5  # Add gap between segments
    
    return {
        "segments": segments,
        "metadata": {
            "word_by_word": True,
            "animation_style": "bounce_in",
            "demo": True
        }
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Demonstrate emotion-aware word positioning in Auto-Caption"
    )
    parser.add_argument(
        "video",
        help="Path to input video file"
    )
    parser.add_argument(
        "--output-dir",
        default="./emotion_positioning_demo",
        help="Directory to save output files (default: ./emotion_positioning_demo)"
    )
    
    args = parser.parse_args()
    
    # Validate video file
    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
    
    # Run demonstration
    try:
        demonstrate_emotion_word_positioning(args.video, args.output_dir)
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)