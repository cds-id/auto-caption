#!/usr/bin/env python3
"""
Example script demonstrating word-by-word caption generation with Auto-Caption.

This example shows how to:
1. Generate word-by-word captions from a video
2. Apply different animation styles
3. Use emotion detection for dynamic styling
4. Save captions in multiple formats
5. Merge styled captions with video
"""

import os
import sys
from pathlib import Path

# Add parent directory to Python path if running from examples folder
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.auto_caption import (
    CaptionGenerator,
    WordTimingProcessor,
    WordAnimationStyle,
    EmotionDetector,
    CaptionStyler,
    StyleIntensity,
    Platform,
    VideoMerger
)


def example_basic_word_timing(video_path: str, output_dir: str):
    """
    Example 1: Basic word-by-word caption generation.
    """
    print("\n" + "="*60)
    print("Example 1: Basic Word-by-Word Captions")
    print("="*60)
    
    # Initialize caption generator with word timestamps enabled
    generator = CaptionGenerator(
        model_name="base",
        language="en",
        verbose=True
    )
    
    # Generate captions with word timestamps
    print("\nGenerating captions with word-level timestamps...")
    result = generator.generate(
        video_path,
        word_timestamps=True  # Enable word-level timestamps from Whisper
    )
    
    # Process into word-by-word timing
    word_processor = WordTimingProcessor(
        animation_style=WordAnimationStyle.TYPEWRITER,
        words_per_second=3.0,  # Average reading speed
        enable_word_timestamps=True
    )
    
    word_segments = word_processor.process_segments(result['segments'])
    
    # Convert to proper format for saving
    result['word_segments'] = []
    for ws in word_segments:
        word_list = []
        for w in ws.words:
            word_dict = {
                'word': w.word,
                'start_time': w.start_time,
                'end_time': w.end_time,
                'duration': w.duration,
                'segment_index': w.segment_index,
                'word_index': w.word_index,
                'emotion': w.emotion.value,
                'confidence': w.confidence,
                'is_emphasized': w.is_emphasized,
                'animation_delay': w.animation_delay,
                'custom_style': w.custom_style
            }
            word_list.append(word_dict)
        
        segment_dict = {
            'segment_index': ws.segment_index,
            'start_time': ws.start_time,
            'end_time': ws.end_time,
            'full_text': ws.full_text,
            'words': word_list,
            'emotion': ws.emotion.value,
            'confidence': ws.confidence
        }
        result['word_segments'].append(segment_dict)
    
    result['word_by_word'] = True
    result['word_animation'] = 'typewriter'
    result['metadata'] = {
        'word_by_word': True,
        'animation_style': 'typewriter'
    }
    
    # Save in different formats
    print("\nSaving word-by-word captions...")
    base_name = Path(video_path).stem
    
    # Save SRT (simple format, one word per subtitle)
    srt_path = os.path.join(output_dir, f"{base_name}_word_by_word.srt")
    generator.save_output(result, srt_path, "srt")
    print(f"✓ Saved SRT: {srt_path}")
    
    # Save ASS (advanced format with styling)
    ass_path = os.path.join(output_dir, f"{base_name}_word_by_word.ass")
    generator.save_output(result, ass_path, "ass")
    print(f"✓ Saved ASS: {ass_path}")
    
    # Save JSON (complete data)
    json_path = os.path.join(output_dir, f"{base_name}_word_by_word.json")
    generator.save_output(result, json_path, "json")
    print(f"✓ Saved JSON: {json_path}")
    
    return result


def example_with_emotions(video_path: str, output_dir: str):
    """
    Example 2: Word-by-word captions with emotion detection and styling.
    """
    print("\n" + "="*60)
    print("Example 2: Word-by-Word with Emotion Detection")
    print("="*60)
    
    # Initialize components
    generator = CaptionGenerator(
        model_name="base",
        language="en",
        verbose=True,
        enable_emotion_detection=True
    )
    
    # Generate captions with emotion detection
    print("\nGenerating captions with emotion detection...")
    result = generator.generate(
        video_path,
        word_timestamps=True,
        detect_emotions=True,
        style_captions=True,
        style_intensity=StyleIntensity.MEDIUM,
        platform=Platform.TIKTOK
    )
    
    # Process with emotion-aware word timing
    word_processor = WordTimingProcessor(
        animation_style=WordAnimationStyle.EMPHASIS,  # Emphasize emotional words
        words_per_second=2.8,
        emphasis_duration_multiplier=1.5  # Longer duration for emphasized words
    )
    
    word_segments = word_processor.process_segments(
        result['segments'],
        result.get('emotion_data')
    )
    
    # Convert and save
    result['word_segments'] = []
    for ws in word_segments:
        word_list = []
        for w in ws.words:
            word_dict = {
                'word': w.word,
                'start_time': w.start_time,
                'end_time': w.end_time,
                'duration': w.duration,
                'segment_index': w.segment_index,
                'word_index': w.word_index,
                'emotion': w.emotion.value,
                'confidence': w.confidence,
                'is_emphasized': w.is_emphasized,
                'animation_delay': w.animation_delay,
                'custom_style': w.custom_style
            }
            word_list.append(word_dict)
        
        segment_dict = {
            'segment_index': ws.segment_index,
            'start_time': ws.start_time,
            'end_time': ws.end_time,
            'full_text': ws.full_text,
            'words': word_list,
            'emotion': ws.emotion.value,
            'confidence': ws.confidence
        }
        result['word_segments'].append(segment_dict)
    
    result['word_by_word'] = True
    result['word_animation'] = 'emphasis'
    result['metadata'] = {
        'word_by_word': True,
        'animation_style': 'emphasis'
    }
    
    # Save emotion-styled word captions
    base_name = Path(video_path).stem
    ass_path = os.path.join(output_dir, f"{base_name}_emotion_words.ass")
    generator.save_output(result, ass_path, "ass")
    print(f"✓ Saved emotion-styled ASS: {ass_path}")
    
    # Show emotion distribution
    if result.get('emotion_data'):
        print("\nDetected emotions:")
        emotion_data = result['emotion_data']
        for score in emotion_data.get('emotion_scores', [])[:3]:
            print(f"  - {score['emotion']}: {score['confidence']:.2%}")
    
    return result


def example_animation_styles(video_path: str, output_dir: str):
    """
    Example 3: Different word animation styles.
    """
    print("\n" + "="*60)
    print("Example 3: Word Animation Styles")
    print("="*60)
    
    # Animation styles to demonstrate
    styles = {
        WordAnimationStyle.TYPEWRITER: "Classic typewriter effect",
        WordAnimationStyle.FADE_IN: "Words fade in smoothly",
        WordAnimationStyle.POP_IN: "Words pop/scale in",
        WordAnimationStyle.WAVE: "Wave pattern animation",
        WordAnimationStyle.KARAOKE: "Karaoke-style highlighting"
    }
    
    generator = CaptionGenerator(model_name="base", verbose=False)
    
    # Generate base captions once
    print("\nGenerating base captions...")
    result = generator.generate(video_path, word_timestamps=True)
    
    base_name = Path(video_path).stem
    
    for style, description in styles.items():
        print(f"\n{style.value}: {description}")
        
        # Process with specific animation style
        word_processor = WordTimingProcessor(
            animation_style=style,
            words_per_second=3.0
        )
        
        word_segments = word_processor.process_segments(result['segments'])
        
        # Prepare result
        styled_result = result.copy()
        styled_result['word_segments'] = []
        
        for ws in word_segments:
            word_list = []
            for w in ws.words:
                word_dict = {
                    'word': w.word,
                    'start_time': w.start_time,
                    'end_time': w.end_time,
                    'duration': w.duration,
                    'segment_index': w.segment_index,
                    'word_index': w.word_index,
                    'emotion': w.emotion.value,
                    'confidence': w.confidence,
                    'is_emphasized': w.is_emphasized,
                    'animation_delay': w.animation_delay,
                    'custom_style': w.custom_style
                }
                word_list.append(word_dict)
            
            segment_dict = {
                'segment_index': ws.segment_index,
                'start_time': ws.start_time,
                'end_time': ws.end_time,
                'full_text': ws.full_text,
                'words': word_list,
                'emotion': ws.emotion.value,
                'confidence': ws.confidence
            }
            styled_result['word_segments'].append(segment_dict)
        
        styled_result['word_by_word'] = True
        styled_result['word_animation'] = style.value
        styled_result['metadata'] = {
            'word_by_word': True,
            'animation_style': style.value
        }
        
        # Save ASS file for this style
        ass_path = os.path.join(output_dir, f"{base_name}_{style.value}.ass")
        generator.save_output(styled_result, ass_path, "ass")
        print(f"  ✓ Saved: {ass_path}")


def example_merge_with_video(video_path: str, caption_json: str, output_dir: str):
    """
    Example 4: Merge word-by-word captions with video.
    """
    print("\n" + "="*60)
    print("Example 4: Merge Word-by-Word Captions with Video")
    print("="*60)
    
    # Initialize video merger
    merger = VideoMerger(
        platform=Platform.TIKTOK,
        quality="high",
        style_intensity=StyleIntensity.MEDIUM,
        verbose=True
    )
    
    base_name = Path(video_path).stem
    output_path = os.path.join(output_dir, f"{base_name}_with_word_captions.mp4")
    
    print(f"\nMerging captions with video...")
    merged_video = merger.merge_with_video(
        video_path,
        caption_json,
        output_path,
        subtitle_format="ass"  # Use ASS for rich styling
    )
    
    print(f"✓ Created video with word-by-word captions: {merged_video}")
    
    return merged_video


def main():
    """
    Run all word-by-word caption examples.
    """
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python word_by_word_example.py <video_file> [output_dir]")
        print("\nThis example demonstrates word-by-word caption generation.")
        print("Provide a video file to process.")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./word_by_word_output"
    
    # Validate input
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Processing video: {video_path}")
    print(f"Output directory: {output_dir}")
    
    try:
        # Run examples
        # 1. Basic word-by-word
        result = example_basic_word_timing(video_path, output_dir)
        
        # 2. With emotion detection (optional - requires emotion models)
        try:
            emotion_result = example_with_emotions(video_path, output_dir)
        except Exception as e:
            print(f"\nSkipping emotion example: {e}")
        
        # 3. Different animation styles
        example_animation_styles(video_path, output_dir)
        
        # 4. Merge with video
        json_path = os.path.join(output_dir, f"{Path(video_path).stem}_word_by_word.json")
        if os.path.exists(json_path):
            example_merge_with_video(video_path, json_path, output_dir)
        
        print("\n" + "="*60)
        print("✓ All examples completed successfully!")
        print(f"✓ Check output in: {output_dir}")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()