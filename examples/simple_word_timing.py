#!/usr/bin/env python3
"""
Simple example of generating word-by-word captions using Auto-Caption.

This script demonstrates:
1. Generating captions with word timestamps
2. Processing segments for word-by-word display
3. Creating different animation styles
4. Saving output in ASS format
"""

import os
import sys
from pathlib import Path

# Add src to path if running from examples directory
src_path = Path(__file__).parent.parent / 'src'
if src_path.exists():
    sys.path.insert(0, str(src_path))

from auto_caption.caption_generator import CaptionGenerator
from auto_caption.word_timing import WordTimingProcessor, WordAnimationStyle
from auto_caption.subtitle import ASSGenerator


def generate_word_by_word_captions(video_path, output_dir="output/word_timing"):
    """
    Generate word-by-word captions for a video.
    
    Args:
        video_path: Path to the input video file
        output_dir: Directory to save output files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("🎬 Auto-Caption Word-by-Word Example")
    print("=" * 50)
    
    # Step 1: Initialize caption generator
    print("\n1. Initializing caption generator...")
    caption_gen = CaptionGenerator(
        model_name="base",
        verbose=True
    )
    
    # Step 2: Generate captions with word timestamps
    print("\n2. Generating captions with word timestamps...")
    result = caption_gen.generate(
        video_path=video_path,
        word_timestamps=True,  # Enable word-level timestamps
        temperature=0.0  # More deterministic results
    )
    
    print(f"   ✅ Generated {len(result['segments'])} segments")
    print(f"   🔤 Language: {result['language']}")
    
    # Step 3: Process segments for word-by-word timing
    print("\n3. Processing segments for word-by-word display...")
    
    # Initialize word timing processor
    word_processor = WordTimingProcessor(
        animation_style=WordAnimationStyle.POP_IN,
        words_per_second=3.0,  # Average reading speed
        min_word_duration=0.15,
        max_word_duration=0.8
    )
    
    # Process segments
    word_segments = word_processor.process_segments(result['segments'])
    
    # Step 4: Create word-by-word caption data
    print("\n4. Creating word-by-word caption data...")
    
    # Copy original result and add word segment data
    word_result = result.copy()
    word_result['word_segments'] = []
    
    for segment in word_segments:
        segment_data = {
            'segment_index': segment.segment_index,
            'words': []
        }
        
        for word in segment.words:
            word_data = {
                'word': word.word,
                'start_time': word.start_time,
                'end_time': word.end_time,
                'duration': word.duration,
                'is_emphasized': word.is_emphasized
            }
            segment_data['words'].append(word_data)
        
        word_result['word_segments'].append(segment_data)
    
    # Mark as word-by-word captions
    word_result['word_by_word'] = True
    word_result['word_animation'] = WordAnimationStyle.POP_IN.value
    
    # Step 5: Save outputs
    print("\n5. Saving caption files...")
    
    # Save JSON for inspection
    json_path = os.path.join(output_dir, "captions_word.json")
    caption_gen.save_output(word_result, json_path, "json")
    print(f"   ✅ JSON: {json_path}")
    
    # Generate ASS subtitle file
    ass_gen = ASSGenerator()
    ass_path = os.path.join(output_dir, "captions_word.ass")
    ass_gen.generate_ass_file(word_result, ass_path, "Word-by-Word Captions")
    print(f"   ✅ ASS: {ass_path}")
    
    # Step 6: Generate different animation styles
    print("\n6. Generating different animation styles...")
    
    styles_to_demo = [
        WordAnimationStyle.TYPEWRITER,
        WordAnimationStyle.FADE_IN,
        WordAnimationStyle.KARAOKE
    ]
    
    for style in styles_to_demo:
        # Create processor with this style
        style_processor = WordTimingProcessor(
            animation_style=style,
            words_per_second=3.0
        )
        
        # Process segments
        styled_segments = style_processor.process_segments(result['segments'])
        
        # Create result for this style
        styled_result = result.copy()
        styled_result['word_segments'] = []
        
        for segment in styled_segments:
            segment_data = {
                'segment_index': segment.segment_index,
                'words': [
                    {
                        'word': w.word,
                        'start_time': w.start_time,
                        'end_time': w.end_time,
                        'duration': w.duration,
                        'is_emphasized': w.is_emphasized
                    }
                    for w in segment.words
                ]
            }
            styled_result['word_segments'].append(segment_data)
        
        styled_result['word_by_word'] = True
        styled_result['word_animation'] = style.value
        
        # Save ASS file
        style_path = os.path.join(output_dir, f"captions_{style.value}.ass")
        ass_gen.generate_ass_file(styled_result, style_path, f"{style.value.title()} Style")
        print(f"   ✅ {style.value}: {style_path}")
    
    # Display sample timing
    print("\n7. Sample word timing (first 10 words):")
    word_count = 0
    for segment in word_segments:
        for word in segment.words:
            if word_count >= 10:
                break
            print(f"   {word_count + 1:2d}. '{word.word}' "
                  f"[{word.start_time:.2f}s - {word.end_time:.2f}s] "
                  f"duration: {word.duration:.2f}s")
            word_count += 1
        if word_count >= 10:
            break
    
    print(f"\n✨ Complete! All files saved to: {output_dir}")
    print("\nTo apply captions to your video, use FFmpeg:")
    print(f"ffmpeg -i {video_path} -vf \"subtitles={ass_path}\" output_with_words.mp4")
    
    return word_result


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate word-by-word captions")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument(
        "--output-dir",
        default="output/word_timing",
        help="Output directory (default: output/word_timing)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
    
    # Generate word-by-word captions
    generate_word_by_word_captions(args.video, args.output_dir)