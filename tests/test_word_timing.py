#!/usr/bin/env python3
"""
Test script for word-by-word caption timing functionality.
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.auto_caption.caption_generator import CaptionGenerator
from src.auto_caption.word_timing import WordTimingProcessor, WordAnimationStyle
from src.auto_caption.emotion_detector import EmotionCategory
from src.auto_caption.caption_styler import StyleIntensity, Platform


def test_word_timing_basic():
    """Test basic word timing functionality."""
    print("Testing basic word timing...")
    
    # Create test segments
    test_segments = [
        {
            "start": 0.0,
            "end": 3.0,
            "text": "Hello world, this is a test.",
            "emotion_metadata": {
                "emotion": "happy",
                "confidence": 0.9
            }
        },
        {
            "start": 3.0,
            "end": 6.0,
            "text": "Word timing should work properly!",
            "emotion_metadata": {
                "emotion": "excited",
                "confidence": 0.85
            }
        }
    ]
    
    # Initialize processor
    processor = WordTimingProcessor(
        animation_style=WordAnimationStyle.TYPEWRITER,
        words_per_second=3.0
    )
    
    # Process segments
    word_segments = processor.process_segments(test_segments)
    
    print(f"✓ Generated {len(word_segments)} word segments")
    
    # Verify results
    for ws in word_segments:
        print(f"\nSegment {ws.segment_index}:")
        print(f"  Full text: '{ws.full_text}'")
        print(f"  Words: {len(ws.words)}")
        print(f"  Emotion: {ws.emotion.value}")
        
        for word in ws.words[:3]:  # Show first 3 words
            print(f"    - '{word.word}' @ {word.start_time:.2f}s - {word.end_time:.2f}s")
    
    return word_segments


def test_save_formats(word_segments):
    """Test saving word-by-word captions in different formats."""
    print("\n\nTesting save formats...")
    
    # Create test result with word segments
    test_result = {
        "text": "Hello world, this is a test. Word timing should work properly!",
        "segments": [
            {
                "start": 0.0,
                "end": 3.0,
                "text": "Hello world, this is a test.",
                "emotion_metadata": {
                    "emotion": "happy",
                    "confidence": 0.9
                }
            },
            {
                "start": 3.0,
                "end": 6.0,
                "text": "Word timing should work properly!",
                "emotion_metadata": {
                    "emotion": "excited",
                    "confidence": 0.85
                }
            }
        ],
        "word_by_word": True,
        "word_animation": "typewriter",
        "word_segments": [],
        "duration": 6.0,
        "language": "en",
        "model": "base",
        "task": "transcribe",
        "metadata": {
            "word_by_word": True,
            "animation_style": "typewriter"
        }
    }
    
    # Convert word segments to dict format
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
        test_result['word_segments'].append(segment_dict)
    
    # Initialize generator
    generator = CaptionGenerator(model_name="base", verbose=True)
    
    # Test each format
    with tempfile.TemporaryDirectory() as tmpdir:
        formats = ['srt', 'vtt', 'txt', 'json', 'ass']
        
        for fmt in formats:
            output_path = os.path.join(tmpdir, f"test_word_timing.{fmt}")
            try:
                generator.save_output(test_result, output_path, fmt)
                
                # Check if file exists and has content
                if os.path.exists(output_path):
                    size = os.path.getsize(output_path)
                    print(f"✓ {fmt.upper()}: Generated ({size} bytes)")
                    
                    # Show sample of content
                    with open(output_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        lines = content.split('\n')
                        print(f"  First few lines:")
                        for line in lines[:5]:
                            if line.strip():
                                print(f"    {line[:80]}...")
                else:
                    print(f"✗ {fmt.upper()}: File not created")
                    
            except Exception as e:
                print(f"✗ {fmt.upper()}: Error - {str(e)}")


def test_animation_styles():
    """Test different word animation styles."""
    print("\n\nTesting animation styles...")
    
    test_segment = {
        "start": 0.0,
        "end": 3.0,
        "text": "Testing different animation styles!",
        "emotion_metadata": {
            "emotion": "excited",
            "confidence": 0.9
        }
    }
    
    styles = [
        WordAnimationStyle.TYPEWRITER,
        WordAnimationStyle.FADE_IN,
        WordAnimationStyle.POP_IN,
        WordAnimationStyle.WAVE,
        WordAnimationStyle.EMPHASIS
    ]
    
    for style in styles:
        processor = WordTimingProcessor(
            animation_style=style,
            words_per_second=3.0
        )
        
        word_segments = processor.process_segments([test_segment])
        
        print(f"\n{style.value}:")
        if word_segments:
            for word in word_segments[0].words[:3]:
                print(f"  '{word.word}' - delay: {word.animation_delay:.3f}s, "
                      f"emphasized: {word.is_emphasized}")


def test_word_emphasis():
    """Test word emphasis detection."""
    print("\n\nTesting word emphasis...")
    
    test_segments = [
        {
            "start": 0.0,
            "end": 3.0,
            "text": "This is AMAZING and wonderful!",
            "emotion_metadata": {
                "emotion": "happy",
                "confidence": 0.9
            }
        },
        {
            "start": 3.0,
            "end": 6.0,
            "text": "I really totally love this.",
            "emotion_metadata": {
                "emotion": "sarcastic",
                "confidence": 0.8
            }
        }
    ]
    
    processor = WordTimingProcessor(
        animation_style=WordAnimationStyle.EMPHASIS
    )
    
    word_segments = processor.process_segments(test_segments)
    
    for ws in word_segments:
        print(f"\nSegment (emotion: {ws.emotion.value}):")
        for word in ws.words:
            if word.is_emphasized:
                print(f"  ★ '{word.word}' - EMPHASIZED")
            else:
                print(f"    '{word.word}'")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Word-by-Word Caption Timing Test Suite")
    print("=" * 60)
    
    # Run tests
    word_segments = test_word_timing_basic()
    test_save_formats(word_segments)
    test_animation_styles()
    test_word_emphasis()
    
    print("\n" + "=" * 60)
    print("Tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()