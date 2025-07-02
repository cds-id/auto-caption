#!/usr/bin/env python3
"""Test script to verify word timing imports and basic functionality."""

import sys
from pathlib import Path

# Add src to path if needed
src_path = Path(__file__).parent.parent / 'src'
if src_path.exists():
    sys.path.insert(0, str(src_path))

print("Testing imports...")

try:
    from auto_caption.caption_generator import CaptionGenerator
    print("✓ CaptionGenerator imported successfully")
except ImportError as e:
    print(f"✗ Failed to import CaptionGenerator: {e}")

try:
    from auto_caption.word_timing import WordTimingProcessor, WordAnimationStyle
    print("✓ WordTimingProcessor and WordAnimationStyle imported successfully")
except ImportError as e:
    print(f"✗ Failed to import word timing modules: {e}")

try:
    from auto_caption.subtitle import ASSGenerator
    print("✓ ASSGenerator imported successfully")
except ImportError as e:
    print(f"✗ Failed to import ASSGenerator: {e}")

try:
    from auto_caption.emotion_detector import EmotionDetector
    print("✓ EmotionDetector imported successfully")
except ImportError as e:
    print(f"✗ Failed to import EmotionDetector: {e}")

try:
    from auto_caption.caption_styler import CaptionStyler, Platform, StyleIntensity
    print("✓ CaptionStyler, Platform, and StyleIntensity imported successfully")
except ImportError as e:
    print(f"✗ Failed to import caption styling modules: {e}")

print("\nTesting basic instantiation...")

try:
    # Test CaptionGenerator
    gen = CaptionGenerator(model_name="base", verbose=False)
    print("✓ CaptionGenerator instantiated successfully")
except Exception as e:
    print(f"✗ Failed to instantiate CaptionGenerator: {e}")

try:
    # Test WordTimingProcessor
    processor = WordTimingProcessor(
        animation_style=WordAnimationStyle.POP_IN,
        words_per_second=3.0
    )
    print("✓ WordTimingProcessor instantiated successfully")
    
    # Test available animation styles
    print("\nAvailable animation styles:")
    for style in WordAnimationStyle:
        print(f"  - {style.value}")
        
except Exception as e:
    print(f"✗ Failed to instantiate WordTimingProcessor: {e}")

try:
    # Test ASSGenerator
    ass_gen = ASSGenerator()
    print("\n✓ ASSGenerator instantiated successfully")
except Exception as e:
    print(f"✗ Failed to instantiate ASSGenerator: {e}")

print("\nAll imports and basic instantiation tests completed!")