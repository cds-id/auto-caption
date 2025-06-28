#!/usr/bin/env python3
"""
Test script for emotion-aware caption generation.

This script tests the emotion detection and caption styling features
of the Auto-Caption tool.
"""

import os
import sys
import json
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from auto_caption import (
    CaptionGenerator,
    EmotionDetector,
    CaptionStyler,
    EmotionCategory,
    StyleIntensity,
    Platform
)


def test_basic_caption_generation():
    """Test basic caption generation without emotion detection."""
    print("=" * 50)
    print("TEST 1: Basic Caption Generation")
    print("=" * 50)
    
    # Create a test video path (you'll need to provide an actual video)
    video_path = "test_video.mp4"
    
    if not os.path.exists(video_path):
        print(f"⚠️  No test video found at '{video_path}'")
        print("   Creating mock caption result for demonstration...")
        
        # Mock result for demonstration
        mock_result = {
            "text": "Hello, this is a test video. I'm feeling great today!",
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.5,
                    "text": "Hello, this is a test video."
                },
                {
                    "start": 2.5,
                    "end": 5.0,
                    "text": "I'm feeling great today!"
                }
            ],
            "language": "en",
            "duration": 5.0
        }
        
        print("\nMock Caption Result:")
        print(f"Language: {mock_result['language']}")
        print(f"Duration: {mock_result['duration']}s")
        print("\nSegments:")
        for i, seg in enumerate(mock_result['segments'], 1):
            print(f"{i}. [{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}")
        
        return mock_result
    
    try:
        generator = CaptionGenerator(model_name="base", verbose=True)
        result = generator.generate(video_path)
        
        print(f"\n✅ Captions generated successfully!")
        print(f"Language: {result.get('language', 'unknown')}")
        print(f"Duration: {result.get('duration', 0):.1f}s")
        print(f"Segments: {len(result.get('segments', []))}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def test_emotion_detection():
    """Test emotion detection on video."""
    print("\n" + "=" * 50)
    print("TEST 2: Emotion Detection")
    print("=" * 50)
    
    video_path = "test_video.mp4"
    
    if not os.path.exists(video_path):
        print(f"⚠️  No test video found at '{video_path}'")
        print("   Creating mock emotion result for demonstration...")
        
        # Mock emotion result
        from auto_caption import EmotionScore, EmotionDetectionResult
        
        mock_emotion_result = EmotionDetectionResult(
            dominant_emotion=EmotionCategory.HAPPY,
            emotion_scores=[
                EmotionScore(emotion=EmotionCategory.HAPPY, confidence=0.85, modality="combined"),
                EmotionScore(emotion=EmotionCategory.EXCITED, confidence=0.65, modality="combined"),
                EmotionScore(emotion=EmotionCategory.NEUTRAL, confidence=0.35, modality="combined")
            ],
            temporal_emotions=[
                {
                    "start": 0.0,
                    "end": 2.5,
                    "dominant_emotion": "neutral",
                    "confidence": 0.7
                },
                {
                    "start": 2.5,
                    "end": 5.0,
                    "dominant_emotion": "happy",
                    "confidence": 0.85
                }
            ],
            visual_emotions=[],
            audio_emotions=[],
            metadata={"duration": 5.0, "visual_samples": 5, "audio_samples": 2}
        )
        
        print("\nMock Emotion Detection Result:")
        print(f"Dominant Emotion: {mock_emotion_result.dominant_emotion.value}")
        print("\nTop 3 Emotions:")
        for i, score in enumerate(mock_emotion_result.emotion_scores[:3], 1):
            print(f"  {i}. {score.emotion.value}: {score.confidence:.1%}")
        
        return mock_emotion_result
    
    try:
        detector = EmotionDetector(verbose=True)
        result = detector.detect_emotions(video_path)
        
        print(f"\n✅ Emotions detected successfully!")
        print(f"Dominant emotion: {result.dominant_emotion.value}")
        print(f"\nTop 3 emotions:")
        for i, score in enumerate(result.emotion_scores[:3], 1):
            print(f"  {i}. {score.emotion.value}: {score.confidence:.1%}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def test_caption_styling():
    """Test emotion-aware caption styling."""
    print("\n" + "=" * 50)
    print("TEST 3: Emotion-Aware Caption Styling")
    print("=" * 50)
    
    # Test text
    test_texts = [
        "I'm fine",
        "This is amazing",
        "I can't believe this happened",
        "Whatever you say"
    ]
    
    # Test emotions
    test_emotions = [
        (EmotionCategory.HAPPY, 0.9),
        (EmotionCategory.SAD, 0.85),
        (EmotionCategory.ANGRY, 0.8),
        (EmotionCategory.SARCASTIC, 0.75)
    ]
    
    styler = CaptionStyler()
    
    print("\nTesting different emotions on the same text:")
    print("-" * 40)
    
    for text in test_texts[:2]:  # Test first two texts
        print(f"\nOriginal: '{text}'")
        print("Styled versions:")
        
        for emotion, confidence in test_emotions:
            styled = styler.style_caption(
                text,
                emotion,
                confidence,
                StyleIntensity.MEDIUM,
                Platform.TIKTOK
            )
            
            print(f"  {emotion.value:12} → {styled['styled_text']}")


def test_platform_variations():
    """Test platform-specific styling."""
    print("\n" + "=" * 50)
    print("TEST 4: Platform-Specific Styling")
    print("=" * 50)
    
    test_text = "Check out this incredible moment"
    emotion = EmotionCategory.EXCITED
    confidence = 0.85
    
    platforms = [Platform.TIKTOK, Platform.INSTAGRAM, Platform.YOUTUBE_SHORTS, Platform.GENERAL]
    
    print(f"\nOriginal text: '{test_text}'")
    print(f"Emotion: {emotion.value} (confidence: {confidence:.1%})")
    print("\nPlatform variations:")
    
    for platform in platforms:
        styler = CaptionStyler(default_platform=platform)
        styled = styler.style_caption(
            test_text,
            emotion,
            confidence,
            StyleIntensity.MEDIUM
        )
        
        print(f"\n{platform.value}:")
        print(f"  Text: {styled['styled_text']}")
        
        # Show visual suggestions
        if styled.get('visual_suggestions'):
            suggestions = styled['visual_suggestions']
            if suggestions.get('text_animation'):
                print(f"  Animation: {', '.join(suggestions['text_animation'][:2])}")
            if suggestions.get('color_scheme'):
                print(f"  Colors: {suggestions['color_scheme'].get('primary', 'N/A')}")


def test_intensity_variations():
    """Test different styling intensities."""
    print("\n" + "=" * 50)
    print("TEST 5: Styling Intensity Variations")
    print("=" * 50)
    
    test_text = "This is unbelievable"
    emotion = EmotionCategory.SURPRISED
    confidence = 0.9
    
    intensities = [StyleIntensity.SUBTLE, StyleIntensity.MEDIUM, StyleIntensity.INTENSE]
    
    print(f"\nOriginal text: '{test_text}'")
    print(f"Emotion: {emotion.value} (confidence: {confidence:.1%})")
    print("\nIntensity variations:")
    
    styler = CaptionStyler()
    
    for intensity in intensities:
        styled = styler.style_caption(
            test_text,
            emotion,
            confidence,
            intensity,
            Platform.GENERAL
        )
        
        print(f"\n{intensity.value}:")
        print(f"  {styled['styled_text']}")


def test_full_integration():
    """Test full integration with caption generation and emotion styling."""
    print("\n" + "=" * 50)
    print("TEST 6: Full Integration Test")
    print("=" * 50)
    
    # Simulate a complete workflow
    print("\nSimulating complete emotion-aware caption generation workflow...")
    
    # Mock caption segments
    segments = [
        {"start": 0.0, "end": 2.0, "text": "Welcome to our channel"},
        {"start": 2.0, "end": 4.0, "text": "Today we have something special"},
        {"start": 4.0, "end": 6.0, "text": "You won't believe what happens next"},
        {"start": 6.0, "end": 8.0, "text": "This is absolutely incredible"}
    ]
    
    # Mock temporal emotions
    temporal_emotions = [
        {"start": 0.0, "end": 2.0, "dominant_emotion": "neutral", "confidence": 0.7},
        {"start": 2.0, "end": 4.0, "dominant_emotion": "happy", "confidence": 0.8},
        {"start": 4.0, "end": 6.0, "dominant_emotion": "excited", "confidence": 0.85},
        {"start": 6.0, "end": 8.0, "dominant_emotion": "surprised", "confidence": 0.9}
    ]
    
    styler = CaptionStyler(
        default_intensity=StyleIntensity.MEDIUM,
        default_platform=Platform.TIKTOK
    )
    
    print("\nStyled captions with temporal emotions:")
    print("-" * 50)
    
    for segment in segments:
        # Find matching temporal emotion
        segment_emotion = None
        for temp_emotion in temporal_emotions:
            if temp_emotion["start"] <= segment["start"] < temp_emotion["end"]:
                segment_emotion = temp_emotion
                break
        
        if segment_emotion:
            emotion = EmotionCategory(segment_emotion["dominant_emotion"])
            confidence = segment_emotion["confidence"]
            
            styled = styler.style_caption(
                segment["text"],
                emotion,
                confidence
            )
            
            print(f"\n[{segment['start']:.1f}s - {segment['end']:.1f}s]")
            print(f"Original:  {segment['text']}")
            print(f"Emotion:   {emotion.value} ({confidence:.1%})")
            print(f"Styled:    {styled['styled_text']}")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("AUTO-CAPTION EMOTION-AWARE TESTING")
    print("=" * 60)
    print("\nThis script tests emotion detection and caption styling features.")
    print("For best results, place a video file named 'test_video.mp4' in this directory.")
    
    try:
        # Run tests
        test_basic_caption_generation()
        test_emotion_detection()
        test_caption_styling()
        test_platform_variations()
        test_intensity_variations()
        test_full_integration()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()