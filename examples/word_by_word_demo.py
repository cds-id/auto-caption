"""
Word-by-Word Caption Demo for Auto-Caption

This script demonstrates how to use word-by-word caption generation
with different animation styles and emotion-based formatting.
"""

import os
import json
import tempfile
from pathlib import Path
from datetime import timedelta

# Import Auto-Caption modules
from auto_caption.caption_generator import CaptionGenerator
from auto_caption.emotion_detector import EmotionCategory
from auto_caption.caption_styler import StyleIntensity, Platform
from auto_caption.word_timing import WordTimingProcessor, WordAnimationStyle
from auto_caption.subtitle import ASSGenerator
from auto_caption.video_merger import VideoMerger


def create_sample_segments():
    """Create sample segments for demonstration."""
    segments = [
        {
            "start": 0.0,
            "end": 4.0,
            "text": "Welcome to this amazing demonstration of word-by-word captions!",
            "emotion_metadata": {
                "emotion": "excited",
                "confidence": 0.95
            }
        },
        {
            "start": 4.5,
            "end": 8.0,
            "text": "Each word appears individually with perfect timing and emotion.",
            "emotion_metadata": {
                "emotion": "happy",
                "confidence": 0.90
            }
        },
        {
            "start": 8.5,
            "end": 12.0,
            "text": "This creates a more dynamic and engaging viewing experience.",
            "emotion_metadata": {
                "emotion": "neutral",
                "confidence": 0.85
            }
        },
        {
            "start": 12.5,
            "end": 16.0,
            "text": "Are you ready to see different animation styles in action?",
            "emotion_metadata": {
                "emotion": "excited",
                "confidence": 0.92
            }
        },
        {
            "start": 16.5,
            "end": 20.0,
            "text": "Let's explore typewriter, fade, pop, and wave effects!",
            "emotion_metadata": {
                "emotion": "happy",
                "confidence": 0.88
            }
        }
    ]
    
    return segments


def demonstrate_animation_styles():
    """Demonstrate different word animation styles."""
    
    print("=== Word-by-Word Animation Styles ===\n")
    
    # Define animation styles with descriptions
    animation_styles = {
        WordAnimationStyle.TYPEWRITER: {
            "description": "Words appear one by one at their natural timing",
            "best_for": "Natural reading flow, documentaries, tutorials"
        },
        WordAnimationStyle.FADE_IN: {
            "description": "Words fade in with opacity transition",
            "best_for": "Gentle, contemplative content"
        },
        WordAnimationStyle.POP_IN: {
            "description": "Words scale from 0 to full size",
            "best_for": "Energetic, exciting content"
        },
        WordAnimationStyle.SLIDE_IN: {
            "description": "Words slide in from the side",
            "best_for": "Dynamic presentations, action content"
        },
        WordAnimationStyle.BOUNCE_IN: {
            "description": "Words bounce in with elastic effect",
            "best_for": "Fun, playful content"
        },
        WordAnimationStyle.WAVE: {
            "description": "Words appear in a wave pattern",
            "best_for": "Musical content, rhythmic speech"
        },
        WordAnimationStyle.KARAOKE: {
            "description": "Highlight words as they're spoken",
            "best_for": "Sing-along content, emphasis on timing"
        },
        WordAnimationStyle.EMPHASIS: {
            "description": "Key words appear with special emphasis",
            "best_for": "Educational content, important messages"
        }
    }
    
    for style, info in animation_styles.items():
        print(f"{style.value.upper()}:")
        print(f"  Description: {info['description']}")
        print(f"  Best for: {info['best_for']}")
        print()


def process_word_by_word(segments, animation_style=WordAnimationStyle.TYPEWRITER):
    """Process segments into word-by-word timing."""
    
    # Create word timing processor
    processor = WordTimingProcessor(
        animation_style=animation_style,
        words_per_second=3.0,  # Average reading speed
        min_word_duration=0.15,
        max_word_duration=0.8,
        emphasis_duration_multiplier=1.3
    )
    
    # Process segments
    word_segments = processor.process_segments(segments)
    
    # Display statistics
    total_words = sum(len(ws.words) for ws in word_segments)
    print(f"\nProcessed {len(segments)} segments into {total_words} individual words")
    
    # Show timing breakdown for first segment
    if word_segments:
        first_segment = word_segments[0]
        print(f"\nFirst segment breakdown ({first_segment.emotion.value} emotion):")
        print("-" * 60)
        
        for word in first_segment.words[:5]:  # Show first 5 words
            print(f"  '{word.word}' - Start: {word.start_time:.2f}s, "
                  f"Duration: {word.duration:.2f}s"
                  f"{' [EMPHASIZED]' if word.is_emphasized else ''}")
        
        if len(first_segment.words) > 5:
            print(f"  ... and {len(first_segment.words) - 5} more words")
    
    return word_segments


def generate_word_by_word_subtitles(word_segments, output_dir="word_demos"):
    """Generate subtitle files for each animation style."""
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # Test different animation styles
    styles_to_test = [
        WordAnimationStyle.TYPEWRITER,
        WordAnimationStyle.FADE_IN,
        WordAnimationStyle.POP_IN,
        WordAnimationStyle.WAVE,
        WordAnimationStyle.KARAOKE
    ]
    
    for style in styles_to_test:
        print(f"\nGenerating {style.value} style subtitles...")
        
        # Create processor with specific style
        processor = WordTimingProcessor(animation_style=style)
        
        # Convert to subtitle format
        segments = []
        for ws in word_segments:
            for word in ws.words:
                segment = {
                    "start": word.start_time,
                    "end": word.end_time,
                    "text": word.word,
                    "emotion_metadata": {
                        "emotion": word.emotion.value,
                        "confidence": word.confidence,
                        "is_emphasized": word.is_emphasized,
                        "word_index": word.word_index,
                        "animation_style": style.value
                    }
                }
                segments.append(segment)
        
        # Create caption data
        caption_data = {
            "segments": segments,
            "metadata": {
                "word_by_word": True,
                "animation_style": style.value
            }
        }
        
        # Generate ASS file
        ass_gen = ASSGenerator(
            platform=Platform.GENERAL,
            style_intensity=StyleIntensity.MEDIUM,
            video_resolution=(1920, 1080)
        )
        
        output_file = os.path.join(output_dir, f"word_by_word_{style.value}.ass")
        ass_gen.generate_ass_file(
            caption_data,
            output_file,
            title=f"Word-by-Word Demo - {style.value}"
        )
        
        print(f"  ✓ Created: {output_file}")


def create_comparison_script(output_dir="word_demos"):
    """Create a comparison script showing regular vs word-by-word."""
    
    segments = create_sample_segments()
    
    # Regular captions
    regular_data = {
        "segments": segments,
        "metadata": {"word_by_word": False}
    }
    
    # Word-by-word captions
    processor = WordTimingProcessor(animation_style=WordAnimationStyle.TYPEWRITER)
    word_segments = processor.process_segments(segments)
    
    word_data_segments = []
    for ws in word_segments:
        for word in ws.words:
            word_data_segments.append({
                "start": word.start_time,
                "end": word.end_time,
                "text": word.word,
                "emotion_metadata": {
                    "emotion": word.emotion.value,
                    "confidence": word.confidence
                }
            })
    
    word_data = {
        "segments": word_data_segments,
        "metadata": {"word_by_word": True}
    }
    
    # Generate both versions
    ass_gen = ASSGenerator()
    
    # Regular version
    regular_file = os.path.join(output_dir, "captions_regular.ass")
    ass_gen.generate_ass_file(regular_data, regular_file, "Regular Captions")
    
    # Word-by-word version
    word_file = os.path.join(output_dir, "captions_word_by_word.ass")
    ass_gen.generate_ass_file(word_data, word_file, "Word-by-Word Captions")
    
    print(f"\nComparison files created:")
    print(f"  Regular: {regular_file}")
    print(f"  Word-by-word: {word_file}")
    
    # Create README
    readme_content = """
Word-by-Word Caption Comparison
===============================

This directory contains examples of regular vs word-by-word captions.

Files:
- captions_regular.ass: Traditional full-line captions
- captions_word_by_word.ass: Word-by-word display
- word_by_word_*.ass: Different animation styles

To test with a video:
ffmpeg -i your_video.mp4 -vf "subtitles=captions_word_by_word.ass" output.mp4

Benefits of Word-by-Word:
1. Better reading flow synchronization
2. Improved engagement and attention
3. Enhanced emotion emphasis on key words
4. More dynamic visual presentation
5. Better for social media platforms

Animation Styles:
- TYPEWRITER: Natural sequential appearance
- FADE_IN: Gentle opacity transition
- POP_IN: Energetic scaling effect
- WAVE: Rhythmic wave pattern
- KARAOKE: Highlight-style emphasis
    """
    
    with open(os.path.join(output_dir, "README.txt"), "w") as f:
        f.write(readme_content)


def demonstrate_emotion_word_emphasis():
    """Show how emotions affect word emphasis."""
    
    print("\n=== Emotion-Based Word Emphasis ===\n")
    
    test_sentences = {
        EmotionCategory.HAPPY: "This is absolutely amazing and wonderful!",
        EmotionCategory.SAD: "I feel so lost and alone without you.",
        EmotionCategory.ANGRY: "This is completely ridiculous and unacceptable!",
        EmotionCategory.EXCITED: "OMG this is incredible and unbelievable!",
        EmotionCategory.SARCASTIC: "Oh sure, that's totally believable, right?",
    }
    
    processor = WordTimingProcessor()
    
    for emotion, text in test_sentences.items():
        print(f"{emotion.value.upper()}:")
        print(f"  Text: {text}")
        
        # Create segment
        segment = [{
            "start": 0.0,
            "end": 3.0,
            "text": text,
            "emotion_metadata": {
                "emotion": emotion.value,
                "confidence": 0.95
            }
        }]
        
        # Process words
        word_segments = processor.process_segments(segment)
        
        if word_segments:
            emphasized_words = [
                w.word for w in word_segments[0].words 
                if w.is_emphasized
            ]
            print(f"  Emphasized: {', '.join(emphasized_words) if emphasized_words else 'None'}")
        
        print()


def main():
    """Run the word-by-word caption demo."""
    
    print("\n" + "="*70)
    print("AUTO-CAPTION WORD-BY-WORD DEMO")
    print("="*70 + "\n")
    
    # Show animation styles
    demonstrate_animation_styles()
    
    # Create sample segments
    segments = create_sample_segments()
    
    # Process word-by-word
    print("\n" + "="*70)
    print("PROCESSING WORD TIMING")
    print("="*70)
    word_segments = process_word_by_word(segments)
    
    # Generate subtitle files
    print("\n" + "="*70)
    print("GENERATING SUBTITLE FILES")
    print("="*70)
    generate_word_by_word_subtitles(word_segments)
    
    # Create comparison
    create_comparison_script()
    
    # Demonstrate emotion emphasis
    print("\n" + "="*70)
    print("EMOTION-BASED EMPHASIS")
    print("="*70)
    demonstrate_emotion_word_emphasis()
    
    print("\n" + "="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    print("\nWord-by-word captions offer:")
    print("- Better synchronization with speech")
    print("- More engaging visual presentation")
    print("- Enhanced emotion expression")
    print("- Improved retention and comprehension")
    print("- Perfect for social media platforms")
    print("\nCheck the 'word_demos' directory for generated examples!")


if __name__ == "__main__":
    main()