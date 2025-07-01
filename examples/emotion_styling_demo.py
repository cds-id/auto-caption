"""
Emotion Styling Demo for Auto-Caption

This script demonstrates how different emotions affect:
- Font selection (MADE AVENUE, CINEMATOGRAFICA, ALMOST TEXTUAL)
- Text positioning on screen
- Text size and scale
- Colors and visual effects

Run this demo to see how captions adapt to emotional context.
"""

import os
import json
from pathlib import Path
from datetime import datetime

# Import Auto-Caption modules
from auto_caption.emotion_detector import EmotionCategory
from auto_caption.caption_styler import StyleIntensity, Platform
from auto_caption.subtitle import ASSGenerator
from auto_caption.video_merger import VideoMerger


def create_emotion_samples():
    """Create sample caption data for each emotion type."""
    
    # Define sample text for each emotion
    emotion_samples = {
        EmotionCategory.HAPPY: {
            "text": "This is amazing! I'm so happy right now!",
            "description": "Font: MADE AVENUE | Position: High/Top | Size: Large (130%)",
            "characteristics": "Bright yellow, bouncy, elevated position"
        },
        EmotionCategory.SAD: {
            "text": "I feel so down... everything seems hopeless",
            "description": "Font: ALMOST TEXTUAL | Position: Low/Bottom | Size: Small (75%)",
            "characteristics": "Deep blue, drooping, weighted down position"
        },
        EmotionCategory.ANGRY: {
            "text": "THIS IS UNACCEPTABLE! I'M FURIOUS!",
            "description": "Font: CINEMATOGRAFICA | Position: Center | Size: Very Large (150%)",
            "characteristics": "Intense red, bold, dominating center screen"
        },
        EmotionCategory.EXCITED: {
            "text": "OMG! This is incredible! Can't contain my excitement!",
            "description": "Font: MADE AVENUE | Position: Very High/Top | Size: Large (135%)",
            "characteristics": "Electric magenta, energetic, jumping high"
        },
        EmotionCategory.FEARFUL: {
            "text": "I'm scared... what if something happens?",
            "description": "Font: ALMOST TEXTUAL | Position: Low Left | Size: Small (80%)",
            "characteristics": "Pale ghostly blue, shrinking, hiding in corner"
        },
        EmotionCategory.SARCASTIC: {
            "text": "Oh yeah, that's TOTALLY believable... sure",
            "description": "Font: CINEMATOGRAFICA | Position: Right Side | Size: Medium+ (110%)",
            "characteristics": "Sharp lime green, italicized, sideways delivery"
        },
        EmotionCategory.ANXIOUS: {
            "text": "I don't know... maybe? I'm not sure about this",
            "description": "Font: ALMOST TEXTUAL | Position: Center-Low | Size: Small (85%)",
            "characteristics": "Nervous pale green, uncertain, frozen in middle"
        },
        EmotionCategory.CONTEMPLATIVE: {
            "text": "Hmm... let me think about this deeply",
            "description": "Font: ALMOST TEXTUAL | Position: High/Top | Size: Medium (90%)",
            "characteristics": "Thoughtful lavender, elevated for thinking, soft focus"
        }
    }
    
    return emotion_samples


def generate_demo_captions(emotion_samples, duration=3.0):
    """Generate caption data for each emotion."""
    
    demo_segments = []
    current_time = 0.0
    
    for emotion, sample in emotion_samples.items():
        segment = {
            "start": current_time,
            "end": current_time + duration,
            "text": sample["text"],
            "emotion_metadata": {
                "emotion": emotion.value,
                "confidence": 0.95,
                "visual_suggestions": {
                    "description": sample["description"],
                    "characteristics": sample["characteristics"]
                }
            }
        }
        demo_segments.append(segment)
        current_time += duration + 0.5  # Add gap between segments
    
    return {
        "segments": demo_segments,
        "emotion_data": {
            "dominant_emotion": "neutral",
            "metadata": {
                "demo": True,
                "created": datetime.now().isoformat()
            }
        }
    }


def create_emotion_style_comparison():
    """Create a visual comparison of all emotion styles."""
    
    print("=== Auto-Caption Emotion Styling Demo ===\n")
    
    # Font assignments
    print("FONT ASSIGNMENTS:")
    print("-" * 50)
    print("MADE AVENUE (Clean, Modern, Friendly):")
    print("  - Happy, Excited, Neutral, Humorous")
    print("\nCINEMATOGRAFICA (Bold, Dramatic, Intense):")
    print("  - Angry, Sarcastic, Surprised, Disgusted")
    print("\nALMOST TEXTUAL (Soft, Emotional, Introspective):")
    print("  - Sad, Fearful, Anxious, Contemplative")
    print("\n")
    
    # Position and size effects
    print("POSITION & SIZE EFFECTS:")
    print("-" * 50)
    
    emotion_effects = {
        "Happy": "Position: 60px higher | Size: 130% | Top-center alignment",
        "Sad": "Position: 80px lower | Size: 75% | Bottom-center alignment",
        "Angry": "Position: 40px higher | Size: 150% | Center screen (confrontational)",
        "Excited": "Position: 70px higher | Size: 135% | Top-center (bouncing)",
        "Fearful": "Position: 50px lower | Size: 80% | Bottom-left (cornered)",
        "Sarcastic": "Position: 10px higher | Size: 110% | Right side (sideways)",
        "Anxious": "Position: 30px lower | Size: 85% | Center (frozen)",
        "Contemplative": "Position: 30px higher | Size: 90% | Top-center (thinking)"
    }
    
    for emotion, effect in emotion_effects.items():
        print(f"{emotion:15} | {effect}")
    
    print("\n")
    
    # Color schemes
    print("COLOR SCHEMES:")
    print("-" * 50)
    
    color_schemes = {
        "Happy": "Bright sunny yellow with orange glow",
        "Sad": "Deep ocean blue with heavy shadow",
        "Angry": "Intense red with red glow effect",
        "Excited": "Electric magenta with pink glow",
        "Fearful": "Pale ghostly blue, deep shadows",
        "Sarcastic": "Sharp lime green with edge",
        "Anxious": "Nervous pale green, blurred",
        "Contemplative": "Thoughtful lavender, soft focus"
    }
    
    for emotion, colors in color_schemes.items():
        print(f"{emotion:15} | {colors}")


def generate_ass_preview(output_dir="emotion_style_preview"):
    """Generate ASS subtitle files for each emotion style."""
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Get emotion samples
    emotion_samples = create_emotion_samples()
    
    # Generate ASS files for different style intensities
    for intensity in [StyleIntensity.SUBTLE, StyleIntensity.MEDIUM, StyleIntensity.INTENSE]:
        print(f"\nGenerating {intensity.value} intensity previews...")
        
        # Create ASS generator
        ass_gen = ASSGenerator(
            platform=Platform.GENERAL,
            style_intensity=intensity,
            video_resolution=(1920, 1080)
        )
        
        # Generate caption data
        caption_data = generate_demo_captions(emotion_samples)
        
        # Generate ASS file
        output_file = os.path.join(output_dir, f"emotions_{intensity.value}.ass")
        ass_gen.generate_ass_file(
            caption_data,
            output_file,
            title=f"Emotion Styles - {intensity.value.capitalize()} Intensity"
        )
        
        print(f"  Created: {output_file}")
    
    # Generate individual emotion samples
    print("\nGenerating individual emotion samples...")
    for emotion, sample in emotion_samples.items():
        # Create single emotion caption
        single_caption = {
            "segments": [{
                "start": 0,
                "end": 3,
                "text": sample["text"],
                "emotion_metadata": {
                    "emotion": emotion.value,
                    "confidence": 0.95
                }
            }]
        }
        
        # Generate ASS file
        ass_gen = ASSGenerator(
            platform=Platform.GENERAL,
            style_intensity=StyleIntensity.INTENSE,
            video_resolution=(1920, 1080)
        )
        
        output_file = os.path.join(output_dir, f"emotion_{emotion.value}.ass")
        ass_gen.generate_ass_file(
            single_caption,
            output_file,
            title=f"Emotion Style - {emotion.value.capitalize()}"
        )
        
        print(f"  Created: {output_file}")
    
    print(f"\n✓ All ASS files generated in: {output_dir}/")
    
    # Create info file
    info_content = """
    EMOTION STYLING REFERENCE
    ========================
    
    This directory contains ASS subtitle files demonstrating how emotions affect:
    
    1. FONT SELECTION:
       - MADE AVENUE: Happy, Excited, Neutral (clean, modern)
       - CINEMATOGRAFICA: Angry, Sarcastic (bold, dramatic)
       - ALMOST TEXTUAL: Sad, Anxious, Contemplative (soft, emotional)
    
    2. POSITIONING:
       - Happy/Excited: Higher on screen (uplifted)
       - Sad/Fearful: Lower on screen (weighted down)
       - Angry: Center screen (confrontational)
       - Sarcastic: Side position (sideways delivery)
    
    3. SIZE SCALING:
       - Intense emotions (Angry): 150% size
       - Positive emotions (Happy): 130% size
       - Negative emotions (Sad): 75% size
       - Subtle emotions: 85-110% size
    
    4. VISUAL EFFECTS:
       - Colors matched to emotional tone
       - Shadow/glow effects for emphasis
       - Blur effects for uncertainty/sadness
       - Bold/italic based on emotion
    
    To use these with a video:
    ffmpeg -i video.mp4 -vf "subtitles=emotion_style.ass" output.mp4
    """
    
    with open(os.path.join(output_dir, "README.txt"), "w") as f:
        f.write(info_content)


def main():
    """Run the emotion styling demo."""
    
    print("\n" + "="*60)
    print("AUTO-CAPTION EMOTION STYLING DEMO")
    print("="*60 + "\n")
    
    # Show style comparison
    create_emotion_style_comparison()
    
    # Generate ASS preview files
    print("\n" + "="*60)
    print("GENERATING PREVIEW FILES")
    print("="*60)
    generate_ass_preview()
    
    print("\n" + "="*60)
    print("DEMO COMPLETE!")
    print("="*60)
    print("\nThe generated ASS files demonstrate how each emotion affects:")
    print("- Font selection (MADE AVENUE / CINEMATOGRAFICA / ALMOST TEXTUAL)")
    print("- Position on screen (high/center/low, left/center/right)")
    print("- Text size (75% to 150% scaling)")
    print("- Colors and visual effects")
    print("\nUse these files with your videos to see the emotion-adaptive styling in action!")


if __name__ == "__main__":
    main()