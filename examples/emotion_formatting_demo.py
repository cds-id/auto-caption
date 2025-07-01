#!/usr/bin/env python3
"""
Emotion-Adaptive Text Formatting Demo

This script demonstrates how Auto-Caption formats text based on detected emotions,
using punctuation, capitalization, and emphasis to convey emotional context.
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from auto_caption.emotion_detector import EmotionDetector, EmotionCategory
from auto_caption.caption_styler import CaptionStyler, StyleIntensity, Platform
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def demonstrate_emotion_formatting():
    """Show how different emotions affect text formatting."""
    
    # Initialize the caption styler
    styler = CaptionStyler(
        default_intensity=StyleIntensity.MEDIUM,
        default_platform=Platform.GENERAL
    )
    
    # Sample text that could have different meanings based on emotion
    sample_texts = [
        "I can't believe this happened",
        "That was really something",
        "Oh great just what I needed",
        "This is amazing",
        "I don't know what to say"
    ]
    
    # Emotions to demonstrate
    emotions = [
        (EmotionCategory.HAPPY, 0.9),
        (EmotionCategory.SAD, 0.85),
        (EmotionCategory.ANGRY, 0.88),
        (EmotionCategory.SARCASTIC, 0.92),
        (EmotionCategory.EXCITED, 0.95),
        (EmotionCategory.ANXIOUS, 0.8),
        (EmotionCategory.CONTEMPLATIVE, 0.87),
        (EmotionCategory.NEUTRAL, 0.9)
    ]
    
    console.print(Panel.fit(
        "[bold cyan]Emotion-Adaptive Text Formatting Demo[/bold cyan]\n"
        "This demo shows how the same text is formatted differently based on detected emotions.\n"
        "The formatting uses punctuation, capitalization, and emphasis to convey emotional context.",
        title="Auto-Caption Demo"
    ))
    
    # Process each sample text
    for text in sample_texts:
        console.print(f"\n[bold yellow]Original text:[/bold yellow] '{text}'")
        console.print("[dim]Formatted based on emotion:[/dim]\n")
        
        # Create a table for this text
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Emotion", style="cyan", width=15)
        table.add_column("Confidence", justify="center", width=10)
        table.add_column("Formatted Text", width=50)
        table.add_column("Style Details", width=30)
        
        for emotion, confidence in emotions:
            # Apply emotion-based formatting
            result = styler.style_caption(
                text,
                emotion,
                confidence,
                StyleIntensity.MEDIUM
            )
            
            # Extract formatting details
            formatting = result['formatting_metadata']
            style_details = f"Cap: {formatting['capitalization']}\n"
            style_details += f"Punct: {formatting['punctuation_style']}\n"
            style_details += f"Emphasis: {formatting.get('text_emphasis', 'none')}"
            
            table.add_row(
                emotion.value,
                f"{confidence:.0%}",
                result['formatted_text'],
                style_details
            )
        
        console.print(table)


def demonstrate_intensity_levels():
    """Show how intensity affects formatting."""
    
    console.print("\n" + "="*80 + "\n")
    console.print(Panel.fit(
        "[bold cyan]Formatting Intensity Levels[/bold cyan]\n"
        "The same emotion can be expressed with different intensity levels.",
        title="Intensity Demo"
    ))
    
    styler = CaptionStyler()
    
    # Test text and emotion
    test_text = "This is incredible"
    test_emotion = EmotionCategory.EXCITED
    confidence = 0.9
    
    console.print(f"\n[bold yellow]Text:[/bold yellow] '{test_text}'")
    console.print(f"[bold yellow]Emotion:[/bold yellow] {test_emotion.value}\n")
    
    # Show all intensity levels
    intensities = [StyleIntensity.SUBTLE, StyleIntensity.MEDIUM, StyleIntensity.INTENSE]
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Intensity", style="cyan", width=15)
    table.add_column("Formatted Text", width=50)
    
    for intensity in intensities:
        result = styler.style_caption(
            test_text,
            test_emotion,
            confidence,
            intensity
        )
        
        table.add_row(
            intensity.value,
            result['formatted_text']
        )
    
    console.print(table)


def demonstrate_platform_differences():
    """Show platform-specific formatting adjustments."""
    
    console.print("\n" + "="*80 + "\n")
    console.print(Panel.fit(
        "[bold cyan]Platform-Specific Formatting[/bold cyan]\n"
        "Different platforms may have different text length limits and conventions.",
        title="Platform Demo"
    ))
    
    # Longer text to show truncation
    long_text = "This is a really long caption that might need to be truncated depending on the platform requirements and limitations"
    emotion = EmotionCategory.HAPPY
    confidence = 0.85
    
    console.print(f"\n[bold yellow]Original text:[/bold yellow] '{long_text}'")
    console.print(f"[bold yellow]Emotion:[/bold yellow] {emotion.value}\n")
    
    platforms = [Platform.TIKTOK, Platform.INSTAGRAM, Platform.YOUTUBE_SHORTS, Platform.GENERAL]
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Platform", style="cyan", width=15)
    table.add_column("Formatted Text", width=65)
    
    for platform in platforms:
        styler = CaptionStyler(default_platform=platform)
        result = styler.style_caption(
            long_text,
            emotion,
            confidence
        )
        
        table.add_row(
            platform.value,
            result['formatted_text']
        )
    
    console.print(table)


def demonstrate_real_world_examples():
    """Show real-world caption formatting examples."""
    
    console.print("\n" + "="*80 + "\n")
    console.print(Panel.fit(
        "[bold cyan]Real-World Caption Examples[/bold cyan]\n"
        "Examples of how captions would be formatted for different types of content.",
        title="Real Examples"
    ))
    
    styler = CaptionStyler()
    
    # Real-world scenarios
    scenarios = [
        {
            "context": "Tutorial video explaining a complex topic",
            "text": "Now this is the important part",
            "emotion": EmotionCategory.CONFIDENT,
            "confidence": 0.9,
            "intensity": StyleIntensity.MEDIUM
        },
        {
            "context": "Comedy sketch with punchline",
            "text": "And then he actually said yes",
            "emotion": EmotionCategory.HUMOROUS,
            "confidence": 0.95,
            "intensity": StyleIntensity.INTENSE
        },
        {
            "context": "Emotional personal story",
            "text": "I never thought this would happen to me",
            "emotion": EmotionCategory.SAD,
            "confidence": 0.88,
            "intensity": StyleIntensity.MEDIUM
        },
        {
            "context": "Exciting product reveal",
            "text": "Check out what we just made",
            "emotion": EmotionCategory.EXCITED,
            "confidence": 0.92,
            "intensity": StyleIntensity.INTENSE
        },
        {
            "context": "Sarcastic reaction video",
            "text": "Oh wow that's totally normal",
            "emotion": EmotionCategory.SARCASTIC,
            "confidence": 0.9,
            "intensity": StyleIntensity.MEDIUM
        }
    ]
    
    for scenario in scenarios:
        console.print(f"\n[bold green]Context:[/bold green] {scenario['context']}")
        console.print(f"[bold yellow]Original:[/bold yellow] '{scenario['text']}'")
        
        result = styler.style_caption(
            scenario['text'],
            scenario['emotion'],
            scenario['confidence'],
            scenario['intensity']
        )
        
        console.print(f"[bold cyan]Formatted:[/bold cyan] '{result['formatted_text']}'")
        console.print(f"[dim]Emotion: {scenario['emotion'].value} | "
                     f"Intensity: {scenario['intensity'].value}[/dim]")


def main():
    """Run all demonstrations."""
    console.print("\n" + "="*80)
    console.print("[bold]Auto-Caption: Emotion-Adaptive Text Formatting[/bold]")
    console.print("="*80 + "\n")
    
    # Run demonstrations
    demonstrate_emotion_formatting()
    demonstrate_intensity_levels()
    demonstrate_platform_differences()
    demonstrate_real_world_examples()
    
    console.print("\n" + "="*80 + "\n")
    console.print("[bold green]Demo Complete![/bold green]")
    console.print("\nThis demo showed how Auto-Caption formats text based on emotions detected")
    console.print("from facial expressions, using punctuation, capitalization, and emphasis")
    console.print("to convey the emotional context of the video.\n")


if __name__ == "__main__":
    main()