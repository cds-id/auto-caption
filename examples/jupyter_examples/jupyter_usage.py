#!/usr/bin/env python3
"""
Jupyter Notebook Usage Examples for Auto-Caption

This module provides examples and utilities for using Auto-Caption
in Jupyter notebooks, including interactive widgets and visualizations.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Video, display, HTML, Audio
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed

# Import auto-caption modules
from auto_caption import (
    CaptionGenerator,
    EmotionDetector,
    CaptionStyler,
    VideoMerger,
    EmotionCategory,
    StyleIntensity,
    Platform
)
from auto_caption.subtitle import ASSGenerator
from auto_caption.visualization import EmotionVisualizer


class JupyterCaptionHelper:
    """
    Helper class for using Auto-Caption in Jupyter notebooks.
    
    Provides convenient methods for interactive caption generation,
    emotion detection, and visualization.
    """
    
    def __init__(self, model_name: str = "base", device: str = "cuda"):
        """Initialize the helper with specified model and device."""
        self.caption_generator = CaptionGenerator(
            model_name=model_name,
            device=device
        )
        self.emotion_detector = EmotionDetector(device=device)
        self.caption_styler = CaptionStyler()
        self.current_video_path = None
        self.current_results = {}
    
    def process_video_interactive(self, video_path: str) -> Dict[str, Any]:
        """
        Process video with interactive progress display.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing captions and emotions
        """
        self.current_video_path = video_path
        
        # Display video info
        print(f"Processing: {video_path}")
        
        # Generate captions with progress
        print("\n📝 Generating captions...")
        captions = self.caption_generator.generate_captions(
            video_path=video_path,
            word_level=True,
            verbose=True
        )
        
        # Detect emotions
        print("\n😊 Detecting emotions...")
        emotions = self.emotion_detector.process_video(
            video_path=video_path,
            sample_rate=0.5,
            show_progress=True
        )
        
        # Match emotions to caption segments
        print("\n🔗 Matching emotions to captions...")
        matched_segments = self._match_emotions_to_segments(
            captions['segments'],
            emotions
        )
        
        # Style captions based on emotions
        print("\n✨ Styling captions...")
        styled_segments = self._style_segments(matched_segments)
        
        # Store results
        self.current_results = {
            'video_path': video_path,
            'duration': captions['duration'],
            'segments': styled_segments,
            'emotion_timeline': emotions,
            'original_captions': captions
        }
        
        print("\n✅ Processing complete!")
        return self.current_results
    
    def _match_emotions_to_segments(
        self,
        segments: List[Dict],
        emotions: List[Dict]
    ) -> List[Dict]:
        """Match detected emotions to caption segments."""
        matched = []
        
        for segment in segments:
            # Find emotions within segment time range
            segment_emotions = [
                e for e in emotions
                if segment['start'] <= e['timestamp'] <= segment['end']
            ]
            
            # Get dominant emotion
            if segment_emotions:
                # Use emotion with highest confidence
                dominant = max(segment_emotions, key=lambda x: x['confidence'])
                emotion = dominant['emotion']
                confidence = dominant['confidence']
            else:
                emotion = EmotionCategory.NEUTRAL.value
                confidence = 0.5
            
            matched.append({
                **segment,
                'emotion': emotion,
                'confidence': confidence
            })
        
        return matched
    
    def _style_segments(self, segments: List[Dict]) -> List[Dict]:
        """Apply emotion-based styling to segments."""
        styled = []
        
        for segment in segments:
            # Convert emotion string to enum
            try:
                emotion_enum = EmotionCategory(segment['emotion'])
            except ValueError:
                emotion_enum = EmotionCategory.NEUTRAL
            
            # Style the text
            style_result = self.caption_styler.style_caption(
                text=segment['text'],
                emotion=emotion_enum,
                confidence=segment['confidence']
            )
            
            styled.append({
                **segment,
                'formatted_text': style_result['formatted_text'],
                'format_type': style_result['format_type']
            })
        
        return styled
    
    def create_emotion_widget(self):
        """Create interactive widget for emotion styling demo."""
        text_input = widgets.Text(
            value='Hello world!',
            placeholder='Enter text to style',
            description='Text:',
            style={'description_width': 'initial'}
        )
        
        emotion_dropdown = widgets.Dropdown(
            options=[e.value for e in EmotionCategory],
            value=EmotionCategory.NEUTRAL.value,
            description='Emotion:',
            style={'description_width': 'initial'}
        )
        
        intensity_slider = widgets.SelectionSlider(
            options=['subtle', 'medium', 'intense'],
            value='medium',
            description='Intensity:',
            style={'description_width': 'initial'}
        )
        
        output = widgets.Output()
        
        def update_styling(text, emotion, intensity):
            with output:
                output.clear_output(wait=True)
                
                emotion_enum = EmotionCategory(emotion)
                intensity_enum = StyleIntensity(intensity)
                
                styler = CaptionStyler(default_intensity=intensity_enum)
                result = styler.style_caption(
                    text=text,
                    emotion=emotion_enum,
                    confidence=0.9
                )
                
                print(f"Original: {text}")
                print(f"Emotion: {emotion} (Intensity: {intensity})")
                print(f"Styled: {result['formatted_text']}")
                print(f"Format Type: {result['format_type']}")
        
        # Create interactive widget
        interactive_widget = widgets.interactive(
            update_styling,
            text=text_input,
            emotion=emotion_dropdown,
            intensity=intensity_slider
        )
        
        return widgets.VBox([interactive_widget, output])
    
    def visualize_emotion_timeline(self, figsize=(15, 8)):
        """Create an interactive emotion timeline visualization."""
        if not self.current_results:
            print("No results to visualize. Process a video first.")
            return
        
        emotions = self.current_results.get('emotion_timeline', [])
        if not emotions:
            print("No emotion data available.")
            return
        
        # Extract data
        timestamps = [e['timestamp'] for e in emotions]
        emotion_names = [e['emotion'] for e in emotions]
        confidences = [e['confidence'] for e in emotions]
        
        # Create color map
        unique_emotions = list(set(emotion_names))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_emotions)))
        emotion_colors = dict(zip(unique_emotions, colors))
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
        
        # Plot emotion timeline
        for emotion in unique_emotions:
            emotion_times = [t for t, e in zip(timestamps, emotion_names) if e == emotion]
            emotion_confs = [c for e, c in zip(emotion_names, confidences) if e == emotion]
            
            ax1.scatter(
                emotion_times,
                [emotion] * len(emotion_times),
                s=[c * 200 for c in emotion_confs],
                c=[emotion_colors[emotion]],
                alpha=0.7,
                label=emotion
            )
        
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Emotion')
        ax1.set_title('Emotion Timeline')
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot confidence over time
        ax2.plot(timestamps, confidences, 'b-', alpha=0.7)
        ax2.fill_between(timestamps, confidences, alpha=0.3)
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Confidence')
        ax2.set_title('Emotion Detection Confidence')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.show()
    
    def display_styled_captions(self, max_segments: int = 10):
        """Display styled captions in a formatted table."""
        if not self.current_results:
            print("No results to display. Process a video first.")
            return
        
        segments = self.current_results.get('segments', [])[:max_segments]
        
        # Create HTML table
        html = """
        <style>
        .caption-table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }
        .caption-table th, .caption-table td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        .caption-table th {
            background-color: #4CAF50;
            color: white;
        }
        .caption-table tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        .emotion-happy { color: #FFD700; font-weight: bold; }
        .emotion-sad { color: #4169E1; }
        .emotion-angry { color: #FF0000; font-weight: bold; }
        .emotion-excited { color: #FF1493; font-weight: bold; }
        .emotion-neutral { color: #666666; }
        </style>
        <table class="caption-table">
        <tr>
            <th>Time</th>
            <th>Original Text</th>
            <th>Emotion</th>
            <th>Styled Text</th>
        </tr>
        """
        
        for seg in segments:
            time_range = f"{seg['start']:.1f}s - {seg['end']:.1f}s"
            emotion_class = f"emotion-{seg['emotion']}"
            
            html += f"""
            <tr>
                <td>{time_range}</td>
                <td>{seg['text']}</td>
                <td class="{emotion_class}">{seg['emotion']}</td>
                <td>{seg.get('formatted_text', seg['text'])}</td>
            </tr>
            """
        
        html += "</table>"
        
        display(HTML(html))
    
    def export_for_video_editing(self, output_dir: str = "output"):
        """Export results in formats suitable for video editing."""
        if not self.current_results:
            print("No results to export. Process a video first.")
            return
        
        Path(output_dir).mkdir(exist_ok=True)
        
        # Export JSON data
        json_path = Path(output_dir) / "caption_data.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.current_results, f, indent=2, ensure_ascii=False)
        
        # Generate ASS subtitle file
        ass_generator = ASSGenerator()
        ass_path = ass_generator.generate_ass_file(
            caption_data=self.current_results,
            output_path=str(Path(output_dir) / "styled_subtitles.ass")
        )
        
        # Generate SRT for compatibility
        srt_path = Path(output_dir) / "subtitles.srt"
        self._export_srt(self.current_results['segments'], str(srt_path))
        
        print(f"✅ Exported files to {output_dir}/")
        print(f"  - JSON data: {json_path}")
        print(f"  - ASS subtitles: {ass_path}")
        print(f"  - SRT subtitles: {srt_path}")
        
        return {
            'json': str(json_path),
            'ass': str(ass_path),
            'srt': str(srt_path)
        }
    
    def _export_srt(self, segments: List[Dict], output_path: str):
        """Export segments as SRT file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(segments, 1):
                start_time = self._seconds_to_srt_time(segment['start'])
                end_time = self._seconds_to_srt_time(segment['end'])
                text = segment.get('formatted_text', segment['text'])
                
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{text}\n\n")
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')


# Convenience functions for quick usage
def quick_process(video_path: str, model: str = "base") -> Dict[str, Any]:
    """
    Quick function to process a video and return results.
    
    Example:
        results = quick_process("my_video.mp4")
        for seg in results['segments'][:5]:
            print(f"{seg['start']:.1f}s: {seg['formatted_text']}")
    """
    helper = JupyterCaptionHelper(model_name=model)
    return helper.process_video_interactive(video_path)


def demo_emotion_styling():
    """Run an interactive emotion styling demo."""
    helper = JupyterCaptionHelper()
    return helper.create_emotion_widget()


def visualize_video_emotions(video_path: str):
    """Process a video and visualize its emotion timeline."""
    helper = JupyterCaptionHelper()
    helper.process_video_interactive(video_path)
    helper.visualize_emotion_timeline()
    helper.display_styled_captions()


# Example notebook code snippets
EXAMPLE_SNIPPETS = {
    "basic_usage": """
# Basic usage
from auto_caption.examples.jupyter_examples.jupyter_usage import quick_process

results = quick_process("video.mp4")
print(f"Processed {len(results['segments'])} segments")
""",
    
    "interactive_demo": """
# Interactive emotion styling demo
from auto_caption.examples.jupyter_examples.jupyter_usage import demo_emotion_styling

widget = demo_emotion_styling()
display(widget)
""",
    
    "full_pipeline": """
# Full processing pipeline
from auto_caption.examples.jupyter_examples.jupyter_usage import JupyterCaptionHelper

helper = JupyterCaptionHelper(model_name="base", device="cuda")
results = helper.process_video_interactive("video.mp4")
helper.visualize_emotion_timeline()
helper.display_styled_captions()
exports = helper.export_for_video_editing()
""",
    
    "custom_styling": """
# Custom emotion styling
from auto_caption import CaptionStyler, EmotionCategory, StyleIntensity

styler = CaptionStyler(
    default_intensity=StyleIntensity.INTENSE,
    default_platform=Platform.TIKTOK
)

text = "This is amazing!"
for emotion in [EmotionCategory.HAPPY, EmotionCategory.EXCITED, EmotionCategory.SARCASTIC]:
    result = styler.style_caption(text, emotion, 0.9)
    print(f"{emotion.value}: {result['formatted_text']}")
"""
}


if __name__ == "__main__":
    print("Auto-Caption Jupyter Usage Examples")
    print("===================================")
    print("\nThis module provides helper functions for using Auto-Caption in Jupyter notebooks.")
    print("\nExample usage:")
    print(EXAMPLE_SNIPPETS["basic_usage"])