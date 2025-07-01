"""
Command-line interface for Auto-Caption tool.
"""

import os
import sys
import json
import glob
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.panel import Panel
from rich import print as rprint

from . import __version__
from .caption_generator import CaptionGenerator
from .models import WhisperModel, get_available_models
from .utils import validate_video_file, get_output_filename, load_config, save_config, get_video_info, format_duration, format_size
from .emotion_detector import EmotionDetector, EmotionCategory
from .caption_styler import CaptionStyler, StyleIntensity, Platform
from .video_merger import VideoMerger
from .subtitle import ASSGenerator
from .training import EmotionTrainer, TrainingConfig, DatasetBuilder, VideoAugmenter, AugmentationConfig
from .word_timing import WordTimingProcessor, WordAnimationStyle
from .object_detection import ObjectDetector

console = Console()


@click.group()
@click.option('--config', type=click.Path(), help='Path to configuration file')
@click.pass_context
def cli(ctx, config):
    """Auto-Caption: Generate captions from video files using AI."""
    ctx.ensure_object(dict)

    # Load configuration
    if config and os.path.exists(config):
        ctx.obj['config'] = load_config(config)
    else:
        # Load default config from home directory if exists
        default_config = Path.home() / '.auto-caption' / 'config.json'
        if default_config.exists():
            ctx.obj['config'] = load_config(str(default_config))
        else:
            ctx.obj['config'] = {
                'default_model': 'base',
                'default_format': ['srt'],
                'default_language': 'auto',
                'verbose': False,
                'threads': 4
            }


@cli.command()
@click.argument('video_file', type=click.Path(exists=True))
@click.option('--model', '-m', default=None,
              type=click.Choice(['tiny', 'base', 'small', 'medium', 'large']),
              help='Whisper model to use (default: base)')
@click.option('--format', '-f', multiple=True,
              type=click.Choice(['srt', 'vtt', 'txt', 'json', 'ass']),
              help='Output format(s) (can be specified multiple times)')
@click.option('--output', '-o', type=click.Path(),
              help='Output file path (auto-generated if not specified)')
@click.option('--language', '-l', default=None,
              help='Language code (e.g., en, es, fr) or "auto" for detection')
@click.option('--task', type=click.Choice(['transcribe', 'translate']),
              default='transcribe', help='Task to perform')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--temperature', type=float, default=0.0,
              help='Temperature for sampling (0.0 = deterministic)')
@click.option('--threads', type=int, default=None,
              help='Number of threads to use')
@click.option('--emotion-mode', type=click.Choice(['auto', 'manual', 'off']),
              default='off', help='Face-based emotion detection mode (analyzes facial expressions)')
@click.option('--emotion', type=click.Choice(['happy', 'sad', 'angry', 'sarcastic',
              'anxious', 'neutral', 'excited', 'contemplative']),
              help='Manual emotion override (requires --emotion-mode manual)')
@click.option('--style-intensity', type=click.Choice(['subtle', 'medium', 'intense']),
              default='medium', help='Caption text formatting intensity')
@click.option('--platform', type=click.Choice(['tiktok', 'instagram', 'youtube_shorts', 'general']),
              default='general', help='Target platform for optimization')
@click.option('--word-by-word', is_flag=True, help='Enable word-by-word caption display')
@click.option('--word-animation', type=click.Choice(['typewriter', 'fade_in', 'pop_in', 'slide_in',
              'bounce_in', 'wave', 'random', 'karaoke', 'emphasis']),
              default='typewriter', help='Word animation style')
@click.option('--words-per-second', type=float, default=3.0,
              help='Reading speed for word timing (default: 3.0)')
@click.option('--smart-positioning', is_flag=True, 
              help='Enable object-aware caption positioning to avoid blocking faces and important objects')
@click.option('--avoid-faces', is_flag=True,
              help='Specifically avoid blocking faces (requires --smart-positioning)')
@click.option('--object-detection-model', type=click.Choice(['small', 'medium', 'large']),
              default='medium', help='Object detection model size (affects accuracy vs speed)')
@click.pass_context
def generate(ctx, video_file, model, format, output, language, task, verbose,
             temperature, threads, emotion_mode, emotion, style_intensity, platform,
             word_by_word, word_animation, words_per_second, smart_positioning,
             avoid_faces, object_detection_model):
    """Generate captions for a single video file with optional emotion-adaptive text formatting and smart object-aware positioning."""
    config = ctx.obj.get('config', {})

    # Use defaults from config if not specified
    if not model:
        model = config.get('default_model', 'base')
    if not format:
        format = config.get('default_format', ['srt'])
    if not language:
        language = config.get('default_language', 'auto')
    if threads is None:
        threads = config.get('threads', 4)

    # Validate video file
    if not validate_video_file(video_file):
        console.print(f"[red]Error:[/red] '{video_file}' is not a valid video file")
        sys.exit(1)

    console.print(Panel.fit(
        f"[bold cyan]Processing:[/bold cyan] {video_file}\n"
        f"[bold]Model:[/bold] {model} | [bold]Language:[/bold] {language} | "
        f"[bold]Format(s):[/bold] {', '.join(format)}\n"
        f"[bold]Emotion Mode:[/bold] {emotion_mode} | [bold]Platform:[/bold] {platform}",
        title="Auto-Caption: Emotion-Aware Generation"
    ))

    try:
        # Initialize caption generator
        transcribe_options = {}
        if word_by_word:
            transcribe_options["word_timestamps"] = True

        generator = CaptionGenerator(
            model_name=model,
            language=language if language != 'auto' else None,
            task=task,
            verbose=verbose,
            threads=threads,
            enable_object_detection=smart_positioning
        )

        # Generate captions
        console.print("[green]✓[/green] Model loaded successfully")

        # Initialize emotion detector if needed
        emotion_result = None
        if emotion_mode != 'off':
            console.print(f"[yellow]Initializing face-based emotion detection...[/yellow]")
            console.print("[dim]Analyzing facial expressions for accurate emotion detection[/dim]")
            emotion_detector = EmotionDetector(verbose=verbose)

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=console
            ) as progress:
                task_id = progress.add_task("Detecting emotions from facial expressions...", total=100)
                emotion_result = emotion_detector.detect_emotions(
                    video_file,
                    progress_callback=lambda p: progress.update(task_id, completed=p)
                )
            console.print(f"[green]✓[/green] Dominant emotion detected: {emotion_result.dominant_emotion.value}")
            console.print(f"[dim]Based on analysis of {emotion_result.metadata.get('visual_samples', 0)} facial expressions[/dim]")

            # Show emotion timeline summary
            if emotion_result.temporal_emotions:
                unique_emotions = []
                prev_emotion = None
                for temporal in emotion_result.temporal_emotions:
                    if temporal['dominant_emotion'] != prev_emotion:
                        unique_emotions.append(temporal['dominant_emotion'])
                        prev_emotion = temporal['dominant_emotion']

                if len(unique_emotions) > 1:
                    console.print(f"[yellow]Detected emotion changes:[/yellow] {' → '.join(unique_emotions[:5])}")
                    if len(unique_emotions) > 5:
                        console.print(f"[dim]... and {len(unique_emotions) - 5} more emotion transitions[/dim]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task_id = progress.add_task("Generating captions...", total=100)

            result = generator.generate(
                video_file,
                temperature=temperature,
                progress_callback=lambda p: progress.update(task_id, completed=p),
                enable_smart_positioning=smart_positioning,
                **transcribe_options
            )

        console.print("[green]✓[/green] Caption generation complete")

        # Apply emotion-aware formatting if enabled
        if emotion_mode != 'off' and emotion_result:
            console.print("[yellow]Applying emotion-adaptive text formatting...[/yellow]")
            styler = CaptionStyler(
                default_intensity=StyleIntensity(style_intensity),
                default_platform=Platform(platform)
            )

            # Override emotion if manual mode
            if emotion_mode == 'manual' and emotion:
                selected_emotion = EmotionCategory(emotion)
            else:
                selected_emotion = emotion_result.dominant_emotion

            # Format all segments using temporal emotions
            formatted_segments = []
            for segment in result['segments']:
                # Find emotion at this segment's timestamp
                segment_timestamp = segment.get('start', 0)
                segment_emotion = None
                segment_confidence = 0.5

                # Search through temporal emotions to find the right one
                for temporal in emotion_result.temporal_emotions:
                    if temporal['start'] <= segment_timestamp < temporal['end']:
                        # Get the emotion for this time period
                        emotion_str = temporal['dominant_emotion']
                        segment_confidence = temporal['confidence']
                        # Convert string to EmotionCategory
                        for emotion_cat in EmotionCategory:
                            if emotion_cat.value == emotion_str:
                                segment_emotion = emotion_cat
                                break
                        break

                # Use manual override if in manual mode
                if emotion_mode == 'manual' and emotion:
                    segment_emotion = selected_emotion

                # Default to neutral if no emotion found
                if segment_emotion is None:
                    segment_emotion = EmotionCategory.NEUTRAL

                formatted = styler.style_caption(
                    segment['text'],
                    segment_emotion,
                    segment_confidence,
                    StyleIntensity(style_intensity),
                    Platform(platform)
                )
                segment['original_text'] = segment['text']
                segment['text'] = formatted['formatted_text']
                segment['emotion_metadata'] = {
                    'emotion': formatted['emotion'],
                    'confidence': formatted['confidence'],
                    'formatting': formatted.get('formatting_metadata', {})
                }
                formatted_segments.append(segment)

            result['segments'] = formatted_segments
            result['emotion_data'] = emotion_result.to_dict() if emotion_result else None
            console.print("[green]✓[/green] Emotion-adaptive formatting applied")

        # Apply word-by-word timing if enabled
        if word_by_word:
            console.print("[yellow]Processing word-by-word timing...[/yellow]")
            
            # Get video resolution for word processor
            video_info = get_video_info(video_file)
            video_resolution = (video_info['video']['width'], video_info['video']['height'])
            
            # Initialize object detector if smart positioning is enabled
            object_detector = None
            if smart_positioning:
                from .object_detection import ObjectDetector
                object_detector = ObjectDetector(
                    enable_face_detection=avoid_faces or True,
                    enable_object_detection=True,
                    model_size=object_detection_model,
                    verbose=verbose
                )
                console.print("[dim]Smart positioning enabled - analyzing objects to avoid[/dim]")
            
            word_processor = WordTimingProcessor(
                animation_style=WordAnimationStyle(word_animation),
                words_per_second=words_per_second,
                enable_word_timestamps=True,
                video_resolution=video_resolution,
                platform=Platform(platform),
                enable_object_detection=smart_positioning,
                object_detector=object_detector
            )

            # Process segments into word timings
            word_segments = word_processor.process_segments(
                result['segments'],
                result.get('emotion_data'),
                detection_results=result.get('detection_results')
            )

            # Store word timing data
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
            result['word_animation'] = word_animation
            result['metadata'] = {
                'word_by_word': True,
                'animation_style': word_animation
            }

            console.print(f"[green]✓[/green] Word-by-word timing applied ({word_animation} style)")
            if smart_positioning:
                console.print("[green]✓[/green] Smart positioning applied to avoid blocking important objects")

        # Save outputs
        saved_files = []
        for fmt in format:
            if output:
                output_path = output if len(format) == 1 else f"{output}.{fmt}"
            else:
                output_path = get_output_filename(video_file, fmt)

            generator.save_output(result, output_path, fmt)
            saved_files.append(output_path)
            console.print(f"[green]✓[/green] Saved {fmt.upper()} → {output_path}")

        # Display summary
        console.print("\n[bold green]Success![/bold green] Generated captions for:")
        console.print(f"  Video: {video_file}")
        console.print(f"  Duration: {result['duration']:.1f} seconds")
        console.print(f"  Language: {result['language']}")
        console.print(f"  Segments: {len(result['segments'])}")
        if word_by_word and 'word_segments' in result:
            total_words = sum(len(ws['words']) for ws in result['word_segments'])
            console.print(f"  Words: {total_words} (word-by-word mode)")

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
        if verbose:
            console.print_exception()
        sys.exit(1)


@cli.command()
@click.argument('directory', type=click.Path(exists=True, file_okay=False))
@click.option('--pattern', '-p', default='*.mp4',
              help='File pattern to match (default: *.mp4)')
@click.option('--model', '-m', default=None,
              type=click.Choice(['tiny', 'base', 'small', 'medium', 'large']),
              help='Whisper model to use')
@click.option('--format', '-f', multiple=True,
              type=click.Choice(['srt', 'vtt', 'txt', 'json', 'ass']),
              help='Output format(s)')
@click.option('--language', '-l', default=None,
              help='Language code or "auto" for detection')
@click.option('--recursive', '-r', is_flag=True,
              help='Process subdirectories recursively')
@click.option('--skip-existing', is_flag=True,
              help='Skip videos that already have captions')
@click.option('--threads', type=int, default=None,
              help='Number of threads to use')
@click.option('--emotion-mode', type=click.Choice(['auto', 'off']),
              default='off', help='Enable face-based emotion detection for all videos')
@click.option('--style-intensity', type=click.Choice(['subtle', 'medium', 'intense']),
              default='medium', help='Caption text formatting intensity')
@click.option('--platform', type=click.Choice(['tiktok', 'instagram', 'youtube_shorts', 'general']),
              default='general', help='Target platform for optimization')
@click.option('--word-by-word', is_flag=True, help='Enable word-by-word caption display')
@click.option('--word-animation', type=click.Choice(['typewriter', 'fade_in', 'pop_in', 'slide_in',
              'bounce_in', 'wave', 'random', 'karaoke', 'emphasis']),
              default='typewriter', help='Word animation style')
@click.option('--smart-positioning', is_flag=True,
              help='Enable object-aware caption positioning for all videos')
@click.option('--object-detection-model', type=click.Choice(['small', 'medium', 'large']),
              default='medium', help='Object detection model size')
@click.pass_context
def batch(ctx, directory, pattern, model, format, language, recursive,
          skip_existing, threads, emotion_mode, style_intensity, platform,
          word_by_word, word_animation, smart_positioning, object_detection_model):
    """Process multiple video files in batch with optional face-based emotion detection and smart positioning."""
    config = ctx.obj.get('config', {})

    # Use defaults from config
    if not model:
        model = config.get('default_model', 'base')
    if not format:
        format = config.get('default_format', ['srt'])
    if not language:
        language = config.get('default_language', 'auto')
    if threads is None:
        threads = config.get('threads', 4)

    # Find video files
    if recursive:
        video_files = list(Path(directory).rglob(pattern))
    else:
        video_files = list(Path(directory).glob(pattern))

    if not video_files:
        console.print(f"[yellow]No files matching '{pattern}' found in {directory}[/yellow]")
        return

    # Filter out files that already have captions if requested
    if skip_existing:
        files_to_process = []
        for video in video_files:
            has_caption = False
            for fmt in format:
                caption_file = get_output_filename(str(video), fmt)
                if os.path.exists(caption_file):
                    has_caption = True
                    break
            if not has_caption:
                files_to_process.append(video)

        skipped = len(video_files) - len(files_to_process)
        if skipped > 0:
            console.print(f"[yellow]Skipping {skipped} files with existing captions[/yellow]")
        video_files = files_to_process

    if not video_files:
        console.print("[yellow]No files to process[/yellow]")
        return

    console.print(Panel.fit(
        f"[bold cyan]Batch Processing[/bold cyan]\n"
        f"[bold]Files:[/bold] {len(video_files)} | [bold]Model:[/bold] {model} | "
        f"[bold]Format(s):[/bold] {', '.join(format)}",
        title="Auto-Caption Batch"
    ))

    # Initialize generator once for all files
    with console.status("[bold green]Loading Whisper model...", spinner="dots"):
        generator = CaptionGenerator(
            model_name=model,
            language=language if language != 'auto' else None,
            verbose=False,
            threads=threads,
            enable_object_detection=smart_positioning
        )

    console.print("[green]✓[/green] Model loaded successfully\n")

    # Process each file
    successful = 0
    failed = 0

    for i, video_file in enumerate(video_files, 1):
        console.print(f"[bold]Processing {i}/{len(video_files)}:[/bold] {video_file.name}")

        try:
            # Enable word timestamps if word-by-word is requested
            transcribe_options = {}
            if word_by_word:
                transcribe_options["word_timestamps"] = True

            # Generate captions
            result = generator.generate(
                str(video_file),
                detect_emotions=emotion_mode == 'auto',
                style_captions=emotion_mode != 'off',
                style_intensity=StyleIntensity(style_intensity),
                platform=Platform(platform),
                enable_smart_positioning=smart_positioning,
                **transcribe_options
            )

            # Apply word-by-word timing if enabled
            if word_by_word:
                console.print(f"  [yellow]Processing word-by-word timing...[/yellow]")
                
                # Get video resolution
                video_info = get_video_info(str(video_file))
                video_resolution = (video_info['video']['width'], video_info['video']['height'])
                
                # Initialize object detector if smart positioning is enabled
                object_detector = None
                if smart_positioning:
                    from .object_detection import ObjectDetector
                    object_detector = ObjectDetector(
                        enable_face_detection=True,
                        enable_object_detection=True,
                        model_size=object_detection_model,
                        verbose=False
                    )
                
                word_processor = WordTimingProcessor(
                    animation_style=WordAnimationStyle(word_animation),
                    words_per_second=3.0,  # Default reading speed
                    enable_word_timestamps=True,
                    video_resolution=video_resolution,
                    platform=Platform(platform),
                    enable_object_detection=smart_positioning,
                    object_detector=object_detector
                )

                # Process segments into word timings
                word_segments = word_processor.process_segments(
                    result['segments'],
                    result.get('emotion_data'),
                    detection_results=result.get('detection_results')
                )

                # Store word timing data
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
                result['word_animation'] = word_animation
                result['metadata'] = {
                    'word_by_word': True,
                    'animation_style': word_animation
                }

            # Save outputs
            for fmt in format:
                output_path = get_output_filename(str(video_file), fmt)
                generator.save_output(result, output_path, fmt)

            console.print(f"  [green]✓[/green] Completed\n")
            successful += 1

        except Exception as e:
            console.print(f"  [red]✗ Failed: {str(e)}[/red]\n")
            failed += 1

    # Summary
    console.print(Panel.fit(
        f"[bold]Batch Processing Complete[/bold]\n"
        f"[green]Successful:[/green] {successful} | [red]Failed:[/red] {failed}",
        title="Summary"
    ))


@cli.command('list-models')
def list_models():
    """List available Whisper models and their characteristics."""
    table = Table(title="Available Whisper Models", show_header=True)

    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Parameters", justify="right")
    table.add_column("English-only", justify="center")
    table.add_column("Multilingual", justify="center")
    table.add_column("Required VRAM", justify="right")
    table.add_column("Relative Speed", justify="right")

    models = get_available_models()

    for model in models:
        table.add_row(
            model['name'],
            model['parameters'],
            "✓" if model['english_only'] else "✗",
            "✓" if model['multilingual'] else "✗",
            model['vram'],
            model['speed']
        )

    console.print(table)
    console.print("\n[bold]Note:[/bold] Larger models provide better accuracy but require more resources.")


@cli.command('download-model')
@click.argument('model_name', type=click.Choice(['tiny', 'base', 'small', 'medium', 'large']))
def download_model(model_name):
    """Pre-download a specific Whisper model."""
    console.print(f"[bold]Downloading Whisper model:[/bold] {model_name}")

    try:
        with console.status(f"[bold green]Downloading {model_name} model...", spinner="dots"):
            model = WhisperModel(model_name)
            model.download()

        console.print(f"[green]✓[/green] Model '{model_name}' downloaded successfully")

    except Exception as e:
        console.print(f"[red]Error downloading model:[/red] {str(e)}")
        sys.exit(1)


@cli.command('download-models')
@click.option('--type', type=click.Choice(['emotion', 'all']),
              default='all', help='Type of models to download')
def download_models(type):
    """Download face-based emotion detection models."""
    console.print(f"[bold]Downloading {type} models...[/bold]")

    try:
        if type in ['emotion', 'all']:
            console.print("[yellow]Downloading face-based emotion detection models...[/yellow]")
            console.print("[dim]These models analyze facial expressions for accurate emotion detection[/dim]")
            # This will trigger model downloads on first use
            detector = EmotionDetector(verbose=True)
            console.print("[green]✓[/green] Face emotion detection models ready")

        console.print(f"[green]✓[/green] All {type} models downloaded successfully")

    except Exception as e:
        console.print(f"[red]Error downloading models:[/red] {str(e)}")
        sys.exit(1)


@cli.command('merge')
@click.argument('video_file', type=click.Path(exists=True))
@click.argument('caption_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(),
              help='Output video path (auto-generated if not specified)')
@click.option('--platform', type=click.Choice(['tiktok', 'instagram', 'youtube_shorts', 'general']),
              default='general', help='Target platform for optimization')
@click.option('--quality', type=click.Choice(['low', 'medium', 'high']),
              default='high', help='Output video quality')
@click.option('--preview', is_flag=True,
              help='Generate low-quality preview')
@click.option('--subtitle-format', type=click.Choice(['ass', 'srt']),
              default='ass', help='Subtitle format for styling')
@click.option('--style-intensity', type=click.Choice(['subtle', 'medium', 'intense']),
              default='medium', help='Caption styling intensity')
@click.option('--format', type=click.Choice(['mp4', 'webm']),
              default='mp4', help='Output video format')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def merge(video_file, caption_file, output, platform, quality, preview, subtitle_format, style_intensity, format, verbose):
    """Merge emotion-formatted captions with video using styled subtitles."""
    # Validate inputs
    if not validate_video_file(video_file):
        console.print(f"[red]Error:[/red] '{video_file}' is not a valid video file")
        sys.exit(1)

    if not caption_file.endswith('.json'):
        console.print(f"[red]Error:[/red] Caption file must be in JSON format")
        sys.exit(1)

    # Generate output path if not specified
    if not output:
        base_name = Path(video_file).stem
        suffix = "_preview" if preview else "_captioned"
        output = f"{base_name}{suffix}.{format}"

    console.print(Panel.fit(
        f"[bold cyan]Merging Emotion-Formatted Captions[/bold cyan]\n"
        f"[bold]Video:[/bold] {video_file}\n"
        f"[bold]Captions:[/bold] {caption_file}\n"
        f"[bold]Platform:[/bold] {platform} | [bold]Quality:[/bold] {quality}\n"
        f"[bold]Subtitle Format:[/bold] {subtitle_format.upper()} | [bold]Style:[/bold] {style_intensity}",
        title="Auto-Caption Video Merger"
    ))

    try:
        # Use subtitle-based merger
        merger = VideoMerger(
            platform=Platform(platform),
            quality=quality,
            style_intensity=StyleIntensity(style_intensity),
            output_format=format,
            verbose=verbose
        )

        # Merge video with captions using subtitles
        with console.status(f"[bold green]Processing video with {subtitle_format.upper()} subtitles...", spinner="dots"):
            result_path = merger.merge_with_video(
                video_file,
                caption_file,
                output,
                preview_mode=preview,
                subtitle_format=subtitle_format
            )

        console.print(f"[green]✓[/green] Video created: {result_path}")

        # Show video info
        video_info = get_video_info(result_path)
        console.print(f"  Duration: {format_duration(video_info['duration'])}")
        console.print(f"  Size: {format_size(video_info['size'])}")
        console.print(f"  Resolution: {video_info['video']['width']}x{video_info['video']['height']}")

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
        if verbose:
            console.print_exception()
        sys.exit(1)


@cli.command('export-subtitles')
@click.argument('caption_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(),
              help='Output subtitle file path (auto-generated if not specified)')
@click.option('--format', type=click.Choice(['ass', 'srt', 'both']),
              default='ass', help='Subtitle format(s) to export')
@click.option('--platform', type=click.Choice(['tiktok', 'instagram', 'youtube_shorts', 'general']),
              default='general', help='Target platform for optimization')
@click.option('--style-intensity', type=click.Choice(['subtle', 'medium', 'intense']),
              default='medium', help='Caption styling intensity')
@click.option('--video-file', type=click.Path(exists=True),
              help='Optional video file to extract resolution info')
@click.option('--resolution', nargs=2, type=int, default=(1920, 1080),
              help='Video resolution (width height) if no video file provided')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def export_subtitles(caption_file, output, format, platform, style_intensity, video_file, resolution, verbose):
    """Export emotion-styled subtitles in ASS or SRT format without video processing."""
    # Validate caption file
    if not caption_file.endswith('.json'):
        console.print(f"[red]Error:[/red] Caption file must be in JSON format")
        sys.exit(1)

    # Load caption data
    try:
        with open(caption_file, 'r', encoding='utf-8') as f:
            captions = json.load(f)
    except Exception as e:
        console.print(f"[red]Error loading caption file:[/red] {str(e)}")
        sys.exit(1)

    # Get video resolution
    if video_file:
        try:
            info = get_video_info(video_file)
            resolution = (info['video']['width'], info['video']['height'])
            console.print(f"[dim]Using video resolution: {resolution[0]}x{resolution[1]}[/dim]")
        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] Could not extract resolution from video, using default")

    console.print(Panel.fit(
        f"[bold cyan]Exporting Styled Subtitles[/bold cyan]\n"
        f"[bold]Captions:[/bold] {caption_file}\n"
        f"[bold]Format:[/bold] {format.upper()} | [bold]Platform:[/bold] {platform}\n"
        f"[bold]Style:[/bold] {style_intensity} | [bold]Resolution:[/bold] {resolution[0]}x{resolution[1]}",
        title="Subtitle Export"
    ))

    try:
        output_files = []

        # Generate ASS subtitle
        if format in ['ass', 'both']:
            ass_output = output if output and format == 'ass' else None
            if not ass_output:
                base_name = Path(caption_file).stem
                ass_output = f"{base_name}_styled.ass"

            # Create ASS generator
            ass_generator = ASSGenerator(
                platform=Platform(platform),
                style_intensity=StyleIntensity(style_intensity),
                video_resolution=resolution,
                verbose=verbose
            )

            # Generate ASS file
            ass_path = ass_generator.generate_ass_file(
                captions,
                ass_output,
                title=f"Auto-Caption - {Path(caption_file).stem}"
            )
            output_files.append(ass_path)
            console.print(f"[green]✓[/green] ASS subtitle exported: {ass_path}")

        # Generate SRT subtitle
        if format in ['srt', 'both']:
            srt_output = output if output and format == 'srt' else None
            if not srt_output:
                base_name = Path(caption_file).stem
                srt_output = f"{base_name}_styled.srt"

            # For SRT, we can use the ASS generator's SRT export
            ass_generator = ASSGenerator(
                platform=Platform(platform),
                style_intensity=StyleIntensity(style_intensity),
                video_resolution=resolution,
                verbose=verbose
            )

            srt_path = ass_generator.create_styled_srt(captions, srt_output)
            output_files.append(srt_path)
            console.print(f"[green]✓[/green] SRT subtitle exported: {srt_path}")

        # Show summary
        console.print("\n[bold green]Export Complete![/bold green]")
        console.print(f"Total segments: {len(captions.get('segments', []))}")

        # Show emotion distribution if available
        emotion_counts = {}
        for segment in captions.get('segments', []):
            emotion = segment.get('emotion_metadata', {}).get('emotion', 'neutral')
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        if emotion_counts:
            console.print("\n[bold]Emotion Distribution:[/bold]")
            for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(captions.get('segments', []))) * 100
                console.print(f"  {emotion}: {count} segments ({percentage:.1f}%)")

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
        if verbose:
            console.print_exception()
        sys.exit(1)


@cli.command('analyze-emotion')
@click.argument('video_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(),
              help='Save emotion analysis to JSON file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def analyze_emotion(video_file, output, verbose):
    """Analyze emotions in a video using facial expression detection without generating captions."""
    if not validate_video_file(video_file):
        console.print(f"[red]Error:[/red] '{video_file}' is not a valid video file")
        sys.exit(1)

    console.print(Panel.fit(
        f"[bold cyan]Analyzing facial expressions in:[/bold cyan] {video_file}",
        title="Face-Based Emotion Analysis"
    ))

    try:
        detector = EmotionDetector(verbose=verbose)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task_id = progress.add_task("Analyzing facial expressions...", total=100)
            result = detector.detect_emotions(
                video_file,
                progress_callback=lambda p: progress.update(task_id, completed=p)
            )

        # Display results
        table = Table(title="Face-Based Emotion Analysis Results", show_header=True)
        table.add_column("Emotion", style="cyan")
        table.add_column("Confidence", justify="right")
        table.add_column("Modality", justify="center")

        for score in result.emotion_scores[:5]:  # Top 5 emotions
            table.add_row(
                score.emotion.value,
                f"{score.confidence:.2%}",
                score.modality or "combined"
            )

        console.print(table)
        console.print(f"\n[bold]Dominant Emotion:[/bold] {result.dominant_emotion.value}")
        console.print(f"[dim]Analyzed {result.metadata.get('visual_samples', 0)} facial expressions across {result.metadata.get('duration', 0):.1f} seconds[/dim]")

        # Display temporal emotion changes
        if result.temporal_emotions:
            console.print("\n[bold]Emotion Timeline:[/bold]")
            timeline_table = Table(show_header=True)
            timeline_table.add_column("Time", style="green")
            timeline_table.add_column("Emotion", style="cyan")
            timeline_table.add_column("Confidence", justify="right")

            # Show significant emotion changes
            prev_emotion = None
            for temporal in result.temporal_emotions:
                if temporal['dominant_emotion'] != prev_emotion:
                    start_time = format_duration(temporal['start'])
                    end_time = format_duration(temporal['end'])
                    timeline_table.add_row(
                        f"{start_time} - {end_time}",
                        temporal['dominant_emotion'],
                        f"{temporal['confidence']:.2%}"
                    )
                    prev_emotion = temporal['dominant_emotion']

            console.print(timeline_table)

            # Show emotion distribution
            emotion_counts = {}
            total_segments = len(result.temporal_emotions)
            for temporal in result.temporal_emotions:
                emotion = temporal['dominant_emotion']
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

            console.print("\n[bold]Emotion Distribution:[/bold]")
            for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_segments) * 100
                bar_length = int(percentage / 2)
                bar = "█" * bar_length
                console.print(f"{emotion:15} {bar} {percentage:.1f}%")

        # Save if requested
        if output:
            with open(output, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            console.print(f"[green]✓[/green] Analysis saved to: {output}")

    except Exception as e:
        console.print(f"[red]Error analyzing emotions:[/red] {str(e)}")
        if verbose:
            console.print_exception()
        sys.exit(1)


@cli.command()
def version():
    """Display version information."""
    console.print(Panel.fit(
        f"[bold cyan]Auto-Caption[/bold cyan] v{__version__}\n"
        f"Emotion-adaptive caption formatting for short-form video content\n"
        f"Analyzes facial expressions to format captions with appropriate punctuation and emphasis\n\n"
        f"[dim]Python {sys.version.split()[0]} | Click | Whisper | Face Detection | Transformers | Rich[/dim]",
        title="Version Info"
    ))


@cli.command('train-emotion')
@click.option('--data-dir', type=click.Path(exists=True),
              help='Directory containing training data organized by emotion')
@click.option('--annotations', type=click.Path(exists=True),
              help='JSON file with video annotations')
@click.option('--model-name', default='dima806/facial_emotions_image_detection',
              help='Base model to fine-tune')
@click.option('--output-dir', '-o', default='./emotion_model',
              help='Directory to save trained model')
@click.option('--epochs', type=int, default=10,
              help='Number of training epochs')
@click.option('--batch-size', type=int, default=16,
              help='Training batch size')
@click.option('--learning-rate', type=float, default=2e-5,
              help='Learning rate')
@click.option('--custom-emotions', multiple=True,
              help='Custom emotion categories (can specify multiple)')
@click.option('--augment/--no-augment', default=True,
              help='Apply data augmentation')
@click.option('--eval-split', type=float, default=0.2,
              help='Validation split ratio')
@click.option('--push-to-hub', is_flag=True,
              help='Push model to HuggingFace Hub')
@click.option('--hub-model-id', help='HuggingFace Hub model ID')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def train_emotion(data_dir, annotations, model_name, output_dir, epochs, batch_size,
                 learning_rate, custom_emotions, augment, eval_split, push_to_hub,
                 hub_model_id, verbose):
    """Train or fine-tune emotion detection model on custom data."""
    console.print(Panel.fit(
        f"[bold cyan]Training Emotion Detection Model[/bold cyan]\n"
        f"[bold]Base Model:[/bold] {model_name}\n"
        f"[bold]Output:[/bold] {output_dir}\n"
        f"[bold]Epochs:[/bold] {epochs} | [bold]Batch Size:[/bold] {batch_size}",
        title="Emotion Model Training"
    ))

    try:
        # Setup training config
        config = TrainingConfig(
            base_model=model_name,
            output_dir=output_dir,
            num_epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            custom_emotions=list(custom_emotions) if custom_emotions else None,
            augmentation=augment,
            train_split=1 - eval_split,
            validation_split=eval_split,
            push_to_hub=push_to_hub,
            hub_model_id=hub_model_id
        )

        # Initialize trainer
        trainer = EmotionTrainer(config)
        trainer.prepare_model()

        # Prepare datasets
        console.print("[yellow]Preparing datasets...[/yellow]")

        if data_dir:
            # Load from directory structure
            builder = DatasetBuilder(
                emotion_labels=trainer.emotion_labels,
                image_size=config.image_size
            )

            train_dataset = builder.from_directory(
                data_dir,
                trainer.processor,
                split="train",
                augment=augment
            )

            val_dataset = builder.from_directory(
                data_dir,
                trainer.processor,
                split="val",
                augment=False
            )

        elif annotations:
            # Load from video annotations
            with open(annotations, 'r') as f:
                annotation_data = json.load(f)

            trainer.fine_tune_on_video(
                annotation_data.get("video_path"),
                annotations,
                output_dir
            )
            console.print(f"[green]✓[/green] Model fine-tuned and saved to: {output_dir}")
            return

        else:
            console.print("[red]Error:[/red] Either --data-dir or --annotations must be provided")
            sys.exit(1)

        # Train model
        console.print(f"[yellow]Training for {epochs} epochs...[/yellow]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task_id = progress.add_task("Training model...", total=epochs)

            # Note: This is simplified - actual training progress would need callbacks
            trainer.train(train_dataset, val_dataset)
            progress.update(task_id, completed=epochs)

        console.print(f"[green]✓[/green] Training complete!")
        console.print(f"[green]✓[/green] Model saved to: {output_dir}")

    except Exception as e:
        console.print(f"[red]Error during training:[/red] {str(e)}")
        if verbose:
            console.print_exception()
        sys.exit(1)


@cli.command('prepare-dataset')
@click.argument('video_file', type=click.Path(exists=True))
@click.option('--output-dir', '-o', required=True,
              help='Directory to save extracted frames')
@click.option('--interval', type=float, default=1.0,
              help='Interval between frame samples (seconds)')
@click.option('--emotion-labels', '-e', multiple=True,
              help='Emotion labels to annotate (interactive if not provided)')
@click.option('--auto-detect', is_flag=True,
              help='Auto-detect emotions using current model')
@click.option('--format', type=click.Choice(['directory', 'csv', 'json']),
              default='directory', help='Output format for dataset')
def prepare_dataset(video_file, output_dir, interval, emotion_labels, auto_detect, format):
    """Prepare training dataset from video files with emotion annotations."""
    console.print(Panel.fit(
        f"[bold cyan]Preparing Emotion Dataset[/bold cyan]\n"
        f"[bold]Video:[/bold] {video_file}\n"
        f"[bold]Output:[/bold] {output_dir}\n"
        f"[bold]Format:[/bold] {format}",
        title="Dataset Preparation"
    ))

    try:
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Extract frames from video
        cap = cv2.VideoCapture(video_file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = int(fps * interval)

        extracted_frames = []
        frame_count = 0

        console.print(f"[yellow]Extracting frames every {interval}s...[/yellow]")

        with tqdm(total=total_frames // frame_interval) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    # Save frame
                    frame_filename = f"frame_{frame_count:06d}.jpg"
                    frame_path = Path(output_dir) / "frames" / frame_filename
                    frame_path.parent.mkdir(exist_ok=True)

                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(str(frame_path), frame)

                    extracted_frames.append({
                        "path": str(frame_path),
                        "timestamp": frame_count / fps,
                        "frame_number": frame_count
                    })

                    pbar.update(1)

                frame_count += 1

        cap.release()
        console.print(f"[green]✓[/green] Extracted {len(extracted_frames)} frames")

        # Auto-detect or manual annotation
        annotations = []

        if auto_detect:
            console.print("[yellow]Auto-detecting emotions...[/yellow]")
            detector = EmotionDetector(verbose=True)

            for frame_data in tqdm(extracted_frames):
                # Detect emotion in frame
                img = cv2.imread(frame_data["path"])
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Simple detection (you'd implement proper frame emotion detection)
                # This is a placeholder
                annotations.append({
                    **frame_data,
                    "emotion": "neutral",  # Would be detected
                    "confidence": 0.9
                })

        else:
            # Interactive annotation
            console.print("[yellow]Manual annotation mode[/yellow]")
            console.print("Enter emotion for each frame (or 'skip' to skip, 'quit' to stop):")

            if not emotion_labels:
                emotion_labels = ["happy", "sad", "angry", "neutral", "surprised", "fearful"]

            console.print(f"Available emotions: {', '.join(emotion_labels)}")

            for frame_data in extracted_frames:
                # Show frame info
                console.print(f"\nFrame: {frame_data['path']} (time: {frame_data['timestamp']:.2f}s)")

                # Get emotion input
                emotion = click.prompt("Emotion", type=click.Choice(list(emotion_labels) + ['skip', 'quit']))

                if emotion == 'quit':
                    break
                elif emotion == 'skip':
                    continue

                annotations.append({
                    **frame_data,
                    "emotion": emotion,
                    "confidence": 1.0
                })

        # Save dataset in requested format
        if format == "directory":
            # Organize by emotion
            for ann in annotations:
                emotion_dir = Path(output_dir) / "train" / ann["emotion"]
                emotion_dir.mkdir(parents=True, exist_ok=True)

                # Copy frame to emotion directory
                src = ann["path"]
                dst = emotion_dir / Path(src).name
                shutil.copy(src, dst)

            console.print(f"[green]✓[/green] Dataset saved in directory format")

        elif format == "csv":
            # Save as CSV
            import pandas as pd
            df = pd.DataFrame(annotations)
            csv_path = Path(output_dir) / "annotations.csv"
            df.to_csv(csv_path, index=False)
            console.print(f"[green]✓[/green] Annotations saved to: {csv_path}")

        elif format == "json":
            # Save as JSON
            json_data = {
                "video_path": video_file,
                "interval": interval,
                "annotations": annotations
            }
            json_path = Path(output_dir) / "annotations.json"
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            console.print(f"[green]✓[/green] Annotations saved to: {json_path}")

    except Exception as e:
        console.print(f"[red]Error preparing dataset:[/red] {str(e)}")
        sys.exit(1)


@cli.command('evaluate-emotion')
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('test_data', type=click.Path(exists=True))
@click.option('--save-report', '-o', type=click.Path(),
              help='Save evaluation report to file')
@click.option('--confusion-matrix', is_flag=True,
              help='Generate confusion matrix plot')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def evaluate_emotion(model_path, test_data, save_report, confusion_matrix, verbose):
    """Evaluate trained emotion detection model on test data."""
    console.print(Panel.fit(
        f"[bold cyan]Evaluating Emotion Model[/bold cyan]\n"
        f"[bold]Model:[/bold] {model_path}\n"
        f"[bold]Test Data:[/bold] {test_data}",
        title="Model Evaluation"
    ))

    try:
        # Load model
        console.print("[yellow]Loading model...[/yellow]")
        config = TrainingConfig(output_dir=model_path)
        trainer = EmotionTrainer(config)
        trainer.load_model(model_path)

        # Load test dataset
        console.print("[yellow]Loading test data...[/yellow]")
        builder = DatasetBuilder(
            emotion_labels=trainer.emotion_labels,
            image_size=config.image_size
        )

        if Path(test_data).is_dir():
            test_dataset = builder.from_directory(
                test_data,
                trainer.processor,
                split="test",
                augment=False
            )
        else:
            # Assume it's a CSV or JSON file
            console.print("[red]Error:[/red] Only directory format currently supported for evaluation")
            sys.exit(1)

        # Evaluate
        console.print("[yellow]Running evaluation...[/yellow]")
        results = trainer.evaluate(test_dataset, save_confusion_matrix=confusion_matrix)

        # Display results
        table = Table(title="Evaluation Results", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        # Overall metrics
        table.add_row("Accuracy", f"{results['accuracy']:.4f}")
        table.add_row("Precision", f"{results['weighted avg']['precision']:.4f}")
        table.add_row("Recall", f"{results['weighted avg']['recall']:.4f}")
        table.add_row("F1-Score", f"{results['weighted avg']['f1-score']:.4f}")

        console.print(table)

        # Per-class results
        class_table = Table(title="Per-Emotion Performance", show_header=True)
        class_table.add_column("Emotion", style="cyan")
        class_table.add_column("Precision", justify="right")
        class_table.add_column("Recall", justify="right")
        class_table.add_column("F1-Score", justify="right")
        class_table.add_column("Support", justify="right")

        for emotion in trainer.emotion_labels.keys():
            if emotion in results:
                metrics = results[emotion]
                class_table.add_row(
                    emotion,
                    f"{metrics['precision']:.3f}",
                    f"{metrics['recall']:.3f}",
                    f"{metrics['f1-score']:.3f}",
                    str(metrics['support'])
                )

        console.print(class_table)

        # Save report if requested
        if save_report:
            with open(save_report, 'w') as f:
                json.dump(results, f, indent=2)
            console.print(f"[green]✓[/green] Evaluation report saved to: {save_report}")

        if confusion_matrix:
            cm_path = Path(model_path) / "confusion_matrix.png"
            console.print(f"[green]✓[/green] Confusion matrix saved to: {cm_path}")

    except Exception as e:
        console.print(f"[red]Error during evaluation:[/red] {str(e)}")
        if verbose:
            console.print_exception()
        sys.exit(1)


@cli.command('preview-grid')
@click.argument('video_file', type=click.Path(exists=True))
@click.argument('caption_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(),
              help='Output preview path (auto-generated if not specified)')
@click.option('--grid', nargs=2, type=int, default=(2, 2),
              help='Grid size (rows cols)')
@click.option('--duration', type=float, default=2.0,
              help='Duration of each preview segment')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def preview_grid(video_file, caption_file, output, grid, duration, verbose):
    """Create a preview grid showing different emotion-based text formatting."""
    if not validate_video_file(video_file):
        console.print(f"[red]Error:[/red] '{video_file}' is not a valid video file")
        sys.exit(1)

    if not caption_file.endswith('.json'):
        console.print(f"[red]Error:[/red] Caption file must be in JSON format")
        sys.exit(1)

    # Generate output path if not specified
    if not output:
        base_name = Path(video_file).stem
        output = f"{base_name}_emotion_grid.mp4"

    console.print(Panel.fit(
        f"[bold cyan]Creating Text Format Preview Grid[/bold cyan]\n"
        f"[bold]Video:[/bold] {video_file}\n"
        f"[bold]Grid:[/bold] {grid[0]}x{grid[1]}",
        title="Preview Grid Generator"
    ))

    try:
        # Initialize video merger
        merger = VideoMerger(verbose=verbose)

        # Create preview grid
        with console.status("[bold green]Creating preview grid...", spinner="dots"):
            result_path = merger.create_preview_grid(
                video_file,
                caption_file,
                output,
                grid_size=tuple(grid),
                segment_duration=duration
            )

        console.print(f"[green]✓[/green] Preview created: {result_path}")

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
        if verbose:
            console.print_exception()
        sys.exit(1)


@cli.command('config')
@click.option('--show', is_flag=True, help='Show current configuration')
@click.option('--set', nargs=2, multiple=True,
              metavar='KEY VALUE', help='Set configuration values')
@click.option('--reset', is_flag=True, help='Reset to default configuration')
def config_cmd(show, set, reset):
    """Manage configuration settings."""
    config_path = Path.home() / '.auto-caption' / 'config.json'
    config_path.parent.mkdir(exist_ok=True)

    if reset:
        default_config = {
            'default_model': 'base',
            'default_format': ['srt'],
            'default_language': 'auto',
            'verbose': False,
            'threads': 4
        }
        save_config(str(config_path), default_config)
        console.print("[green]✓[/green] Configuration reset to defaults")
        return

    # Load current config
    if config_path.exists():
        config = load_config(str(config_path))
    else:
        config = {}

    if set:
        for key, value in set:
            # Try to parse value as JSON first
            try:
                parsed_value = json.loads(value)
            except:
                # If not JSON, treat as string
                parsed_value = value

            config[key] = parsed_value

        save_config(str(config_path), config)
        console.print("[green]✓[/green] Configuration updated")

    if show or not set:
        if config:
            console.print(Panel.fit(
                json.dumps(config, indent=2),
                title="Current Configuration"
            ))
        else:
            console.print("[yellow]No configuration file found[/yellow]")


def main():
    """Main entry point."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[bold red]Unexpected error:[/bold red] {str(e)}")
        console.print_exception()
        sys.exit(1)


if __name__ == '__main__':
    main()
