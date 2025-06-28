"""
Command-line interface for Auto-Caption tool.
"""

import os
import sys
import json
import glob
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
from .utils import validate_video_file, get_output_filename, load_config, save_config

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
              type=click.Choice(['srt', 'vtt', 'txt', 'json']),
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
@click.pass_context
def generate(ctx, video_file, model, format, output, language, task, verbose,
             temperature, threads):
    """Generate captions for a single video file."""
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
        f"[bold]Format(s):[/bold] {', '.join(format)}",
        title="Auto-Caption"
    ))
    
    try:
        # Initialize caption generator
        with console.status("[bold green]Loading Whisper model...", spinner="dots"):
            generator = CaptionGenerator(
                model_name=model,
                language=language if language != 'auto' else None,
                task=task,
                verbose=verbose,
                threads=threads
            )
        
        # Generate captions
        console.print("[green]✓[/green] Model loaded successfully")
        
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
                progress_callback=lambda p: progress.update(task_id, completed=p)
            )
        
        console.print("[green]✓[/green] Caption generation complete")
        
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
              type=click.Choice(['srt', 'vtt', 'txt', 'json']),
              help='Output format(s)')
@click.option('--language', '-l', default=None,
              help='Language code or "auto" for detection')
@click.option('--recursive', '-r', is_flag=True,
              help='Process subdirectories recursively')
@click.option('--skip-existing', is_flag=True,
              help='Skip videos that already have captions')
@click.option('--threads', type=int, default=None,
              help='Number of threads to use')
@click.pass_context
def batch(ctx, directory, pattern, model, format, language, recursive, 
          skip_existing, threads):
    """Process multiple video files in batch."""
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
            threads=threads
        )
    
    console.print("[green]✓[/green] Model loaded successfully\n")
    
    # Process each file
    successful = 0
    failed = 0
    
    for i, video_file in enumerate(video_files, 1):
        console.print(f"[bold]Processing {i}/{len(video_files)}:[/bold] {video_file.name}")
        
        try:
            # Generate captions
            result = generator.generate(str(video_file))
            
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


@cli.command()
def version():
    """Display version information."""
    console.print(Panel.fit(
        f"[bold cyan]Auto-Caption[/bold cyan] v{__version__}\n"
        f"Automatic video captioning using OpenAI Whisper\n\n"
        f"[dim]Python {sys.version.split()[0]} | Click | Whisper | Rich[/dim]",
        title="Version Info"
    ))


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