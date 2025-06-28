# Auto-Caption

A powerful command-line tool for automatically generating captions/subtitles from video files using OpenAI's Whisper speech recognition model.

## Features

- 🎥 Extract audio from video files and generate accurate captions
- 🌍 Support for multiple languages (95+ languages)
- 📝 Multiple output formats: SRT, VTT, TXT, JSON
- 🎯 Various Whisper model sizes for speed/accuracy trade-offs
- 🔧 Customizable parameters (language detection, timestamps, etc.)
- 📊 Progress bars and detailed logging
- 🎨 Rich CLI interface with colored output
- 🚀 Easy-to-use command-line interface

## Requirements

- Python 3.8 or higher
- ffmpeg (for audio/video processing)
- ~1-10GB of free disk space (depending on the Whisper model size)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/cds-id/auto-caption.git
cd auto-caption
```

### 2. Run the setup script

The easiest way to set up the project is using the provided setup script:

```bash
chmod +x setup.sh
./setup.sh
```

This script will:
- Create a Python virtual environment
- Install all required dependencies
- Set up the CLI tool for development

### 3. Manual installation (alternative)

If you prefer to set up manually:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install the package
pip install -e .
```

### 4. Install ffmpeg

Make sure ffmpeg is installed on your system:

- **Ubuntu/Debian**: `sudo apt-get install ffmpeg`
- **macOS**: `brew install ffmpeg`
- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html)
- **Fedora**: `sudo dnf install ffmpeg`
- **Arch Linux**: `sudo pacman -S ffmpeg`

## Usage

### Basic Usage

Generate captions for a video file:

```bash
auto-caption generate video.mp4
```

### Advanced Options

```bash
# Specify output format
auto-caption generate video.mp4 --format srt

# Use a specific Whisper model (tiny, base, small, medium, large)
auto-caption generate video.mp4 --model medium

# Specify language (auto-detect by default)
auto-caption generate video.mp4 --language en

# Custom output file
auto-caption generate video.mp4 --output my_captions.srt

# Generate multiple formats at once
auto-caption generate video.mp4 --format srt --format vtt --format txt

# Verbose output for debugging
auto-caption generate video.mp4 --verbose
```

### Batch Processing

Process multiple videos at once:

```bash
# Process all MP4 files in a directory
auto-caption batch /path/to/videos --pattern "*.mp4"

# Process with specific settings
auto-caption batch /path/to/videos --model small --format srt --format vtt
```

### Available Commands

- `generate`: Generate captions for a single video
- `batch`: Process multiple videos
- `list-models`: Show available Whisper models
- `download-model`: Pre-download a specific model
- `version`: Show version information

### Whisper Models

| Model  | Parameters | English-only | Multilingual | Required VRAM | Relative Speed |
|--------|------------|--------------|--------------|---------------|----------------|
| tiny   | 39 M       | ✓            | ✓            | ~1 GB         | ~32x           |
| base   | 74 M       | ✓            | ✓            | ~1 GB         | ~16x           |
| small  | 244 M      | ✓            | ✓            | ~2 GB         | ~6x            |
| medium | 769 M      | ✓            | ✓            | ~5 GB         | ~2x            |
| large  | 1550 M     | ✗            | ✓            | ~10 GB        | 1x             |

Choose a model based on your needs:
- `tiny` or `base`: Fast processing, good for quick drafts
- `small` or `medium`: Balanced performance and accuracy
- `large`: Best accuracy, especially for challenging audio

## Output Formats

### SRT (SubRip Subtitle)
Standard subtitle format supported by most video players.

```srt
1
00:00:00,000 --> 00:00:03,000
Hello, welcome to our video.

2
00:00:03,500 --> 00:00:07,000
Today we'll be discussing auto-captioning.
```

### VTT (WebVTT)
Web Video Text Tracks format, ideal for HTML5 video.

```vtt
WEBVTT

00:00:00.000 --> 00:00:03.000
Hello, welcome to our video.

00:00:03.500 --> 00:00:07.000
Today we'll be discussing auto-captioning.
```

### TXT (Plain Text)
Simple text format with timestamps.

```txt
[00:00:00.000 --> 00:00:03.000] Hello, welcome to our video.
[00:00:03.500 --> 00:00:07.000] Today we'll be discussing auto-captioning.
```

### JSON
Structured format with detailed information.

```json
{
  "text": "Hello, welcome to our video. Today we'll be discussing auto-captioning.",
  "segments": [
    {
      "id": 0,
      "seek": 0,
      "start": 0.0,
      "end": 3.0,
      "text": "Hello, welcome to our video.",
      "tokens": [50364, 2425, 11, 2928, 281, 527, 960, 13, 50514],
      "temperature": 0.0,
      "avg_logprob": -0.2761423448
    }
  ],
  "language": "en"
}
```

## Configuration

You can create a configuration file at `~/.auto-caption/config.json`:

```json
{
  "default_model": "small",
  "default_format": ["srt", "vtt"],
  "default_language": "auto",
  "verbose": false,
  "threads": 4
}
```

## Troubleshooting

### Common Issues

1. **"No module named 'whisper'"**
   - Make sure you've activated the virtual environment: `source venv/bin/activate`

2. **"ffmpeg not found"**
   - Install ffmpeg using your system's package manager

3. **Out of memory errors**
   - Use a smaller model (tiny, base, or small)
   - Process shorter video segments

4. **Slow processing**
   - Use a smaller model for faster processing
   - Ensure you have CUDA installed for GPU acceleration (if available)

### Performance Tips

- For GPU acceleration, install PyTorch with CUDA support
- Process videos in batches during off-peak hours
- Use the `--threads` option to control CPU usage
- Consider splitting long videos into segments

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for the amazing speech recognition model
- [ffmpeg](https://ffmpeg.org/) for audio/video processing
- [Click](https://click.palletsprojects.com/) for the CLI framework
- [Rich](https://rich.readthedocs.io/) for beautiful terminal output

## Support

If you encounter any issues or have questions:

1. Check the [FAQ](docs/FAQ.md)
2. Search existing [issues](https://github.com/cds-id/auto-caption/issues)
3. Create a new issue with detailed information

---

Made with ❤️ by the Auto-Caption Team
