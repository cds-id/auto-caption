# ASS Subtitle Guide for Auto-Caption

## Overview

Auto-Caption now supports ASS (Advanced SubStation Alpha) subtitles, providing rich formatting capabilities for emotion-based caption styling. This guide covers everything you need to know about using ASS subtitles with Auto-Caption.

## Why ASS Subtitles?

### Advantages over SRT
- **Rich Formatting**: Colors, fonts, sizes, and effects
- **Emotion-Based Styling**: Different styles for different emotions
- **Better Performance**: Subtitle burning with ffmpeg is faster than frame-by-frame rendering
- **Platform Optimization**: Tailored styles for TikTok, Instagram, YouTube
- **Professional Quality**: Industry-standard format used in anime and professional video

### When to Use ASS vs SRT
- **ASS**: When you need emotion-based styling, color coding, or special effects
- **SRT**: When you need maximum compatibility or simple, unstyled subtitles

## Generating ASS Subtitles

### Basic Generation

```bash
# Generate captions with emotion detection and export as ASS
auto-caption generate video.mp4 --emotion-mode auto --format json
auto-caption export-subtitles video.json --format ass
```

### Platform-Specific Styles

```bash
# TikTok optimized (larger text, higher position)
auto-caption export-subtitles video.json --format ass --platform tiktok

# Instagram Reels (medium size, safe zones)
auto-caption export-subtitles video.json --format ass --platform instagram

# YouTube Shorts (balanced styling)
auto-caption export-subtitles video.json --format ass --platform youtube_shorts
```

### Style Intensity Levels

```bash
# Subtle - minimal emotion differentiation
auto-caption export-subtitles video.json --style-intensity subtle

# Medium - balanced emotion styling (default)
auto-caption export-subtitles video.json --style-intensity medium

# Intense - maximum emotion expression
auto-caption export-subtitles video.json --style-intensity intense
```

## Merging ASS Subtitles with Video

### Basic Merge

```bash
# Merge using ASS subtitles (recommended)
auto-caption merge video.mp4 captions.json --subtitle-format ass
```

### Quality Options

```bash
# High quality (slower, best output)
auto-caption merge video.mp4 captions.json --quality high

# Medium quality (balanced)
auto-caption merge video.mp4 captions.json --quality medium

# Preview mode (fast, lower quality)
auto-caption merge video.mp4 captions.json --preview
```

### Complete Pipeline Example

```bash
# 1. Generate captions with emotion detection
auto-caption generate input.mp4 --emotion-mode auto --format json -o captions.json

# 2. Export styled ASS subtitles
auto-caption export-subtitles captions.json --format ass --platform tiktok --style-intensity medium

# 3. Merge with video
auto-caption merge input.mp4 captions.json --platform tiktok --quality high -o final_video.mp4
```

## Understanding ASS Styles

### Emotion Color Mapping

| Emotion | Primary Color | Font Style | Effect |
|---------|--------------|------------|---------|
| Happy | Bright Yellow (#FFE033) | Bold, Rounded | Bounce animation |
| Sad | Muted Blue-Gray (#C4A484) | Italic | Fade, smaller |
| Angry | Bright Red (#FF2020) | Bold, Impact | Shake, larger |
| Excited | Magenta (#FF00FF) | Bold | Pop effect |
| Sarcastic | Cyan (#99FFFF) | Italic, Mono | Slide |
| Anxious | Pale Yellow-Green (#CCCC99) | Regular | Subtle shake |
| Neutral | White (#FFFFFF) | Regular | Simple fade |
| Contemplative | Soft Beige (#E6D4B3) | Serif | Slow fade |

### ASS File Structure

```ass
[Script Info]
Title: Your Video Title
PlayResX: 1920  ; Video width
PlayResY: 1080  ; Video height

[V4+ Styles]
; Each emotion gets its own style definition
Style: Emotion_happy,Arial Rounded MT Bold,56,&H00FFE033,&H00FFB300,&H00000000,&H80000000...

[Events]
; Timed subtitles with emotion-specific styles
Dialogue: 0,0:00:00.00,0:00:03.00,Emotion_happy,,0,0,0,,Welcome to our video!
```

## Customization

### Using Custom Fonts

Create a custom font mapping in your code:

```python
from auto_caption import ASSGenerator, Platform, StyleIntensity

generator = ASSGenerator(
    platform=Platform.TIKTOK,
    style_intensity=StyleIntensity.MEDIUM,
    custom_fonts={
        "happy": "Comic Sans MS",
        "sad": "Georgia",
        "angry": "Impact"
    }
)
```

### Manual ASS Editing

You can manually edit the generated ASS files in:
- **Aegisub** (recommended) - Professional subtitle editor
- **Subtitle Edit** - Free, cross-platform
- Any text editor (for simple changes)

### Common Modifications

1. **Change Colors**: Edit PrimaryColour in styles
   - Format: &HAABBGGRR (Alpha, Blue, Green, Red in hex)
   
2. **Adjust Positioning**: Modify MarginV for vertical position
   - Higher values = lower on screen
   
3. **Font Size**: Change Fontsize value
   - Scale based on video resolution

## Troubleshooting

### FFmpeg Not Found

```bash
# Install ffmpeg
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### Subtitles Not Showing

1. Check video player subtitle support
2. Ensure ASS file is in correct format
3. Verify subtitle timing matches video
4. Try burning subtitles directly:
   ```bash
   ffmpeg -i video.mp4 -vf "subtitles=subtitles.ass" output.mp4
   ```

### Performance Issues

- Use `--quality medium` or `--preview` for faster processing
- Process videos in batches during off-peak hours
- Consider hardware acceleration with `--use-hardware-accel`

## Advanced Usage

### Batch Processing

```bash
# Process multiple videos with consistent styling
for video in *.mp4; do
    auto-caption generate "$video" --emotion-mode auto --format json
    auto-caption merge "$video" "${video%.mp4}.json" --platform tiktok
done
```

### Creating Preview Grids

```bash
# Show different emotion styles in a grid
auto-caption preview-grid video.mp4 captions.json --grid 2 2
```

### API Usage

```python
from auto_caption import ASSGenerator, VideoMerger
from auto_caption import Platform, StyleIntensity

# Generate ASS
generator = ASSGenerator(
    platform=Platform.TIKTOK,
    style_intensity=StyleIntensity.INTENSE
)
ass_path = generator.generate_ass_file(caption_data, "output.ass")

# Merge with video
merger = VideoMerger(platform=Platform.TIKTOK)
merger.merge_with_video("video.mp4", caption_data, "output.mp4")
```

## Best Practices

1. **Always Preview First**: Use `--preview` to quickly check styling
2. **Match Platform**: Use platform-specific settings for optimal display
3. **Test on Target Device**: Verify subtitles look good on actual phones/tablets
4. **Keep Originals**: Always keep original video and caption files
5. **Batch Similar Content**: Process videos with similar emotion profiles together

## Examples

### TikTok Drama Content
```bash
auto-caption generate drama.mp4 --emotion-mode auto --style-intensity intense
auto-caption merge drama.mp4 drama.json --platform tiktok --quality high
```

### Instagram Educational Content
```bash
auto-caption generate tutorial.mp4 --emotion-mode auto --style-intensity subtle
auto-caption merge tutorial.mp4 tutorial.json --platform instagram
```

### YouTube Shorts Comedy
```bash
auto-caption generate comedy.mp4 --emotion-mode auto --style-intensity medium
auto-caption merge comedy.mp4 comedy.json --platform youtube_shorts
```

## Resources

- [ASS Subtitle Specification](http://www.tcax.org/docs/ass-specs.htm)
- [Aegisub Documentation](http://docs.aegisub.org/)
- [FFmpeg Subtitle Options](https://ffmpeg.org/ffmpeg-filters.html#subtitles-1)
- [Auto-Caption GitHub](https://github.com/cds-id/auto-caption)