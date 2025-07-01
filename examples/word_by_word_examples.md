# Word-by-Word Caption Examples

This guide provides comprehensive examples for generating word-by-word captions with emotion-aware styling using the Auto-Caption tool.

## Table of Contents
- [Basic Word-by-Word Generation](#basic-word-by-word-generation)
- [Platform-Specific Examples](#platform-specific-examples)
- [Animation Styles](#animation-styles)
- [Emotion Detection Examples](#emotion-detection-examples)
- [Advanced Configurations](#advanced-configurations)
- [Batch Processing](#batch-processing)

## Basic Word-by-Word Generation

### Simple word-by-word caption generation
```bash
# Generate word-by-word captions with default settings
auto-caption generate video.mp4 --word-by-word

# Generate with specific output format
auto-caption generate video.mp4 --word-by-word --format ass -o output_wordbyword.ass

# Generate with word timing information in JSON
auto-caption generate video.mp4 --word-by-word --format json -o word_timing_data.json
```

### Adjusting reading speed
```bash
# Slower reading speed (2 words per second)
auto-caption generate video.mp4 --word-by-word --words-per-second 2.0

# Faster reading speed (4 words per second)
auto-caption generate video.mp4 --word-by-word --words-per-second 4.0

# Dynamic speed based on content complexity
auto-caption generate video.mp4 --word-by-word --words-per-second 3.5
```

## Platform-Specific Examples

### TikTok optimized word-by-word captions
```bash
# TikTok with word-by-word animation
auto-caption generate video.mp4 \
  --word-by-word \
  --platform tiktok \
  --word-animation bounce_in \
  --style-intensity intense

# TikTok with emotion detection
auto-caption generate video.mp4 \
  --word-by-word \
  --platform tiktok \
  --emotion-mode auto \
  --word-animation wave
```

### Instagram Reels word-by-word captions
```bash
# Instagram with fade-in animation
auto-caption generate video.mp4 \
  --word-by-word \
  --platform instagram \
  --word-animation fade_in \
  --style-intensity medium

# Instagram with pop effect
auto-caption generate video.mp4 \
  --word-by-word \
  --platform instagram \
  --word-animation pop_in \
  --words-per-second 2.5
```

### YouTube Shorts word-by-word captions
```bash
# YouTube Shorts with typewriter effect
auto-caption generate video.mp4 \
  --word-by-word \
  --platform youtube_shorts \
  --word-animation typewriter \
  --style-intensity subtle

# YouTube Shorts with karaoke style
auto-caption generate video.mp4 \
  --word-by-word \
  --platform youtube_shorts \
  --word-animation karaoke
```

## Animation Styles

### Typewriter effect (default)
```bash
# Classic typewriter - words appear one by one
auto-caption generate video.mp4 \
  --word-by-word \
  --word-animation typewriter
```

### Fade in effect
```bash
# Words fade in smoothly
auto-caption generate video.mp4 \
  --word-by-word \
  --word-animation fade_in \
  --words-per-second 2.8
```

### Pop in effect
```bash
# Words pop/scale in dynamically
auto-caption generate video.mp4 \
  --word-by-word \
  --word-animation pop_in \
  --style-intensity intense
```

### Slide in effect
```bash
# Words slide in from the side
auto-caption generate video.mp4 \
  --word-by-word \
  --word-animation slide_in
```

### Bounce in effect
```bash
# Words bounce in with energy
auto-caption generate video.mp4 \
  --word-by-word \
  --word-animation bounce_in \
  --platform tiktok
```

### Wave effect
```bash
# Words appear in a wave pattern
auto-caption generate video.mp4 \
  --word-by-word \
  --word-animation wave \
  --words-per-second 3.0
```

### Random appearance
```bash
# Words appear randomly
auto-caption generate video.mp4 \
  --word-by-word \
  --word-animation random
```

### Karaoke style
```bash
# Highlight words as they're spoken
auto-caption generate video.mp4 \
  --word-by-word \
  --word-animation karaoke \
  --style-intensity medium
```

### Emphasis animation
```bash
# Key words appear with extra emphasis
auto-caption generate video.mp4 \
  --word-by-word \
  --word-animation emphasis \
  --emotion-mode auto
```

## Emotion Detection Examples

### Auto emotion detection with word-by-word
```bash
# Detect emotions and apply word-by-word styling
auto-caption generate video.mp4 \
  --word-by-word \
  --emotion-mode auto \
  --word-animation wave \
  --style-intensity intense

# With specific platform optimization
auto-caption generate video.mp4 \
  --word-by-word \
  --emotion-mode auto \
  --platform tiktok \
  --word-animation bounce_in
```

### Manual emotion override
```bash
# Happy emotion with bouncy words
auto-caption generate video.mp4 \
  --word-by-word \
  --emotion-mode manual \
  --emotion happy \
  --word-animation bounce_in

# Sad emotion with slow fade-in
auto-caption generate video.mp4 \
  --word-by-word \
  --emotion-mode manual \
  --emotion sad \
  --word-animation fade_in \
  --words-per-second 2.0

# Excited emotion with pop effect
auto-caption generate video.mp4 \
  --word-by-word \
  --emotion-mode manual \
  --emotion excited \
  --word-animation pop_in \
  --style-intensity intense
```

## Advanced Configurations

### Multi-language support
```bash
# Spanish word-by-word captions
auto-caption generate video.mp4 \
  --word-by-word \
  --language es \
  --word-animation typewriter

# French with emotion detection
auto-caption generate video.mp4 \
  --word-by-word \
  --language fr \
  --emotion-mode auto \
  --word-animation fade_in
```

### Custom model selection
```bash
# Use large model for better accuracy
auto-caption generate video.mp4 \
  --word-by-word \
  --model large \
  --word-animation wave

# Use tiny model for faster processing
auto-caption generate video.mp4 \
  --word-by-word \
  --model tiny \
  --word-animation typewriter
```

### Multiple output formats
```bash
# Generate both ASS and SRT formats
auto-caption generate video.mp4 \
  --word-by-word \
  --format ass \
  --format srt \
  --word-animation bounce_in
```

### Merge with video
```bash
# Generate word-by-word captions and merge with video
auto-caption generate video.mp4 \
  --word-by-word \
  --word-animation wave \
  --format json -o captions.json

# Then merge
auto-caption merge video.mp4 captions.json \
  --output final_video.mp4 \
  --quality high
```

## Batch Processing

### Process multiple videos with word-by-word
```bash
# Process all videos in a directory
auto-caption batch ./videos \
  --word-by-word \
  --word-animation fade_in \
  --platform tiktok \
  --emotion-mode auto

# Process with specific pattern
auto-caption batch ./videos \
  --pattern "*.mp4" \
  --word-by-word \
  --word-animation bounce_in \
  --style-intensity medium

# Recursive processing
auto-caption batch ./videos \
  --recursive \
  --word-by-word \
  --word-animation wave \
  --platform instagram
```

## Complete Examples

### TikTok viral video with emotion-aware word-by-word
```bash
auto-caption generate tiktok_video.mp4 \
  --word-by-word \
  --platform tiktok \
  --emotion-mode auto \
  --word-animation bounce_in \
  --style-intensity intense \
  --words-per-second 3.5 \
  --format ass \
  --output tiktok_viral_captions.ass
```

### Instagram Reel with stylized word timing
```bash
auto-caption generate instagram_reel.mp4 \
  --word-by-word \
  --platform instagram \
  --emotion-mode auto \
  --word-animation pop_in \
  --style-intensity medium \
  --words-per-second 3.0 \
  --format json \
  --output instagram_word_data.json

# Then merge with video
auto-caption merge instagram_reel.mp4 instagram_word_data.json \
  --output instagram_final.mp4 \
  --quality high \
  --subtitle-format ass
```

### YouTube Shorts with karaoke-style highlighting
```bash
auto-caption generate youtube_short.mp4 \
  --word-by-word \
  --platform youtube_shorts \
  --word-animation karaoke \
  --style-intensity subtle \
  --words-per-second 2.8 \
  --format ass \
  --output youtube_karaoke.ass
```

### Educational content with emphasis on key words
```bash
auto-caption generate lecture_video.mp4 \
  --word-by-word \
  --word-animation emphasis \
  --emotion-mode manual \
  --emotion neutral \
  --style-intensity medium \
  --words-per-second 2.5 \
  --format ass
```

## Tips and Best Practices

1. **Reading Speed**: Adjust `--words-per-second` based on content:
   - Educational/Technical: 2.0-2.5 words/second
   - Entertainment: 3.0-3.5 words/second
   - Fast-paced content: 3.5-4.0 words/second

2. **Animation Selection**:
   - `typewriter`: Classic, good for most content
   - `fade_in`: Smooth, professional look
   - `pop_in`: Energetic, great for upbeat content
   - `bounce_in`: Fun, perfect for TikTok/Reels
   - `wave`: Dynamic, good for music videos
   - `karaoke`: Best for sing-along or emphasis
   - `emphasis`: Highlights important words automatically

3. **Platform Optimization**:
   - TikTok: Use `bounce_in` or `pop_in` with `intense` styling
   - Instagram: Use `fade_in` or `wave` with `medium` styling
   - YouTube Shorts: Use `typewriter` or `karaoke` with `subtle` to `medium` styling

4. **Emotion Detection**:
   - Always use `--emotion-mode auto` for dynamic content
   - Manual override for consistent styling throughout video
   - Combine with appropriate animation style for best results

5. **Performance**:
   - Use smaller models (`tiny`, `base`) for faster processing
   - Larger models (`medium`, `large`) for better accuracy
   - Batch processing is more efficient for multiple videos

## Troubleshooting

### Words appearing too fast/slow
```bash
# Adjust reading speed
auto-caption generate video.mp4 \
  --word-by-word \
  --words-per-second 2.5  # Slower than default 3.0
```

### Words overlapping or cut off
```bash
# Use platform-specific safe zones
auto-caption generate video.mp4 \
  --word-by-word \
  --platform tiktok  # Automatically adjusts for TikTok UI
```

### Emotion detection not working well
```bash
# Try manual emotion setting
auto-caption generate video.mp4 \
  --word-by-word \
  --emotion-mode manual \
  --emotion happy
```

### Animation too subtle/intense
```bash
# Adjust style intensity
auto-caption generate video.mp4 \
  --word-by-word \
  --style-intensity subtle  # or medium, intense
```
