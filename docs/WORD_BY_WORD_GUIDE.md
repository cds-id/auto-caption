# Word-by-Word Caption Guide

This guide explains how to use Auto-Caption's word-by-word caption feature, which displays text one word at a time synchronized with speech for enhanced engagement and readability.

## Overview

Word-by-word captions display each word individually as it's spoken, creating a more dynamic and engaging viewing experience. This feature is particularly effective for:

- Social media content (TikTok, Instagram Reels, YouTube Shorts)
- Educational videos
- Music videos and lyric displays
- Emotional or dramatic content
- Accessibility-focused content

## Benefits

1. **Better Synchronization**: Words appear exactly when spoken
2. **Increased Engagement**: Viewers focus on each word as it appears
3. **Improved Comprehension**: Easier to follow along with speech
4. **Enhanced Emotion**: Emphasizes key words based on emotion
5. **Modern Aesthetic**: Popular style for social media platforms

## Basic Usage

### Command Line

Enable word-by-word captions with the `--word-by-word` flag:

```bash
# Basic word-by-word generation
auto-caption generate video.mp4 --word-by-word

# With emotion detection and styling
auto-caption generate video.mp4 --word-by-word --emotion-mode auto

# Specify animation style
auto-caption generate video.mp4 --word-by-word --word-animation pop_in

# Adjust reading speed (words per second)
auto-caption generate video.mp4 --word-by-word --words-per-second 4.0
```

### Python API

```python
from auto_caption import CaptionGenerator
from auto_caption.word_timing import WordTimingProcessor, WordAnimationStyle

# Initialize generator with word timestamps enabled
generator = CaptionGenerator(model_name="base")

# Generate captions with word timestamps
result = generator.generate(
    "video.mp4",
    word_timestamps=True  # Enable word-level timestamps from Whisper
)

# Process into word-by-word timing
processor = WordTimingProcessor(
    animation_style=WordAnimationStyle.TYPEWRITER,
    words_per_second=3.0
)

word_segments = processor.process_segments(result['segments'])
```

## Animation Styles

### TYPEWRITER
- **Description**: Words appear sequentially at their natural timing
- **Best for**: Natural speech flow, documentaries, tutorials
- **Command**: `--word-animation typewriter`

### FADE_IN
- **Description**: Words fade in with smooth opacity transition
- **Best for**: Contemplative, emotional content
- **Command**: `--word-animation fade_in`

### POP_IN
- **Description**: Words scale from 0 to full size with bounce
- **Best for**: Energetic, exciting content
- **Command**: `--word-animation pop_in`

### SLIDE_IN
- **Description**: Words slide in from the side
- **Best for**: Dynamic presentations, action content
- **Command**: `--word-animation slide_in`

### BOUNCE_IN
- **Description**: Words bounce in with elastic effect
- **Best for**: Fun
, playful content
- **Command**: `--word-animation bounce_in`

### WAVE
- **Description**: Words appear in a wave pattern
- **Best for**: Musical content, rhythmic speech
- **Command**: `--word-animation wave`

### KARAOKE
- **Description**: Highlights words as they're spoken
- **Best for**: Sing-along content, emphasis on timing
- **Command**: `--word-animation karaoke`

### EMPHASIS
- **Description**: Key words appear with special emphasis
- **Best for**: Educational content, important messages
- **Command**: `--word-animation emphasis`

## Emotion-Based Word Emphasis

When combined with emotion detection, certain words are automatically emphasized based on the detected emotion:

### Happy/Excited
- Emphasized words: "amazing", "wonderful", "great", "love", "awesome"
- Effect: Larger size, brighter colors, bounce animation

### Sad
- Emphasized words: "sorry", "miss", "lost", "alone", "hurt"
- Effect: Smaller size, blue tint, slow fade

### Angry
- Emphasized words: "hate", "stupid", "wrong", "never", "stop"
- Effect: Larger size, red color, aggressive pop

### Sarcastic
- Emphasized words: "really", "totally", "obviously", "sure"
- Effect: Italic style, green tint, sideways slide

## Technical Details

### Word Timing Calculation

When Whisper doesn't provide word-level timestamps, the system calculates timing based on:

1. **Word Length**: Longer words get more duration
2. **Punctuation**: Pauses after periods, commas
3. **Emphasis**: Important words get extra time
4. **Total Duration**: Distributed across segment time

### Timing Parameters

```python
WordTimingProcessor(
    words_per_second=3.0,        # Average reading speed
    min_word_duration=0.15,      # Minimum time per word (seconds)
    max_word_duration=0.8,       # Maximum time per word (seconds)
    emphasis_duration_multiplier=1.3,  # Extra time for emphasized words
    punctuation_pause=0.2        # Pause after punctuation (seconds)
)
```

## Output Formats

### ASS Subtitles (Recommended)

Word-by-word captions work best with ASS format for precise timing and styling:

```bash
auto-caption generate video.mp4 --word-by-word --format ass
```

### JSON Format

Includes detailed word timing data:

```json
{
  "segments": [...],
  "word_segments": [
    {
      "segment_index": 0,
      "words": [
        {
          "word": "Hello",
          "start_time": 0.0,
          "end_time": 0.3,
          "duration": 0.3,
          "emotion": "happy",
          "is_emphasized": false
        }
      ]
    }
  ],
  "word_by_word": true,
  "word_animation": "typewriter"
}
```

## Best Practices

### 1. Choose Appropriate Animation Style
- Match animation to content mood
- Use subtle styles for serious content
- Use dynamic styles for energetic content

### 2. Optimize Reading Speed
- Default: 3 words/second
- Slower (2-2.5) for complex content
- Faster (3.5-4) for simple, energetic content

### 3. Platform Optimization
```bash
# TikTok - fast-paced, dynamic
auto-caption generate video.mp4 --word-by-word --word-animation pop_in --platform tiktok

# Instagram - balanced, stylish
auto-caption generate video.mp4 --word-by-word --word-animation fade_in --platform instagram

# YouTube Shorts - clear, readable
auto-caption generate video.mp4 --word-by-word --word-animation typewriter --platform youtube_shorts
```

### 4. Combine with Emotion Detection
```bash
# Full emotion-aware word-by-word captions
auto-caption generate video.mp4 \
  --word-by-word \
  --emotion-mode auto \
  --style-intensity intense \
  --word-animation emphasis
```

## Examples

### Example 1: Educational Content
```bash
auto-caption generate lecture.mp4 \
  --word-by-word \
  --word-animation typewriter \
  --words-per-second 2.5 \
  --format ass
```

### Example 2: Music Video
```bash
auto-caption generate music_video.mp4 \
  --word-by-word \
  --word-animation karaoke \
  --emotion-mode auto \
  --style-intensity intense
```

### Example 3: Social Media Reel
```bash
auto-caption generate reel.mp4 \
  --word-by-word \
  --word-animation pop_in \
  --platform instagram \
  --emotion-mode auto
```

### Example 4: Batch Processing
```bash
auto-caption batch /videos \
  --word-by-word \
  --word-animation wave \
  --format ass \
  --recursive
```

## Performance Considerations

1. **Processing Time**: Word-by-word processing adds ~10-20% to generation time
2. **File Size**: Word-by-word ASS files are larger due to individual word entries
3. **Whisper Model**: Larger models provide better word-level accuracy
4. **Memory Usage**: Minimal additional memory required

## Troubleshooting

### Words Appearing Too Fast/Slow
Adjust the reading speed:
```bash
--words-per-second 2.5  # Slower
--words-per-second 4.0  # Faster
```

### Words Not Synchronized
Use a larger Whisper model for better accuracy:
```bash
--model medium --word-by-word
```

### Animation Not Smooth
Ensure video has sufficient frame rate (30+ fps recommended)

### Emphasized Words Not Showing
Check emotion detection is enabled:
```bash
--emotion-mode auto --word-animation emphasis
```

## Advanced Usage

### Custom Word Emphasis
```python
from auto_caption.word_timing import create_word_emphasis_rules

# Create custom emphasis rules
rules = create_word_emphasis_rules(
    emotion=EmotionCategory.HAPPY,
    custom_words=["birthday", "celebration", "party"]
)
```

### Programmatic Word Timing
```python
#
 Fine-tune word timing
processor = WordTimingProcessor()
word_segments = processor.process_segments(segments)

# Adjust specific words
for word in word_segments[0].words:
    if word.word.lower() in ["important", "critical"]:
        word.duration *= 1.5  # Make important words last longer
        word.is_emphasized = True
```

### Export Word Timing Data
```python
# Export for custom rendering
word_data = []
for segment in word_segments:
    for word in segment.words:
        word_data.append({
            "text": word.word,
            "start": word.start_time,
            "end": word.end_time,
            "style": {
                "emotion": word.emotion.value,
                "emphasized": word.is_emphasized
            }
        })

# Save for external use
with open("word_timing.json", "w") as f:
    json.dump(word_data, f, indent=2)
```

## Integration with Video Editing

The word-by-word ASS files can be used with any video editor that supports ASS subtitles:

### FFmpeg
```bash
ffmpeg -i video.mp4 -vf "subtitles=captions_word_by_word.ass" output.mp4
```

### DaVinci Resolve
Import the ASS file as a subtitle track

### Adobe Premiere Pro
Use the Subtitles panel to import ASS format

## Future Enhancements

Planned improvements for word-by-word captions:

1. **Custom Word Animations**: Define per-word animation effects
2. **Multi-Line Support**: Better handling of long sentences
3. **Word Grouping**: Option to show phrases instead of individual words
4. **Speed Ramping**: Variable speed within segments
5. **Visual Effects**: Particle effects, glows, shadows per word