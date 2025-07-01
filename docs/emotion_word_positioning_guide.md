# Emotion-Based Word Positioning Guide

## Overview

Auto-Caption's emotion-based word positioning feature dynamically adjusts the position, size, and animation of individual words based on the detected emotion in the video. This creates visually engaging captions that reflect the emotional context of the content.

## How It Works

When word-by-word mode is enabled with emotion detection, each word is:
1. Analyzed for its emotional context
2. Positioned on screen based on the dominant emotion
3. Sized according to emotional intensity
4. Animated with emotion-appropriate effects

## Emotion Positioning Map

### Visual Layout Guide

```
┌─────────────────────────────────────────────┐
│                                             │ ← EXCITED/HAPPY (Very High)
│         "AMAZING!" "WOW!"                   │    y: -60 to -120px
│                                             │    Large size (1.4x)
├─────────────────────────────────────────────┤
│     "Great" "Love"                          │ ← HAPPY (High)
│                                             │    y: -50 to -100px
│                                             │    Large size (1.3x)
├─────────────────────────────────────────────┤
│   "Think" "Maybe"                           │ ← CONTEMPLATIVE
│                                             │    y: -20 to -40px
│                                             │    Normal size (0.9x)
├─────────────────────────────────────────────┤
│ "Really?" "Sure..."                         │ ← SARCASTIC
│                                             │    y: -10 to -20px
│                                             │    Tilted angle
├─────────────────────────────────────────────┤
│ Normal text appears here                    │ ← NEUTRAL (Center)
│                                             │    y: 0px
│                                             │    Normal size (1.0x)
├─────────────────────────────────────────────┤
│         "worried" "nervous"                 │ ← ANXIOUS
│                                             │    y: 0 to +30px
│                                             │    Smaller (0.85x)
├─────────────────────────────────────────────┤
│             "scared" "help"                 │ ← FEARFUL (Low)
│                                             │    y: +30 to +70px
│                                             │    Small size (0.7x)
├─────────────────────────────────────────────┤
│                 "sad" "alone"               │ ← SAD (Very Low)
│                                             │    y: +50 to +100px
│                                             │    Small size (0.8x)
└─────────────────────────────────────────────┘
```

## Emotion-Specific Effects

### 😊 HAPPY
- **Position**: Higher on screen (-50 to -100px)
- **Size**: 1.3x larger
- **Spacing**: Wide (30px spread)
- **Animation**: Bouncy, upward movement
- **Rotation**: Slight playful tilt (-5° to +5°)
- **Special**: Yellow color tint, glow effect on emphasized words

### 😢 SAD
- **Position**: Lower on screen (+50 to +100px)
- **Size**: 0.8x smaller
- **Spacing**: Tight (10px spread)
- **Animation**: Slow fade-in, drooping
- **Rotation**: None
- **Special**: Blue color tint, reduced opacity

### 😠 ANGRY
- **Position**: Upper-middle (-30 to -60px)
- **Size**: 1.5x larger (aggressive)
- **Spacing**: Very wide (50px spread)
- **Animation**: Aggressive shake, slam-in effect
- **Rotation**: Chaotic (-10° to +10°)
- **Special**: Red color tint, strong shadows

### 🤩 EXCITED
- **Position**: Very high (-60 to -120px)
- **Size**: 1.4x larger
- **Spacing**: Wide (40px spread)
- **Animation**: Bouncing, energetic
- **Rotation**: Dynamic (-15° to +15°)
- **Special**: Magenta color, pulsing effect

### 😰 FEARFUL
- **Position**: Lower (+30 to +70px)
- **Size**: 0.7x smaller (shrinking)
- **Spacing**: Very tight (5px spread)
- **Animation**: Trembling, fade-in
- **Rotation**: Slight shake (-2° to +2°)
- **Special**: Pale colors, clustered together

### 😟 ANXIOUS
- **Position**: Middle-low (0 to +30px)
- **Size**: 0.85x smaller
- **Spacing**: Scattered (20px spread)
- **Animation**: Nervous jitter
- **Rotation**: Small shake (-3° to +3°)
- **Special**: Random positioning offsets

### 😏 SARCASTIC
- **Position**: Slightly elevated (-10 to -20px)
- **Size**: 1.1x larger
- **Spacing**: Moderate (25px spread)
- **Animation**: Slide-in with tilt
- **Rotation**: One-sided tilt (-8° to 0°)
- **Special**: Lime green tint, italic style

### 😐 NEUTRAL
- **Position**: Center (0px)
- **Size**: Normal (1.0x)
- **Spacing**: Standard (20px spread)
- **Animation**: Simple fade-in
- **Rotation**: None
- **Special**: White color, standard styling

### 🤔 CONTEMPLATIVE
- **Position**: Slightly elevated (-20 to -40px)
- **Size**: 0.9x slightly smaller
- **Spacing**: Moderate (15px spread)
- **Animation**: Slow fade-in
- **Rotation**: Minimal (-1° to +1°)
- **Special**: Lavender tint, soft edges

## Word Emphasis Rules

Words are automatically emphasized based on emotion:

- **HAPPY**: "amazing", "wonderful", "great", "love", "awesome"
- **SAD**: "sorry", "miss", "lost", "alone", "hurt"
- **ANGRY**: "hate", "stupid", "wrong", "never", "stop"
- **EXCITED**: "wow", "omg", "yes", "incredible", "unbelievable"
- **FEARFUL**: "scared", "afraid", "worry", "danger", "help"
- **SARCASTIC**: "really", "totally", "obviously", "clearly", "sure"

Emphasized words receive:
- 1.3x additional size boost
- Stronger color intensity
- Special effects (glow, shadow, etc.)

## Animation Styles

### Available Animations

1. **TYPEWRITER**
   - Words appear one by one at their designated positions
   - Clean and simple

2. **BOUNCE_IN**
   - Words bounce from above/below based on emotion
   - Happy emotions bounce higher
   - Sad emotions have smaller bounces

3. **WAVE**
   - Words appear in wave pattern
   - Wave amplitude varies by emotion
   - Happy: Large waves (25px)
   - Sad: Small waves (8px)

4. **POP_IN**
   - Words scale from 0 to emotion-based size
   - Energetic for happy emotions
   - Subtle for sad emotions

5. **EMPHASIS**
   - Special effects based on emotion:
   - ANGRY: Shake effect
   - HAPPY: Jump effect
   - Default: Standard appearance

## Usage Examples

### Basic Word-by-Word with Emotion Positioning

```bash
auto-caption generate video.mp4 \
  --word-by-word \
  --emotion-mode auto \
  --style-intensity intense \
  --word-animation bounce_in
```

### Custom Animation Style

```python
from auto_caption import WordTimingProcessor, WordAnimationStyle

processor = WordTimingProcessor(
    animation_style=WordAnimationStyle.WAVE,
    words_per_second=3.0,
    emphasis_duration_multiplier=1.5
)
```

### Platform-Specific Adjustments

Different platforms have different safe zones:

- **TikTok**: Words avoid top 20% and bottom 20% of screen
- **Instagram**: Words stay within 85% safe zone
- **YouTube Shorts**: Similar to TikTok with 90% safe zone

## Configuration Options

### Style Intensity Levels

1. **SUBTLE**
   - Minimal position variation
   - Size multiplier: 0.85x of base
   - Reduced animation effects

2. **MEDIUM**
   - Standard position variation
   - Size multiplier: 1.0x of base
   - Normal animation effects

3. **INTENSE**
   - Maximum position variation
   - Size multiplier: 1.35x of base
   - Enhanced animation effects

### Custom Position Overrides

You can customize positioning for specific use cases:

```python
# Custom emotion position styles
CUSTOM_POSITIONS = {
    EmotionCategory.HAPPY: {
        "y_offset_range": (-100, -150),  # Even higher
        "x_spread": 50,                  # Wider spread
        "size_base": 1.5,                # Larger
        "rotation_range": (-10, 10),     # More rotation
    }
}
```

## Best Practices

1. **Content Type Considerations**
   - Educational: Use subtle positioning
   - Entertainment: Use intense positioning
   - Emotional content: Match intensity to content

2. **Readability**
   - Ensure contrast between text and background
   - Don't overuse rotation for important information
   - Keep emphasized words to 20-30% of total

3. **Performance**
   - Word-by-word rendering is more CPU intensive
   - Use preview mode for testing
   - Batch process multiple videos overnight

## Troubleshooting

### Words Overlapping
- Reduce x_spread values
- Decrease size multipliers
- Use fewer emphasized words

### Words Off-Screen
- Check video resolution settings
- Adjust y_offset_range values
- Use platform-specific settings

### Animation Too Fast/Slow
- Adjust words_per_second parameter
- Modify animation_delay values
- Check video FPS settings

## Visual Examples

### Happy Emotion Pattern
```
        "This"    "is"     "AMAZING!"
          ↗         ↗          ↗
      (bounce)  (bounce)  (big bounce)
```

### Sad Emotion Pattern
```
                     "feel"
                       ↓
                    "so"
                      ↓
                   "alone..."
                      ↓
               (slow fall)
```

### Angry Emotion Pattern
```
"THIS" ← → "IS" ← → "WRONG!"
  ↕         ↕         ↕
(shake)  (shake)  (shake)
```

## Advanced Features

### Multi-Emotion Segments
When a video segment contains emotion transitions, words smoothly animate between position styles.

### Emphasized Word Detection
- ALL CAPS words get automatic emphasis
- Words ending with "!" in happy/excited contexts
- Emotion-specific keywords

### Custom Styling Hooks
You can add custom CSS-like styling through the ASS format:
- Blur effects for dreamy emotions
- Shadow effects for dramatic emotions
- Gradient colors for transitions