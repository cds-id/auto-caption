# Auto-Caption Emotion Styling Guide

This guide explains how different emotions affect the visual presentation of captions, including font selection, positioning, sizing, and color schemes.

## Overview

Auto-Caption uses three custom fonts and dynamic positioning to match the emotional context of speech:

- **MADE AVENUE**: Clean, modern, friendly - used for positive and neutral emotions
- **CINEMATOGRAFICA**: Bold, dramatic, impactful - used for intense and sarcastic emotions  
- **ALMOST TEXTUAL**: Soft, handwritten, emotional - used for vulnerable and introspective emotions

## Emotion-to-Style Mappings

### 😊 HAPPY
- **Font**: MADE AVENUE
- **Position**: 60px higher than baseline (top-center alignment)
- **Size**: 130% of base size
- **Color**: Bright sunny yellow (#FFFF00) with orange glow
- **Effects**: Bold text, sharp focus, strong shadow for "pop"
- **Characteristics**: Elevated position suggests uplift and joy, larger size for enthusiasm

### 😢 SAD
- **Font**: ALMOST TEXTUAL
- **Position**: 80px lower than baseline (bottom-center)
- **Size**: 75% of base size
- **Color**: Deep ocean blue (#6699FF) with heavy shadow
- **Effects**: Italic text, 2.0x blur (tearful effect), thin outline
- **Characteristics**: Weighted down position, smaller size suggesting diminished energy

### 😠 ANGRY
- **Font**: CINEMATOGRAFICA
- **Position**: 40px higher than baseline (center screen)
- **Size**: 150% of base size (largest)
- **Color**: Pure intense red (#FF0000) with red glow effect
- **Effects**: Bold text, 2.0x outline thickness, very sharp focus
- **Characteristics**: Dominating center position for confrontation, maximum size for intensity

### 🤩 EXCITED
- **Font**: MADE AVENUE
- **Position**: 70px higher than baseline (top-center)
- **Size**: 135% of base size
- **Color**: Electric magenta (#FF00FF) with pink glow
- **Effects**: Bold text, dynamic shadow, slight bounce
- **Characteristics**: Very high position for bouncing energy, large size for enthusiasm

### 😨 FEARFUL
- **Font**: ALMOST TEXTUAL
- **Position**: 50px lower than baseline (bottom-left corner)
- **Size**: 80% of base size
- **Color**: Pale ghostly blue (#AACCEE) with deep shadows
- **Effects**: 1.8x blur (shaky/unclear), thin outline
- **Characteristics**: Corner position suggesting hiding, small size for shrinking away

### 😏 SARCASTIC
- **Font**: CINEMATOGRAFICA
- **Position**: 10px higher than baseline (right side)
- **Size**: 110% of base size
- **Color**: Sharp lime green (#CCFF33) with dark edge
- **Effects**: Bold + italic (for attitude), sharp focus
- **Characteristics**: Side position for "sideways" delivery, moderate size increase

### 😰 ANXIOUS
- **Font**: ALMOST TEXTUAL
- **Position**: 30px lower than baseline (center)
- **Size**: 85% of base size
- **Color**: Nervous pale green (#CCDDBB) with blurred shadow
- **Effects**: 1.5x blur, thin outline, slight shake
- **Characteristics**: Center position (frozen in place), smaller size for uncertainty

### 🤔 CONTEMPLATIVE
- **Font**: ALMOST TEXTUAL
- **Position**: 30px higher than baseline (top-center)
- **Size**: 90% of base size
- **Color**: Thoughtful lavender (#96B8DD) with soft shadow
- **Effects**: Italic text, 1.2x blur (soft focus)
- **Characteristics**: Elevated position for "looking up" thinking, slightly reduced size

### 😐 NEUTRAL
- **Font**: MADE AVENUE
- **Position**: Baseline (bottom-center)
- **Size**: 100% (standard)
- **Color**: Pure white (#FFFFFF) with standard shadow
- **Effects**: Normal text, standard outline
- **Characteristics**: Default positioning and sizing

## Style Intensity Levels

The styling can be adjusted with three intensity levels:

### SUBTLE (85% effect)
- Smaller position adjustments
- More modest size changes
- Lighter color variations
- Minimal special effects

### MEDIUM (100% effect)
- Standard position adjustments as listed above
- Normal size variations
- Full color schemes
- Moderate special effects

### INTENSE (135% effect)
- Exaggerated position changes
- Maximum size variations
- Vibrant colors with strong glows
- Heavy special effects and animations

## Platform-Specific Adjustments

### TikTok
- Base font size: 56px
- Margin: 150px from bottom
- Extra outline for visibility
- 120% size multiplier

### Instagram Reels
- Base font size: 52px
- Margin: 120px from bottom
- Moderate outline
- 110% size multiplier

### YouTube Shorts
- Base font size: 54px
- Margin: 100px from bottom
- Standard outline
- 115% size multiplier

## Visual Effects by Emotion Type

### Positive Emotions (Happy, Excited)
- Bright, warm colors
- Higher screen position
- Larger text size
- Strong shadows/glows
- Sharp, clear focus

### Negative Emotions (Sad, Fearful, Anxious)
- Cool, muted colors
- Lower screen position
- Smaller text size
- Soft/blurred edges
- Subtle shadows

### Intense Emotions (Angry, Sarcastic)
- Bold, saturated colors
- Center or side positions
- Larger text size
- Thick outlines
- Sharp contrast

### Contemplative Emotions (Contemplative, Nostalgic)
- Soft, pastel colors
- Variable positions
- Medium text size
- Gentle effects
- Soft focus

## Implementation Examples

### ASS Subtitle Style Definition
```ass
Style: Emotion_happy,MADE AVENUE,72,&H0000FFFF,&H0000D4FF,&H00FF6600,&H60FF8800,1,0,0,0,115,120,0,0,1,3.9,1.95,8,10,10,70,1
```

### Position Calculation
```python
# Base margin from bottom: 80px
# Happy emotion adjustment: -60px (moves up)
# Final position: 20px from bottom (higher on screen)
margin_v = base_margin + emotion_adjustment
```

### Size Calculation
```python
# Base font size: 48px
# Happy size multiplier: 1.3
# Intensity multiplier (intense): 1.35
# Final size: 48 * 1.3 * 1.35 = 84px
font_size = base_size * emotion_multiplier * intensity_multiplier
```

## Best Practices

1. **Consistency**: Maintain consistent styling for the same emotion throughout a video
2. **Readability**: Ensure contrast between text and background remains high
3. **Context**: Consider the video content when choosing intensity levels
4. **Transitions**: Smooth transitions between different emotional states
5. **Platform**: Adjust base settings according to target platform requirements

## Color Reference (Hex to ASS Format)

| Emotion | Hex Color | ASS Format | Description |
|---------|-----------|------------|-------------|
| Happy | #FFFF00 | &H0000FFFF | Bright yellow |
| Sad | #6699FF | &H00FF9966 | Ocean blue |
| Angry | #FF0000 | &H000000FF | Pure red |
| Excited | #FF00FF | &H00FF00FF | Magenta |
| Fearful | #AACCEE | &H00EECCAA | Pale blue |
| Sarcastic | #33FFCC | &H0033FFCC | Lime green |
| Anxious | #CCDDBB | &H00BBDDCC | Pale green |
| Contemplative | #96B8DD | &H00DDB896 | Lavender |

Note: ASS format uses AABBGGRR (Alpha, Blue, Green, Red) color notation.