# Quick Start Guide: Emotion-Aware Caption Generation

Welcome to Auto-Caption! This guide will help you get started with generating emotionally-intelligent captions for your short-form videos in minutes.

## 🚀 5-Minute Setup

### 1. Install Auto-Caption

```bash
# Clone the repository
git clone https://github.com/yourusername/auto-caption.git
cd auto-caption

# Run the automated setup
chmod +x setup.sh
./setup.sh

# Activate the environment
source venv/bin/activate
```

### 2. Download Required Models

```bash
# Download emotion detection models (first-time setup)
auto-caption download-models --type emotion

# Download speech recognition model
auto-caption download-model base
```

### 3. Generate Your First Emotion-Aware Caption

```bash
# Basic emotion-aware captioning
auto-caption generate my_video.mp4 --emotion-mode auto
```

## 🎭 Understanding Emotion Modes

Auto-Caption can detect and apply various emotional styles to your captions:

### Automatic Emotion Detection
```bash
# Let AI detect emotions from video content
auto-caption generate video.mp4 --emotion-mode auto
```

### Manual Emotion Override
```bash
# Force a specific emotional style
auto-caption generate video.mp4 --emotion happy --intensity medium
```

### Available Emotions
- **Primary**: happy, sad, angry, fearful, surprised, neutral
- **Complex**: sarcastic, anxious, excited, contemplative
- **Content Moods**: motivational, humorous, dramatic, casual

## 📱 Platform-Specific Generation

### TikTok
```bash
auto-caption generate video.mp4 --platform tiktok --emotion-mode auto

# Output characteristics:
# - Shorter, punchier captions
# - Trend-aware formatting (POV:, etc.)
# - Optimized timing for quick cuts
```

### Instagram Reels
```bash
auto-caption generate video.mp4 --platform instagram --emotion-mode auto

# Output characteristics:
# - Slightly longer captions allowed
# - Emphasis on visual appeal
# - Strategic emoji placement
```

### YouTube Shorts
```bash
auto-caption generate video.mp4 --platform youtube_shorts --emotion-mode auto

# Output characteristics:
# - Full-length captions supported
# - SEO-friendly formatting
# - Clear punctuation
```

## 🎨 Styling Examples

### Same Text, Different Emotions

Original text: "I can't believe this happened"

**Happy Style:**
```
"I CAN'T BELIEVE this
