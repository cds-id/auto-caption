
# Auto-Caption: Emotion-Adaptive Caption Generation for Short-Form Videos

An advanced AI-powered tool that generates emotionally-adaptive captions for short-form video content. Unlike traditional speech-to-text tools, Auto-Caption analyzes facial expressions to understand emotional context and formats captions with appropriate punctuation, capitalization, and emphasis to match not just what is said, but *how* it's said.

## 🎯 Project Vision

In the era of TikTok, Instagram Reels, and YouTube Shorts, captions have evolved beyond mere accessibility tools. They've become an integral part of the creative language, used to:
- Guide viewer interpretation through punctuation and emphasis
- Convey emotional tone through text formatting
- Match the speaker's emotional state
- Enhance narrative meaning without visual distractions

However, existing captioning tools fall short by focusing solely on literal transcription accuracy, missing the emotional nuances that make content engaging. Auto-Caption bridges this gap by using face-based emotion detection to adaptively format text with appropriate punctuation, capitalization, and emphasis patterns.

## 🌟 Key Features

### Emotion-Adaptive Formatting
- **Face-Based Detection**: Analyzes facial expressions every 0.5 seconds for accurate emotion tracking
- **Adaptive Text Formatting**: Formats captions with emotion-appropriate punctuation and capitalization
- **Nuanced Expression**: Same words formatted differently based on detected emotion (e.g., "great" vs "GREAT!" vs "great...")

### Multi-Modal Analysis
- **Primary Face Detection**: Analyzes facial expressions with 85% weight for emotion detection
- **Supplementary Audio Analysis**: Uses voice tone as 15% supplementary data
- **Temporal Tracking**: Maintains emotion continuity across frames for smooth transitions

### Intelligent Text Formatting
- **Emotion-Based Punctuation**: Adds appropriate punctuation (!!!, ..., ?!) based on emotion
- **Adaptive Capitalization**: Uses CAPS, Mixed Case, or lowercase to convey tone
- **Natural Emphasis**: Emphasizes key words based on emotional intensity

### Advanced Subtitle Support
- **ASS Format**: Rich subtitle formatting with emotion-based colors, fonts, and effects
- **Style Presets**: Platform-optimized subtitle styles for TikTok, Instagram, YouTube
- **FFmpeg Integration**: Reliable subtitle burning with high performance
- **Fallback Support**: OpenCV renderer when ffmpeg is unavailable

### Platform Optimization
- **Format Support**: Exports for TikTok, Instagram Reels, YouTube Shorts
- **Style Templates**: Pre-configured styles for different content types
- **Batch Processing**: Handle multiple videos with consistent styling

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- FFmpeg (for video processing)
- Git

### Step 1: Clone and Setup

```bash
# Clone the repository
git clone https://github.com/cds-id/auto-caption.git
cd auto-caption

# Run the automated setup script
chmod +x setup.sh
./setup.sh

# Or manual setup
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### Step 2: Download Models (Optional)

```bash
# Download emotion detection models
auto-caption download-models --type emotion

# Pre-download a specific Whisper model
auto-caption download-model base
```

## 📖 Step-by-Step Usage Guide

### Basic Caption Generation

#### Step 1: Generate Simple Captions
```bash
# Basic transcription without emotion
auto-caption generate video.mp4

# Output: video.srt (subtitle file)
```

#### Step 2: Add Emotion Detection
```bash
# Automatic emotion detection
auto-caption generate video.mp4 --emotion-mode auto

# The tool will:
# 1. Analyze facial expressions
# 2. Detect voice emotions
# 3. Apply appropriate styling
```

#### Step 3: Manual Emotion Override
```bash
# Force a specific emotion style
auto-caption generate video.mp4 --emotion-mode manual --emotion happy --style-intensity intense
```

### Advanced Caption Styling

#### Generate Styled Subtitles (ASS Format)
```bash
# Export emotion-styled ASS subtitles
auto-caption export-subtitles video.json --format ass --platform tiktok

# Export both ASS and SRT formats
auto-caption export-subtitles video.json --format both

# Custom resolution for subtitle scaling
auto-caption export-subtitles video.json --video-file video.mp4 --style-intensity intense
```

#### Merge Captions with Video
```bash
# Merge using ASS subtitles (recommended)
auto-caption merge video.mp4 video.json --subtitle-format ass

# Platform-specific optimization
auto-caption merge video.mp4 video.json --platform tiktok --quality high

# Quick preview with lower quality
auto-caption merge video.mp4 video.json --preview

# Available emotions:
# happy, sad, angry, sarcastic, anxious, neutral, excited, contemplative
```

### Advanced Styling Options

#### Platform-Specific Generation
```bash
# Optimize for TikTok
auto-caption generate video.mp4 --emotion-mode auto --platform tiktok

# Optimize for Instagram Reels
auto-caption generate video.mp4 --emotion-mode auto --platform instagram

# Optimize for YouTube Shorts
auto-caption generate video.mp4 --emotion-mode auto --platform youtube_shorts
```

#### Style Intensity Levels
```bash
# Subtle styling (minimal changes)
auto-caption generate video.mp4 --emotion-mode auto --style-intensity subtle

# Medium styling (balanced)
auto-caption generate video.mp4 --emotion-mode auto --style-intensity medium

# Intense styling (maximum effect)
auto-caption generate video.mp4 --emotion-mode auto --style-intensity intense
```

### 🎬 Creating Videos with Styled Captions

#### Step 1: Generate Caption Data
```bash
# Generate JSON file with emotion data
auto-caption generate video.mp4 --emotion-mode auto --format json -o captions.json
```

#### Step 2: Merge Captions with Video
```bash
# Use OpenCV renderer (recommended - more reliable)
auto-caption merge video.mp4 captions.json --use-opencv --output final_video.mp4

# Or use MoviePy renderer (requires ImageMagick)
auto-caption merge video.mp4 captions.json --output final_video.mp4
```

#### Step 3: Platform-Specific Video Output
```bash
# Create TikTok-optimized video with captions
auto-caption merge video.mp4 captions.json --use-opencv --platform tiktok

# Create Instagram Reels version
auto-caption merge video.mp4 captions.json --use-opencv --platform instagram

# Preview mode (lower quality, faster processing)
auto-caption merge video.mp4 captions.json --use-opencv --preview
```

### Complete Workflow Examples

#### Example 1: TikTok Comedy Video
```bash
# Step 1: Generate captions with sarcastic tone
auto-caption generate funny_video.mp4 \
  --emotion-mode manual \
  --emotion sarcastic \
  --style-intensity intense \
  --format json \
  -o funny_captions.json

# Step 2: Create video with styled captions
auto-caption merge funny_video.mp4 funny_captions.json \
  --use-opencv \
  --platform tiktok \
  --output funny_final.mp4
```

#### Example 2: Emotional Story Video
```bash
# Step 1: Auto-detect emotions throughout the video
auto-caption generate story.mp4 \
  --emotion-mode auto \
  --format json \
  -o story_captions.json

# Step 2: Create video with emotion-adaptive captions
auto-caption merge story.mp4 story_captions.json \
  --use-opencv \
  --quality high \
  --output story_with_captions.mp4
```

#### Example 3: Batch Processing
```bash
# Process multiple videos in a folder
auto-caption batch /path/to/videos \
  --pattern "*.mp4" \
  --emotion-mode auto \
  --format json,srt

# Then merge all videos with captions
for video in /path/to/videos/*.mp4; do
  json_file="${video%.mp4}.json"
  output="${video%.mp4}_captioned.mp4"
  auto-caption merge "$video" "$json_file" --use-opencv --output "$output"
done
```

### 🎨 Emotion Analysis Tools

#### Analyze Video Emotions
```bash
# Get detailed emotion analysis
auto-caption analyze-emotion video.mp4 -o emotion_report.json

# View emotion timeline
auto-caption analyze-emotion video.mp4 --verbose
```

#### Preview Different Styles
```bash
# Create a grid showing different emotion styles
auto-caption preview-grid video.mp4 captions.json \
  --grid 2 2 \
  --output emotion_preview.mp4
```

### Output Format Options

```bash
# Generate multiple formats
auto-caption generate video.mp4 --format srt,vtt,txt,json

# Format descriptions:
# - srt: Standard subtitle format
# - vtt: WebVTT format for web players
# - txt: Plain text with timestamps
# - json: Complete data including emotions and styling
```

### Troubleshooting Common Issues

#### ImageMagick Policy Error
If you see ImageMagick errors when merging:
```bash
# Use the OpenCV renderer instead
auto-caption merge video.mp4 captions.json --use-opencv
```

#### Model Download Issues
```bash
# Manually download models
auto-caption download-models --type all

# Check available models
auto-caption list-models
```

#### Performance Optimization
```bash
# Use smaller model for faster processing
auto-caption generate video.mp4 --model tiny

# Use preview mode for testing
auto-caption merge video.mp4 captions.json --preview --use-opencv
```

## 🧠 How It Works

### 1. Multi-Modal Analysis
- **Video Analysis**: Extract visual features using computer vision models
- **Audio Processing**: Analyze speech patterns, tone, and prosody
- **Context Integration**: Combine signals to understand overall emotional context

### 2. Emotion Detection
- **Facial Expression Recognition**: Detect micro-expressions and emotional states
- **Voice Emotion Analysis**: Identify emotional cues in speech patterns
- **Scene Context**: Consider visual context and environment

### 3. Caption Generation
- **Emotion-Aware Language Model**: Generate text that matches detected emotion
- **Style Transfer**: Apply appropriate linguistic style and tone
- **Creative Enhancement**: Add emphasis, punctuation, and formatting

### 4. Output Optimization
- **Platform-Specific Formatting**: Optimize for target platform requirements
- **Timing Synchronization**: Ensure perfect sync with emotional beats
- **Visual Suggestions**: Recommend text effects and animations

## 📊 Supported Emotions & Visual Effects

### Emotion Categories

| Emotion | Text Style | Visual Effects | Use Case |
|---------|------------|----------------|----------|
| **Happy** | Uppercase emphasis, exclamation marks | Bounce animation, confetti, bright colors | Celebration, positive content |
| **Sad** | Lowercase, ellipses | Fade/drip animation, rain effect, blue tones | Emotional stories, melancholic content |
| **Angry** | ALL CAPS, multiple exclamation | Shake animation, fire effect, red colors | Rants, intense reactions |
| **Sarcastic** | MiXeD CaSe, tildes | Tilt/wave animation, eye roll effect | Comedy, ironic content |
| **Excited** | CAPS, vibrant punctuation | Bounce/sparkle, fireworks, orange/yellow | Announcements, energetic content |
| **Anxious** | Stuttering, question marks | Jitter/shake, dark tones | Nervous content, suspense |
| **Neutral** | Standard formatting | Simple fade, minimal effects | Information, tutorials |

### Complete Emotion List
- **Primary**: Happy, Sad, Angry, Fearful, Surprised, Disgusted, Neutral
- **Complex**: Sarcastic, Ironic, Contemplative, Excited, Melancholic, Anxious, Confident, Confused
- **Content**: Motivational, Humorous, Dramatic, Casual, Professional, Romantic, Nostalgic

### Visual Effect Animations
- **Text Animations**: fade, slide, bounce, shake, pop, wave, typewriter, glow, sparkle, fire, drip
- **Background Effects**: confetti, stars, hearts, rain, snow, fire, lightning, sparkles, blur
- **Color Schemes**: Automatically matched to emotion with customizable palettes

## 🛠️ Technical Architecture

### Core Components

1. **Emotion Detection Module**
   - Visual emotion recognition (CNN-based)
   - Audio emotion analysis (RNN/Transformer-based)
   - Multi-modal fusion network

2. **Caption Generation Module**
   - Emotion-conditioned language model
   - Style transfer mechanisms
   - Context-aware text generation

3. **Integration Pipeline**
   - Real-time processing capabilities
   - Scalable batch processing
   - API endpoints for integration

### Technology Stack

- **Deep Learning**: PyTorch, Transformers, OpenCV
- **Audio Processing**: Whisper, librosa, pyAudioAnalysis
- **Vision Models**: Face recognition, emotion detection CNNs
- **NLP Models**: GPT-based generation with emotion conditioning
- **Framework**: FastAPI for API, Click for CLI

## 📈 Use Cases

### Content Creators
- Generate engaging captions that match video mood
- Save time on manual caption editing
- Maintain consistent style across content

### Digital Marketers
- Create emotionally resonant ad captions
- A/B test different emotional approaches
- Scale content production

### Accessibility Advocates
- Provide context-rich captions for better understanding
- Include emotional cues for hearing-impaired viewers
- Enhance overall content accessibility

## 🔧 Configuration

Create a configuration file at `~/.auto-caption/config.json`:

```json
{
  "emotion_detection": {
    "visual_weight": 0.6,
    "audio_weight": 0.4,
    "threshold": 0.7
  },
  "caption_generation": {
    "style_intensity": "medium",
    "platform_defaults": {
      "tiktok": {
        "max_length": 100,
        "style": "casual"
      },
      "instagram": {
        "max_length": 125,
        "style": "engaging"
      }
    }
  },
  "output": {
    "default_format": ["srt", "json"],
    "include_emotion_data": true,
    "include_suggestions": true
  }
}
```

## 🎨 Caption Style Examples

### Same Text, Different Emotions

**Original**: "I can't believe this happened"

| Emotion | Styled Output | Visual Effect |
|---------|--------------|---------------|
| **Happy** | "I CAN'T BELIEVE this happened!!!! 🎉" | Bounce + confetti |
| **Sad** | "i can't believe... this happened..." | Fade + rain |
| **Angry** | "I CAN'T BELIEVE THIS HAPPENED!!!!" | Shake + fire |
| **Sarcastic** | "I cAn'T bELiEvE this happened 🙄" | Tilt + eye roll |
| **Anxious** | "I can't... can't believe this happened???" | Jitter + pulse |
| **Excited** | "***I CAN'T BELIEVE*** THIS HAPPENED!!! 🔥" | Vibrate + sparkles |

### Platform-Specific Styling

#### TikTok Style
- Larger text (1.3x scale)
- Bottom 80% positioning
- High contrast colors
- Quick animations

#### Instagram Reels
- Medium text (1.2x scale)  
- Bottom 75% positioning
- Aesthetic color palettes
- Smooth transitions

#### YouTube Shorts
- Balanced text (1.15x scale)
- Bottom 85% positioning
- Clear readability
- Professional appearance

### Real-World Examples

#### Comedy Skit
```bash
# Input: "That was supposed to be easy"
# Output with sarcasm: "That was SUPPOSED to be 'easy' 🙄"
# Effect: Tilt animation with air quotes gesture
```

#### Motivational Speech
```bash
# Input: "You can do this"
# Output with motivation: "YOU CAN DO THIS! 💪"
# Effect: Power rise animation with impact effect
```

#### Sad Story
```bash
# Input: "I miss those days"
# Output with sadness: "i miss those days..."
# Effect: Slow fade with falling rain
```

## 🚧 Roadmap

### Phase 1: Core Functionality ✅
- Basic emotion detection
- Simple caption generation
- CLI interface

### Phase 2: Advanced Features 🚧
- Multi-modal emotion fusion
- Platform-specific optimization
- Batch processing capabilities

### Phase 3: Creative Tools 📅
- Real-time preview
- Custom style creation
- Effect recommendations

### Phase 4: Integration 📅
- API development
- Plugin system
- Third-party integrations

## 🛡️ Best Practices

### For Content Creators
1. **Test Different Emotions**: Try multiple emotion modes to find what fits best
2. **Platform Optimization**: Always specify your target platform for best results
3. **Preview First**: Use preview mode to test before final rendering
4. **Batch Processing**: Process multiple videos at once for consistency

### For Developers
1. **Use JSON Output**: Get full emotion data for custom processing
2. **OpenCV Renderer**: More reliable than MoviePy for cross-platform compatibility
3. **Custom Styling**: Override visual suggestions in the JSON before merging
4. **API Integration**: Use the emotion detection separately from caption generation

### Performance Tips
- Use `tiny` or `base` models for faster processing
- Enable preview mode for testing
- Process videos in batches for efficiency
- Use the `--threads` option to optimize CPU usage

## 🤝 Contributing

We welcome contributions! Areas where you can help:

1. **Emotion Models**: Improve emotion detection accuracy
2. **Language Styles**: Add support for more languages and cultural contexts  
3. **Platform Features**: Add support for new platforms
4. **Visual Effects**: Create new animation styles and effects
5. **UI/UX**: Develop web interface or mobile app

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

## 📚 Research & References

This project builds upon research in:
- Affective Computing
- Multi-modal Emotion Recognition
- Emotion-Conditioned Text Generation
- Cross-Modal Learning

Key papers and resources:
- [Emotion Recognition in Context](https://arxiv.org/abs/xxxxx)
- [Multi-modal Sentiment Analysis](https://arxiv.org/abs/xxxxx)
- [Affective Text Generation](https://arxiv.org/abs/xxxxx)

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI Whisper for speech recognition foundation
- Emotion recognition research community
- Content creators who inspired this project
- Open source contributors

---

**Note**: This project addresses the growing need for emotionally-intelligent content creation tools in the age of short-form video. By bridging the gap between technical accuracy and creative expression, we aim to empower creators to produce more engaging and accessible content at scale.
