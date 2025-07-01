# Smart Object-Aware Caption Positioning

Auto-Caption's smart positioning feature uses advanced computer vision to detect faces, people, and other important objects in your videos, automatically positioning captions to avoid blocking key visual elements.

## Overview

Traditional captioning tools place text at fixed positions (usually bottom-center), often blocking important content like faces, products, or existing on-screen text. Smart positioning solves this by:

- 🎯 **Face Detection & Tracking** - Identifies and tracks faces throughout the video
- 👥 **Person Detection** - Detects full bodies to avoid blocking people
- 📝 **Text Detection** - Avoids placing captions over existing text
- 🎨 **Object Recognition** - Identifies products, pets, food, and other key objects
- 🔄 **Motion Prediction** - Predicts object movement to keep captions clear
- ✨ **Dynamic Positioning** - Adjusts caption placement frame-by-frame

## How It Works

### 1. Object Detection Pipeline

The system analyzes your video using multiple detection methods:

```python
# Initialize object detector
detector = ObjectDetector(
    enable_face_detection=True,    # MediaPipe face detection
    enable_object_detection=True,  # YOLO object detection
    enable_text_detection=True,    # Text region detection
    tracking_enabled=True,         # Object tracking across frames
    model_size="medium"           # Balance of speed vs accuracy
)
```

### 2. Safe Zone Creation

Detected objects are converted into "safe zones" with importance scores:

- **Faces**: Importance 1.0 (highest priority)
- **People**: Importance 0.9
- **Text**: Importance 0.85
- **Hands**: Importance 0.8 (for tutorials)
- **Products**: Importance 0.7
- **Pets**: Importance 0.65

### 3. Position Optimization

The position optimizer finds optimal caption placement:

```python
optimizer = PositionOptimizer(
    video_resolution=(1920, 1080),
    platform=Platform.TIKTOK,
    default_strategy=PositioningStrategy.DYNAMIC
)
```

## Usage

### Command Line

Enable smart positioning with the `--smart-positioning` flag:

```bash
# Basic usage with smart positioning
auto-caption generate video.mp4 --smart-positioning

# Word-by-word captions with face avoidance
auto-caption generate video.mp4 --word-by-word --smart-positioning --avoid-faces

# Specify detection model size
auto-caption generate video.mp4 --smart-positioning --object-detection-model large
```

### Python API

```python
from auto_caption import CaptionGenerator, WordTimingProcessor
from auto_caption.object_detection import ObjectDetector
from auto_caption.caption_styler import Platform

# Initialize with object detection
generator = CaptionGenerator(
    model_name="base",
    enable_object_detection=True
)

# Generate captions with smart positioning
result = generator.generate(
    "video.mp4",
    word_timestamps=True,
    enable_smart_positioning=True
)

# Process word-by-word with object awareness
word_processor = WordTimingProcessor(
    enable_object_detection=True,
    platform=Platform.TIKTOK
)

word_segments = word_processor.process_segments(
    result['segments'],
    detection_results=result.get('detection_results')
)
```

## Positioning Strategies

### Dynamic (Default)
Continuously adjusts position based on detected objects:
```python
strategy=PositioningStrategy.DYNAMIC
```

### Flow Around
Captions flow around objects like water:
```python
strategy=PositioningStrategy.FLOW_AROUND
```

### Sides
Places captions on left or right side:
```python
strategy=PositioningStrategy.SIDES
```

### Corner
Uses corner positions when center is blocked:
```python
strategy=PositioningStrategy.CORNER
```

## Platform-Specific Optimization

Different platforms have different safe zones:

### TikTok
- Top 15% reserved for user info
- Bottom 20% for controls
- Prefers higher caption placement

### Instagram Reels
- Top 12% for UI
- Bottom 18% for controls
- Slightly lower caption preference

### YouTube Shorts
- Top 10% for UI
- Bottom 15% for controls
- More flexible positioning

## Advanced Features

### Face Tracking with Prediction

Track faces and predict their movement:

```python
from auto_caption.object_detection import FaceTracker

tracker = FaceTracker(
    enable_speaking_detection=True,
    prediction_horizon=15  # Predict 15 frames ahead
)

# Get safe zones for upcoming caption
safe_zones = tracker.get_safe_zones_for_caption(
    caption_duration=2.0,
    fps=30.0
)
```

### Custom Object Importance

Adjust importance scores for specific use cases:

```python
detector = ObjectDetector()

# Increase importance for products in a product review
detector.OBJECT_IMPORTANCE[ObjectType.PRODUCT] = 0.95

# Lower importance for background objects
detector.OBJECT_IMPORTANCE[ObjectType.VEHICLE] = 0.3
```

### Visualization Tools

Visualize detected objects and safe zones:

```python
# Create visualization video
from examples.smart_positioning_demo import visualize_object_detection

visualize_object_detection(
    "input_video.mp4",
    "output_visualization.mp4",
    detector
)
```

## Performance Optimization

### Model Size Selection

Choose model size based on your needs:

- **Small**: Fast, lower accuracy, good for real-time
- **Medium**: Balanced (default)
- **Large**: Best accuracy, slower processing

```bash
# Fast processing
auto-caption generate video.mp4 --smart-positioning --object-detection-model small

# High accuracy
auto-caption generate video.mp4 --smart-positioning --object-detection-model large
```

### Batch Processing

Process multiple videos efficiently:

```bash
auto-caption batch /videos --smart-positioning --threads 4
```

### GPU Acceleration

Automatically uses GPU if available:

```python
detector = ObjectDetector(
    device="cuda"  # or "cpu" to force CPU
)
```

## Emotion-Aware Positioning

Combine smart positioning with emotion detection:

```python
# Happy emotions prefer higher placement
# Sad emotions prefer lower placement
# Angry emotions use dramatic positioning

result = generator.generate(
    "video.mp4",
    detect_emotions=True,
    style_captions=True,
    enable_smart_positioning=True
)
```

## Examples

### Tutorial Video
Avoids hands and demonstration objects:
```bash
auto-caption generate tutorial.mp4 --smart-positioning --word-by-word
```

### Interview/Vlog
Prioritizes face tracking:
```bash
auto-caption generate interview.mp4 --smart-positioning --avoid-faces
```

### Product Review
Ensures products remain visible:
```bash
auto-caption generate review.mp4 --smart-positioning --platform youtube_shorts
```

## Troubleshooting

### Captions Still Blocking Objects

1. Try a larger detection model:
   ```bash
   --object-detection-model large
   ```

2. Increase safe zone padding:
   ```python
   detector.SAFE_ZONE_PADDING[ObjectType.FACE] = 80  # Increase from 50
   ```

3. Use a different positioning strategy:
   ```python
   strategy=PositioningStrategy.SIDES
   ```

### Slow Processing

1. Use a smaller model:
   ```bash
   --object-detection-model small
   ```

2. Reduce detection frequency:
   ```python
   detector = ObjectDetector(
       confidence_threshold=0.7  # Higher threshold = fewer detections
   )
   ```

3. Disable unnecessary detection types:
   ```python
   detector = ObjectDetector(
       enable_text_detection=False  # If not needed
   )
   ```

### Erratic Caption Movement

1. Enable tracking for stability:
   ```python
   detector = ObjectDetector(
       tracking_enabled=True
   )
   ```

2. Increase position optimization grid resolution:
   ```python
   optimizer = PositionOptimizer(
       grid_resolution=10  # Finer grid (default: 20)
   )
   ```

## Requirements

The smart positioning feature requires additional dependencies:

```bash
pip install mediapipe>=0.10.8  # For face detection
pip install ultralytics>=8.0.228  # For object detection
```

## Demo

Run the interactive demo to see smart positioning in action:

```bash
# With your own video
python examples/smart_positioning_demo.py your_video.mp4

# With test video
python examples/smart_positioning_demo.py --test-mode
```

This creates:
- Object detection visualization video
- Position comparison (traditional vs smart)
- ASS subtitles with optimized positioning
- JSON data with position information

## API Reference

### ObjectDetector

Main object detection class:

```python
detector = ObjectDetector(
    enable_face_detection=True,
    enable_object_detection=True,
    enable_text_detection=True,
    device="cuda",  # or "cpu"
    confidence_threshold=0.5,
    tracking_enabled=True,
    model_size="medium",  # "small", "medium", "large"
    verbose=False
)

# Detect objects in frame
result = detector.detect_objects(frame, frame_index, timestamp)
```

### PositionOptimizer

Optimizes caption placement:

```python
optimizer = PositionOptimizer(
    video_resolution=(1920, 1080),
    platform=Platform.TIKTOK,
    default_strategy=PositioningStrategy.DYNAMIC,
    grid_resolution=20
)

# Find optimal position
optimization_result = optimizer.optimize_position(
    caption_text="Hello world",
    detection_result=detection_result,
    caption_size=(200, 50),
    emotion=EmotionCategory.HAPPY
)
```

### FaceTracker

Tracks faces across frames:

```python
tracker = FaceTracker(
    max_tracks=10,
    enable_speaking_detection=True,
    prediction_horizon=10
)

# Update with new detections
tracking_result = tracker.update(detected_faces, frame, frame_index)

# Get speaking faces
speaking_faces = tracker.get_speaking_faces()
```

## Best Practices

1. **Test with your content**: Different video types benefit from different strategies
2. **Monitor performance**: Larger models = better accuracy but slower processing
3. **Platform awareness**: Use platform-specific settings for optimal results
4. **Combine with emotions**: Let emotions influence positioning for better storytelling
5. **Preview first**: Use visualization tools to verify detection quality

## Future Enhancements

- [ ] Custom object training for specific use cases
- [ ] Real-time processing mode
- [ ] Advanced text detection with OCR
- [ ] 3D scene understanding
- [ ] Gesture recognition for tutorials
- [ ] Automatic strategy selection based on content

## Contributing

Help improve smart positioning:

1. Test with diverse video content
2. Report edge cases and failures
3. Contribute new positioning strategies
4. Optimize detection algorithms
5. Add support for new object types

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.