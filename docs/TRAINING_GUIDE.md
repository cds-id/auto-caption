# Auto-Caption Emotion Detection Training Guide

This guide will walk you through training custom emotion detection models for Auto-Caption, allowing you to improve accuracy or add custom emotion categories for your specific use case.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Preparing Training Data](#preparing-training-data)
4. [Training Your Model](#training-your-model)
5. [Evaluating Model Performance](#evaluating-model-performance)
6. [Using Your Custom Model](#using-your-custom-model)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

## Overview

Auto-Caption's emotion detection is based on facial expression analysis. By training your own model, you can:

- **Improve accuracy** for specific demographics or video styles
- **Add custom emotions** beyond the default categories
- **Fine-tune** for specific lighting conditions or camera angles
- **Adapt** to domain-specific expressions (e.g., gaming, education, vlogs)

### Default vs Custom Emotions

**Default emotions:**
- Happy, Sad, Angry, Fearful, Surprised, Disgusted, Neutral, Sarcastic, Anxious, Excited, Contemplative

**Custom emotions you might add:**
- Focused, Bored, Confused, Amused, Frustrated, Confident, Skeptical, etc.

## Prerequisites

### 1. Install Training Dependencies

```bash
# Install base auto-caption
pip install -r requirements.txt

# The training dependencies are already included:
# - albumentations (data augmentation)
# - scikit-learn (metrics)
# - matplotlib & seaborn (visualization)
# - pandas (data handling)
```

### 2. Hardware Requirements

- **GPU recommended** (NVIDIA with CUDA support)
- **RAM:** 16GB minimum, 32GB recommended
- **Storage:** 10-50GB depending on dataset size

### 3. Download Base Model

```bash
# This happens automatically, but you can pre-download:
auto-caption download-models --type emotion
```

## Preparing Training Data

### Method 1: Extract from Videos (Recommended for Beginners)

#### Step 1: Extract Frames

```bash
# Extract frames every 1 second from your video
auto-caption prepare-dataset my_video.mp4 \
  --output-dir ./my_dataset \
  --interval 1.0 \
  --format directory
```

#### Step 2: Auto-Detect Initial Labels (Optional)

```bash
# Use existing model to pre-label frames (you can correct them later)
auto-caption prepare-dataset my_video.mp4 \
  --output-dir ./my_dataset \
  --interval 1.0 \
  --auto-detect \
  --format directory
```

#### Step 3: Manual Annotation

If not using auto-detect, you'll be prompted to label each frame:

```bash
# Interactive annotation mode
auto-caption prepare-dataset my_video.mp4 \
  --output-dir ./my_dataset \
  --interval 0.5 \
  --emotion-labels happy sad angry neutral confused focused
```

### Method 2: Organize Existing Images

Create a directory structure:

```
my_dataset/
├── train/
│   ├── happy/
│   │   ├── person1_happy_001.jpg
│   │   ├── person1_happy_002.jpg
│   │   └── person2_happy_001.jpg
│   ├── sad/
│   │   ├── person1_sad_001.jpg
│   │   └── person3_sad_001.jpg
│   ├── angry/
│   │   └── person2_angry_001.jpg
│   └── focused/  # Custom emotion
│       └── person4_focused_001.jpg
├── val/
│   └── (same structure with 20% of data)
└── test/
    └── (same structure with 10% of data)
```

### Method 3: Video Annotations (Advanced)

Create a JSON annotation file:

```json
{
  "videos": [
    {
      "path": "interview_video.mp4",
      "annotations": [
        {
          "timestamp": 1.5,
          "emotion": "happy",
          "confidence": 0.95
        },
        {
          "timestamp": 3.2,
          "emotion": "contemplative",
          "confidence": 0.87
        }
      ]
    },
    {
      "path": "presentation_video.mp4",
      "annotations": [
        {
          "timestamp": 0.5,
          "emotion": "focused",
          "confidence": 0.92,
          "face_bbox": [100, 50, 200, 200]  # Optional: x, y, w, h
        }
      ]
    }
  ]
}
```

### Data Requirements

- **Minimum images per emotion:** 50-100 for fine-tuning, 500+ for training from scratch
- **Image quality:** Clear facial expressions, good lighting
- **Diversity:** Multiple people, angles, lighting conditions
- **Balance:** Try to have similar numbers for each emotion

## Training Your Model

### Basic Training

```bash
# Train with default settings
auto-caption train-emotion \
  --data-dir ./my_dataset \
  --output-dir ./models/my_emotion_model \
  --epochs 10
```

### Training with Custom Emotions

```bash
# Add your own emotion categories
auto-caption train-emotion \
  --data-dir ./my_dataset \
  --output-dir ./models/custom_emotions \
  --custom-emotions happy sad angry neutral focused bored confused \
  --epochs 20 \
  --batch-size 16
```

### Fine-Tuning on Specific Videos

```bash
# Fine-tune existing model on your annotated videos
auto-caption train-emotion \
  --annotations ./video_annotations.json \
  --output-dir ./models/finetuned_model \
  --epochs 5 \
  --learning-rate 1e-5
```

### Advanced Training Options

```bash
# Full control over training process
auto-caption train-emotion \
  --data-dir ./my_dataset \
  --output-dir ./models/advanced_model \
  --model-name "dima806/facial_emotions_image_detection" \
  --epochs 30 \
  --batch-size 32 \
  --learning-rate 2e-5 \
  --augment \
  --eval-split 0.2 \
  --verbose
```

### Training Parameters Explained

- **`--epochs`**: Number of training iterations (10-30 typical)
- **`--batch-size`**: Images per batch (16-32 typical, lower if GPU memory limited)
- **`--learning-rate`**: How fast model learns (1e-5 to 5e-5 typical)
- **`--augment`**: Apply data augmentation (recommended)
- **`--eval-split`**: Validation data percentage (0.2 = 20%)

## Evaluating Model Performance

### Run Evaluation

```bash
# Evaluate on test set
auto-caption evaluate-emotion \
  ./models/my_emotion_model \
  ./my_dataset/test \
  --confusion-matrix \
  --save-report ./evaluation_report.json
```

### Understanding Metrics

The evaluation will show:

1. **Overall Accuracy**: Percentage of correct predictions
2. **Per-Emotion Metrics**:
   - **Precision**: Of all faces labeled as "happy", how many were actually happy?
   - **Recall**: Of all actually happy faces, how many were detected?
   - **F1-Score**: Balanced measure of precision and recall

Example output:
```
Evaluation Results
┌──────────┬───────┐
│ Metric   │ Value │
├──────────┼───────┤
│ Accuracy │ 0.8543│
│ Precision│ 0.8612│
│ Recall   │ 0.8501│
│ F1-Score │ 0.8556│
└──────────┴───────┘

Per-Emotion Performance
┌──────────────┬───────────┬────────┬─────────┬─────────┐
│ Emotion      │ Precision │ Recall │ F1-Score│ Support │
├──────────────┼───────────┼────────┼─────────┼─────────┤
│ happy        │   0.912   │  0.895 │  0.903  │   245   │
│ sad          │   0.834   │  0.867 │  0.850  │   198   │
│ angry        │   0.801   │  0.772 │  0.786  │   167   │
│ focused      │   0.889   │  0.901 │  0.895  │   201   │
└──────────────┴───────────┴────────┴─────────┴─────────┘
```

### Confusion Matrix

The confusion matrix shows which emotions are confused with each other:

![Confusion Matrix](confusion_matrix.png)

- **Diagonal**: Correct predictions (should be bright)
- **Off-diagonal**: Confusion between emotions (should be dark)

## Using Your Custom Model

### Option 1: Update Configuration (Coming Soon)

```bash
# Set your model as default
auto-caption config --set emotion_model ./models/my_emotion_model
```

### Option 2: Specify Model Path (Coming Soon)

```bash
# Use custom model for generation
auto-caption generate video.mp4 \
  --emotion-mode auto \
  --emotion-model ./models/my_emotion_model
```

### Option 3: Replace Default Model

```bash
# Backup original model
mv ~/.cache/huggingface/hub/models--dima806--facial_emotions_image_detection \
   ~/.cache/huggingface/hub/models--dima806--facial_emotions_image_detection.backup

# Copy your model
cp -r ./models/my_emotion_model \
      ~/.cache/huggingface/hub/models--dima806--facial_emotions_image_detection
```

## Best Practices

### 1. Data Collection

- **Lighting**: Collect data in various lighting conditions
- **Angles**: Include different head poses and camera angles
- **Demographics**: Ensure diversity in age, gender, ethnicity
- **Expressions**: Get natural, not exaggerated expressions

### 2. Data Augmentation

Built-in augmentations include:
- **Rotation**: ±15 degrees
- **Brightness/Contrast**: Simulates different lighting
- **Occlusions**: Sunglasses, hands, masks
- **Blur**: Motion and gaussian blur

### 3. Training Strategy

1. **Start with fine-tuning**: Don't train from scratch unless necessary
2. **Use early stopping**: Monitor validation loss
3. **Balance your dataset**: Equal samples per emotion
4. **Progressive training**: Start with fewer emotions, add more gradually

### 4. Quality Control

- **Review misclassified examples**: Often reveals labeling errors
- **Test on unseen videos**: Ensures generalization
- **A/B test**: Compare with default model

## Troubleshooting

### Common Issues

#### 1. "CUDA out of memory"
```bash
# Reduce batch size
--batch-size 8  # or even 4
```

#### 2. "No faces detected during training"
- Check image quality
- Ensure faces are clearly visible
- Try manual face cropping

#### 3. "Low accuracy on custom emotions"
- Need more training data (500+ images)
- Emotions might be too similar
- Check for labeling consistency

#### 4. "Model overfitting"
Signs: High training accuracy, low validation accuracy

Solutions:
- Add more augmentation
- Collect more diverse data
- Reduce learning rate
- Add dropout (in code)

### Performance Tips

1. **GPU Training**:
   ```bash
   # Check GPU is available
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Mixed Precision Training**:
   ```bash
   # Faster training with minimal accuracy loss
   --fp16
   ```

3. **Data Loading**:
   - Use SSD for dataset storage
   - Pre-resize images to target size
   - Use multiple workers (in code)

## Example: Gaming Content Emotions

Here's a complete example for training a model for gaming content:

```bash
# 1. Prepare dataset from gaming videos
auto-caption prepare-dataset gameplay_compilation.mp4 \
  --output-dir ./gaming_dataset \
  --interval 0.5 \
  --emotion-labels happy frustrated excited focused rage_quit victory

# 2. Train model
auto-caption train-emotion \
  --data-dir ./gaming_dataset \
  --output-dir ./models/gaming_emotions \
  --custom-emotions happy frustrated excited focused rage_quit victory neutral \
  --epochs 25 \
  --batch-size 24 \
  --learning-rate 3e-5

# 3. Evaluate
auto-caption evaluate-emotion \
  ./models/gaming_emotions \
  ./gaming_dataset/test \
  --confusion-matrix

# 4. Use for captions
auto-caption generate gaming_video.mp4 \
  --emotion-mode auto \
  --style-intensity intense \
  --platform youtube_shorts
```

## Next Steps

1. **Share your model**: Upload to HuggingFace Hub
   ```bash
   --push-to-hub --hub-model-id your-username/auto-caption-gaming
   ```

2. **Continuous improvement**: Retrain periodically with new data

3. **Community models**: Check for pre-trained models for your domain

## Support

- **GitHub Issues**: Report bugs or request features
- **Discord**: Join community discussions
- **Documentation**: Check other guides in `/docs`

Happy training! 🎉