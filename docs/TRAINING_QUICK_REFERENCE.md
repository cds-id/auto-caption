# Auto-Caption Training Quick Reference

## 🚀 Common Training Scenarios

### 1. Quick Fine-Tune (Improve Accuracy)
```bash
# Extract frames from your videos
auto-caption prepare-dataset video.mp4 -o ./dataset --interval 1.0 --auto-detect

# Fine-tune for 5 epochs
auto-caption train-emotion --data-dir ./dataset -o ./models/finetuned --epochs 5
```

### 2. Add Custom Emotions
```bash
# Prepare dataset with custom emotions
auto-caption prepare-dataset video.mp4 -o ./dataset \
  --emotion-labels happy sad angry focused confused bored

# Train with custom emotions
auto-caption train-emotion --data-dir ./dataset -o ./models/custom \
  --custom-emotions happy sad angry focused confused bored --epochs 15
```

### 3. Domain-Specific Training

#### Gaming Content
```bash
auto-caption train-emotion --data-dir ./gaming_data -o ./models/gaming \
  --custom-emotions happy frustrated excited focused rage_quit victory \
  --epochs 20 --batch-size 24
```

#### Educational Content
```bash
auto-caption train-emotion --data-dir ./edu_data -o ./models/education \
  --custom-emotions focused confused understanding thinking neutral \
  --epochs 20 --learning-rate 2e-5
```

#### Vlog/Lifestyle
```bash
auto-caption train-emotion --data-dir ./vlog_data -o ./models/vlog \
  --custom-emotions happy excited contemplative surprised casual \
  --epochs 15 --augment
```

## 📊 Dataset Preparation Cheatsheet

### From Single Video
```bash
# Basic extraction
auto-caption prepare-dataset video.mp4 -o ./data --interval 1.0

# With auto-labeling
auto-caption prepare-dataset video.mp4 -o ./data --interval 0.5 --auto-detect

# Custom emotions interactive
auto-caption prepare-dataset video.mp4 -o ./data \
  --emotion-labels happy sad focused bored
```

### From Multiple Videos
```bash
# Process each video
for video in *.mp4; do
  auto-caption prepare-dataset "$video" -o ./data --interval 1.0 --auto-detect
done
```

### Required Directory Structure
```
dataset/
├── train/     # 70% of data
│   ├── happy/
│   ├── sad/
│   └── angry/
├── val/       # 20% of data
│   └── (same structure)
└── test/      # 10% of data
    └── (same structure)
```

## ⚙️ Training Parameters Guide

| Parameter | Low-End GPU | Mid-Range GPU | High-End GPU |
|-----------|-------------|---------------|--------------|
| Batch Size | 4-8 | 16-24 | 32-64 |
| Learning Rate | 1e-5 | 2e-5 | 3e-5 |
| Epochs (Fine-tune) | 5-10 | 10-15 | 15-20 |
| Epochs (New Model) | 15-20 | 20-30 | 30-50 |

## 🔍 Evaluation Commands

```bash
# Basic evaluation
auto-caption evaluate-emotion ./models/my_model ./data/test

# With confusion matrix
auto-caption evaluate-emotion ./models/my_model ./data/test --confusion-matrix

# Save detailed report
auto-caption evaluate-emotion ./models/my_model ./data/test \
  --save-report ./reports/evaluation.json --confusion-matrix
```

## 💡 Tips & Tricks

### Memory Issues?
```bash
# Reduce batch size
--batch-size 4

# Enable mixed precision
--fp16

# Reduce image size (in code)
```

### Low Accuracy?
```bash
# More epochs
--epochs 30

# Lower learning rate
--learning-rate 1e-5

# More augmentation
--augment

# Balance dataset
auto-caption balance-dataset ./data  # (if implemented)
```

### Training Too Slow?
```bash
# Increase batch size (if memory allows)
--batch-size 32

# Use mixed precision
--fp16

# Reduce validation frequency (in code)
```

## 📁 Data Requirements

| Training Type | Minimum Images/Emotion | Recommended | Ideal |
|--------------|----------------------|-------------|-------|
| Fine-tuning | 50 | 200 | 500+ |
| New Emotions | 100 | 500 | 1000+ |
| From Scratch | 500 | 2000 | 5000+ |

## 🎯 Quick Troubleshooting

### No Faces Detected
- Check image brightness/contrast
- Ensure faces are > 48x48 pixels
- Try different face angles

### Overfitting (High train, Low val accuracy)
```bash
# Add more augmentation
--augment

# Reduce learning rate
--learning-rate 1e-5

# Add more diverse data
```

### Emotion Confusion
- Similar emotions need more data
- Check labeling consistency
- Consider merging similar emotions

## 📈 Expected Performance

| Scenario | Expected Accuracy |
|----------|------------------|
| Fine-tuning (same domain) | 85-95% |
| New emotions (5-7 total) | 75-85% |
| New emotions (8-12 total) | 65-75% |
| Cross-domain | 60-70% |

## 🚄 Speed Reference

| GPU | Images/Second | Time for 10k images |
|-----|---------------|-------------------|
| RTX 3060 | ~50 | ~3.3 minutes |
| RTX 3080 | ~120 | ~1.4 minutes |
| RTX 4090 | ~250 | ~0.7 minutes |
| CPU only | ~5 | ~33 minutes |

## 📝 Example Workflows

### 1. Personal Vlog Model (2 hours)
```bash
# 30 min: Extract & label frames
auto-caption prepare-dataset vlog1.mp4 vlog2.mp4 -o ./vlog_data

# 90 min: Train model
auto-caption train-emotion --data-dir ./vlog_data -o ./vlog_model --epochs 15

# 10 min: Test
auto-caption generate test_vlog.mp4 --emotion-mode auto
```

### 2. Quick Accuracy Boost (30 min)
```bash
# 10 min: Extract problem frames
auto-caption prepare-dataset problem_video.mp4 -o ./fixes --interval 2.0

# 15 min: Fine-tune
auto-caption train-emotion --data-dir ./fixes -o ./improved --epochs 5

# 5 min: Verify improvement
auto-caption evaluate-emotion ./improved ./fixes/test
```

### 3. Multi-Person Model (4 hours)
```bash
# 1 hour: Collect diverse data
for person in person1 person2 person3; do
  auto-caption prepare-dataset ${person}.mp4 -o ./diverse_data
done

# 2.5 hours: Train robust model
auto-caption train-emotion --data-dir ./diverse_data -o ./robust_model \
  --epochs 25 --augment --learning-rate 2e-5

# 30 min: Comprehensive testing
auto-caption evaluate-emotion ./robust_model ./diverse_data/test --confusion-matrix
```

---

**Remember**: Better data > More epochs > Bigger model