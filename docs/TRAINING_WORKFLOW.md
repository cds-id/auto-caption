# Auto-Caption Emotion Training Workflow

## Visual Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AUTO-CAPTION TRAINING WORKFLOW                        │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   VIDEO FILES   │     │  IMAGE DATASET  │     │  ANNOTATIONS    │
│  ┌───┐ ┌───┐  │     │ ┌───┐ ┌───┐    │     │   JSON FILE     │
│  │MP4│ │AVI│  │ OR  │ │JPG│ │PNG│    │ OR  │ {timestamp:     │
│  └───┘ └───┘  │     │ └───┘ └───┘    │     │  emotion: ...}  │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                        │
         └───────────────────────┴────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   PREPARE DATASET      │
                    │ auto-caption prepare-  │
                    │      dataset           │
                    └────────────┬───────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   AUTO-DETECT   │    │ MANUAL LABELING │    │   IMPORT JSON   │
│ Use existing    │    │ Interactive CLI │    │ Pre-annotated   │
│ model to label  │    │ annotation      │    │ video frames    │
└────────┬────────┘    └────────┬────────┘    └────────┬────────┘
         └───────────────────────┴────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   DATASET STRUCTURE    │
                    │   dataset/             │
                    │   ├── train/ (70%)     │
                    │   ├── val/   (20%)     │
                    │   └── test/  (10%)     │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   DATA AUGMENTATION    │
                    │ • Rotation ±15°        │
                    │ • Brightness/Contrast  │
                    │ • Face occlusions      │
                    │ • Lighting variations  │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │      TRAIN MODEL       │
                    │ auto-caption train-    │
                    │      emotion           │
                    └────────────┬───────────┘
                                 │
                ┌────────────────┴────────────────┐
                ▼                                 ▼
    ┌───────────────────────┐       ┌───────────────────────┐
    │     FINE-TUNING       │       │   CUSTOM EMOTIONS     │
    │ • Existing model base │       │ • New categories      │
    │ • 5-15 epochs        │       │ • 15-30 epochs        │
    │ • Lower learning rate │       │ • More training data  │
    └───────────┬───────────┘       └───────────┬───────────┘
                └────────────────┬────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   TRAINING PROCESS     │
                    │ ┌──────────────────┐  │
                    │ │ Epoch 1: ████░░░ │  │
                    │ │ Loss: 0.892      │  │
                    │ │ Acc: 0.623       │  │
                    │ └──────────────────┘  │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │     EVALUATION         │
                    │ auto-caption evaluate- │
                    │      emotion           │
                    └────────────┬───────────┘
                                 │
                ┌────────────────┴────────────────┐
                ▼                                 ▼
    ┌───────────────────────┐       ┌───────────────────────┐
    │   CONFUSION MATRIX    │       │   METRICS REPORT      │
    │   ┌─────┬─────┬─────┐ │       │ • Accuracy: 85.4%     │
    │   │ H│S │ 92 │  8  │ │       │ • Precision: 86.1%    │
    │   ├─────┼─────┼─────┤ │       │ • Recall: 85.0%       │
    │   │ S│A │  5  │ 87 │ │       │ • F1-Score: 85.6%     │
    │   └─────┴─────┴─────┘ │       └───────────────────────┘
    └───────────────────────┘
                │
                ▼
    ┌────────────────────────┐
    │   MODEL DEPLOYMENT     │
    │ • Save to disk         │
    │ • Export optimized     │
    │ • Push to Hub          │
    └────────────┬───────────┘
                 │
                 ▼
    ┌────────────────────────┐
    │    USE IN CAPTIONS     │
    │ auto-caption generate  │
    │ --emotion-mode auto    │
    └────────────────────────┘
```

## Step-by-Step Process

### 1️⃣ Data Collection Phase

```
Input Sources:
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Videos    │    │   Images    │    │ Annotations │
│   (.mp4)    │ OR │   (.jpg)    │ OR │   (.json)   │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       └──────────────────┴───────────────────┘
                          │
                          ▼
                   Extract Frames
                   (if from video)
```

### 2️⃣ Annotation Phase

```
┌─────────────────────────────────────────────────┐
│              ANNOTATION METHODS                  │
├─────────────────────────────────────────────────┤
│                                                 │
│  AUTO-DETECT        MANUAL           HYBRID     │
│  ┌─────────┐      ┌─────────┐     ┌─────────┐ │
│  │ Model   │      │  User   │     │ Model + │ │
│  │ Labels  │      │ Labels  │     │ Review  │ │
│  └─────────┘      └─────────┘     └─────────┘ │
│      Fast           Accurate         Balanced  │
│                                                 │
└─────────────────────────────────────────────────┘
```

### 3️⃣ Dataset Organization

```
dataset/
│
├── train/ (70% of data)
│   ├── happy/
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   ├── sad/
│   ├── angry/
│   └── [custom_emotion]/
│
├── val/ (20% of data)
│   └── (same structure)
│
└── test/ (10% of data)
    └── (same structure)
```

### 4️⃣ Training Configuration

```
┌─────────────────────────────────────────────────┐
│           TRAINING PARAMETERS                    │
├─────────────────────────────────────────────────┤
│                                                 │
│  Base Model: ┌─────────────────────────┐       │
│              │ dima806/facial_emotions │       │
│              └─────────────────────────┘       │
│                                                 │
│  Hyperparameters:                              │
│  ┌──────────────┬──────────────┬─────────────┐│
│  │ Learning Rate│ Batch Size   │ Epochs      ││
│  │ 2e-5         │ 16           │ 10-30       ││
│  └──────────────┴──────────────┴─────────────┘│
│                                                 │
│  Augmentation: [✓] Enabled                     │
│  Mixed Precision: [✓] FP16                     │
│                                                 │
└─────────────────────────────────────────────────┘
```

### 5️⃣ Training Progress

```
Training Progress:
┌─────────────────────────────────────────────────┐
│ Epoch 1/20  ████████████████████░░░░░░░  80%   │
│ Loss: 0.453  Acc: 0.823  Val_Loss: 0.501       │
├─────────────────────────────────────────────────┤
│ Epoch 2/20  ████████████████████████░░  95%   │
│ Loss: 0.312  Acc: 0.891  Val_Loss: 0.387       │
└─────────────────────────────────────────────────┘

Learning Curves:
Loss ↓                          Accuracy ↑
1.0 ┤╲                          1.0 ┤      ╱─────
0.8 ┤ ╲                         0.8 ┤    ╱
0.6 ┤  ╲___                     0.6 ┤  ╱
0.4 ┤      ╲___                 0.4 ┤╱
0.2 ┤          ────             0.2 ┤
    └──────────────→ Epochs        └──────────────→
```

### 6️⃣ Model Evaluation

```
┌─────────────────────────────────────────────────┐
│              EVALUATION METRICS                  │
├─────────────────────────────────────────────────┤
│                                                 │
│  Overall Performance:                           │
│  ┌────────────┬─────────┐                      │
│  │ Accuracy   │  85.4%  │                      │
│  │ Precision  │  86.1%  │                      │
│  │ Recall     │  85.0%  │                      │
│  │ F1-Score   │  85.6%  │                      │
│  └────────────┴─────────┘                      │
│                                                 │
│  Confusion Matrix:                              │
│         Predicted                               │
│     ┌───┬───┬───┬───┐                         │
│   A │ H │ S │ A │ N │                         │
│  ┌─┼───┼───┼───┼───┤                         │
│ c│H│ 92│  5│  2│  1│                         │
│ t│S│  8│ 87│  3│  2│                         │
│ u│A│  3│  4│ 89│  4│                         │
│ a│N│  1│  2│  2│ 95│                         │
│ l└─┴───┴───┴───┴───┘                         │
│                                                 │
└─────────────────────────────────────────────────┘
```

## Common Workflows

### Quick Improvement (1-2 hours)
```
Video → Extract Frames → Auto-label → Review → Train (5 epochs) → Test
```

### Custom Emotions (4-6 hours)
```
Multiple Videos → Extract → Manual Label → Balance Dataset → Train (20 epochs) → Evaluate
```

### Production Model (1-2 days)
```
Large Dataset → Augmentation → Train → Evaluate → Fine-tune → Export → Deploy
```

## Decision Tree

```
Start
  │
  ├─ Have existing model?
  │     │
  │     ├─ YES → Want to improve accuracy?
  │     │          │
  │     │          ├─ YES → Fine-tune (5-10 epochs)
  │     │          │
  │     │          └─ NO → Want new emotions?
  │     │                    │
  │     │                    ├─ YES → Add emotions + train
  │     │                    │
  │     │                    └─ NO → Use as-is
  │     │
  │     └─ NO → Start from pre-trained base model
  │
  └─ Data preparation needed?
        │
        ├─ YES → Extract from videos
        │
        └─ NO → Organize existing images
```

## Tips for Each Stage

### 📸 Data Collection
- **Quality > Quantity**: 100 good images > 1000 bad ones
- **Diversity**: Multiple people, lighting, angles
- **Balance**: Equal samples per emotion

### 🏷️ Annotation
- **Consistency**: Use clear emotion definitions
- **Validation**: Have multiple people verify labels
- **Edge cases**: Document ambiguous expressions

### 🚀 Training
- **Start small**: Test with subset first
- **Monitor**: Watch validation loss for overfitting
- **Save checkpoints**: Keep best models

### 📊 Evaluation
- **Test set**: Never seen during training
- **Real videos**: Test on actual use cases
- **A/B test**: Compare with original model