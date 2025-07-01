"""
Training module for Auto-Caption emotion detection models.

This module provides functionality to:
- Fine-tune emotion detection models on custom datasets
- Create custom emotion categories
- Improve accuracy for specific use cases
"""

from .emotion_trainer import EmotionTrainer, TrainingConfig
from .dataset import EmotionDataset, DatasetBuilder
from .augmentation import VideoAugmenter, AugmentationConfig

__all__ = [
    "EmotionTrainer",
    "TrainingConfig",
    "EmotionDataset",
    "DatasetBuilder",
    "VideoAugmenter",
    "AugmentationConfig",
]