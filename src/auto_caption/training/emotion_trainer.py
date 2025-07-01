"""
Emotion trainer module for fine-tuning emotion detection models.

This module provides functionality to train and fine-tune emotion detection
models on custom datasets for improved accuracy and custom emotion categories.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import (
    AutoModelForImageClassification,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    EvalPrediction
)
from PIL import Image
import cv2

from ..emotion_detector import EmotionCategory


@dataclass
class TrainingConfig:
    """Configuration for emotion detection training."""
    # Model settings
    base_model: str = "dima806/facial_emotions_image_detection"
    model_type: str = "visual"  # 'visual' or 'audio'
    custom_emotions: Optional[List[str]] = None

    # Training parameters
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 10
    warmup_steps: int = 500
    weight_decay: float = 0.01

    # Data settings
    train_split: float = 0.8
    validation_split: float = 0.1
    test_split: float = 0.1
    image_size: Tuple[int, int] = (224, 224)
    augmentation: bool = True

    # Output settings
    output_dir: str = "./emotion_model"
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 50

    # Advanced settings
    gradient_accumulation_steps: int = 1
    fp16: bool = False
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary."""
        return cls(**config_dict)


class EmotionTrainer:
    """
    Trainer for emotion detection models.

    Supports fine-tuning pre-trained models on custom datasets
    with custom emotion categories.
    """

    def __init__(
        self,
        config: TrainingConfig,
        device: Optional[str] = None
    ):
        """
        Initialize the emotion trainer.

        Args:
            config: Training configuration
            device: Device to use ('cuda' or 'cpu')
        """
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model and processor
        self.model = None
        self.processor = None
        self.emotion_labels = self._setup_emotion_labels()

        # Training state
        self.best_accuracy = 0.0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }

    def _setup_emotion_labels(self) -> Dict[str, int]:
        """Setup emotion label mappings."""
        if self.config.custom_emotions:
            # Use custom emotions
            labels = {emotion: idx for idx, emotion in enumerate(self.config.custom_emotions)}
        else:
            # Use default emotions
            labels = {
                emotion.value: idx
                for idx, emotion in enumerate(EmotionCategory)
            }

        return labels

    def prepare_model(self):
        """Prepare model for training."""
        print(f"Loading base model: {self.config.base_model}")

        if self.config.model_type == "visual":
            # Load visual emotion model
            self.processor = AutoProcessor.from_pretrained(self.config.base_model)
            self.model = AutoModelForImageClassification.from_pretrained(
                self.config.base_model,
                num_labels=len(self.emotion_labels),
                ignore_mismatched_sizes=True
            )

            # Update model config with our labels
            self.model.config.label2id = self.emotion_labels
            self.model.config.id2label = {v: k for k, v in self.emotion_labels.items()}

        else:
            raise NotImplementedError("Audio emotion training not yet implemented")

        # Move model to device
        self.model = self.model.to(self.device)
        print(f"Model loaded on {self.device}")

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None
    ):
        """
        Train the emotion detection model.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset
        """
        if self.model is None:
            self.prepare_model()

        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy="steps" if val_dataset else "no",
            save_strategy="steps",
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="accuracy" if val_dataset else None,
            greater_is_better=True,
            fp16=self.config.fp16,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            push_to_hub=self.config.push_to_hub,
            hub_model_id=self.config.hub_model_id,
            remove_unused_columns=False,
        )

        # Setup trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self._compute_metrics,
            tokenizer=self.processor,
        )

        # Train
        print("Starting training...")
        train_result = trainer.train()

        # Save model
        trainer.save_model()

        # Save training history
        self._save_training_history()

        # Evaluate on test set if provided
        if test_dataset:
            print("\nEvaluating on test set...")
            test_results = trainer.evaluate(test_dataset)
            self._save_test_results(test_results)

        return train_result

    def _compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """Compute evaluation metrics."""
        predictions = np.argmax(eval_pred.predictions, axis=1)
        labels = eval_pred.label_ids

        # Calculate accuracy
        accuracy = (predictions == labels).mean()

        # Calculate per-class metrics
        report = classification_report(
            labels,
            predictions,
            target_names=list(self.emotion_labels.keys()),
            output_dict=True
        )

        # Extract key metrics
        metrics = {
            "accuracy": accuracy,
            "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"],
            "f1": report["weighted avg"]["f1-score"]
        }

        # Add per-emotion accuracy
        for emotion, emotion_id in self.emotion_labels.items():
            emotion_mask = labels == emotion_id
            if emotion_mask.any():
                emotion_acc = (predictions[emotion_mask] == labels[emotion_mask]).mean()
                metrics[f"accuracy_{emotion}"] = emotion_acc

        return metrics

    def fine_tune_on_video(
        self,
        video_path: str,
        annotations_path: str,
        output_dir: Optional[str] = None
    ):
        """
        Fine-tune model on a single annotated video.

        Args:
            video_path: Path to video file
            annotations_path: Path to annotations JSON file
            output_dir: Output directory for fine-tuned model
        """
        # Load annotations
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)

        # Extract frames and create dataset
        dataset = self._create_dataset_from_video(video_path, annotations)

        # Split into train/val
        train_size = int(len(dataset) * 0.8)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        # Fine-tune
        if output_dir:
            self.config.output_dir = output_dir

        self.train(train_dataset, val_dataset)

    def _create_dataset_from_video(
        self,
        video_path: str,
        annotations: Dict[str, Any]
    ) -> Dataset:
        """Create dataset from annotated video."""
        from .dataset import EmotionDataset

        # Extract frames
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        frames = []
        labels = []

        for annotation in annotations.get("annotations", []):
            timestamp = annotation["timestamp"]
            emotion = annotation["emotion"]

            # Seek to frame
            frame_number = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                labels.append(self.emotion_labels[emotion])

        cap.release()

        # Create dataset
        return EmotionDataset(
            images=frames,
            labels=labels,
            processor=self.processor,
            image_size=self.config.image_size,
            augment=self.config.augmentation
        )

    def evaluate(
        self,
        test_dataset: Dataset,
        save_confusion_matrix: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate model on test dataset.

        Args:
            test_dataset: Test dataset
            save_confusion_matrix: Whether to save confusion matrix plot

        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call prepare_model() first.")

        self.model.eval()

        # Prepare data loader
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                inputs = batch["pixel_values"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        report = classification_report(
            all_labels,
            all_predictions,
            target_names=list(self.emotion_labels.keys()),
            output_dict=True
        )

        # Create confusion matrix
        if save_confusion_matrix:
            self._plot_confusion_matrix(all_labels, all_predictions)

        return report

    def _plot_confusion_matrix(self, true_labels: List[int], predictions: List[int]):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(true_labels, predictions)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=list(self.emotion_labels.keys()),
            yticklabels=list(self.emotion_labels.keys())
        )
        plt.title('Emotion Detection Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        # Save plot
        output_path = Path(self.config.output_dir) / "confusion_matrix.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Confusion matrix saved to: {output_path}")

    def save_model(self, path: Optional[str] = None):
        """Save trained model and configuration."""
        if self.model is None:
            raise ValueError("No model to save")

        save_path = path or self.config.output_dir
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)

        # Save emotion labels
        with open(save_path / "emotion_labels.json", 'w') as f:
            json.dump(self.emotion_labels, f, indent=2)

        # Save training config
        with open(save_path / "training_config.json", 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)

        print(f"Model saved to: {save_path}")

    def load_model(self, path: str):
        """Load trained model from path."""
        load_path = Path(path)

        # Load config
        with open(load_path / "training_config.json", 'r') as f:
            config_dict = json.load(f)
            self.config = TrainingConfig.from_dict(config_dict)

        # Load emotion labels
        with open(load_path / "emotion_labels.json", 'r') as f:
            self.emotion_labels = json.load(f)

        # Load model and processor
        self.processor = AutoProcessor.from_pretrained(load_path)
        self.model = AutoModelForImageClassification.from_pretrained(load_path)
        self.model = self.model.to(self.device)

        print(f"Model loaded from: {load_path}")
    
    def _save_training_history(self):
        """Save training history to file."""
        history_path = Path(self.config.output_dir) / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def _save_test_results(self, results: Dict[str, Any]):
        """Save test results to file."""
        results_path = Path(self.config.output_dir) / "test_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    def export_for_inference(self, export_path: str, optimize: bool = True):
        """
        Export model for efficient inference.
        
        Args:
            export_path: Path to save exported model
            optimize: Whether to optimize model for inference
        """
        if self.model is None:
            raise ValueError("No model to export")
        
        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)
        
        # Put model in eval mode
        self.model.eval()
        
        if optimize and self.device == "cuda":
            # Optimize with TorchScript
            example_input = torch.randn(1, 3, *self.config.image_size).to(self.device)
            traced_model = torch.jit.trace(self.model, example_input)
            torch.jit.save(traced_model, export_path / "model_optimized.pt")
            print(f"Optimized model saved to: {export_path / 'model_optimized.pt'}")
        
        # Save regular model
        self.model.save_pretrained(export_path)
        self.processor.save_pretrained(export_path)
        
        # Save config and labels
        shutil.copy(
            Path(self.config.output_dir) / "emotion_labels.json",
            export_path / "emotion_labels.json"
        )
        
        print(f"Model exported to: {export_path}")