"""
Dataset module for emotion detection training.

This module provides dataset classes and utilities for preparing
training data for emotion detection models.
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import numpy as np
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


@dataclass
class EmotionSample:
    """A single emotion training sample."""
    image_path: str
    emotion_label: str
    confidence: float = 1.0
    face_bbox: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h
    metadata: Optional[Dict[str, Any]] = None


class EmotionDataset(Dataset):
    """
    PyTorch dataset for emotion detection training.

    Handles loading, preprocessing, and augmentation of facial emotion data.
    """

    def __init__(
        self,
        images: Union[List[np.ndarray], List[str]],
        labels: List[int],
        processor: Any,
        image_size: Tuple[int, int] = (224, 224),
        augment: bool = True,
        face_detection: bool = True
    ):
        """
        Initialize emotion dataset.

        Args:
            images: List of image arrays or paths
            labels: List of emotion labels (as integers)
            processor: HuggingFace processor for the model
            image_size: Target image size
            augment: Whether to apply data augmentation
            face_detection: Whether to detect and crop faces
        """
        self.images = images
        self.labels = labels
        self.processor = processor
        self.image_size = image_size
        self.augment = augment
        self.face_detection = face_detection

        # Setup augmentation transforms
        if self.augment:
            self.augmentation = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1)
                ),
            ])
        else:
            self.augmentation = None

        # Initialize face detector if needed
        if self.face_detection:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single item from the dataset."""
        # Load image
        if isinstance(self.images[idx], str):
            # Load from path
            image = Image.open(self.images[idx]).convert('RGB')
            image = np.array(image)
        else:
            # Already numpy array
            image = self.images[idx]

        # Detect and crop face if enabled
        if self.face_detection:
            face_image = self._extract_face(image)
            if face_image is not None:
                image = face_image

        # Convert to PIL for processing
        image = Image.fromarray(image)

        # Resize
        image = image.resize(self.image_size, Image.Resampling.LANCZOS)

        # Apply augmentation if training
        if self.augmentation is not None:
            image = self.augmentation(image)

        # Process with model processor
        inputs = self.processor(images=image, return_tensors="pt")

        # Prepare output
        item = {
            "pixel_values": inputs["pixel_values"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

        return item

    def _extract_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract face from image."""
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) > 0:
            # Use the largest face
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face

            # Add padding
            padding = int(min(w, h) * 0.2)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)

            # Crop face
            face_image = image[y:y+h, x:x+w]
            return face_image

        return None


class DatasetBuilder:
    """
    Builder class for creating emotion datasets from various sources.
    """

    def __init__(
        self,
        emotion_labels: Dict[str, int],
        image_size: Tuple[int, int] = (224, 224),
        face_detection: bool = True
    ):
        """
        Initialize dataset builder.

        Args:
            emotion_labels: Mapping of emotion names to label indices
            image_size: Target image size
            face_detection: Whether to detect and crop faces
        """
        self.emotion_labels = emotion_labels
        self.image_size = image_size
        self.face_detection = face_detection

    def from_directory(
        self,
        data_dir: str,
        processor: Any,
        split: str = "train",
        augment: bool = True
    ) -> EmotionDataset:
        """
        Create dataset from directory structure.

        Expected structure:
        data_dir/
            train/
                happy/
                    image1.jpg
                    image2.jpg
                sad/
                    image1.jpg
                angry/
                    ...
            val/
                ...
            test/
                ...
        """
        data_path = Path(data_dir) / split

        if not data_path.exists():
            raise ValueError(f"Data directory not found: {data_path}")

        images = []
        labels = []

        # Load images from each emotion directory
        for emotion_name, emotion_id in self.emotion_labels.items():
            emotion_dir = data_path / emotion_name

            if emotion_dir.exists():
                for image_path in emotion_dir.glob("*.jpg"):
                    images.append(str(image_path))
                    labels.append(emotion_id)

                for image_path in emotion_dir.glob("*.png"):
                    images.append(str(image_path))
                    labels.append(emotion_id)

        print(f"Loaded {len(images)} images from {split} split")

        return EmotionDataset(
            images=images,
            labels=labels,
            processor=processor,
            image_size=self.image_size,
            augment=augment and split == "train",
            face_detection=self.face_detection
        )

    def from_video_annotations(
        self,
        annotations_file: str,
        processor: Any,
        video_dir: Optional[str] = None,
        augment: bool = True
    ) -> EmotionDataset:
        """
        Create dataset from video annotations.

        Annotations format:
        {
            "videos": [
                {
                    "path": "video1.mp4",
                    "annotations": [
                        {
                            "timestamp
": 1.5,
                            "emotion": "happy",
                            "confidence": 0.9,
                            "face_bbox": [x, y, w, h]  # optional
                        }
                    ]
                }
            ]
        }
        """
        with open(annotations_file, 'r') as f:
            data = json.load(f)
        
        images = []
        labels = []
        
        for video_data in tqdm(data["videos"], desc="Processing videos"):
            video_path = video_data["path"]
            
            if video_dir:
                video_path = os.path.join(video_dir, video_path)
            
            # Extract frames from video
            frames = self._extract_frames_from_video(
                video_path,
                video_data["annotations"]
            )
            
            for frame, annotation in frames:
                emotion = annotation["emotion"]
                if emotion in self.emotion_labels:
                    images.append(frame)
                    labels.append(self.emotion_labels[emotion])
        
        print(f"Extracted {len(images)} frames from videos")
        
        return EmotionDataset(
            images=images,
            labels=labels,
            processor=processor,
            image_size=self.image_size,
            augment=augment,
            face_detection=self.face_detection
        )
    
    def _extract_frames_from_video(
        self,
        video_path: str,
        annotations: List[Dict[str, Any]]
    ) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        """Extract annotated frames from video."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frames = []
        
        for annotation in annotations:
            timestamp = annotation["timestamp"]
            frame_number = int(timestamp * fps)
            
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Crop to face bbox if provided
                if "face_bbox" in annotation:
                    x, y, w, h = annotation["face_bbox"]
                    frame = frame[y:y+h, x:x+w]
                
                frames.append((frame, annotation))
        
        cap.release()
        return frames
    
    def from_csv(
        self,
        csv_file: str,
        image_dir: str,
        processor: Any,
        image_col: str = "image",
        emotion_col: str = "emotion",
        augment: bool = True
    ) -> EmotionDataset:
        """
        Create dataset from CSV file.
        
        CSV format:
        image,emotion
        img001.jpg,happy
        img002.jpg,sad
        """
        import pandas as pd
        
        df = pd.read_csv(csv_file)
        
        images = []
        labels = []
        
        for _, row in df.iterrows():
            image_path = os.path.join(image_dir, row[image_col])
            emotion = row[emotion_col]
            
            if emotion in self.emotion_labels and os.path.exists(image_path):
                images.append(image_path)
                labels.append(self.emotion_labels[emotion])
        
        print(f"Loaded {len(images)} images from CSV")
        
        return EmotionDataset(
            images=images,
            labels=labels,
            processor=processor,
            image_size=self.image_size,
            augment=augment,
            face_detection=self.face_detection
        )
    
    def create_balanced_dataset(
        self,
        dataset: EmotionDataset,
        samples_per_class: Optional[int] = None
    ) -> EmotionDataset:
        """
        Create a balanced dataset with equal samples per emotion.
        
        Args:
            dataset: Original dataset
            samples_per_class: Number of samples per class (None for minimum)
        """
        # Group samples by label
        label_groups = {}
        for idx, label in enumerate(dataset.labels):
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)
        
        # Determine samples per class
        if samples_per_class is None:
            samples_per_class = min(len(indices) for indices in label_groups.values())
        
        # Sample from each class
        balanced_indices = []
        for label, indices in label_groups.items():
            if len(indices) >= samples_per_class:
                sampled = random.sample(indices, samples_per_class)
            else:
                # Oversample if needed
                sampled = random.choices(indices, k=samples_per_class)
            balanced_indices.extend(sampled)
        
        # Create new dataset
        balanced_images = [dataset.images[i] for i in balanced_indices]
        balanced_labels = [dataset.labels[i] for i in balanced_indices]
        
        return EmotionDataset(
            images=balanced_images,
            labels=balanced_labels,
            processor=dataset.processor,
            image_size=dataset.image_size,
            augment=dataset.augment,
            face_detection=dataset.face_detection
        )