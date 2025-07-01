"""
Augmentation module for emotion detection training.

This module provides data augmentation techniques specifically designed
for facial emotion detection to improve model robustness and generalization.
"""

import random
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""
    # Geometric transformations
    rotation_range: Tuple[float, float] = (-15, 15)
    scale_range: Tuple[float, float] = (0.9, 1.1)
    translation_range: Tuple[float, float] = (-0.1, 0.1)
    horizontal_flip: float = 0.5

    # Color augmentations
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    saturation_range: Tuple[float, float] = (0.8, 1.2)
    hue_shift: float = 0.05

    # Noise and blur
    gaussian_noise_var: float = 0.01
    blur_limit: int = 3
    motion_blur: bool = True

    # Advanced augmentations
    cutout_prob: float = 0.1
    cutout_size: Tuple[int, int] = (20, 20)
    mixup_alpha: float = 0.2
    temporal_consistency: bool = True

    # Emotion-specific augmentations
    expression_intensity_scale: Tuple[float, float] = (0.8, 1.2)
    facial_occlusion: bool = True
    lighting_variation: bool = True


class VideoAugmenter:
    """
    Augmentation pipeline for video-based emotion detection training.

    Provides various augmentation techniques to improve model robustness
    while maintaining temporal consistency for video data.
    """

    def __init__(self, config: AugmentationConfig):
        """
        Initialize video augmenter.

        Args:
            config: Augmentation configuration
        """
        self.config = config
        self._setup_augmentations()

    def _setup_augmentations(self):
        """Setup augmentation pipelines."""
        # Albumentations pipeline for spatial augmentations
        self.spatial_transform = A.Compose([
            A.Rotate(
                limit=self.config.rotation_range[1],
                p=0.5
            ),
            A.Affine(
                scale=self.config.scale_range,
                translate_percent={
                    "x": self.config.translation_range,
                    "y": self.config.translation_range
                },
                p=0.5
            ),
            A.HorizontalFlip(p=self.config.horizontal_flip),
        ])

        # Color augmentations
        self.color_transform = A.Compose([
            A.ColorJitter(
                brightness=self.config.brightness_range,
                contrast=self.config.contrast_range,
                saturation=self.config.saturation_range,
                hue=self.config.hue_shift,
                p=0.8
            ),
            A.CLAHE(p=0.3),
            A.RandomGamma(p=0.3),
        ])

        # Noise and blur
        self.noise_transform = A.Compose([
            A.GaussNoise(
                var_limit=(0, self.config.gaussian_noise_var),
                p=0.3
            ),
            A.OneOf([
                A.MotionBlur(blur_limit=self.config.blur_limit, p=1),
                A.MedianBlur(blur_limit=self.config.blur_limit, p=1),
                A.GaussianBlur(blur_limit=self.config.blur_limit, p=1),
            ], p=0.2),
        ])

        # Advanced augmentations
        self.advanced_transform = A.Compose([
            A.CoarseDropout(
                max_holes=8,
                max_height=self.config.cutout_size[0],
                max_width=self.config.cutout_size[1],
                p=self.config.cutout_prob
            ),
            A.RandomShadow(p=0.2),
            A.RandomFog(p=0.1),
        ])

    def augment_frame(
        self,
        image: np.ndarray,
        emotion: str,
        maintain_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Augment a single frame.

        Args:
            image: Input image (RGB)
            emotion: Emotion label for emotion-specific augmentations
            maintain_params: Parameters to maintain temporal consistency

        Returns:
            Augmented image and parameters used
        """
        # Apply spatial transformations
        if maintain_params and self.config.temporal_consistency:
            # Use same parameters for temporal consistency
            spatial_params = maintain_params.get("spatial", {})
            augmented = self.spatial_transform(
                image=image,
                replay=spatial_params
            )
        else:
            augmented = self.spatial_transform(image=image)
            spatial_params = augmented.get("replay", {})

        image = augmented["image"]

        # Apply color augmentations (with some variation for realism)
        color_augmented = self.color_transform(image=image)
        image = color_augmented["image"]

        # Apply noise and blur
        noise_augmented = self.noise_transform(image=image)
        image = noise_augmented["image"]

        # Apply advanced augmentations
        advanced_augmented = self.advanced_transform(image=image)
        image = advanced_augmented["image"]

        # Apply emotion-specific augmentations
        if emotion:
            image = self._apply_emotion_specific_augmentation(image, emotion)

        # Store parameters for temporal consistency
        params = {
            "spatial": spatial_params,
            "emotion": emotion
        }

        return image, params

    def _apply_emotion_specific_augmentation(
        self,
        image: np.ndarray,
        emotion: str
    ) -> np.ndarray:
        """Apply augmentations specific to certain emotions."""
        if emotion in ["happy", "excited"] and random.random() < 0.3:
            # Brighten for positive emotions
            image = self._adjust_brightness(image, factor=1.1)

        elif emotion in ["sad", "contemplative"] and random.random() < 0.3:
            # Darken/desaturate for sad emotions
            image = self._adjust_brightness(image, factor=0.9)
            image = self._adjust_saturation(image, factor=0.8)

        elif emotion == "angry" and random.random() < 0.3:
            # Add red tint for angry
            image = self._add_color_tint(image, (10, -5, -5))

        return image

    def augment_video_sequence(
        self,
        frames: List[np.ndarray],
        emotions: List[str],
        consistent_augmentation: bool = True
    ) -> List[np.ndarray]:
        """
        Augment a sequence of video frames.

        Args:
            frames: List of video frames
            emotions: List of emotions for each frame
            consistent_augmentation: Apply consistent spatial augmentation

        Returns:
            List of augmented frames
        """
        augmented_frames = []
        params = None

        for frame, emotion in zip(frames, emotions):
            if consistent_augmentation:
                aug_frame, params = self.augment_frame(frame, emotion, params)
            else:
                aug_frame, _ = self.augment_frame(frame, emotion)

            augmented_frames.append(aug_frame)

        return augmented_frames

    def apply_facial_occlusion(
        self,
        image: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> np.ndarray:
        """
        Apply realistic facial occlusions (glasses, hands, etc.).

        Args:
            image: Input image
            face_bbox: Face bounding box (x, y, w, h)

        Returns:
            Image with facial occlusion
        """
        if not self.config.facial_occlusion or random.random() > 0.3:
            return image

        occlusion_type = random.choice(["sunglasses", "hand", "mask"])

        if occlusion_type == "sunglasses":
            return self._add_sunglasses(image, face_bbox)
        elif occlusion_type == "hand":
            return self._add_hand_occlusion(image, face_bbox)
        elif occlusion_type == "mask":
            return self._add_face_mask(image, face_bbox)

        return image

    def _add_sunglasses(
        self,
        image: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]]
    ) -> np.ndarray:
        """Add sunglasses overlay to face."""
        if face_bbox is None:
            return image

        x, y, w, h = face_bbox

        # Create simple sunglasses shape
        overlay = image.copy()

        # Calculate sunglasses position (upper third of face)
        glasses_y = y + int(h * 0.25)
        glasses_height = int(h * 0.2)

        # Draw sunglasses
        cv2.rectangle(
            overlay,
            (x + int(w * 0.1), glasses_y),
            (x + int(w * 0.9), glasses_y + glasses_height),
            (20, 20, 20),
            -1
        )

        # Blend with original
        alpha = 0.7
        image = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

        return image

    def _add_hand_occlusion(
        self,
        image: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]]
    ) -> np.ndarray:
        """Add hand gesture occlusion."""
        if face_bbox is None:
            return image

        x, y, w, h = face_bbox

        # Create elliptical mask for hand
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Random hand position
        hand_x = x + random.randint(0, w)
        hand_y = y + random.randint(int(h * 0.3), int(h * 0.7))
        hand_size = (int(w * 0.4), int(h * 0.3))

        cv2.ellipse(
            mask,
            (hand_x, hand_y),
            hand_size,
            random.randint(0, 45),
            0,
            360,
            255,
            -1
        )

        # Blur the mask for soft edges
        mask = cv2.GaussianBlur(mask, (21, 21), 0)

        # Apply skin tone color
        skin_color = (random.randint(180, 220), random.randint(140, 180), random.randint(100, 140))
        overlay = np.full_like(image, skin_color)

        # Blend
        mask_3d = mask[:, :, np.newaxis] / 255.0
        image = image * (1 - mask_3d * 0.8) + overlay * mask_3d * 0.8

        return image.astype(np.uint8)

    def _add_face_mask(
        self,
        image: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]]
    ) -> np.ndarray:
        """Add face mask (COVID-style) occlusion."""
        if face_bbox is None:
            return image

        x, y, w, h = face_bbox

        # Mask covers lower half of face
        mask_y = y + int(h * 0.5)
        mask_height = int(h * 0.5)

        # Create mask shape
        overlay = image.copy()
        mask_color = random.choice([
            (200, 200, 200),  # White
            (100, 150, 200),  # Blue
            (50, 50, 50)      # Black
        ])

        cv2.rectangle(
            overlay,
            (x, mask_y),
            (x + w, y + h),
            mask_color,
            -1
        )

        # Blend
        alpha = 0.9
        image = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

        return image

    def apply_lighting_variation(
        self,
        image: np.ndarray,
        light_type: Optional[str] = None
    ) -> np.ndarray:
        """
        Apply realistic lighting variations.

        Args:
            image: Input image
            light_type: Type of lighting ('bright', 'dim', 'side', 'backlight')

        Returns:
            Image with lighting variation
        """
        if not self.config.lighting_variation or random.random() > 0.4:
            return image

        if light_type is None:
            light_type = random.choice(['bright', 'dim', 'side', 'backlight'])

        if light_type == 'bright':
            # Simulate bright lighting
            image = self._adjust_brightness(image, factor=random.uniform(1.2, 1.5))
            image = self._adjust_contrast(image, factor=random.uniform(1.1, 1.2))

        elif light_type == 'dim':
            # Simulate dim lighting
            image = self._adjust_brightness(image, factor=random.uniform(0.5, 0.8))
            image = self._add_noise(image, intensity=0.02)

        elif light_type == 'side':
            # Simulate side lighting
            image = self._apply_gradient_lighting(image, direction='horizontal')

        elif light_type == 'backlight':
            # Simulate backlighting
            image = self._apply_vignette(image, intensity=0.5)
            image = self._adjust_brightness(image, factor=0.7)

        return image

    def _adjust_brightness(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image brightness."""
        pil_image = Image.fromarray(image)
        enhancer = ImageEnhance.Brightness(pil_image)
        enhanced = enhancer.enhance(factor)
        return np.array(enhanced)

    def _adjust_contrast(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image contrast."""
        pil_image = Image.fromarray(image)
        enhancer = ImageEnhance.Contrast(pil_image)
        enhanced = enhancer.enhance(factor)
        return np.array(enhanced)

    def _adjust_saturation(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image saturation."""
        pil_image = Image.fromarray(image)
        enhancer = ImageEnhance.Color(pil_image)
        enhanced = enhancer.enhance(factor)
        return np.array(enhanced)

    def _add_color_tint(self, image: np.ndarray, tint: Tuple[int, int, int]) -> np.ndarray:
        """Add color tint to image."""
        return np.clip(image + np.array(tint), 0, 255).astype(np.uint8)

    def _add_noise(self, image: np.ndarray, intensity: float) -> np.ndarray:
        """Add Gaussian noise to image."""
        noise = np.random.normal(0, intensity * 255, image.shape)
        return np.clip(image + noise, 0, 255).astype(np.uint8)

    def _apply_gradient_lighting(
        self,
        image: np.ndarray,
        direction: str = 'horizontal'
    ) -> np.ndarray:

        """Apply gradient lighting effect."""
        h, w = image.shape[:2]
        
        if direction == 'horizontal':
            gradient = np.linspace(0.5, 1.5, w)
            gradient = np.tile(gradient, (h, 1))
        else:
            gradient = np.linspace(0.5, 1.5, h)
            gradient = np.tile(gradient.reshape(-1, 1), (1, w))
        
        gradient = np.stack([gradient] * 3, axis=-1)
        return np.clip(image * gradient, 0, 255).astype(np.uint8)
    
    def _apply_vignette(self, image: np.ndarray, intensity: float) -> np.ndarray:
        """Apply vignette effect."""
        h, w = image.shape[:2]
        
        # Create radial gradient
        center_x, center_y = w // 2, h // 2
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        vignette = 1 - (dist_from_center / max_dist) * intensity
        vignette = np.stack([vignette] * 3, axis=-1)
        
        return np.clip(image * vignette, 0, 255).astype(np.uint8)
    
    def mixup(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        label1: int,
        label2: int,
        alpha: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply mixup augmentation.
        
        Args:
            image1: First image
            image2: Second image
            label1: First label
            label2: Second label
            alpha: Mixup parameter (None to use config)
            
        Returns:
            Mixed image and mixed label (one-hot)
        """
        if alpha is None:
            alpha = self.config.mixup_alpha
        
        # Sample lambda from beta distribution
        lam = np.random.beta(alpha, alpha)
        
        # Mix images
        mixed_image = lam * image1 + (1 - lam) * image2
        mixed_image = mixed_image.astype(np.uint8)
        
        # Mix labels (assuming one-hot encoding needed)
        # This would need to be adjusted based on your label format
        mixed_label = np.array([lam if i == label1 else (1 - lam) if i == label2 else 0
                               for i in range(max(label1, label2) + 1)])
        
        return mixed_image, mixed_label