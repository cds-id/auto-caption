"""
Emotion detection module for analyzing emotional context in videos.

This module provides multi-modal emotion detection by analyzing:
- Visual cues (facial expressions, body language)
- Audio cues (tone, pitch, prosody)
- Combined context for accurate emotion classification
"""

import os
import tempfile
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from enum import Enum

import cv2
import torch
import torch.nn.functional as F
from transformers import (
    AutoProcessor, 
    AutoModelForAudioClassification,
    AutoModelForImageClassification,
    pipeline
)
import librosa
import soundfile as sf
from moviepy.editor import VideoFileClip
from tqdm import tqdm
from collections import deque
from PIL import Image


class EmotionCategory(Enum):
    """Enumeration of supported emotion categories."""
    # Primary emotions
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    SURPRISED = "surprised"
    DISGUSTED = "disgusted"
    NEUTRAL = "neutral"
    
    # Complex emotional states
    SARCASTIC = "sarcastic"
    IRONIC = "ironic"
    CONTEMPLATIVE = "contemplative"
    EXCITED = "excited"
    MELANCHOLIC = "melancholic"
    ANXIOUS = "anxious"
    CONFIDENT = "confident"
    CONFUSED = "confused"
    
    # Content moods
    MOTIVATIONAL = "motivational"
    HUMOROUS = "humorous"
    DRAMATIC = "dramatic"
    CASUAL = "casual"
    PROFESSIONAL = "professional"
    ROMANTIC = "romantic"
    NOSTALGIC = "nostalgic"


@dataclass
class EmotionScore:
    """Data class for emotion detection results."""
    emotion: EmotionCategory
    confidence: float
    timestamp: Optional[float] = None
    modality: Optional[str] = None  # 'visual', 'audio', or 'combined'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "emotion": self.emotion.value,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "modality": self.modality
        }


@dataclass
class EmotionDetectionResult:
    """Complete emotion detection result for a video."""
    dominant_emotion: EmotionCategory
    emotion_scores: List[EmotionScore]
    temporal_emotions: List[Dict[str, Any]]  # Emotions over time
    visual_emotions: List[EmotionScore]
    audio_emotions: List[EmotionScore]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "dominant_emotion": self.dominant_emotion.value,
            "emotion_scores": [score.to_dict() for score in self.emotion_scores],
            "temporal_emotions": self.temporal_emotions,
            "visual_emotions": [score.to_dict() for score in self.visual_emotions],
            "audio_emotions": [score.to_dict() for score in self.audio_emotions],
            "metadata": self.metadata
        }


class EmotionDetector:
    """
    Multi-modal emotion detector for video content.
    
    Combines visual and audio analysis to detect emotional context
    in videos, particularly optimized for short-form content.
    """
    
    def __init__(
        self,
        visual_model: Optional[str] = "dima806/facial_emotions_image_detection",
        audio_model: Optional[str] = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        device: Optional[str] = None,
        visual_weight: float = 0.85,
        audio_weight: float = 0.15,
        confidence_threshold: float = 0.7,
        verbose: bool = False
    ):
        """
        Initialize the emotion detector.
        
        Args:
            visual_model: Model for visual emotion detection
            audio_model: Model for audio emotion detection
            device: Device to run models on ('cuda' or 'cpu')
            visual_weight: Weight for visual emotions in fusion
            audio_weight: Weight for audio emotions in fusion
            confidence_threshold: Minimum confidence for emotion detection
            verbose: Enable verbose output
        """
        self.visual_model_name = visual_model
        self.audio_model_name = audio_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.visual_weight = visual_weight
        self.audio_weight = audio_weight
        self.confidence_threshold = confidence_threshold
        self.verbose = verbose
        
        # Face tracking state
        self._face_tracker = None
        self._tracked_faces = []
        self._face_net = None  # DNN face detector
        
        # Initialize models
        self._init_models()
        
    def _init_models(self):
        """Initialize emotion detection models."""
        if self.verbose:
            print(f"Initializing emotion detection models on {self.device}...")
        
        # Initialize visual emotion detection - primary method
        if self.visual_model_name:
            try:
                self.visual_pipeline = pipeline(
                    "image-classification",
                    model=self.visual_model_name,
                    device=0 if self.device == "cuda" else -1
                )
                if self.verbose:
                    print("Face emotion detection model loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load visual model: {e}")
                print("Face-based emotion detection is highly recommended for accurate results")
                self.visual_pipeline = None
        else:
            self.visual_pipeline = None
        
        # Initialize audio emotion detection - supplementary method
        if self.audio_model_name:
            try:
                self.audio_pipeline = pipeline(
                    "audio-classification",
                    model=self.audio_model_name,
                    device=0 if self.device == "cuda" else -1
                )
                if self.verbose:
                    print("Audio emotion detection loaded as supplementary method")
            except Exception as e:
                if self.verbose:
                    print(f"Info: Audio model not loaded: {e}")
                    print("Continuing with face-based detection only")
                self.audio_pipeline = None
        else:
            self.audio_pipeline = None
        
        if self.verbose:
            print("Models initialized successfully")
            if self.visual_pipeline:
                print("✓ Face-based emotion detection ready (primary method)")
            if self.audio_pipeline:
                print("✓ Audio emotion detection ready (supplementary)")
    
    def detect_emotions(
        self,
        video_path: str,
        sample_rate: int = 16000,
        frame_sample_interval: float = 0.5,
        progress_callback: Optional[callable] = None
    ) -> EmotionDetectionResult:
        """
        Detect emotions in a video file.
        
        Args:
            video_path: Path to the video file
            sample_rate: Audio sample rate for processing
            frame_sample_interval: Interval in seconds between frame samples
            progress_callback: Callback for progress updates
            
        Returns:
            EmotionDetectionResult with comprehensive emotion analysis
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Extract video information
        with VideoFileClip(str(video_path)) as video:
            duration = video.duration
            fps = video.fps
        
        # Analyze visual emotions
        visual_emotions = []
        if self.visual_pipeline:
            visual_emotions = self._analyze_visual_emotions(
                video_path, 
                frame_sample_interval,
                progress_callback
            )
        
        # Analyze audio emotions
        audio_emotions = []
        if self.audio_pipeline:
            audio_emotions = self._analyze_audio_emotions(
                video_path,
                sample_rate,
                progress_callback
            )
        
        # Combine emotions using multi-modal fusion
        combined_result = self._fuse_emotions(
            visual_emotions,
            audio_emotions,
            duration
        )
        
        return combined_result
    
    def _analyze_visual_emotions(
        self,
        video_path: Path,
        frame_interval: float,
        progress_callback: Optional[callable] = None
    ) -> List[EmotionScore]:
        """Analyze emotions from video frames."""
        visual_emotions = []
        
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_skip = int(fps * frame_interval)
        
        if self.verbose:
            print(f"Analyzing facial expressions (sampling every {frame_interval}s for accurate emotion tracking)...")
        
        frame_count = 0
        processed_frames = 0
        
        with tqdm(total=total_frames // frame_skip, disable=not self.verbose) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_skip == 0:
                    # Convert frame to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Detect faces in frame - prioritize face detection for emotion accuracy
                    faces = self._detect_faces(frame_rgb)
                    
                    if faces:
                        # Analyze emotions for detected faces (prioritize largest/most prominent)
                        face_emotions = []
                        for i, face in enumerate(faces[:3]):  # Analyze up to 3 faces
                            emotion_result = self._analyze_face_emotion(face)
                            if emotion_result:
                                # Weight by face prominence (first face = most prominent)
                                prominence_weight = 1.0 / (i + 1)
                                emotion_result.confidence *= prominence_weight
                                face_emotions.append(emotion_result)
                        
                        # Use the highest confidence emotion
                        if face_emotions:
                            best_emotion = max(face_emotions, key=lambda e: e.confidence)
                            best_emotion.timestamp = frame_count / fps
                            best_emotion.modality = "visual"
                            visual_emotions.append(best_emotion)
                    else:
                        # No face detected - use previous emotion with decaying confidence
                        if visual_emotions and len(visual_emotions) > 0:
                            last_emotion = visual_emotions[-1]
                            if (frame_count / fps - last_emotion.timestamp) < 2.0:  # Within 2 seconds
                                decayed_emotion = EmotionScore(
                                    emotion=last_emotion.emotion,
                                    confidence=last_emotion.confidence * 0.8,  # Decay confidence
                                    timestamp=frame_count / fps,
                                    modality="visual"
                                )
                                visual_emotions.append(decayed_emotion)
                    
                    processed_frames += 1
                    pbar.update(1)
                    
                    if progress_callback and processed_frames % 10 == 0:
                        progress = (frame_count / total_frames) * 50  # Visual is 50% of progress
                        progress_callback(progress)
                
                frame_count += 1
        
        cap.release()
        return visual_emotions
    
    def _detect_faces(self, frame: np.ndarray) -> List[np.ndarray]:
        """Detect faces using multiple methods for better accuracy."""
        face_images = []
        
        # Method 1: Try DNN-based face detection (more accurate)
        try:
            # Use OpenCV's DNN face detector if available
            modelFile = "opencv_face_detector_uint8.pb"
            configFile = "opencv_face_detector.pbtxt"
            
            if hasattr(self, '_face_net') and self._face_net is not None:
                net = self._face_net
            else:
                # Try to use pre-trained model if available
                if os.path.exists(modelFile) and os.path.exists(configFile):
                    net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
                    self._face_net = net
                else:
                    net = None
            
            if net is not None:
                blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), 
                                           [104, 117, 123], False, False)
                net.setInput(blob)
                detections = net.forward()
                
                h, w = frame.shape[:2]
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.7:  # Higher confidence threshold
                        x1 = int(detections[0, 0, i, 3] * w)
                        y1 = int(detections[0, 0, i, 4] * h)
                        x2 = int(detections[0, 0, i, 5] * w)
                        y2 = int(detections[0, 0, i, 6] * h)
                        
                        # Ensure coordinates are within bounds
                        x1 = max(0, min(x1, w-1))
                        y1 = max(0, min(y1, h-1))
                        x2 = max(0, min(x2, w-1))
                        y2 = max(0, min(y2, h-1))
                        
                        if x2 > x1 and y2 > y1:
                            face_img = frame[y1:y2, x1:x2]
                            if face_img.size > 0:
                                face_images.append(face_img)
        except Exception as e:
            if self.verbose:
                print(f"DNN face detection not available: {e}")
        
        # Method 2: Fallback to Haar Cascade (more compatible)
        if not face_images:
            try:
                # Try multiple cascades for better detection
                cascade_names = [
                    'haarcascade_frontalface_default.xml',
                    'haarcascade_frontalface_alt.xml',
                    'haarcascade_frontalface_alt2.xml'
                ]
                
                for cascade_name in cascade_names:
                    face_cascade = cv2.CascadeClassifier(
                        cv2.data.haarcascades + cascade_name
                    )
                    
                    if not face_cascade.empty():
                        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                        # Enhance contrast for better detection
                        gray = cv2.equalizeHist(gray)
                        
                        # Detect at multiple scales
                        faces = face_cascade.detectMultiScale(
                            gray, 
                            scaleFactor=1.05,  # Smaller scale factor for better detection
                            minNeighbors=3,    # Lower for more detections
                            minSize=(30, 30),  # Minimum face size
                            flags=cv2.CASCADE_SCALE_IMAGE
                        )
                        
                        if len(faces) > 0:
                            # Sort faces by size (larger faces first)
                            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                            
                            for (x, y, w, h) in faces[:3]:  # Limit to top 3 faces
                                # Add padding around face
                                padding = int(min(w, h) * 0.1)
                                x_pad = max(0, x - padding)
                                y_pad = max(0, y - padding)
                                w_pad = min(frame.shape[1] - x_pad, w + 2 * padding)
                                h_pad = min(frame.shape[0] - y_pad, h + 2 * padding)
                                
                                face_img = frame[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
                                if face_img.size > 0:
                                    face_images.append(face_img)
                            
                            if face_images:
                                break
            except Exception as e:
                if self.verbose:
                    print(f"Haar cascade face detection error: {e}")
        
        return face_images
    
    def _analyze_face_emotion(self, face_image: np.ndarray) -> Optional[EmotionScore]:
        """Analyze emotion in a face image with preprocessing."""
        if not self.visual_pipeline:
            return None
        
        try:
            # Preprocess face image for better emotion detection
            # Resize to expected input size
            face_resized = cv2.resize(face_image, (224, 224))
            
            # Enhance contrast
            lab = cv2.cvtColor(face_resized, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            face_enhanced = cv2.merge([l, a, b])
            face_enhanced = cv2.cvtColor(face_enhanced, cv2.COLOR_LAB2RGB)
            
            # Convert numpy array to PIL Image for the pipeline
            face_pil = Image.fromarray(face_enhanced)
            
            # Run emotion classification
            results = self.visual_pipeline(face_pil)
            
            # Map results to our emotion categories
            emotion_map = {
                "happy": EmotionCategory.HAPPY,
                "sad": EmotionCategory.SAD,
                "angry": EmotionCategory.ANGRY,
                "fear": EmotionCategory.FEARFUL,
                "surprise": EmotionCategory.SURPRISED,
                "disgust": EmotionCategory.DISGUSTED,
                "neutral": EmotionCategory.NEUTRAL
            }
            
            # Find best matching emotion
            for result in results:
                label = result['label'].lower()
                for key, emotion in emotion_map.items():
                    if key in label:
                        return EmotionScore(
                            emotion=emotion,
                            confidence=result['score']
                        )
            
            # Default to neutral if no match
            return EmotionScore(
                emotion=EmotionCategory.NEUTRAL,
                confidence=0.5
            )
            
        except Exception as e:
            if self.verbose:
                print(f"Error in face emotion analysis: {e}")
            return None
    
    def _analyze_audio_emotions(
        self,
        video_path: Path,
        sample_rate: int,
        progress_callback: Optional[callable] = None
    ) -> List[EmotionScore]:
        """Analyze emotions from audio track."""
        audio_emotions = []
        
        if self.verbose:
            print("Extracting and analyzing audio emotions...")
        
        # Extract audio from video
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "audio.wav"
            
            # Extract audio using moviepy
            with VideoFileClip(str(video_path)) as video:
                audio = video.audio
                audio.write_audiofile(
                    str(audio_path),
                    fps=sample_rate,
                    logger=None if not self.verbose else 'bar'
                )
            
            # Load audio
            audio_data, sr = librosa.load(str(audio_path), sr=sample_rate)
            
            # Analyze audio in smaller chunks to match visual emotion granularity
            chunk_duration = 2.0  # seconds - smaller chunks for better synchronization
            chunk_samples = int(chunk_duration * sr)
            total_chunks = len(audio_data) // chunk_samples
            
            for i in range(total_chunks):
                start_idx = i * chunk_samples
                end_idx = min((i + 1) * chunk_samples, len(audio_data))
                chunk = audio_data[start_idx:end_idx]
                
                # Analyze emotion in chunk
                emotion_result = self._analyze_audio_chunk(chunk, sr)
                if emotion_result:
                    emotion_result.timestamp = i * chunk_duration
                    emotion_result.modality = "audio"
                    audio_emotions.append(emotion_result)
                
                if progress_callback:
                    progress = 50 + ((i / total_chunks) * 50)  # Audio is second 50%
                    progress_callback(progress)
        
        return audio_emotions
    
    def _analyze_audio_chunk(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int
    ) -> Optional[EmotionScore]:
        """Analyze emotion in an audio chunk."""
        if not self.audio_pipeline:
            return None
        
        try:
            # Run emotion classification
            results = self.audio_pipeline(audio_chunk)
            
            # Map results to our emotion categories
            emotion_map = {
                "happy": EmotionCategory.HAPPY,
                "sad": EmotionCategory.SAD,
                "angry": EmotionCategory.ANGRY,
                "fear": EmotionCategory.FEARFUL,
                "neutral": EmotionCategory.NEUTRAL,
                "calm": EmotionCategory.CONTEMPLATIVE,
            }
            
            # Find best matching emotion
            for result in results:
                label = result['label'].lower()
                for key, emotion in emotion_map.items():
                    if key in label:
                        return EmotionScore(
                            emotion=emotion,
                            confidence=result['score']
                        )
            
            return EmotionScore(
                emotion=EmotionCategory.NEUTRAL,
                confidence=0.5
            )
            
        except Exception as e:
            if self.verbose:
                print(f"Error in audio emotion analysis: {e}")
            return None
    
    def _fuse_emotions(
        self,
        visual_emotions: List[EmotionScore],
        audio_emotions: List[EmotionScore],
        duration: float
    ) -> EmotionDetectionResult:
        """Fuse visual and audio emotions with strong preference for facial expressions."""
        # Aggregate emotions by category
        emotion_counts = {}
        emotion_confidences = {}
        
        # Process visual emotions with higher priority
        for score in visual_emotions:
            emotion = score.emotion
            # Boost visual weight for more accurate face-based detection
            weight = self.visual_weight * score.confidence * 1.2
            
            if emotion not in emotion_counts:
                emotion_counts[emotion] = 0
                emotion_confidences[emotion] = []
            
            emotion_counts[emotion] += weight
            emotion_confidences[emotion].append(score.confidence)
        
        # Process audio emotions as supplementary data
        for score in audio_emotions:
            emotion = score.emotion
            # Reduce audio influence since facial expressions are more reliable
            weight = self.audio_weight * score.confidence * 0.8
            
            if emotion not in emotion_counts:
                emotion_counts[emotion] = 0
                emotion_confidences[emotion] = []
            
            emotion_counts[emotion] += weight
            emotion_confidences[emotion].append(score.confidence)
        
        # Calculate final emotion scores
        emotion_scores = []
        for emotion, count in emotion_counts.items():
            avg_confidence = np.mean(emotion_confidences[emotion])
            normalized_score = count / (len(visual_emotions) + len(audio_emotions))
            
            emotion_scores.append(EmotionScore(
                emotion=emotion,
                confidence=min(avg_confidence * normalized_score * 2, 1.0),
                modality="combined"
            ))
        
        # Sort by confidence
        emotion_scores.sort(key=lambda x: x.confidence, reverse=True)
        
        # Determine dominant emotion
        dominant_emotion = emotion_scores[0].emotion if emotion_scores else EmotionCategory.NEUTRAL
        
        # Create temporal emotion map
        temporal_emotions = self._create_temporal_map(
            visual_emotions,
            audio_emotions,
            duration
        )
        
        # Compile metadata
        metadata = {
            "duration": duration,
            "visual_samples": len(visual_emotions),
            "audio_samples": len(audio_emotions),
            "visual_weight": self.visual_weight,
            "audio_weight": self.audio_weight,
            "confidence_threshold": self.confidence_threshold
        }
        
        return EmotionDetectionResult(
            dominant_emotion=dominant_emotion,
            emotion_scores=emotion_scores,
            temporal_emotions=temporal_emotions,
            visual_emotions=visual_emotions,
            audio_emotions=audio_emotions,
            metadata=metadata
        )
    
    def _create_temporal_map(
        self,
        visual_emotions: List[EmotionScore],
        audio_emotions: List[EmotionScore],
        duration: float
    ) -> List[Dict[str, Any]]:
        """Create a temporal map of emotions throughout the video."""
        # Combine all emotions with timestamps
        all_emotions = []
        
        for score in visual_emotions:
            if score.timestamp is not None:
                all_emotions.append({
                    "timestamp": score.timestamp,
                    "emotion": score.emotion.value,
                    "confidence": score.confidence,
                    "source": "visual"
                })
        
        for score in audio_emotions:
            if score.timestamp is not None:
                all_emotions.append({
                    "timestamp": score.timestamp,
                    "emotion": score.emotion.value,
                    "confidence": score.confidence,
                    "source": "audio"
                })
        
        # Sort by timestamp
        all_emotions.sort(key=lambda x: x["timestamp"])
        
        # Create time segments (e.g., every 0.5 seconds for finer emotion tracking)
        segment_duration = 0.5
        segments = []
        
        for i in range(int(duration / segment_duration)):
            start_time = i * segment_duration
            end_time = (i + 1) * segment_duration
            
            # Find emotions in this segment
            segment_emotions = [
                e for e in all_emotions
                if start_time <= e["timestamp"] < end_time
            ]
            
            if segment_emotions:
                # Aggregate emotions in segment
                emotion_weights = {}
                for e in segment_emotions:
                    emotion = e["emotion"]
                    weight = e["confidence"]
                    
                    if emotion not in emotion_weights:
                        emotion_weights[emotion] = 0
                    emotion_weights[emotion] += weight
                
                # Find dominant emotion in segment
                dominant = max(emotion_weights.items(), key=lambda x: x[1])
                
                segments.append({
                    "start": start_time,
                    "end": end_time,
                    "dominant_emotion": dominant[0],
                    "confidence": dominant[1] / len(segment_emotions),
                    "all_emotions": emotion_weights
                })
        
        return segments
    
    def detect_emotion_transitions(
        self,
        temporal_emotions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect significant emotion transitions in the video."""
        transitions = []
        
        for i in range(1, len(temporal_emotions)):
            prev_emotion = temporal_emotions[i-1]["dominant_emotion"]
            curr_emotion = temporal_emotions[i]["dominant_emotion"]
            
            if prev_emotion != curr_emotion:
                # Check if transition is significant
                prev_conf = temporal_emotions[i-1]["confidence"]
                curr_conf = temporal_emotions[i]["confidence"]
                
                if prev_conf > self.confidence_threshold and curr_conf > self.confidence_threshold:
                    transitions.append({
                        "timestamp": temporal_emotions[i]["start"],
                        "from_emotion": prev_emotion,
                        "to_emotion": curr_emotion,
                        "confidence": (prev_conf + curr_conf) / 2
                    })
        
        return transitions