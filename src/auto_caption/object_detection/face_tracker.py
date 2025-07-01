"""
Face tracker module for continuous face tracking across video frames.

This module tracks faces throughout video sequences to predict movement
and ensure captions avoid blocking faces even as they move.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import deque
import math

try:
    import mediapipe as mp
except ImportError:
    mp = None

from .object_detector import DetectedObject, ObjectType


@dataclass
class TrackedFace:
    """Represents a tracked face with movement history."""
    track_id: int
    current_bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    position_history: deque = field(default_factory=lambda: deque(maxlen=30))
    velocity: Tuple[float, float] = (0.0, 0.0)  # pixels per frame
    acceleration: Tuple[float, float] = (0.0, 0.0)
    predicted_positions: List[Tuple[int, int]] = field(default_factory=list)
    landmarks: Optional[Dict[str, Tuple[float, float]]] = None
    age: int = 0  # Frames since first detection
    last_seen: int = 0  # Frame when last seen
    is_speaking: bool = False
    emotion_hints: Optional[str] = None
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get center point of face."""
        x, y, w, h = self.current_bbox
        return (x + w // 2, y + h // 2)
    
    @property
    def is_stable(self) -> bool:
        """Check if face position is stable."""
        if len(self.position_history) < 5:
            return False
        
        # Calculate position variance
        recent_positions = list(self.position_history)[-5:]
        centers = [(p[0] + p[2]//2, p[1] + p[3]//2) for p in recent_positions]
        
        # Calculate standard deviation
        x_coords = [c[0] for c in centers]
        y_coords = [c[1] for c in centers]
        
        x_std = np.std(x_coords)
        y_std = np.std(y_coords)
        
        # Stable if movement is minimal
        return x_std < 5 and y_std < 5
    
    def update_position(self, new_bbox: Tuple[int, int, int, int], frame_index: int):
        """Update face position and calculate motion."""
        old_center = self.center
        self.current_bbox = new_bbox
        new_center = self.center
        
        # Update position history
        self.position_history.append(new_bbox)
        
        # Calculate velocity (exponential moving average)
        alpha = 0.3  # Smoothing factor
        new_velocity = (
            new_center[0] - old_center[0],
            new_center[1] - old_center[1]
        )
        self.velocity = (
            alpha * new_velocity[0] + (1 - alpha) * self.velocity[0],
            alpha * new_velocity[1] + (1 - alpha) * self.velocity[1]
        )
        
        # Update tracking info
        self.last_seen = frame_index
        self.age += 1
        
    def predict_future_positions(self, num_frames: int = 10) -> List[Tuple[int, int]]:
        """Predict future positions based on current motion."""
        predictions = []
        current_x, current_y = self.center
        
        for i in range(1, num_frames + 1):
            # Simple linear prediction with velocity
            # Could be enhanced with Kalman filter
            pred_x = int(current_x + self.velocity[0] * i)
            pred_y = int(current_y + self.velocity[1] * i)
            
            predictions.append((pred_x, pred_y))
            
        self.predicted_positions = predictions
        return predictions
    
    def get_expanded_bbox(self, expansion_factor: float = 1.5) -> Tuple[int, int, int, int]:
        """Get expanded bounding box for safety margin."""
        x, y, w, h = self.current_bbox
        
        # Expand based on motion
        motion_magnitude = math.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
        dynamic_expansion = 1.0 + (motion_magnitude / 100.0)  # More expansion for fast movement
        
        total_expansion = expansion_factor * dynamic_expansion
        
        # Calculate expansion
        expand_w = int(w * (total_expansion - 1) / 2)
        expand_h = int(h * (total_expansion - 1) / 2)
        
        return (
            x - expand_w,
            y - expand_h,
            w + 2 * expand_w,
            h + 2 * expand_h
        )


@dataclass
class FaceTrackingResult:
    """Result of face tracking for a frame."""
    tracked_faces: List[TrackedFace]
    new_faces: List[TrackedFace]
    lost_faces: List[int]  # Track IDs
    frame_index: int
    movement_map: Optional[np.ndarray] = None  # Heatmap of face movement


class FaceTracker:
    """
    Advanced face tracker for video sequences.
    
    Features:
    - Multi-face tracking with ID persistence
    - Motion prediction
    - Speaking detection
    - Stability analysis
    """
    
    def __init__(
        self,
        max_tracks: int = 10,
        max_lost_frames: int = 15,
        min_confidence: float = 0.5,
        iou_threshold: float = 0.3,
        enable_landmarks: bool = True,
        enable_speaking_detection: bool = True,
        prediction_horizon: int = 10,
        verbose: bool = False
    ):
        """
        Initialize face tracker.
        
        Args:
            max_tracks: Maximum number of simultaneous face tracks
            max_lost_frames: Frames before a track is considered lost
            min_confidence: Minimum confidence for face detection
            iou_threshold: IoU threshold for matching faces
            enable_landmarks: Enable facial landmark detection
            enable_speaking_detection: Enable speaking detection
            prediction_horizon: Number of frames to predict ahead
            verbose: Enable verbose output
        """
        self.max_tracks = max_tracks
        self.max_lost_frames = max_lost_frames
        self.min_confidence = min_confidence
        self.iou_threshold = iou_threshold
        self.enable_landmarks = enable_landmarks
        self.enable_speaking_detection = enable_speaking_detection
        self.prediction_horizon = prediction_horizon
        self.verbose = verbose
        
        # Tracking state
        self.tracks: Dict[int, TrackedFace] = {}
        self.next_track_id = 0
        self.frame_count = 0
        
        # Initialize MediaPipe if available
        if mp is not None and self.enable_landmarks:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=max_tracks,
                refine_landmarks=True,
                min_detection_confidence=min_confidence,
                min_tracking_confidence=0.5
            )
        else:
            self.face_mesh = None
            
        # Speaking detection state
        self.mouth_history: Dict[int, deque] = {}
        
    def update(
        self,
        detected_faces: List[DetectedObject],
        frame: Optional[np.ndarray] = None,
        frame_index: Optional[int] = None
    ) -> FaceTrackingResult:
        """
        Update tracking with new face detections.
        
        Args:
            detected_faces: List of detected face objects
            frame: Current frame (for landmark detection)
            frame_index: Current frame index
            
        Returns:
            FaceTrackingResult with tracking information
        """
        if frame_index is None:
            frame_index = self.frame_count
            
        self.frame_count = frame_index
        
        # Match detected faces to existing tracks
        matched_tracks, unmatched_detections, lost_tracks = self._match_faces(
            detected_faces,
            frame_index
        )
        
        # Update matched tracks
        for track_id, detection in matched_tracks:
            self.tracks[track_id].update_position(detection.bbox, frame_index)
            self.tracks[track_id].confidence = detection.confidence
            
        # Create new tracks for unmatched detections
        new_faces = []
        for detection in unmatched_detections:
            if len(self.tracks) < self.max_tracks:
                new_track = self._create_new_track(detection, frame_index)
                self.tracks[new_track.track_id] = new_track
                new_faces.append(new_track)
                
        # Handle lost tracks
        lost_face_ids = []
        for track_id in lost_tracks:
            if frame_index - self.tracks[track_id].last_seen > self.max_lost_frames:
                lost_face_ids.append(track_id)
                del self.tracks[track_id]
                if track_id in self.mouth_history:
                    del self.mouth_history[track_id]
                    
        # Update landmarks and detect speaking if enabled
        if frame is not None and self.face_mesh is not None:
            self._update_landmarks(frame)
            
        # Predict future positions
        for track in self.tracks.values():
            track.predict_future_positions(self.prediction_horizon)
            
        # Generate movement heatmap
        movement_map = self._generate_movement_map(frame.shape if frame is not None else None)
        
        return FaceTrackingResult(
            tracked_faces=list(self.tracks.values()),
            new_faces=new_faces,
            lost_faces=lost_face_ids,
            frame_index=frame_index,
            movement_map=movement_map
        )
        
    def _match_faces(
        self,
        detections: List[DetectedObject],
        frame_index: int
    ) -> Tuple[List[Tuple[int, DetectedObject]], List[DetectedObject], List[int]]:
        """Match detected faces to existing tracks using IoU."""
        matched = []
        unmatched_detections = list(detections)
        unmatched_tracks = list(self.tracks.keys())
        
        # Calculate IoU matrix
        if detections and unmatched_tracks:
            iou_matrix = np.zeros((len(detections), len(unmatched_tracks)))
            
            for i, detection in enumerate(detections):
                for j, track_id in enumerate(unmatched_tracks):
                    track = self.tracks[track_id]
                    iou = self._calculate_iou(detection.bbox, track.current_bbox)
                    iou_matrix[i, j] = iou
                    
            # Greedy matching
            while True:
                # Find best match
                if iou_matrix.size == 0:
                    break
                    
                best_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                best_iou = iou_matrix[best_idx]
                
                if best_iou < self.iou_threshold:
                    break
                    
                # Record match
                detection_idx, track_idx = best_idx
                matched.append((unmatched_tracks[track_idx], detections[detection_idx]))
                
                # Remove matched items
                iou_matrix = np.delete(iou_matrix, detection_idx, axis=0)
                iou_matrix = np.delete(iou_matrix, track_idx, axis=1)
                unmatched_detections.pop(detection_idx)
                unmatched_tracks.pop(track_idx)
                
        return matched, unmatched_detections, unmatched_tracks
        
    def _calculate_iou(
        self,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int]
    ) -> float:
        """Calculate Intersection over Union."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 < xi1 or yi2 < yi1:
            return 0.0
            
        intersection = (xi2 - xi1) * (yi2 - yi1)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
        
    def _create_new_track(
        self,
        detection: DetectedObject,
        frame_index: int
    ) -> TrackedFace:
        """Create new face track."""
        track = TrackedFace(
            track_id=self.next_track_id,
            current_bbox=detection.bbox,
            confidence=detection.confidence,
            last_seen=frame_index
        )
        
        self.next_track_id += 1
        
        # Initialize position history
        track.position_history.append(detection.bbox)
        
        # Initialize mouth history for speaking detection
        if self.enable_speaking_detection:
            self.mouth_history[track.track_id] = deque(maxlen=10)
            
        return track
        
    def _update_landmarks(self, frame: np.ndarray):
        """Update facial landmarks for all tracked faces."""
        if self.face_mesh is None:
            return
            
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            # Match landmarks to tracks
            # This is simplified - in production you'd want better matching
            for face_landmarks in results.multi_face_landmarks:
                # Find closest track
                # Extract key landmarks
                landmarks = self._extract_key_landmarks(face_landmarks, frame.shape)
                
                # Find best matching track based on face center
                best_track = None
                min_distance = float('inf')
                
                face_center = landmarks.get('nose_tip', (0, 0))
                
                for track in self.tracks.values():
                    track_center = track.center
                    distance = math.sqrt(
                        (face_center[0] - track_center[0])**2 + 
                        (face_center[1] - track_center[1])**2
                    )
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_track = track
                        
                if best_track and min_distance < 100:  # Reasonable threshold
                    best_track.landmarks = landmarks
                    
                    # Update speaking detection
                    if self.enable_speaking_detection:
                        self._update_speaking_detection(best_track)
                        
    def _extract_key_landmarks(self, face_landmarks, frame_shape):
        """Extract key facial landmarks."""
        h, w = frame_shape[:2]
        landmarks = {}
        
        # Key landmark indices in MediaPipe
        landmark_indices = {
            'nose_tip': 1,
            'mouth_center': 13,
            'left_eye': 33,
            'right_eye': 263,
            'mouth_left': 61,
            'mouth_right': 291,
            'mouth_top': 12,
            'mouth_bottom': 15
        }
        
        for name, idx in landmark_indices.items():
            if idx < len(face_landmarks.landmark):
                landmark = face_landmarks.landmark[idx]
                landmarks[name] = (int(landmark.x * w), int(landmark.y * h))
                
        return landmarks
        
    def _update_speaking_detection(self, track: TrackedFace):
        """Detect if person is speaking based on mouth movement."""
        if track.landmarks is None:
            return
            
        # Calculate mouth openness
        mouth_top = track.landmarks.get('mouth_top')
        mouth_bottom = track.landmarks.get('mouth_bottom')
        
        if mouth_top and mouth_bottom:
            mouth_open = abs(mouth_bottom[1] - mouth_top[1])
            
            # Store in history
            self.mouth_history[track.track_id].append(mouth_open)
            
            # Detect speaking based on mouth movement variance
            if len(self.mouth_history[track.track_id]) >= 5:
                mouth_variance = np.var(list(self.mouth_history[track.track_id]))
                
                # Speaking if significant mouth movement
                track.is_speaking = mouth_variance > 10
                
    def _generate_movement_map(self, frame_shape: Optional[Tuple[int, int, int]]) -> Optional[np.ndarray]:
        """Generate heatmap of predicted face movements."""
        if frame_shape is None or not self.tracks:
            return None
            
        height, width = frame_shape[:2]
        movement_map = np.zeros((height, width), dtype=np.float32)
        
        for track in self.tracks.values():
            # Add current position
            x, y, w, h = track.current_bbox
            cv2.rectangle(movement_map, (x, y), (x + w, y + h), 0.5, -1)
            
            # Add predicted positions with decreasing intensity
            for i, (px, py) in enumerate(track.predicted_positions):
                intensity = 0.3 * (1 - i / len(track.predicted_positions))
                # Simple circle for predicted position
                cv2.circle(movement_map, (px, py), 20, intensity, -1)
                
        # Apply Gaussian blur for smooth heatmap
        movement_map = cv2.GaussianBlur(movement_map, (31, 31), 0)
        
        return movement_map
        
    def get_safe_zones_for_caption(
        self,
        caption_duration: float,
        fps: float = 30.0
    ) -> List[Tuple[int, int, int, int]]:
        """Get safe zones to avoid for caption placement."""
        safe_zones = []
        frames_ahead = int(caption_duration * fps)
        
        for track in self.tracks.values():
            # Get expanded bbox for current position
            safe_zones.append(track.get_expanded_bbox())
            
            # Add predicted positions
            predictions = track.predict_future_positions(frames_ahead)
            for px, py in predictions:
                # Approximate bbox at predicted position
                w, h = track.current_bbox[2:]
                safe_zones.append((
                    px - w//2 - 30,  # Extra margin
                    py - h//2 - 30,
                    w + 60,
                    h + 60
                ))
                
        return safe_zones
        
    def get_speaking_faces(self) -> List[TrackedFace]:
        """Get list of faces that are currently speaking."""
        return [track for track in self.tracks.values() if track.is_speaking]
        
    def reset(self):
        """Reset tracker state."""
        self.tracks.clear()
        self.mouth_history.clear()
        self.next_track_id = 0
        self.frame_count = 0