"""
Object detector module for detecting faces and important objects in video frames.

This module uses computer vision models to detect and track objects that should
not be obscured by captions, ensuring optimal subtitle placement.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import torch
from collections import defaultdict

try:
    import mediapipe as mp
except ImportError:
    mp = None
    print("MediaPipe not installed. Face detection accuracy may be reduced.")

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    print("YOLO not installed. General object detection will be limited.")


class ObjectType(Enum):
    """Types of objects to detect and avoid."""
    FACE = "face"
    PERSON = "person"
    TEXT = "text"  # Existing on-screen text
    LOGO = "logo"
    HAND = "hand"
    PRODUCT = "product"  # For product showcases
    PET = "pet"  # Dogs, cats, etc.
    FOOD = "food"  # For cooking videos
    VEHICLE = "vehicle"
    SPORTS_EQUIPMENT = "sports_equipment"
    MUSICAL_INSTRUMENT = "musical_instrument"
    UNKNOWN = "unknown"


@dataclass
class DetectedObject:
    """Represents a detected object in a frame."""
    type: ObjectType
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    importance: float  # 0-1, how important it is not to block this
    tracking_id: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get center point of the object."""
        x, y, w, h = self.bbox
        return (x + w // 2, y + h // 2)
    
    @property
    def area(self) -> int:
        """Get area of the bounding box."""
        _, _, w, h = self.bbox
        return w * h


@dataclass
class SafeZone:
    """Represents a safe zone where captions should not be placed."""
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    importance: float
    padding: int  # Additional padding around the zone
    object_type: ObjectType
    
    def get_padded_bbox(self) -> Tuple[int, int, int, int]:
        """Get bounding box with padding applied."""
        x, y, w, h = self.bbox
        return (
            max(0, x - self.padding),
            max(0, y - self.padding),
            w + 2 * self.padding,
            h + 2 * self.padding
        )
    
    def contains_point(self, point: Tuple[int, int]) -> bool:
        """Check if a point is within this safe zone."""
        x, y, w, h = self.get_padded_bbox()
        px, py = point
        return x <= px <= x + w and y <= py <= y + h
    
    def overlaps_with_rect(self, rect: Tuple[int, int, int, int]) -> bool:
        """Check if a rectangle overlaps with this safe zone."""
        x1, y1, w1, h1 = self.get_padded_bbox()
        x2, y2, w2, h2 = rect
        
        # Check if rectangles overlap
        return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)


@dataclass
class DetectionResult:
    """Result of object detection on a frame."""
    objects: List[DetectedObject]
    safe_zones: List[SafeZone]
    frame_index: int
    timestamp: float
    heatmap: Optional[np.ndarray] = None  # Importance heatmap


class ObjectDetector:
    """
    Detects and tracks important objects in video frames.
    
    Uses multiple detection methods:
    - MediaPipe for accurate face detection
    - YOLO for general object detection
    - Custom methods for text and UI elements
    """
    
    # Object importance scores (how important it is not to block them)
    OBJECT_IMPORTANCE = {
        ObjectType.FACE: 1.0,  # Most important - never block faces
        ObjectType.PERSON: 0.9,  # Very important
        ObjectType.TEXT: 0.85,  # Don't block existing text
        ObjectType.HAND: 0.8,  # Important for tutorials
        ObjectType.LOGO: 0.75,  # Brand visibility
        ObjectType.PRODUCT: 0.7,  # Product showcases
        ObjectType.PET: 0.65,  # Pet videos
        ObjectType.FOOD: 0.6,  # Cooking content
        ObjectType.MUSICAL_INSTRUMENT: 0.55,
        ObjectType.SPORTS_EQUIPMENT: 0.5,
        ObjectType.VEHICLE: 0.45,
        ObjectType.UNKNOWN: 0.3
    }
    
    # Safe zone padding based on object type (in pixels)
    SAFE_ZONE_PADDING = {
        ObjectType.FACE: 50,  # Large padding for faces
        ObjectType.PERSON: 40,
        ObjectType.TEXT: 20,  # Less padding for text
        ObjectType.HAND: 30,
        ObjectType.LOGO: 25,
        ObjectType.PRODUCT: 30,
        ObjectType.PET: 35,
        ObjectType.FOOD: 25,
        ObjectType.MUSICAL_INSTRUMENT: 20,
        ObjectType.SPORTS_EQUIPMENT: 20,
        ObjectType.VEHICLE: 20,
        ObjectType.UNKNOWN: 15
    }
    
    def __init__(
        self,
        enable_face_detection: bool = True,
        enable_object_detection: bool = True,
        enable_text_detection: bool = True,
        device: Optional[str] = None,
        confidence_threshold: float = 0.5,
        tracking_enabled: bool = True,
        model_size: str = "medium",  # small, medium, large
        verbose: bool = False
    ):
        """
        Initialize object detector.
        
        Args:
            enable_face_detection: Enable face detection
            enable_object_detection: Enable general object detection
            enable_text_detection: Enable text detection
            device: Device to run models on
            confidence_threshold: Minimum confidence for detections
            tracking_enabled: Enable object tracking across frames
            model_size: Model size (affects accuracy vs speed)
            verbose: Enable verbose output
        """
        self.enable_face_detection = enable_face_detection
        self.enable_object_detection = enable_object_detection
        self.enable_text_detection = enable_text_detection
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        self.tracking_enabled = tracking_enabled
        self.model_size = model_size
        self.verbose = verbose
        
        # Initialize detectors
        self._init_detectors()
        
        # Tracking state
        self.tracked_objects = {}
        self.next_tracking_id = 0
        
    def _init_detectors(self):
        """Initialize detection models."""
        # Initialize MediaPipe face detection
        if self.enable_face_detection and mp is not None:
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_face_mesh = mp.solutions.face_mesh
            
            # Use face detection for general face detection
            model_selection = {
                "small": 0,  # Short-range model
                "medium": 1,  # Full-range model
                "large": 1   # Full-range model with higher confidence
            }.get(self.model_size, 1)
            
            self.face_detector = self.mp_face_detection.FaceDetection(
                model_selection=model_selection,
                min_detection_confidence=self.confidence_threshold
            )
            
            # Use face mesh for detailed face tracking
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=5,
                refine_landmarks=True,
                min_detection_confidence=self.confidence_threshold,
                min_tracking_confidence=0.5
            )
        else:
            self.face_detector = None
            self.face_mesh = None
            
        # Initialize YOLO for general object detection
        if self.enable_object_detection and YOLO is not None:
            model_name = {
                "small": "yolov8n.pt",  # Nano
                "medium": "yolov8s.pt",  # Small
                "large": "yolov8m.pt"   # Medium
            }.get(self.model_size, "yolov8s.pt")
            
            try:
                self.yolo_model = YOLO(model_name)
                if self.device == "cuda":
                    self.yolo_model.to('cuda')
            except Exception as e:
                if self.verbose:
                    print(f"Failed to load YOLO model: {e}")
                self.yolo_model = None
        else:
            self.yolo_model = None
            
        # Initialize text detector (using OpenCV)
        if self.enable_text_detection:
            # EAST text detector for scene text detection
            self.text_detector = self._init_text_detector()
        else:
            self.text_detector = None
            
    def _init_text_detector(self):
        """Initialize text detection using OpenCV's EAST detector."""
        # For now, we'll use a simple approach
        # In production, you'd want to use EAST or similar
        return "simple"  # Placeholder
        
    def detect_objects(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
        timestamp: float = 0.0
    ) -> DetectionResult:
        """
        Detect all objects in a frame.
        
        Args:
            frame: Video frame (BGR)
            frame_index: Frame index in video
            timestamp: Timestamp in seconds
            
        Returns:
            DetectionResult with all detected objects
        """
        detected_objects = []
        
        # Detect faces
        if self.enable_face_detection:
            faces = self._detect_faces(frame)
            detected_objects.extend(faces)
            
        # Detect general objects
        if self.enable_object_detection:
            objects = self._detect_general_objects(frame)
            detected_objects.extend(objects)
            
        # Detect text regions
        if self.enable_text_detection:
            text_regions = self._detect_text_regions(frame)
            detected_objects.extend(text_regions)
            
        # Apply tracking if enabled
        if self.tracking_enabled:
            detected_objects = self._apply_tracking(detected_objects, frame_index)
            
        # Create safe zones
        safe_zones = self._create_safe_zones(detected_objects, frame.shape)
        
        # Generate importance heatmap
        heatmap = self._generate_importance_heatmap(safe_zones, frame.shape)
        
        return DetectionResult(
            objects=detected_objects,
            safe_zones=safe_zones,
            frame_index=frame_index,
            timestamp=timestamp,
            heatmap=heatmap
        )
        
    def _detect_faces(self, frame: np.ndarray) -> List[DetectedObject]:
        """Detect faces using MediaPipe."""
        if self.face_detector is None:
            return []
            
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(rgb_frame)
        
        detected_faces = []
        
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w = frame.shape[:2]
                
                # Convert relative coordinates to absolute
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Ensure bounds are valid
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)
                
                detected_faces.append(DetectedObject(
                    type=ObjectType.FACE,
                    bbox=(x, y, width, height),
                    confidence=detection.score[0],
                    importance=self.OBJECT_IMPORTANCE[ObjectType.FACE],
                    metadata={
                        "detection_method": "mediapipe",
                        "landmarks": self._extract_face_landmarks(detection)
                    }
                ))
                
        return detected_faces
        
    def _extract_face_landmarks(self, detection):
        """Extract key face landmarks from MediaPipe detection."""
        landmarks = {}
        if hasattr(detection, 'location_data') and detection.location_data.relative_keypoints:
            for idx, keypoint in enumerate(detection.location_data.relative_keypoints):
                if idx == 0:
                    landmarks['left_eye'] = (keypoint.x, keypoint.y)
                elif idx == 1:
                    landmarks['right_eye'] = (keypoint.x, keypoint.y)
                elif idx == 2:
                    landmarks['nose_tip'] = (keypoint.x, keypoint.y)
                elif idx == 3:
                    landmarks['mouth_center'] = (keypoint.x, keypoint.y)
                elif idx == 4:
                    landmarks['left_ear'] = (keypoint.x, keypoint.y)
                elif idx == 5:
                    landmarks['right_ear'] = (keypoint.x, keypoint.y)
        return landmarks
        
    def _detect_general_objects(self, frame: np.ndarray) -> List[DetectedObject]:
        """Detect general objects using YOLO."""
        if self.yolo_model is None:
            return []
            
        detected_objects = []
        
        try:
            # Run YOLO detection
            results = self.yolo_model(frame, conf=self.confidence_threshold, verbose=False)
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                        
                        # Get class and confidence
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        # Map YOLO class to our ObjectType
                        object_type = self._map_yolo_class_to_type(cls, result.names)
                        
                        if object_type != ObjectType.UNKNOWN:
                            detected_objects.append(DetectedObject(
                                type=object_type,
                                bbox=(x, y, w, h),
                                confidence=conf,
                                importance=self.OBJECT_IMPORTANCE.get(object_type, 0.3),
                                metadata={
                                    "detection_method": "yolo",
                                    "class_name": result.names[cls]
                                }
                            ))
                            
        except Exception as e:
            if self.verbose:
                print(f"YOLO detection error: {e}")
                
        return detected_objects
        
    def _map_yolo_class_to_type(self, class_id: int, class_names: Dict) -> ObjectType:
        """Map YOLO class ID to our ObjectType."""
        class_name = class_names.get(class_id, "").lower()
        
        # Mapping of YOLO classes to our types
        mapping = {
            "person": ObjectType.PERSON,
            "face": ObjectType.FACE,
            "car": ObjectType.VEHICLE,
            "truck": ObjectType.VEHICLE,
            "bus": ObjectType.VEHICLE,
            "motorcycle": ObjectType.VEHICLE,
            "bicycle": ObjectType.VEHICLE,
            "dog": ObjectType.PET,
            "cat": ObjectType.PET,
            "horse": ObjectType.PET,
            "bird": ObjectType.PET,
            "sports ball": ObjectType.SPORTS_EQUIPMENT,
            "tennis racket": ObjectType.SPORTS_EQUIPMENT,
            "skateboard": ObjectType.SPORTS_EQUIPMENT,
            "surfboard": ObjectType.SPORTS_EQUIPMENT,
            "bottle": ObjectType.PRODUCT,
            "cup": ObjectType.PRODUCT,
            "bowl": ObjectType.FOOD,
            "pizza": ObjectType.FOOD,
            "donut": ObjectType.FOOD,
            "cake": ObjectType.FOOD,
            "sandwich": ObjectType.FOOD,
            "hot dog": ObjectType.FOOD,
        }
        
        return mapping.get(class_name, ObjectType.UNKNOWN)
        
    def _detect_text_regions(self, frame: np.ndarray) -> List[DetectedObject]:
        """Detect text regions in the frame."""
        if self.text_detector is None:
            return []
            
        detected_text = []
        
        # Simple text detection using edge detection and contours
        # In production, use EAST or similar
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by aspect ratio (text is usually wider than tall)
                aspect_ratio = w / h if h > 0 else 0
                if 1.5 < aspect_ratio < 10:
                    detected_text.append(DetectedObject(
                        type=ObjectType.TEXT,
                        bbox=(x, y, w, h),
                        confidence=0.7,  # Placeholder confidence
                        importance=self.OBJECT_IMPORTANCE[ObjectType.TEXT],
                        metadata={
                            "detection_method": "edge_detection"
                        }
                    ))
                    
        return detected_text
        
    def _apply_tracking(
        self,
        objects: List[DetectedObject],
        frame_index: int
    ) -> List[DetectedObject]:
        """Apply object tracking to maintain IDs across frames."""
        # Simple IoU-based tracking
        tracked_objects = []
        
        for obj in objects:
            best_iou = 0
            best_track_id = None
            
            # Find best matching tracked object
            for track_id, track_info in self.tracked_objects.items():
                if track_info['type'] == obj.type:
                    iou = self._calculate_iou(obj.bbox, track_info['bbox'])
                    if iou > best_iou and iou > 0.3:  # Minimum IoU threshold
                        best_iou = iou
                        best_track_id = track_id
                        
            if best_track_id is not None:
                # Update existing track
                obj.tracking_id = best_track_id
                self.tracked_objects[best_track_id] = {
                    'bbox': obj.bbox,
                    'type': obj.type,
                    'last_seen': frame_index
                }
            else:
                # Create new track
                obj.tracking_id = self.next_tracking_id
                self.tracked_objects[self.next_tracking_id] = {
                    'bbox': obj.bbox,
                    'type': obj.type,
                    'last_seen': frame_index
                }
                self.next_tracking_id += 1
                
            tracked_objects.append(obj)
            
        # Clean up old tracks
        tracks_to_remove = []
        for track_id, track_info in self.tracked_objects.items():
            if frame_index - track_info['last_seen'] > 30:  # 30 frames = ~1 second
                tracks_to_remove.append(track_id)
                
        for track_id in tracks_to_remove:
            del self.tracked_objects[track_id]
            
        return tracked_objects
        
    def _calculate_iou(
        self,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int]
    ) -> float:
        """Calculate Intersection over Union of two bounding boxes."""
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
        
    def _create_safe_zones(
        self,
        objects: List[DetectedObject],
        frame_shape: Tuple[int, int, int]
    ) -> List[SafeZone]:
        """Create safe zones from detected objects."""
        safe_zones = []
        
        for obj in objects:
            padding = self.SAFE_ZONE_PADDING.get(obj.type, 20)
            
            # Adjust padding based on object size and importance
            if obj.importance > 0.8:
                padding = int(padding * 1.2)  # Extra padding for important objects
                
            # Create safe zone
            safe_zone = SafeZone(
                bbox=obj.bbox,
                importance=obj.importance,
                padding=padding,
                object_type=obj.type
            )
            
            safe_zones.append(safe_zone)
            
        # Merge overlapping safe zones
        safe_zones = self._merge_overlapping_zones(safe_zones)
        
        return safe_zones
        
    def _merge_overlapping_zones(self, zones: List[SafeZone]) -> List[SafeZone]:
        """Merge overlapping safe zones to reduce complexity."""
        if not zones:
            return zones
            
        merged = []
        sorted_zones = sorted(zones, key=lambda z: z.importance, reverse=True)
        
        for zone in sorted_zones:
            merged_with_existing = False
            
            for i, existing in enumerate(merged):
                if zone.overlaps_with_rect(existing.get_padded_bbox()):
                    # Merge zones
                    x1, y1, w1, h1 = existing.get_padded_bbox()
                    x2, y2, w2, h2 = zone.get_padded_bbox()
                    
                    # New merged bbox
                    new_x = min(x1, x2)
                    new_y = min(y1, y2)
                    new_w = max(x1 + w1, x2 + w2) - new_x
                    new_h = max(y1 + h1, y2 + h2) - new_y
                    
                    # Update existing zone
                    merged[i] = SafeZone(
                        bbox=(new_x, new_y, new_w, new_h),
                        importance=max(existing.importance, zone.importance),
                        padding=0,  # Already padded
                        object_type=existing.object_type if existing.importance >= zone.importance else zone.object_type
                    )
                    
                    merged_with_existing = True
                    break
                    
            if not merged_with_existing:
                merged.append(zone)
                
        return merged
        
    def _generate_importance_heatmap(
        self,
        safe_zones: List[SafeZone],
        frame_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """Generate importance heatmap for visualization."""
        height, width = frame_shape[:2]
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        for zone in safe_zones:
            x, y, w, h = zone.get_padded_bbox()
            
            # Ensure bounds
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width, x + w)
            y2 = min(height, y + h)
            
            # Add importance to heatmap
            heatmap[y1:y2, x1:x2] = np.maximum(
                heatmap[y1:y2, x1:x2],
                zone.importance
            )
            
        # Apply Gaussian blur for smooth transitions
        heatmap = cv2.GaussianBlur(heatmap, (31, 31), 0)
        
        return heatmap
        
    def visualize_detection(
        self,
        frame: np.ndarray,
        detection_result: DetectionResult,
        show_heatmap: bool = True
    ) -> np.ndarray:
        """Visualize detection results on frame."""
        vis_frame = frame.copy()
        
        # Draw safe zones
        for zone in detection_result.safe_zones:
            x, y, w, h = zone.get_padded_bbox()
            
            # Color based on importance
            if zone.importance > 0.8:
                color = (0, 0, 255)  # Red for high importance
            elif zone.importance > 0.6:
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 255, 0)  # Green
                
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, 2)
            
            # Add label
            label = f"{zone.object_type.value} ({zone.importance:.2f})"
            cv2.putText(vis_frame, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                       
        # Overlay heatmap if requested
        if show_heatmap and detection_result.heatmap is not None:
            # Convert heatmap to color
            heatmap_color = cv2.applyColorMap(
                (detection_result.heatmap * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            
            # Blend with original frame
            alpha = 0.3
            vis_frame = cv2.addWeighted(vis_frame, 1 - alpha, heatmap_color, alpha, 0)
            
        return vis_frame