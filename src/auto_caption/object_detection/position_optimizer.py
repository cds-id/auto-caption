"""
Position optimizer for smart caption placement.

This module optimizes caption positioning to avoid blocking important objects
while maintaining readability and visual appeal.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import cv2
from scipy.optimize import minimize
from scipy.spatial import distance

from .object_detector import SafeZone, DetectionResult, ObjectType
from ..emotion_detector import EmotionCategory
from ..caption_styler import Platform


class PositioningStrategy(Enum):
    """Caption positioning strategies."""
    BOTTOM_CENTERED = "bottom_centered"  # Traditional bottom center
    TOP_CENTERED = "top_centered"  # Top center
    DYNAMIC = "dynamic"  # Dynamically positioned based on content
    SIDES = "sides"  # Left or right side
    CORNER = "corner"  # Corner positions
    FLOW_AROUND = "flow_around"  # Flow around objects
    SPLIT = "split"  # Split caption if needed
    CURVED = "curved"  # Follow curved paths


class CaptionAlignment(Enum):
    """Caption text alignment."""
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"
    JUSTIFY = "justify"


@dataclass
class PositionConstraint:
    """Constraints for caption positioning."""
    min_margin_top: int = 50
    min_margin_bottom: int = 50
    min_margin_left: int = 50
    min_margin_right: int = 50
    max_width_ratio: float = 0.9  # Maximum width as ratio of screen
    max_height_ratio: float = 0.3  # Maximum height as ratio of screen
    preferred_y_position: float = 0.85  # Preferred Y position (0-1)
    allow_split: bool = True  # Allow splitting captions
    maintain_readability: bool = True  # Prioritize readability
    respect_safe_zones: bool = True
    platform_constraints: Optional[Dict[str, Any]] = None


@dataclass
class OptimizedPosition:
    """Optimized position for a caption."""
    x: int
    y: int
    width: int
    height: int
    alignment: CaptionAlignment
    score: float  # Quality score (0-1)
    overlap_penalty: float  # How much it overlaps with important objects
    readability_score: float  # How readable the position is
    alternative_positions: List[Tuple[int, int]]  # Alternative positions
    split_required: bool = False
    split_positions: Optional[List['OptimizedPosition']] = None


@dataclass
class OptimizationResult:
    """Result of position optimization."""
    primary_position: OptimizedPosition
    strategy_used: PositioningStrategy
    safe_zones_avoided: List[SafeZone]
    optimization_score: float
    metadata: Dict[str, Any]


class PositionOptimizer:
    """
    Optimizes caption positioning to avoid important objects.
    
    Uses various strategies to find the best position for captions
    while avoiding faces, text, and other important visual elements.
    """
    
    # Platform-specific default positions
    PLATFORM_DEFAULTS = {
        Platform.TIKTOK: {
            "preferred_y": 0.75,  # Higher to avoid UI
            "max_width": 0.85,
            "alignment": CaptionAlignment.CENTER
        },
        Platform.INSTAGRAM: {
            "preferred_y": 0.8,
            "max_width": 0.8,
            "alignment": CaptionAlignment.CENTER
        },
        Platform.YOUTUBE_SHORTS: {
            "preferred_y": 0.85,
            "max_width": 0.9,
            "alignment": CaptionAlignment.CENTER
        },
        Platform.GENERAL: {
            "preferred_y": 0.9,
            "max_width": 0.9,
            "alignment": CaptionAlignment.CENTER
        }
    }
    
    # Emotion-based position preferences
    EMOTION_POSITION_BIAS = {
        EmotionCategory.HAPPY: {"y_bias": -0.1, "spread": True},  # Higher, more spread
        EmotionCategory.SAD: {"y_bias": 0.05, "spread": False},  # Lower, compact
        EmotionCategory.ANGRY: {"y_bias": -0.15, "spread": True},  # Higher, aggressive
        EmotionCategory.EXCITED: {"y_bias": -0.2, "spread": True},  # Very high, dynamic
        EmotionCategory.FEARFUL: {"y_bias": 0.1, "spread": False},  # Lower, hiding
        EmotionCategory.CONTEMPLATIVE: {"y_bias": -0.05, "spread": False},  # Slightly higher
    }
    
    def __init__(
        self,
        video_resolution: Tuple[int, int],
        platform: Platform = Platform.GENERAL,
        default_strategy: PositioningStrategy = PositioningStrategy.DYNAMIC,
        grid_resolution: int = 20,  # Grid cells for optimization
        verbose: bool = False
    ):
        """
        Initialize position optimizer.
        
        Args:
            video_resolution: Video resolution (width, height)
            platform: Target platform
            default_strategy: Default positioning strategy
            grid_resolution: Resolution of optimization grid
            verbose: Enable verbose output
        """
        self.video_resolution = video_resolution
        self.platform = platform
        self.default_strategy = default_strategy
        self.grid_resolution = grid_resolution
        self.verbose = verbose
        
        # Calculate grid
        self.grid_width = video_resolution[0] // grid_resolution
        self.grid_height = video_resolution[1] // grid_resolution
        
        # Platform defaults
        self.platform_defaults = self.PLATFORM_DEFAULTS.get(
            platform, 
            self.PLATFORM_DEFAULTS[Platform.GENERAL]
        )
        
    def optimize_position(
        self,
        caption_text: str,
        detection_result: DetectionResult,
        caption_size: Tuple[int, int],  # Estimated width, height of caption
        constraints: Optional[PositionConstraint] = None,
        emotion: Optional[EmotionCategory] = None,
        strategy: Optional[PositioningStrategy] = None
    ) -> OptimizationResult:
        """
        Find optimal position for caption.
        
        Args:
            caption_text: The caption text
            detection_result: Object detection results
            caption_size: Estimated size of caption (width, height)
            constraints: Position constraints
            emotion: Emotion for position bias
            strategy: Positioning strategy to use
            
        Returns:
            OptimizationResult with optimal position
        """
        if constraints is None:
            constraints = self._get_default_constraints()
            
        strategy = strategy or self.default_strategy
        
        # Apply emotion bias to constraints if provided
        if emotion:
            constraints = self._apply_emotion_bias(constraints, emotion)
        
        # Choose optimization method based on strategy
        if strategy == PositioningStrategy.BOTTOM_CENTERED:
            result = self._optimize_bottom_centered(
                caption_size, detection_result, constraints
            )
        elif strategy == PositioningStrategy.DYNAMIC:
            result = self._optimize_dynamic(
                caption_size, detection_result, constraints
            )
        elif strategy == PositioningStrategy.FLOW_AROUND:
            result = self._optimize_flow_around(
                caption_text, caption_size, detection_result, constraints
            )
        elif strategy == PositioningStrategy.SIDES:
            result = self._optimize_sides(
                caption_size, detection_result, constraints
            )
        else:
            # Default to bottom centered
            result = self._optimize_bottom_centered(
                caption_size, detection_result, constraints
            )
            
        return result
        
    def optimize_word_positions(
        self,
        words: List[str],
        word_sizes: List[Tuple[int, int]],
        detection_result: DetectionResult,
        emotion: Optional[EmotionCategory] = None,
        animation_style: Optional[str] = None
    ) -> List[OptimizedPosition]:
        """
        Optimize positions for word-by-word captions.
        
        Args:
            words: List of words
            word_sizes: Size of each word (width, height)
            detection_result: Object detection results
            emotion: Emotion for styling
            animation_style: Animation style for words
            
        Returns:
            List of optimized positions for each word
        """
        positions = []
        
        # Get base position area
        base_constraints = self._get_default_constraints()
        if emotion:
            base_constraints = self._apply_emotion_bias(base_constraints, emotion)
            
        # Find optimal flow path avoiding objects
        flow_path = self._calculate_flow_path(
            detection_result, 
            base_constraints
        )
        
        # Position words along the flow path
        current_x = flow_path[0][0]
        current_y = flow_path[0][1]
        path_index = 0
        
        for i, (word, size) in enumerate(zip(words, word_sizes)):
            width, height = size
            
            # Find position that doesn't overlap with safe zones
            pos = self._find_word_position(
                current_x, current_y,
                width, height,
                detection_result.safe_zones,
                flow_path,
                path_index
            )
            
            # Create optimized position
            opt_pos = OptimizedPosition(
                x=pos[0],
                y=pos[1],
                width=width,
                height=height,
                alignment=CaptionAlignment.CENTER,
                score=self._calculate_position_score(
                    pos[0], pos[1], width, height,
                    detection_result.safe_zones
                ),
                overlap_penalty=0.0,
                readability_score=1.0,
                alternative_positions=[]
            )
            
            positions.append(opt_pos)
            
            # Update current position
            current_x = pos[0] + width + 10  # 10px spacing
            
            # Move to next path point if needed
            if current_x > self.video_resolution[0] - 100:
                path_index = min(path_index + 1, len(flow_path) - 1)
                current_x = flow_path[path_index][0]
                current_y = flow_path[path_index][1]
                
        return positions
        
    def _get_default_constraints(self) -> PositionConstraint:
        """Get default constraints based on platform."""
        return PositionConstraint(
            min_margin_top=int(self.video_resolution[1] * 0.1),
            min_margin_bottom=int(self.video_resolution[1] * 0.1),
            min_margin_left=int(self.video_resolution[0] * 0.05),
            min_margin_right=int(self.video_resolution[0] * 0.05),
            max_width_ratio=self.platform_defaults["max_width"],
            preferred_y_position=self.platform_defaults["preferred_y"],
            platform_constraints=self.platform_defaults
        )
        
    def _apply_emotion_bias(
        self,
        constraints: PositionConstraint,
        emotion: EmotionCategory
    ) -> PositionConstraint:
        """Apply emotion-based bias to constraints."""
        bias = self.EMOTION_POSITION_BIAS.get(emotion, {})
        
        if "y_bias" in bias:
            constraints.preferred_y_position += bias["y_bias"]
            constraints.preferred_y_position = max(0.1, min(0.9, constraints.preferred_y_position))
            
        return constraints
        
    def _optimize_bottom_centered(
        self,
        caption_size: Tuple[int, int],
        detection_result: DetectionResult,
        constraints: PositionConstraint
    ) -> OptimizationResult:
        """Optimize for traditional bottom-centered position."""
        width, height = caption_size
        
        # Start with preferred position
        x = (self.video_resolution[0] - width) // 2
        y = int(self.video_resolution[1] * constraints.preferred_y_position - height // 2)
        
        # Check for overlaps and adjust
        best_y = y
        min_overlap = float('inf')
        
        # Try different Y positions
        for test_y in range(
            constraints.min_margin_top,
            self.video_resolution[1] - height - constraints.min_margin_bottom,
            self.grid_resolution
        ):
            overlap = self._calculate_overlap(
                x, test_y, width, height,
                detection_result.safe_zones
            )
            
            if overlap < min_overlap:
                min_overlap = overlap
                best_y = test_y
                
                if overlap == 0:
                    break  # Found perfect position
                    
        # Create optimized position
        score = self._calculate_position_score(
            x, best_y, width, height,
            detection_result.safe_zones
        )
        
        position = OptimizedPosition(
            x=x,
            y=best_y,
            width=width,
            height=height,
            alignment=CaptionAlignment.CENTER,
            score=score,
            overlap_penalty=min_overlap,
            readability_score=self._calculate_readability_score(x, best_y, width, height),
            alternative_positions=[]
        )
        
        return OptimizationResult(
            primary_position=position,
            strategy_used=PositioningStrategy.BOTTOM_CENTERED,
            safe_zones_avoided=[z for z in detection_result.safe_zones if not z.overlaps_with_rect((x, best_y, width, height))],
            optimization_score=score,
            metadata={"iterations": 1, "overlap": min_overlap}
        )
        
    def _optimize_dynamic(
        self,
        caption_size: Tuple[int, int],
        detection_result: DetectionResult,
        constraints: PositionConstraint
    ) -> OptimizationResult:
        """Dynamic optimization using grid search."""
        width, height = caption_size
        
        # Create score grid
        scores = np.zeros((self.grid_height, self.grid_width))
        
        # Calculate scores for each grid position
        for gy in range(self.grid_height):
            for gx in range(self.grid_width):
                x = gx * self.grid_resolution
                y = gy * self.grid_resolution
                
                # Check if position is valid
                if (x + width <= self.video_resolution[0] and
                    y + height <= self.video_resolution[1] and
                    x >= constraints.min_margin_left and
                    y >= constraints.min_margin_top):
                    
                    scores[gy, gx] = self._calculate_position_score(
                        x, y, width, height,
                        detection_result.safe_zones
                    )
                else:
                    scores[gy, gx] = -1  # Invalid position
                    
        # Find best position
        best_idx = np.unravel_index(np.argmax(scores), scores.shape)
        best_x = best_idx[1] * self.grid_resolution
        best_y = best_idx[0] * self.grid_resolution
        best_score = scores[best_idx]
        
        # Find alternative positions
        alternatives = []
        flat_scores = scores.flatten()
        sorted_indices = np.argsort(flat_scores)[::-1]
        
        for idx in sorted_indices[1:6]:  # Top 5 alternatives
            if flat_scores[idx] > 0:
                ay, ax = np.unravel_index(idx, scores.shape)
                alternatives.append((ax * self.grid_resolution, ay * self.grid_resolution))
                
        # Create optimized position
        position = OptimizedPosition(
            x=best_x,
            y=best_y,
            width=width,
            height=height,
            alignment=CaptionAlignment.CENTER,
            score=best_score,
            overlap_penalty=1.0 - best_score,
            readability_score=self._calculate_readability_score(best_x, best_y, width, height),
            alternative_positions=alternatives
        )
        
        return OptimizationResult(
            primary_position=position,
            strategy_used=PositioningStrategy.DYNAMIC,
            safe_zones_avoided=[z for z in detection_result.safe_zones if not z.overlaps_with_rect((best_x, best_y, width, height))],
            optimization_score=best_score,
            metadata={"grid_resolution": self.grid_resolution, "alternatives": len(alternatives)}
        )
        
    def _optimize_flow_around(
        self,
        caption_text: str,
        caption_size: Tuple[int, int],
        detection_result: DetectionResult,
        constraints: PositionConstraint
    ) -> OptimizationResult:
        """Optimize caption to flow around objects."""
        # This is a more complex optimization that might split text
        # For now, implement a simpler version
        
        # Try to find the largest clear area
        clear_areas = self._find_clear_areas(detection_result, constraints)
        
        if not clear_areas:
            # Fall back to dynamic optimization
            return self._optimize_dynamic(caption_size, detection_result, constraints)
            
        # Use the largest clear area
        best_area = max(clear_areas, key=lambda a: a[2] * a[3])
        x, y, w, h = best_area
        
        # Center caption in clear area
        caption_x = x + (w - caption_size[0]) // 2
        caption_y = y + (h - caption_size[1]) // 2
        
        position = OptimizedPosition(
            x=caption_x,
            y=caption_y,
            width=caption_size[0],
            height=caption_size[1],
            alignment=CaptionAlignment.CENTER,
            score=0.8,  # Good but not perfect
            overlap_penalty=0.0,
            readability_score=self._calculate_readability_score(caption_x, caption_y, caption_size[0], caption_size[1]),
            alternative_positions=[]
        )
        
        return OptimizationResult(
            primary_position=position,
            strategy_used=PositioningStrategy.FLOW_AROUND,
            safe_zones_avoided=detection_result.safe_zones,
            optimization_score=0.8,
            metadata={"clear_areas_found": len(clear_areas)}
        )
        
    def _optimize_sides(
        self,
        caption_size: Tuple[int, int],
        detection_result: DetectionResult,
        constraints: PositionConstraint
    ) -> OptimizationResult:
        """Optimize for side positioning (left or right)."""
        width, height = caption_size
        
        # Try both sides
        left_x = constraints.min_margin_left
        right_x = self.video_resolution[0] - width - constraints.min_margin_right
        y = int(self.video_resolution[1] * 0.5 - height // 2)  # Center vertically
        
        # Calculate scores for both sides
        left_score = self._calculate_position_score(
            left_x, y, width, height,
            detection_result.safe_zones
        )
        
        right_score = self._calculate_position_score(
            right_x, y, width, height,
            detection_result.safe_zones
        )
        
        # Choose better side
        if left_score > right_score:
            best_x = left_x
            alignment = CaptionAlignment.LEFT
            score = left_score
        else:
            best_x = right_x
            alignment = CaptionAlignment.RIGHT
            score = right_score
            
        position = OptimizedPosition(
            x=best_x,
            y=y,
            width=width,
            height=height,
            alignment=alignment,
            score=score,
            overlap_penalty=1.0 - score,
            readability_score=self._calculate_readability_score(best_x, y, width, height),
            alternative_positions=[]
        )
        
        return OptimizationResult(
            primary_position=position,
            strategy_used=PositioningStrategy.SIDES,
            safe_zones_avoided=[z for z in detection_result.safe_zones if not z.overlaps_with_rect((best_x, y, width, height))],
            optimization_score=score,
            metadata={"side": "left" if best_x == left_x else "right"}
        )
        
    def _calculate_overlap(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        safe_zones: List[SafeZone]
    ) -> float:
        """Calculate overlap penalty for a position."""
        total_overlap = 0.0
        caption_area = width * height
        
        for zone in safe_zones:
            # Calculate intersection area
            zx, zy, zw, zh = zone.get_padded_bbox()
            
            # Intersection bounds
            ix1 = max(x, zx)
            iy1 = max(y, zy)
            ix2 = min(x + width, zx + zw)
            iy2 = min(y + height, zy + zh)
            
            if ix2 > ix1 and iy2 > iy1:
                intersection_area = (ix2 - ix1) * (iy2 - iy1)
                # Weight by zone importance
                weighted_overlap = (intersection_area / caption_area) * zone.importance
                total_overlap += weighted_overlap
                
        return min(total_overlap, 1.0)
        
    def _calculate_position_score(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        safe_zones: List[SafeZone]
    ) -> float:
        """Calculate overall quality score for a position."""
        # Start with perfect score
        score = 1.0
        
        # Penalize overlap
        overlap = self._calculate_overlap(x, y, width, height, safe_zones)
        score -= overlap * 0.8  # Heavy penalty for overlap
        
        # Penalize distance from preferred position
        preferred_y = self.video_resolution[1] * self.platform_defaults["preferred_y"]
        y_distance = abs(y + height/2 - preferred_y) / self.video_resolution[1]
        score -= y_distance * 0.2  # Moderate penalty for distance
        
        # Bonus for centered X position
        center_x = self.video_resolution[0] / 2
        x_distance = abs(x + width/2 - center_x) / self.video_resolution[0]
        score += (1 - x_distance) * 0.1  # Small bonus for centering
        
        return max(0, min(1, score))
        
    def _calculate_readability_score(
        self,
        x: int,
        y: int,
        width: int,
        height: int
    ) -> float:
        """Calculate readability score based on position."""
        score = 1.0
        
        # Penalize positions too close to edges
        edge_margin = 50
        if x < edge_margin or x + width > self.video_resolution[0] - edge_margin:
            score -= 0.2
        if y < edge_margin or y + height > self.video_resolution[1] - edge_margin:
            score -= 0.2
            
        # Prefer lower positions (more natural for reading)
        y_ratio = y / self.video_resolution[1]
        if y_ratio < 0.3:  # Too high
            score -= 0.3
        elif y_ratio > 0.7:  # Good position
            score += 0.1
            
        return max(0, min(1, score))
        
    def _find_clear_areas(
        self,
        detection_result: DetectionResult,
        constraints: PositionConstraint
    ) -> List[Tuple[int, int, int, int]]:
        """Find clear rectangular areas in the frame."""
        # Simple implementation - find areas not covered by safe zones
        clear_areas = []
        
        # Create occupancy grid
        grid = np.zeros((self.grid_height, self.grid_width), dtype=bool)
        
        # Mark occupied cells
        for zone in detection_result.safe_zones:
            x, y, w, h = zone.get_padded_bbox()
            gx1 = x // self.grid_resolution
            gy1 = y // self.grid_resolution
            gx2 = (x + w) // self.grid_resolution
            gy2 = (y + h) // self.grid_resolution
            
            grid[gy1:gy2+1, gx1:gx2+1] = True
            
        # Find rectangular clear areas
        # This is a simplified version - you could use more sophisticated algorithms
        for gy in range(self.grid_height - 2):
            for gx in range(self.grid_width - 4):
                # Check for 4x2 grid cells clear area (minimum for caption)
                if not np.any(grid[gy:gy+2, gx:gx+4]):
                    clear_areas.append((
                        gx * self.grid_resolution,
                        gy * self.grid_resolution,
                        4 * self.grid_resolution,
                        2 * self.grid_resolution
                    ))
                    
        return clear_areas
        
    def _calculate_flow_path(
        self,
        detection_result: DetectionResult,
        constraints: PositionConstraint
    ) -> List[Tuple[int, int]]:
        """Calculate flow path avoiding objects."""
        # Simple implementation - create path at preferred Y that avoids objects
        path = []
        
        preferred_y = int(self.video_resolution[1] * constraints.preferred_y_position)
        
        # Check if preferred line is clear
        y_variations = [0, -50, 50, -100, 100, -150, 150]
        
        for y_offset in y_variations:
            test_y = preferred_y + y_offset
            if constraints.min_margin_top <= test_y <= self.video_resolution[1] - constraints.min_margin_bottom:
                # Check if this Y level is relatively clear
                overlaps = 0
                for zone in detection_result.safe_zones:
                    _, zy, _, zh = zone.get_padded_bbox()
                    if zy <= test_y <= zy + zh:
                        overlaps += 1
                        
                if overlaps < 2:  # Acceptable
                    # Create path at this Y level
                    for x in range(constraints.min_margin_left, 
                                 self.video_resolution[0] - constraints.min_margin_right, 
                                 50):
                        path.append((x, test_y))
                    break
                    
        if not path:
            # Fallback to preferred Y
            for x in range(constraints.min_margin_left, 
                         self.video_resolution[0] - constraints.min_margin_right, 
                         50):
                path.append((x, preferred_y))
                
        return path
        
    def _find_word_position(
        self,
        start_x: int,
        start_y: int,
        width: int,
        height: int,
        safe_zones: List[SafeZone],
        flow_path: List[Tuple[int, int]],
        path_index: int
    ) -> Tuple[int, int]:
        """Find position for a single word."""
        # Start from suggested position
        best_x, best_y = start_x, start_y
        min_overlap = self._calculate_overlap(start_x, start_y, width, height, safe_zones)
        
        if min_overlap == 0:
            return (best_x, best_y)  # Already good
            
        # Try nearby positions
        search_radius = 100
        step = 20
        
        for dy in range(-search_radius, search_radius + 1, step):
            for dx in range(-search_radius, search_radius + 1, step):
                test_x = start_x + dx
                test_y = start_y + dy
                
                # Check bounds
                if (0 <= test_x <= self.video_resolution[0] - width and
                    0 <= test_y <= self.video_resolution[1] - height):
                    
                    overlap = self._calculate_overlap(test_x, test_y, width, height, safe_zones)
                    if overlap < min_overlap:
                        min_overlap = overlap
                        best_x, best_y = test_x, test_y
                        
                        if overlap == 0:
                            return (best_x, best_y)
                            
        return (best_x, best_y)