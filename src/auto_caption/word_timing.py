"""
Word-by-word caption timing module for Auto-Caption.

This module provides functionality to break down caption segments into individual
words with precise timing, enabling dynamic word-by-word caption display with
emotion-aware styling.
"""

import re
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from pathlib import Path

from .emotion_detector import EmotionCategory
from .caption_styler import StyleIntensity, Platform


class WordAnimationStyle(Enum):
    """Animation styles for word appearance."""
    TYPEWRITER = "typewriter"      # Words appear one by one
    FADE_IN = "fade_in"            # Words fade in individually
    POP_IN = "pop_in"              # Words pop/scale in
    SLIDE_IN = "slide_in"          # Words slide in from side
    BOUNCE_IN = "bounce_in"        # Words bounce in
    WAVE = "wave"                  # Words appear in wave pattern
    RANDOM = "random"              # Random appearance
    KARAOKE = "karaoke"           # Highlight style like karaoke
    EMPHASIS = "emphasis"          # Key words appear with emphasis


@dataclass
class WordTiming:
    """Represents timing and styling for a single word."""
    word: str
    start_time: float
    end_time: float
    duration: float
    segment_index: int
    word_index: int
    emotion: EmotionCategory
    confidence: float
    is_emphasized: bool = False
    animation_delay: float = 0.0
    custom_style: Optional[Dict[str, Any]] = None
    position_offset: Optional[Tuple[float, float]] = None  # (x, y) offset
    size_multiplier: float = 1.0
    rotation_angle: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "word": self.word,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "segment_index": self.segment_index,
            "word_index": self.word_index,
            "emotion": self.emotion.value,
            "confidence": self.confidence,
            "is_emphasized": self.is_emphasized,
            "animation_delay": self.animation_delay,
            "custom_style": self.custom_style,
            "position_offset": self.position_offset,
            "size_multiplier": self.size_multiplier,
            "rotation_angle": self.rotation_angle
        }


@dataclass
class WordSegment:
    """A segment containing multiple word timings."""
    segment_index: int
    start_time: float
    end_time: float
    full_text: str
    words: List[WordTiming]
    emotion: EmotionCategory
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class WordTimingProcessor:
    """
    Processes caption segments to generate word-by-word timing.
    """

    # Words that typically get emphasized in different emotions
    EMOTION_EMPHASIS_WORDS = {
        EmotionCategory.HAPPY: ["amazing", "wonderful", "great", "love", "awesome", "fantastic", "beautiful"],
        EmotionCategory.SAD: ["sorry", "miss", "lost", "alone", "hurt", "pain", "cry"],
        EmotionCategory.ANGRY: ["hate", "stupid", "wrong", "never", "stop", "enough", "ridiculous"],
        EmotionCategory.EXCITED: ["wow", "omg", "yes", "incredible", "unbelievable", "crazy", "insane"],
        EmotionCategory.FEARFUL: ["scared", "afraid", "worry", "danger", "help", "terrified"],
        EmotionCategory.SARCASTIC: ["really", "totally", "obviously", "clearly", "sure", "right", "definitely"],
    }

    # Emotion-based position and size adjustments (percentages of safe zone)
    EMOTION_POSITION_STYLES = {
        EmotionCategory.HAPPY: {
            "y_position": 0.75,  # 75% down from top of safe zone
            "y_variance": 0.05,  # 5% variance for organic feel
            "x_spread": 0.03,  # 3% horizontal spread between words
            "size_base": 1.2,  # 20% larger than base
            "rotation_range": (-3, 3),  # Subtle playful rotation
            "wave_amplitude": 0.02,  # 2% wave motion
            "bounce_height": 0.03,  # 3% bounce effect
        },
        EmotionCategory.SAD: {
            "y_position": 0.85,  # 85% down - lower on screen
            "y_variance": 0.02,  # Less movement
            "x_spread": 0.01,  # Tight clustering
            "size_base": 0.85,  # Smaller, diminished
            "rotation_range": (0, 0),  # No rotation
            "wave_amplitude": 0,  # No wave
            "bounce_height": 0,  # No bounce
        },
        EmotionCategory.ANGRY: {
            "y_position": 0.7,  # 70% down - higher, confrontational
            "y_variance": 0.03,  # Some variance
            "x_spread": 0.04,  # Wide aggressive spread
            "size_base": 1.4,  # Large and bold
            "rotation_range": (-5, 5),  # More chaotic
            "wave_amplitude": 0,  # No wave
            "bounce_height": 0.01,  # Small shake
        },
        EmotionCategory.EXCITED: {
            "y_position": 0.65,  # 65% down - higher, energetic
            "y_variance": 0.08,  # High variance for bouncy feel
            "x_spread": 0.035,  # Energetic spread
            "size_base": 1.3,  # Large and energetic
            "rotation_range": (-8, 8),  # Dynamic rotation
            "wave_amplitude": 0.03,  # Pronounced wave
            "bounce_height": 0.05,  # Big bounce
        },
        EmotionCategory.FEARFUL: {
            "y_position": 0.88,  # 88% down - very low, hiding
            "y_variance": 0.01,  # Minimal movement
            "x_spread": 0.005,  # Very tight, huddled
            "size_base": 0.75,  # Small and timid
            "rotation_range": (-1, 1),  # Slight trembling
            "wave_amplitude": 0,  # No wave
            "bounce_height": 0,  # No bounce
        },
        EmotionCategory.ANXIOUS: {
            "y_position": 0.8,  # 80% down - lower middle
            "y_variance": 0.04,  # Some nervous movement
            "x_spread": 0.025,  # Moderate scatter
            "size_base": 0.9,  # Slightly smaller
            "rotation_range": (-2, 2),  # Nervous shake
            "wave_amplitude": 0.01,  # Small wave
            "bounce_height": 0.02,  # Nervous bounce
        },
        EmotionCategory.SARCASTIC: {
            "y_position": 0.72,  # 72% down - slightly elevated
            "y_variance": 0.02,  # Controlled variance
            "x_spread": 0.02,  # Moderate spread
            "size_base": 1.1,  # Slightly larger for emphasis
            "rotation_range": (-4, 0),  # One-sided tilt
            "wave_amplitude": 0,  # No wave
            "bounce_height": 0,  # No bounce
        },
        EmotionCategory.NEUTRAL: {
            "y_position": 0.8,  # 80% down - standard position
            "y_variance": 0,  # No variance
            "x_spread": 0.015,  # Normal spread
            "size_base": 1.0,  # Normal size
            "rotation_range": (0, 0),  # No rotation
            "wave_amplitude": 0,  # No wave
            "bounce_height": 0,  # No bounce
        },
        EmotionCategory.CONTEMPLATIVE: {
            "y_position": 0.73,  # 73% down - slightly elevated
            "y_variance": 0.01,  # Minimal variance
            "x_spread": 0.012,  # Tight, focused
            "size_base": 0.95,  # Slightly smaller, introspective
            "rotation_range": (0, 0),  # No rotation
            "wave_amplitude": 0.005,  # Very subtle wave
            "bounce_height": 0,  # No bounce
        }
    }

    # Platform-specific safe zones (to avoid UI elements)
    PLATFORM_SAFE_ZONES = {
        Platform.TIKTOK: {
            "top": 0.15,     # 15% from top (for user info)
            "bottom": 0.2,   # 20% from bottom (for controls)
            "left": 0.05,    # 5% from left
            "right": 0.05,   # 5% from right
        },
        Platform.INSTAGRAM: {
            "top": 0.12,     # 12% from top
            "bottom": 0.18,  # 18% from bottom
            "left": 0.05,    # 5% from left
            "right": 0.05,   # 5% from right
        },
        Platform.YOUTUBE_SHORTS: {
            "top": 0.1,      # 10% from top
            "bottom": 0.15,  # 15% from bottom
            "left": 0.05,    # 5% from left
            "right": 0.05,   # 5% from right
        },
        Platform.GENERAL: {
            "top": 0.1,      # 10% from top
            "bottom": 0.1,   # 10% from bottom
            "left": 0.1,     # 10% from left
            "right": 0.1,    # 10% from right
        }
    }

    def __init__(
        self,
        animation_style: WordAnimationStyle = WordAnimationStyle.TYPEWRITER,
        words_per_second: float = 3.0,
        min_word_duration: float = 0.15,
        max_word_duration: float = 0.8,
        emphasis_duration_multiplier: float = 1.3,
        punctuation_pause: float = 0.2,
        enable_word_timestamps: bool = True,
        video_resolution: Tuple[int, int] = (1920, 1080),
        platform: Platform = Platform.GENERAL
    ):
        """
        Initialize word timing processor.

        Args:
            animation_style: Default animation style for words
            words_per_second: Average reading speed
            min_word_duration: Minimum duration for a word
            max_word_duration: Maximum duration for a word
            emphasis_duration_multiplier: Duration multiplier for emphasized words
            punctuation_pause: Extra pause after punctuation
            enable_word_timestamps: Try to get word-level timestamps from Whisper
            video_resolution: Video resolution (width, height) for positioning
            platform: Target platform for safe zone calculations
        """
        self.animation_style = animation_style
        self.words_per_second = words_per_second
        self.min_word_duration = min_word_duration
        self.max_word_duration = max_word_duration
        self.emphasis_duration_multiplier = emphasis_duration_multiplier
        self.punctuation_pause = punctuation_pause
        self.enable_word_timestamps = enable_word_timestamps
        self.video_resolution = video_resolution
        self.platform = platform
        
        # Calculate safe zone boundaries
        self.safe_zone = self._calculate_safe_zone()

    def _calculate_safe_zone(self) -> Dict[str, float]:
        """Calculate safe zone boundaries in pixels."""
        safe_margins = self.PLATFORM_SAFE_ZONES.get(self.platform, self.PLATFORM_SAFE_ZONES[Platform.GENERAL])
        width, height = self.video_resolution
        
        return {
            "top": height * safe_margins["top"],
            "bottom": height * (1 - safe_margins["bottom"]),
            "left": width * safe_margins["left"],
            "right": width * (1 - safe_margins["right"]),
            "width": width * (1 - safe_margins["left"] - safe_margins["right"]),
            "height": height * (1 - safe_margins["top"] - safe_margins["bottom"])
        }

    def process_segments(
        self,
        segments: List[Dict[str, Any]],
        emotion_data: Optional[Dict[str, Any]] = None,
        video_resolution: Optional[Tuple[int, int]] = None
    ) -> List[WordSegment]:
        """
        Process caption segments into word-level timing.

        Args:
            segments: List of caption segments
            emotion_data: Optional emotion detection data
            video_resolution: Optional video resolution override

        Returns:
            List of WordSegment objects with word-level timing
        """
        # Update resolution if provided
        if video_resolution:
            self.video_resolution = video_resolution
            self.safe_zone = self._calculate_safe_zone()
        
        word_segments = []

        for idx, segment in enumerate(segments):
            # Extract segment info
            start_time = segment.get("start", 0.0)
            end_time = segment.get("end", 0.0)
            text = segment.get("text", "")
            
            # Get emotion for this segment
            emotion_meta = segment.get("emotion_metadata", {})
            emotion_str = emotion_meta.get("emotion", "neutral")
            emotion = self._str_to_emotion(emotion_str)
            confidence = emotion_meta.get("confidence", 0.5)

            # Check if segment has word-level timestamps
            if "words" in segment and self.enable_word_timestamps:
                # Use provided word timestamps
                word_timings = self._process_with_timestamps(
                    segment["words"],
                    idx,
                    emotion,
                    confidence
                )
            else:
                # Generate word timings by interpolation
                word_timings = self._interpolate_word_timings(
                    text,
                    start_time,
                    end_time,
                    idx,
                    emotion,
                    confidence
                )

            # Apply animation delays based on style
            word_timings = self._apply_animation_delays(
                word_timings,
                self.animation_style
            )

            # Create word segment
            word_segment = WordSegment(
                segment_index=idx,
                start_time=start_time,
                end_time=end_time,
                full_text=text,
                words=word_timings,
                emotion=emotion,
                confidence=confidence,
                metadata=segment.get("metadata", {})
            )

            word_segments.append(word_segment)

        return word_segments

    def _interpolate_word_timings(
        self,
        text: str,
        start_time: float,
        end_time: float,
        segment_index: int,
        emotion: EmotionCategory,
        confidence: float
    ) -> List[WordTiming]:
        """Interpolate word timings when word-level timestamps are not available."""
        # Tokenize text into words
        words = self._tokenize_text(text)
        if not words:
            return []

        # Calculate total duration
        total_duration = end_time - start_time
        
        # Calculate word durations based on length and emphasis
        word_durations = self._calculate_word_durations(
            words,
            total_duration,
            emotion
        )

        # Generate word timings
        word_timings = []
        current_time = start_time

        for idx, (word, duration) in enumerate(zip(words, word_durations)):
            # Check if word should be emphasized
            is_emphasized = self._should_emphasize_word(word, emotion)
            
            # Calculate emotion-based position and size
            position_style = self.EMOTION_POSITION_STYLES.get(emotion, self.EMOTION_POSITION_STYLES[EmotionCategory.NEUTRAL])
            
            # Calculate base Y position within safe zone
            base_y = self.safe_zone["top"] + (self.safe_zone["height"] * position_style["y_position"])
            y_variance = self.safe_zone["height"] * position_style["y_variance"]
            y_position = base_y + np.random.uniform(-y_variance, y_variance)
            
            # Calculate X position (centered with spread)
            center_x = self.video_resolution[0] / 2
            total_width = len(words) * position_style["x_spread"] * self.safe_zone["width"]
            start_x = center_x - (total_width / 2)
            x_position = start_x + (idx * position_style["x_spread"] * self.safe_zone["width"])
            
            # Apply wave or other positional effects
            if position_style["wave_amplitude"] > 0:
                wave_offset = position_style["wave_amplitude"] * self.safe_zone["height"] * np.sin(idx * 0.5)
                y_position += wave_offset
            
            # Convert to relative offset from center
            x_offset = x_position - center_x
            y_offset = y_position - (self.video_resolution[1] / 2)
            
            # Calculate size multiplier
            size_multiplier = position_style["size_base"]
            if is_emphasized:
                size_multiplier *= 1.2  # Subtle boost for emphasized words
            
            # Calculate rotation
            rotation = np.random.uniform(position_style["rotation_range"][0], position_style["rotation_range"][1])
            
            word_timing = WordTiming(
                word=word,
                start_time=current_time,
                end_time=current_time + duration,
                duration=duration,
                segment_index=segment_index,
                word_index=idx,
                emotion=emotion,
                confidence=confidence,
                is_emphasized=is_emphasized,
                position_offset=(x_offset, y_offset),
                size_multiplier=size_multiplier,
                rotation_angle=rotation
            )
            
            word_timings.append(word_timing)
            current_time += duration

        return word_timings

    def _process_with_timestamps(
        self,
        word_data: List[Dict[str, Any]],
        segment_index: int,
        emotion: EmotionCategory,
        confidence: float
    ) -> List[WordTiming]:
        """Process words with provided timestamps from Whisper."""
        word_timings = []

        for idx, word_info in enumerate(word_data):
            word = word_info.get("word", "")
            start = word_info.get("start", 0.0)
            end = word_info.get("end", start + self.min_word_duration)
            
            # Check if word should be emphasized
            is_emphasized = self._should_emphasize_word(word, emotion)
            
            # Calculate emotion-based position and size
            position_style = self.EMOTION_POSITION_STYLES.get(emotion, self.EMOTION_POSITION_STYLES[EmotionCategory.NEUTRAL])
            
            # Calculate base Y position within safe zone
            base_y = self.safe_zone["top"] + (self.safe_zone["height"] * position_style["y_position"])
            y_variance = self.safe_zone["height"] * position_style["y_variance"]
            y_position = base_y + np.random.uniform(-y_variance, y_variance)
            
            # Calculate X position (centered with spread)
            center_x = self.video_resolution[0] / 2
            total_width = len(word_data) * position_style["x_spread"] * self.safe_zone["width"]
            start_x = center_x - (total_width / 2)
            x_position = start_x + (idx * position_style["x_spread"] * self.safe_zone["width"])
            
            # Apply wave or other positional effects
            if position_style["wave_amplitude"] > 0:
                wave_offset = position_style["wave_amplitude"] * self.safe_zone["height"] * np.sin(idx * 0.5)
                y_position += wave_offset
            
            # Convert to relative offset from center
            x_offset = x_position - center_x
            y_offset = y_position - (self.video_resolution[1] / 2)
            
            # Calculate size multiplier
            size_multiplier = position_style["size_base"]
            if is_emphasized:
                size_multiplier *= 1.2  # Subtle boost for emphasized words
            
            # Calculate rotation
            rotation = np.random.uniform(position_style["rotation_range"][0], position_style["rotation_range"][1])
            
            word_timing = WordTiming(
                word=word,
                start_time=start,
                end_time=end,
                duration=end - start,
                segment_index=segment_index,
                word_index=idx,
                emotion=emotion,
                confidence=confidence,
                is_emphasized=is_emphasized,
                position_offset=(x_offset, y_offset),
                size_multiplier=size_multiplier,
                rotation_angle=rotation
            )
            
            word_timings.append(word_timing)

        return word_timings

    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into words while preserving punctuation."""
        # Split by whitespace while keeping punctuation attached
        words = re.findall(r'\S+', text)
        return words

    def _calculate_word_durations(
        self,
        words: List[str],
        total_duration: float,
        emotion: EmotionCategory
    ) -> List[float]:
        """Calculate duration for each word based on various factors."""
        # Calculate base weights
        weights = []
        
        for word in words:
            # Base weight from word length
            weight = len(word) ** 0.5
            
            # Adjust for punctuation
            if word.endswith(('.', '!', '?')):
                weight += self.punctuation_pause * self.words_per_second
            elif word.endswith(','):
                weight += (self.punctuation_pause * 0.5) * self.words_per_second
            
            # Adjust for emphasis
            if self._should_emphasize_word(word, emotion):
                weight *= self.emphasis_duration_multiplier
            
            weights.append(weight)
        
        # Normalize weights to fit total duration
        total_weight = sum(weights)
        if total_weight > 0:
            durations = [
                max(self.min_word_duration, 
                    min(self.max_word_duration, 
                        (w / total_weight) * total_duration))
                for w in weights
            ]
        else:
            # Fallback to equal distribution
            durations = [total_duration / len(words)] * len(words)
        
        # Adjust if total doesn't match
        duration_diff = total_duration - sum(durations)
        if duration_diff != 0:
            adjustment = duration_diff / len(durations)
            durations = [d + adjustment for d in durations]
        
        return durations

    def _should_emphasize_word(self, word: str, emotion: EmotionCategory) -> bool:
        """Determine if a word should be emphasized based on emotion."""
        # Clean word for comparison
        clean_word = word.lower().strip('.,!?;:')
        
        # Check emotion-specific emphasis words
        emphasis_words = self.EMOTION_EMPHASIS_WORDS.get(emotion, [])
        if clean_word in emphasis_words:
            return True
        
        # Check for all-caps words (already emphasized by user)
        if word.isupper() and len(word) > 1:
            return True
        
        # Check for exclamation marks
        if emotion in [EmotionCategory.HAPPY, EmotionCategory.EXCITED] and word.endswith('!'):
            return True
        
        return False

    def _apply_animation_delays(
        self,
        word_timings: List[WordTiming],
        animation_style: WordAnimationStyle
    ) -> List[WordTiming]:
        """Apply animation delays based on the selected style."""
        if not word_timings:
            return word_timings
        
        if animation_style == WordAnimationStyle.TYPEWRITER:
            # No delay needed, words appear at their start time
            pass
            
        elif animation_style == WordAnimationStyle.WAVE:
            # Create wave effect with emotion-based amplitude
            for i, word in enumerate(word_timings):
                position_style = self.EMOTION_POSITION_STYLES.get(
                    word.emotion, 
                    self.EMOTION_POSITION_STYLES[EmotionCategory.NEUTRAL]
                )
                
                # Use emotion-specific wave amplitude
                wave_amplitude = position_style.get("wave_amplitude", 0.02)
                
                word.animation_delay = wave_amplitude * abs(np.sin(i * 0.5))
                
                # Wave position is already handled in position calculation
                
        elif animation_style == WordAnimationStyle.RANDOM:
            # Random delays
            for word in word_timings:
                word.animation_delay = np.random.uniform(0, 0.3)
                
        elif animation_style == WordAnimationStyle.EMPHASIS:
            # Delay non-emphasized words
            for word in word_timings:
                if not word.is_emphasized:
                    word.animation_delay = 0.1
        
        return word_timings

    def _str_to_emotion(self, emotion_str: str) -> EmotionCategory:
        """Convert string to EmotionCategory enum."""
        for emotion in EmotionCategory:
            if emotion.value == emotion_str:
                return emotion
        return EmotionCategory.NEUTRAL

    def generate_word_by_word_ass(
        self,
        word_segments: List[WordSegment],
        output_path: str,
        platform: Platform = Platform.GENERAL,
        style_intensity: StyleIntensity = StyleIntensity.MEDIUM,
        video_resolution: Optional[Tuple[int, int]] = None
    ) -> str:
        """
        Generate ASS subtitle file with word-by-word timing.

        Args:
            word_segments: List of word segments
            output_path: Path to save ASS file
            platform: Target platform
            style_intensity: Styling intensity
            video_resolution: Video resolution override

        Returns:
            Path to generated ASS file
        """
        from .subtitle import ASSGenerator
        
        # Use provided resolution or default
        resolution = video_resolution or self.video_resolution
        
        # Convert word segments to standard segments for ASS generation
        segments = []
        
        for word_segment in word_segments:
            for word_timing in word_segment.words:
                segment = {
                    "start": word_timing.start_time + word_timing.animation_delay,
                    "end": word_timing.end_time,
                    "text": word_timing.word,
                    "emotion_metadata": {
                        "emotion": word_timing.emotion.value,
                        "confidence": word_timing.confidence,
                        "is_emphasized": word_timing.is_emphasized,
                        "word_index": word_timing.word_index,
                        "segment_index": word_timing.segment_index,
                        "animation_style": self.animation_style.value,
                        "position_offset": word_timing.position_offset,
                        "size_multiplier": word_timing.size_multiplier,
                        "rotation_angle": word_timing.rotation_angle
                    }
                }
                segments.append(segment)
        
        # Create caption data
        caption_data = {
            "segments": segments,
            "metadata": {
                "word_by_word": True,
                "animation_style": self.animation_style.value
            }
        }
        
        # Generate ASS file
        ass_generator = ASSGenerator(
            platform=platform,
            style_intensity=style_intensity,
            video_resolution=resolution
        )
        
        return ass_generator.generate_ass_file(
            caption_data,
            output_path,
            title="Word-by-Word Captions"
        )

    def create_karaoke_style_timing(
        self,
        segments: List[Dict[str, Any]],
        highlight_duration: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Create karaoke-style timing where words highlight as they're spoken.

        Args:
            segments: Original segments
            highlight_duration: Duration to highlight each word

        Returns:
            Modified segments with karaoke timing
        """
        karaoke_segments = []
        
        for segment in segments:
            word_segments = self.process_segments([segment])
            if not word_segments:
                continue
                
            word_segment = word_segments[0]
            
            # Create karaoke effect segments
            for i, word_timing in enumerate(word_segment.words):
                # Build the full line with current word highlighted
                words_before = [w.word for w in word_segment.words[:i]]
                current_word = word_timing.word
                words_after = [w.word for w in word_segment.words[i+1:]]
                
                # Create formatted text with karaoke effect
                text_parts = []
                
                # Dimmed words before
                if words_before:
                    text_parts.append(f"{{\\alpha&H80&}}{' '.join(words_before)}{{\\alpha&H00&}}")
                
                # Highlighted current word
                text_parts.append(f"{{\\fscx120\\fscy120\\b1}}{current_word}{{\\r}}")
                
                # Normal words after
                if words_after:
                    text_parts.append(' '.join(words_after))
                
                karaoke_segment = {
                    "start": word_timing.start_time,
                    "end": word_timing.start_time + highlight_duration,
                    "text": ' '.join(text_parts),
                    "emotion_metadata": segment.get("emotion_metadata", {})
                }
                
                karaoke_segments.append(karaoke_segment)
        
        return karaoke_segments


def enable_word_timestamps_in_whisper(whisper_options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enable word-level timestamps in Whisper transcription options.

    Args:
        whisper_options: Original Whisper options

    Returns:
        Modified options with word timestamps enabled
    """
    whisper_options["word_timestamps"] = True
    return whisper_options


def create_word_emphasis_rules(
    emotion: EmotionCategory,
    custom_words: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Create word emphasis rules for specific emotion.

    Args:
        emotion: The emotion category
        custom_words: Additional words to emphasize

    Returns:
        Dictionary of emphasis rules
    """
    processor = WordTimingProcessor()
    
    emphasis_rules = {
        "emotion": emotion.value,
        "default_words": processor.EMOTION_EMPHASIS_WORDS.get(emotion, []),
        "custom_words": custom_words or [],
        "rules": {
            "uppercase": True,
            "exclamation": emotion in [EmotionCategory.HAPPY, EmotionCategory.EXCITED],
            "questions": emotion in [EmotionCategory.ANXIOUS, EmotionCategory.CONFUSED],
            "repeated_letters": emotion == EmotionCategory.EXCITED
        }
    }
    
    return emphasis_rules