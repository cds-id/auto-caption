#!/usr/bin/env python3
"""
Smart Positioning Demo for Auto-Caption

This script demonstrates the object-aware caption positioning feature that
automatically detects faces and important objects in videos to ensure captions
don't block key visual elements.

Features demonstrated:
- Face detection and tracking
- Object detection (people, text, etc.)
- Smart caption positioning
- Word-by-word animation with object avoidance
- Visualization of safe zones
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.auto_caption import (
    CaptionGenerator,
    WordTimingProcessor,
    WordAnimationStyle,
    get_video_info
)
from src.auto_caption.emotion_detector import EmotionCategory
from src.auto_caption.caption_styler import Platform, StyleIntensity
from src.auto_caption.object_detection import (
    ObjectDetector,
    PositionOptimizer,
    FaceTracker,
    PositioningStrategy
)
from src.auto_caption.utils import format_duration


def visualize_object_detection(
    video_path: str,
    output_path: str,
    detector: ObjectDetector,
    sample_interval: float = 1.0
):
    """
    Create a visualization video showing detected objects and safe zones.
    
    Args:
        video_path: Input video path
        output_path: Output video path for visualization
        detector: Configured object detector
        sample_interval: Interval between detection samples (seconds)
    """
    print(f"\n🔍 Analyzing objects in video...")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Video writer for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_interval = int(fps * sample_interval)
    frame_count = 0
    detected_objects_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection at intervals
        if frame_count % frame_interval == 0:
            timestamp = frame_count / fps
            detection_result = detector.detect_objects(
                frame,
                frame_index=frame_count,
                timestamp=timestamp
            )
            
            # Visualize detection
            vis_frame = detector.visualize_detection(
                frame,
                detection_result,
                show_heatmap=True
            )
            
            detected_objects_count += len(detection_result.objects)
            
            # Add info text
            info_text = f"Frame: {frame_count} | Objects: {len(detection_result.objects)} | Time: {timestamp:.1f}s"
            cv2.putText(vis_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show for multiple frames to make it visible
            for _ in range(frame_interval):
                out.write(vis_frame)
        else:
            out.write(frame)
        
        frame_count += 1
    
    cap.release()
    out.release()
    
    print(f"✅ Detection visualization saved to: {output_path}")
    print(f"   Total objects detected: {detected_objects_count}")


def demonstrate_smart_positioning(
    video_path: str,
    output_dir: str = "./output",
    platform: Platform = Platform.TIKTOK,
    word_animation: WordAnimationStyle = WordAnimationStyle.POP_IN
):
    """
    Demonstrate smart positioning with a video file.
    
    Args:
        video_path: Path to input video
        output_dir: Directory for output files
        platform: Target platform
        word_animation: Word animation style
    """
    print(f"\n🎬 Smart Positioning Demo")
    print(f"   Video: {video_path}")
    print(f"   Platform: {platform.value}")
    print(f"   Animation: {word_animation.value}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get video info
    video_info = get_video_info(video_path)
    resolution = (video_info['video']['width'], video_info['video']['height'])
    duration = video_info['duration']
    
    print(f"   Resolution: {resolution[0]}x{resolution[1]}")
    print(f"   Duration: {format_duration(duration)}")
    
    # Step 1: Initialize object detector
    print("\n1️⃣ Initializing object detection...")
    detector = ObjectDetector(
        enable_face_detection=True,
        enable_object_detection=True,
        enable_text_detection=True,
        tracking_enabled=True,
        model_size="medium",
        verbose=True
    )
    
    # Step 2: Create detection visualization
    vis_output = output_dir / "object_detection_visualization.mp4"
    visualize_object_detection(video_path, str(vis_output), detector)
    
    # Step 3: Initialize face tracker
    print("\n2️⃣ Initializing face tracking...")
    face_tracker = FaceTracker(
        enable_speaking_detection=True,
        prediction_horizon=15,
        verbose=True
    )
    
    # Step 4: Generate captions with smart positioning
    print("\n3️⃣ Generating captions with smart positioning...")
    generator = CaptionGenerator(
        model_name="base",
        verbose=True,
        enable_object_detection=True,
        object_detector=detector,
        face_tracker=face_tracker
    )
    
    # Generate captions
    result = generator.generate(
        video_path,
        word_timestamps=True,
        enable_smart_positioning=True,
        detect_emotions=True,
        style_captions=True,
        platform=platform
    )
    
    # Step 5: Process word-by-word timing with smart positioning
    print("\n4️⃣ Processing word-by-word timing...")
    word_processor = WordTimingProcessor(
        animation_style=word_animation,
        words_per_second=3.0,
        video_resolution=resolution,
        platform=platform,
        enable_object_detection=True,
        object_detector=detector
    )
    
    # Get detection results from the generation
    detection_results = result.get('detection_results', [])
    
    # Process segments
    word_segments = word_processor.process_segments(
        result['segments'],
        result.get('emotion_data'),
        detection_results=detection_results
    )
    
    # Update result with word segments
    result['word_segments'] = []
    for ws in word_segments:
        word_list = []
        for w in ws.words:
            word_dict = {
                'word': w.word,
                'start_time': w.start_time,
                'end_time': w.end_time,
                'duration': w.duration,
                'segment_index': w.segment_index,
                'word_index': w.word_index,
                'emotion': w.emotion.value,
                'confidence': w.confidence,
                'is_emphasized': w.is_emphasized,
                'animation_delay': w.animation_delay,
                'position_offset': w.position_offset,
                'size_multiplier': w.size_multiplier,
                'rotation_angle': w.rotation_angle,
                'custom_style': w.custom_style
            }
            word_list.append(word_dict)
        
        segment_dict = {
            'segment_index': ws.segment_index,
            'start_time': ws.start_time,
            'end_time': ws.end_time,
            'full_text': ws.full_text,
            'words': word_list,
            'emotion': ws.emotion.value,
            'confidence': ws.confidence
        }
        result['word_segments'].append(segment_dict)
    
    result['word_by_word'] = True
    result['word_animation'] = word_animation.value
    result['smart_positioning_enabled'] = True
    
    # Save results
    output_json = output_dir / f"{Path(video_path).stem}_smart_captions.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Smart captions saved to: {output_json}")
    
    # Step 6: Generate subtitle files
    print("\n5️⃣ Generating subtitle files...")
    
    # ASS subtitle with smart positioning
    ass_output = output_dir / f"{Path(video_path).stem}_smart.ass"
    generator.save_output(result, str(ass_output), 'ass')
    print(f"   ASS subtitle: {ass_output}")
    
    # SRT subtitle (fallback)
    srt_output = output_dir / f"{Path(video_path).stem}_smart.srt"
    generator.save_output(result, str(srt_output), 'srt')
    print(f"   SRT subtitle: {srt_output}")
    
    # Step 7: Create comparison visualization
    print("\n6️⃣ Creating position comparison...")
    create_position_comparison(
        video_path,
        result,
        output_dir,
        detector
    )
    
    print("\n🎉 Smart positioning demo complete!")
    print(f"   Check the output directory: {output_dir}")


def create_position_comparison(
    video_path: str,
    caption_result: dict,
    output_dir: Path,
    detector: ObjectDetector
):
    """Create a visual comparison of regular vs smart positioning."""
    print("   Creating position comparison visualization...")
    
    # Load a sample frame
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Find a frame with both faces and captions
    target_time = caption_result['duration'] / 2  # Middle of video
    target_frame = int(target_time * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("   ⚠️  Could not read frame for comparison")
        return
    
    # Detect objects in frame
    detection_result = detector.detect_objects(frame, target_frame, target_time)
    
    # Create side-by-side comparison
    height, width = frame.shape[:2]
    comparison = np.zeros((height, width * 2, 3), dtype=np.uint8)
    
    # Left side: Regular positioning (bottom center)
    regular_frame = frame.copy()
    regular_y = int(height * 0.85)
    cv2.rectangle(regular_frame, (50, regular_y - 40), (width - 50, regular_y + 10),
                 (0, 0, 255), 2)
    cv2.putText(regular_frame, "Regular Position", (width // 2 - 100, regular_y),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(regular_frame, "TRADITIONAL", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Right side: Smart positioning
    smart_frame = detector.visualize_detection(frame, detection_result, show_heatmap=False)
    
    # Find a word position from the results
    if caption_result.get('word_segments'):
        # Get a word from the middle segment
        mid_segment = caption_result['word_segments'][len(caption_result['word_segments']) // 2]
        if mid_segment['words']:
            word = mid_segment['words'][0]
            pos_offset = word.get('position_offset', (0, 0))
            
            # Convert offset to absolute position
            word_x = width // 2 + pos_offset[0]
            word_y = height // 2 + pos_offset[1]
            
            # Draw smart position
            cv2.rectangle(smart_frame, 
                         (int(word_x - 50), int(word_y - 20)),
                         (int(word_x + 50), int(word_y + 20)),
                         (0, 255, 0), 2)
            cv2.putText(smart_frame, "Smart Position", 
                       (int(word_x - 60), int(word_y + 5)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.putText(smart_frame, "SMART POSITIONING", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Combine frames
    comparison[:, :width] = regular_frame
    comparison[:, width:] = smart_frame
    
    # Save comparison
    comparison_path = output_dir / "positioning_comparison.jpg"
    cv2.imwrite(str(comparison_path), comparison)
    print(f"   Comparison saved to: {comparison_path}")


def main():
    """Main demo function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Demonstrate smart object-aware caption positioning"
    )
    parser.add_argument(
        "video",
        help="Path to video file"
    )
    parser.add_argument(
        "--output-dir",
        default="./smart_positioning_output",
        help="Output directory (default: ./smart_positioning_output)"
    )
    parser.add_argument(
        "--platform",
        choices=["tiktok", "instagram", "youtube_shorts", "general"],
        default="tiktok",
        help="Target platform (default: tiktok)"
    )
    parser.add_argument(
        "--animation",
        choices=["typewriter", "fade_in", "pop_in", "slide_in", 
                "bounce_in", "wave", "random", "karaoke", "emphasis"],
        default="pop_in",
        help="Word animation style (default: pop_in)"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode with a sample video"
    )
    
    args = parser.parse_args()
    
    # Convert string arguments to enums
    platform = Platform(args.platform)
    animation = WordAnimationStyle(args.animation)
    
    # Run demo
    if args.test_mode:
        # Create a simple test video if in test mode
        print("🧪 Running in test mode...")
        test_video = create_test_video_with_face()
        demonstrate_smart_positioning(test_video, args.output_dir, platform, animation)
    else:
        # Check if video exists
        if not os.path.exists(args.video):
            print(f"❌ Error: Video file not found: {args.video}")
            sys.exit(1)
        
        demonstrate_smart_positioning(args.video, args.output_dir, platform, animation)


def create_test_video_with_face():
    """Create a simple test video with a moving face rectangle for testing."""
    output_path = "test_video_with_face.mp4"
    
    # Video properties
    width, height = 1080, 1920  # Portrait mode
    fps = 30
    duration = 5  # seconds
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Generate frames
    for frame_num in range(fps * duration):
        # Create blank frame
        frame = np.ones((height, width, 3), dtype=np.uint8) * 50  # Dark gray
        
        # Add moving "face" rectangle
        t = frame_num / (fps * duration)
        face_x = int(width * (0.3 + 0.4 * np.sin(t * 2 * np.pi)))
        face_y = int(height * 0.3)
        face_size = 200
        
        # Draw face rectangle
        cv2.rectangle(frame,
                     (face_x - face_size//2, face_y - face_size//2),
                     (face_x + face_size//2, face_y + face_size//2),
                     (255, 200, 150), -1)  # Skin tone color
        
        # Add eyes
        cv2.circle(frame, (face_x - 40, face_y - 20), 20, (50, 50, 50), -1)
        cv2.circle(frame, (face_x + 40, face_y - 20), 20, (50, 50, 50), -1)
        
        # Add mouth
        cv2.ellipse(frame, (face_x, face_y + 40), (60, 30), 0, 0, 180, (50, 50, 50), 3)
        
        # Add some text on screen
        cv2.putText(frame, "EXISTING TEXT", (100, height - 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # Add timestamp
        cv2.putText(frame, f"Time: {frame_num/fps:.1f}s", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
        
        out.write(frame)
    
    out.release()
    print(f"✅ Test video created: {output_path}")
    return output_path


if __name__ == "__main__":
    main()