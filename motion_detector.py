# motion_detector.py

import cv2
import numpy as np
import os
from datetime import datetime
import time
from collections import deque
import threading
from typing import Optional, List, Tuple, Deque

class MotionDetector:
    def __init__(
        self,
        pixel_threshold: int = 30,
        motion_threshold: float = 0.01,
        buffer_seconds: float = 3.0,
        fps: int = 30,
        save_dir: str = "motion_captures",
        min_recording_time: float = 5.0
    ):
        """
        Initialize motion detector with background subtraction.
        
        Args:
            pixel_threshold: Threshold for pixel difference detection
            motion_threshold: Percentage of frame that must change to trigger motion
            buffer_seconds: Seconds of video to save after motion stops
            fps: Frames per second of the video stream
            save_dir: Directory to save motion clips
            min_recording_time: Minimum recording time in seconds
        """
        # Create background subtractors - we'll use two methods for better results
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False
        )
        self.bg_subtractor_knn = cv2.createBackgroundSubtractorKNN(
            history=500, dist2Threshold=400.0, detectShadows=False
        )
        
        # Parameters
        self.pixel_threshold = pixel_threshold
        self.motion_threshold = motion_threshold
        self.buffer_frames = int(buffer_seconds * fps)
        self.min_recording_frames = int(min_recording_time * fps)
        self.fps = fps
        
        # Create directory for saving
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Internal state
        self.prev_frame = None
        self.motion_detected = False
        self.motion_counter = 0
        self.frame_buffer: Deque[np.ndarray] = deque(maxlen=self.buffer_frames)
        self.recording = False
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.frame_count = 0
        self.last_saved_time = None
        
        # Thread lock for video writing operations
        self.lock = threading.Lock()
    
    def detect_motion_pixel_diff(self, frame: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        Detect motion by comparing pixel values between consecutive frames.
        
        Args:
            frame: Current frame
        
        Returns:
            Tuple of (motion_detected, difference_mask)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Initialize prev_frame if it's the first frame
        if self.prev_frame is None:
            self.prev_frame = gray
            return False, np.zeros_like(gray)
        
        # Calculate absolute difference between current and previous frame
        frame_delta = cv2.absdiff(self.prev_frame, gray)
        
        # Apply threshold to delta
        thresh = cv2.threshold(frame_delta, self.pixel_threshold, 255, cv2.THRESH_BINARY)[1]
        
        # Dilate threshold image to fill in holes
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Update previous frame
        self.prev_frame = gray
        
        # Calculate percentage of changed pixels
        changed_pixels = np.count_nonzero(thresh)
        total_pixels = thresh.size
        motion_percent = changed_pixels / total_pixels
        
        # Determine if motion is detected
        motion_detected = motion_percent > self.motion_threshold
        
        return motion_detected, thresh
    
    def detect_motion_bg_subtraction(self, frame: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        Detect motion using background subtraction models.
        
        Args:
            frame: Current frame
        
        Returns:
            Tuple of (motion_detected, foreground_mask)
        """
        # Apply background subtraction
        fg_mask_mog2 = self.bg_subtractor.apply(frame)
        fg_mask_knn = self.bg_subtractor_knn.apply(frame)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(fg_mask_mog2, fg_mask_knn)
        
        # Apply threshold to remove noise
        thresh = cv2.threshold(combined_mask, 128, 255, cv2.THRESH_BINARY)[1]
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Calculate percentage of foreground pixels
        foreground_pixels = np.count_nonzero(thresh)
        total_pixels = thresh.size
        motion_percent = foreground_pixels / total_pixels
        
        # Determine if motion is detected
        motion_detected = motion_percent > self.motion_threshold
        
        return motion_detected, thresh
    
    def start_recording(self, frame: np.ndarray) -> None:
        """Start recording motion sequence."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(self.save_dir, f"motion_{timestamp}.mp4")
        
        h, w = frame.shape[:2]
        
        with self.lock:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                video_path, fourcc, self.fps, (w, h)
            )
            
            # Write buffered frames first
            for buffered_frame in self.frame_buffer:
                self.video_writer.write(buffered_frame)
            
            # Write current frame
            self.video_writer.write(frame)
            
            self.recording = True
            self.frame_count = len(self.frame_buffer) + 1
            self.last_saved_time = time.time()
            
            print(f"Started recording motion to {video_path}")
    
    def stop_recording(self) -> None:
        """Stop the current recording."""
        with self.lock:
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            
            self.recording = False
            self.frame_count = 0
            
            print("Stopped recording motion.")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Process a frame from the video stream to detect and record motion.
        
        Args:
            frame: Video frame to process
        
        Returns:
            Tuple of (annotated_frame, motion_detected)
        """
        # Create a copy for annotation
        annotated_frame = frame.copy()
        
        # Always add frame to buffer
        self.frame_buffer.append(frame.copy())
        
        # Detect motion using both methods
        motion_pixel, diff_mask = self.detect_motion_pixel_diff(frame)
        motion_bg, bg_mask = self.detect_motion_bg_subtraction(frame)
        
        # Combine results (motion detected if either method detects it)
        current_motion = motion_pixel or motion_bg
        
        # Annotate frame
        cv2.putText(
            annotated_frame,
            f"Pixel Motion: {'Yes' if motion_pixel else 'No'}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255) if motion_pixel else (0, 255, 0),
            2
        )
        
        cv2.putText(
            annotated_frame,
            f"BG Motion: {'Yes' if motion_bg else 'No'}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255) if motion_bg else (0, 255, 0),
            2
        )
        
        # Handle motion state
        if current_motion:
            self.motion_detected = True
            self.motion_counter = self.buffer_frames
            
            # Draw rectangle to indicate motion
            cv2.putText(
                annotated_frame,
                "Motion Detected",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
            
            # Start recording if not already recording
            if not self.recording:
                self.start_recording(frame)
            # Add current frame to recording
            elif self.video_writer is not None:
                with self.lock:
                    self.video_writer.write(frame)
                    self.frame_count += 1
        else:
            # Decrease motion counter
            if self.motion_counter > 0:
                self.motion_counter -= 1
            else:
                self.motion_detected = False
            
            # Continue recording for buffer period
            if self.recording:
                if self.motion_counter == 0 and self.frame_count >= self.min_recording_frames:
                    self.stop_recording()
                elif self.video_writer is not None:
                    with self.lock:
                        self.video_writer.write(frame)
                        self.frame_count += 1
        
        # Add recording status
        if self.recording:
            cv2.putText(
                annotated_frame,
                f"Recording: {self.frame_count // self.fps}s",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
        
        return annotated_frame, self.motion_detected

def create_motion_detector(
    pixel_threshold: int = 30,
    motion_threshold: float = 0.01,
    buffer_seconds: float = 3.0,
    fps: int = 30,
    save_dir: str = "motion_captures",
    min_recording_time: float = 5.0
) -> MotionDetector:
    """
    Create and return a configured MotionDetector instance.
    
    This function can be used to create the detector that will be
    integrated with a FastAPI endpoint.
    """
    return MotionDetector(
        pixel_threshold=pixel_threshold,
        motion_threshold=motion_threshold,
        buffer_seconds=buffer_seconds,
        fps=fps,
        save_dir=save_dir,
        min_recording_time=min_recording_time
    )