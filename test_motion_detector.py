import cv2
from motion_detector import create_motion_detector
import time

def main():
    # Create motion detector
    detector = create_motion_detector(
        pixel_threshold=30,
        motion_threshold=0.01,
        buffer_seconds=3.0,
        fps=30,
        save_dir="motion_captures",
        min_recording_time=5.0
    )
    
    # Open webcam or video file
    # For webcam use: cap = cv2.VideoCapture(0)
    # For video file use: cap = cv2.VideoCapture("path/to/video.mp4")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    # Get FPS information
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # Default FPS if not available
    
    print(f"Video source opened successfully, FPS: {fps}")
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("End of video stream reached")
                break
            
            # Process frame with motion detector
            annotated_frame, motion_detected = detector.process_frame(frame)
            
            # Display the resulting frame
            cv2.imshow('Motion Detection', annotated_frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Control frame rate for simulation
            time.sleep(1/fps)
    
    finally:
        # Clean up
        if detector.recording:
            detector.stop_recording()
        
        cap.release()
        cv2.destroyAllWindows()
        print("Resources released")

if __name__ == "__main__":
    main()