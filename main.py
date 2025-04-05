import asyncio
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from motion_detector import create_motion_detector
import io
import base64
from typing import Dict, List, Set
import uvicorn
import json
import os
from datetime import datetime
from collections import deque

app = FastAPI()

# Mount static files directory
os.makedirs("static", exist_ok=True)
if os.path.exists("index.html"):
    html_path = "index.html"
elif os.path.exists("static/index.html"):
    html_path = "static/index.html"
else:
    with open("static/index.html", "w") as f:
        f.write("<!-- Placeholder HTML file -->")
    html_path = "static/index.html"

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_html():
    try:
        with open(html_path, "r") as f:
            return f.read()
    except Exception as e:
        return f"""
        <html>
            <head><title>Motion Detector</title></head>
            <body>
                <h1>Error loading HTML template: {str(e)}</h1>
                <p>Please make sure the index.html file is in the correct location.</p>
            </body>
        </html>
        """

# Create a motion detector
motion_detector = create_motion_detector(
    pixel_threshold=30,
    motion_threshold=0.01,
    buffer_seconds=3.0,
    fps=30,
    save_dir="motion_captures",
    min_recording_time=5.0
)

# Store for active connections
active_connections: Set[WebSocket] = set()

@app.websocket("/ws/video-stream")
async def video_stream(websocket: WebSocket):
    await websocket.accept()
    active_connections.add(websocket)
    print("connection open")
    try:
        while True:
            # Receive frame data from client
            try:
                data = await websocket.receive_text()
                
                # Parse the frame data
                try:
                    frame_data = json.loads(data)
                    frame_base64 = frame_data.get("frame")
                    timestamp = frame_data.get("timestamp", datetime.now().isoformat())
                    
                    if not frame_base64:
                        await asyncio.sleep(0.01)  # Small delay to prevent CPU hogging
                        continue  # Skip if no frame data was received
                    
                    # Decode base64 frame - handle different formats
                    try:
                        # Try to extract the base64 part if it includes a data URL prefix
                        if "," in frame_base64:
                            frame_base64 = frame_base64.split(",")[1]
                        
                        # Decode the base64 string
                        frame_bytes = base64.b64decode(frame_base64)
                        
                        # Convert to numpy array
                        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
                        
                        # Decode the image
                        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                        
                        # Check if frame was successfully decoded
                        if frame is None or frame.size == 0:
                            print("Warning: Received empty or invalid frame")
                            await asyncio.sleep(0.01)
                            continue
                        
                        # Process the frame for motion detection
                        annotated_frame, motion_detected = motion_detector.process_frame(frame)
                        
                        # Encode the processed frame back to JPEG
                        ret, buffer = cv2.imencode('.jpg', annotated_frame)
                        if not ret:
                            print("Warning: Failed to encode processed frame")
                            await asyncio.sleep(0.01)
                            continue
                        
                        # Convert to base64 for sending
                        frame_base64 = base64.b64encode(buffer).decode('utf-8')
                        
                        # Send result back to client
                        response = {
                            "frame": f"data:image/jpeg;base64,{frame_base64}",
                            "motion_detected": motion_detected,
                            "timestamp": timestamp
                        }
                        
                        await websocket.send_json(response)
                        
                    except Exception as e:
                        print(f"Frame processing error: {str(e)}")
                        await asyncio.sleep(0.01)
                        continue
                        
                except json.JSONDecodeError:
                    print("Error: Invalid JSON received")
                    await asyncio.sleep(0.01)
                    continue
                    
            except WebSocketDisconnect:
                raise  # Re-raise to be caught by the outer try-except
            except Exception as e:
                print(f"Error receiving frame: {str(e)}")
                await asyncio.sleep(0.1)  # Add a small delay before retrying
                # Check if the connection is still alive
                try:
                    # Send a ping to check if the connection is still open
                    pong = await websocket.receive_text()
                    # If we get here, the connection is still open
                except:
                    # If an exception occurs, the connection is probably closed
                    print("WebSocket connection appears to be closed")
                    break
                
    except WebSocketDisconnect:
        print("connection closed")
        if websocket in active_connections:
            active_connections.remove(websocket)
    except Exception as e:
        print(f"Error processing video stream: {str(e)}")
        if websocket in active_connections:
            active_connections.remove(websocket)

# Endpoint to list captured motion videos
@app.get("/motion-captures")
async def list_captures():
    captures = []
    
    try:
        if os.path.exists(motion_detector.save_dir):
            for file in os.listdir(motion_detector.save_dir):
                if file.endswith(".mp4"):
                    file_path = os.path.join(motion_detector.save_dir, file)
                    file_stats = os.stat(file_path)
                    
                    captures.append({
                        "filename": file,
                        "path": file_path,
                        "size_bytes": file_stats.st_size,
                        "created": datetime.fromtimestamp(file_stats.st_ctime).isoformat()
                    })
    except Exception as e:
        print(f"Error listing captures: {str(e)}")
    
    return {"captures": sorted(captures, key=lambda x: x["created"], reverse=True)}

# Endpoint to serve a captured video
@app.get("/motion-captures/{filename}")
async def serve_capture(filename: str):
    file_path = os.path.join(motion_detector.save_dir, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    def video_generator():
        try:
            with open(file_path, "rb") as video_file:
                yield from video_file
        except Exception as e:
            print(f"Error serving video file: {str(e)}")
            raise HTTPException(status_code=500, detail="Error reading video file")
    
    return StreamingResponse(
        video_generator(),
        media_type="video/mp4"
    )

# Settings endpoint
@app.post("/api/settings")
async def update_settings(settings: dict):
    try:
        # Update motion detector settings
        if "pixel_threshold" in settings:
            motion_detector.pixel_threshold = int(settings["pixel_threshold"])
        if "motion_threshold" in settings:
            motion_detector.motion_threshold = float(settings["motion_threshold"]) / 100.0  # Convert from percentage
        if "buffer_seconds" in settings:
            new_buffer = int(settings["buffer_seconds"])
            motion_detector.buffer_frames = int(new_buffer * motion_detector.fps)
            motion_detector.frame_buffer = deque(maxlen=motion_detector.buffer_frames)
        if "min_recording_time" in settings:
            motion_detector.min_recording_frames = int(float(settings["min_recording_time"]) * motion_detector.fps)
        
        return {"status": "success", "message": "Settings updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating settings: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)