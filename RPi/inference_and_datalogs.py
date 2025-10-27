import cv2
import numpy as np
import time
from datetime import datetime
import os
from ultralytics import YOLO

# Configuration
CAPTURE_DURATION = 60  # seconds
CAPTURE_INTERVAL = 10  # seconds
CAMERA_0_INDEX = 0
CAMERA_1_INDEX = 2

# YOLO Configuration
YOLO_MODEL_PATH = "yolo11n.pt"  # Make sure this file is in your working directory

# Database configuration (choose your option below)
USE_SUPABASE = True  # Set to True for Supabase, False for local storage

if USE_SUPABASE:
    # Supabase configuration
    from supabase import create_client, Client
    SUPABASE_URL = "https://bnyapmfutnmeroujhdjn.supabase.co"
    SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJueWFwbWZ1dG5tZXJvdWpoZGpuIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjE1NTkzNzUsImV4cCI6MjA3NzEzNTM3NX0.yLYZOllU_7LaIHtynS8f-r9QK3oV1wXH_3xenbKaubg"
    BUCKET_NAME = "camera-images"
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def load_yolo_model():
    """Load YOLO model once at startup"""
    try:
        print("Loading YOLO model...")
        model = YOLO(YOLO_MODEL_PATH)
        print("✓ YOLO model loaded successfully")
        return model
    except Exception as e:
        print(f"✗ Failed to load YOLO model: {str(e)}")
        print("Make sure yolo11n.pt is in the current directory")
        return None

def run_yolo_prediction(model, image_path):
    """
    Run YOLO prediction on an image and save the result
    Returns the path to the predicted image
    """
    try:
        # Run inference
        results = model(image_path, verbose=False)
        
        # Generate predicted filename
        base_name = os.path.splitext(image_path)[0]
        predicted_path = f"{base_name}_predicted.jpg"
        
        # Save the prediction result
        for result in results:
            result.save(filename=predicted_path)
        
        print(f"  ✓ YOLO prediction saved: {predicted_path}")
        return predicted_path
    
    except Exception as e:
        print(f"  ✗ YOLO prediction failed for {image_path}: {str(e)}")
        return None

def initialize_cameras():
    """Initialize both cameras with low resource settings"""
    video_capture_0 = cv2.VideoCapture(CAMERA_0_INDEX)
    video_capture_1 = cv2.VideoCapture(CAMERA_1_INDEX)
    
    if not video_capture_0.isOpened() or not video_capture_1.isOpened():
        print("Error: Could not open cameras")
        return None, None
    
    # Set low resolution
    for cap in [video_capture_0, video_capture_1]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
        cap.set(cv2.CAP_PROP_FPS, 10)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    
    print("Warming up cameras...")
    time.sleep(2)
    
    return video_capture_0, video_capture_1

def release_cameras(video_capture_0, video_capture_1):
    """Release camera resources"""
    if video_capture_0:
        video_capture_0.release()
    if video_capture_1:
        video_capture_1.release()
    cv2.destroyAllWindows()

def upload_to_supabase(filename, filepath):
    """Upload image to Supabase storage"""
    try:
        with open(filepath, 'rb') as f:
            supabase.storage.from_(BUCKET_NAME).upload(
                file=f,
                path=filename,
                file_options={"content-type": "image/jpeg"}
            )
        
        # Store metadata in database table
        supabase.table('captures').insert({
            'filename': filename,
            'timestamp': datetime.now().isoformat(),
            'camera': 0 if 'camera_0' in filename else 1
        }).execute()
        
        print(f"  ✓ Uploaded {filename} to Supabase")
        return True
    except Exception as e:
        print(f"  ✗ Failed to upload {filename}: {str(e)}")
        return False

def save_locally(filename, frame):
    """Save image locally"""
    try:
        cv2.imwrite(filename, frame)
        print(f"  ✓ Saved {filename} locally")
        return True
    except Exception as e:
        print(f"  ✗ Failed to save {filename}: {str(e)}")
        return False

def cleanup_files(*filepaths):
    """Delete files from the Raspberry Pi"""
    for filepath in filepaths:
        try:
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
                print(f"  ✓ Deleted {filepath}")
        except Exception as e:
            print(f"  ✗ Failed to delete {filepath}: {str(e)}")

def capture_sequence(yolo_model):
    # Run the full capture sequence
    print("\n" + "="*50)
    print("STARTING CAPTURE SEQUENCE")
    print("="*50)
    
    if yolo_model is None:
        print("ERROR: YOLO model not loaded. Cannot continue.")
        return
    
    # Initialize cameras
    video_capture_0, video_capture_1 = initialize_cameras()
    if not video_capture_0 or not video_capture_1:
        return
    
    window_name = "Dual Camera View"
    cv2.namedWindow(window_name)
    
    start_time = time.time()
    last_capture_time = start_time
    capture_count = 0
    
    print(f"Running {CAPTURE_DURATION}s capture session...")
    print(f"Captures every {CAPTURE_INTERVAL}s")
    
    try:
        while True:
            result_0, video_frame_0 = video_capture_0.read()
            result_1, video_frame_1 = video_capture_1.read()
            
            if not result_0 or not result_1:
                continue
            
            elapsed_time = time.time() - start_time
            remaining_time = CAPTURE_DURATION - elapsed_time
            
            # Check if session complete
            if elapsed_time >= CAPTURE_DURATION:
                print(f"\n✓ Session complete! Total captures: {capture_count}")
                break
            
            # Check if time to capture
            time_since_last_capture = time.time() - last_capture_time
            if time_since_last_capture >= CAPTURE_INTERVAL:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                filename_0 = f"camera_0_{timestamp}.jpg"
                filename_1 = f"camera_1_{timestamp}.jpg"
                
                print(f"\n--- Processing capture #{capture_count + 1} ---")
                
                # Step 1: Save original images locally
                save_locally(filename_0, video_frame_0)
                save_locally(filename_1, video_frame_1)
                
                # Step 2: Run YOLO predictions
                print("Running YOLO predictions...")
                predicted_0 = run_yolo_prediction(yolo_model, filename_0)
                predicted_1 = run_yolo_prediction(yolo_model, filename_1)
                
                # Step 3: Upload predicted images to Supabase
                if USE_SUPABASE:
                    print("Uploading to Supabase...")
                    if predicted_0:
                        upload_to_supabase(os.path.basename(predicted_0), predicted_0)
                    if predicted_1:
                        upload_to_supabase(os.path.basename(predicted_1), predicted_1)
                
                # Step 4: Cleanup - delete both original and predicted images
                print("Cleaning up local files...")
                cleanup_files(filename_0, filename_1, predicted_0, predicted_1)
                
                capture_count += 1
                last_capture_time = time.time()
                print(f"✓ Capture pair #{capture_count} complete")
            
            # Display preview
            combined_frame = np.hstack((video_frame_0, video_frame_1))
            display_frame = combined_frame.copy()
            
            # Add overlay
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 0), (320, 80), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(display_frame, f"Time Left: {int(remaining_time)}s", 
                       (10, 25), font, 0.6, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Captures: {capture_count}", 
                       (10, 55), font, 0.6, (255, 255, 0), 2)
            
            next_capture_in = CAPTURE_INTERVAL - time_since_last_capture
            if next_capture_in > 0:
                cv2.putText(display_frame, f"Next: {int(next_capture_in)}s", 
                           (180, 25), font, 0.5, (255, 100, 100), 2)
            
            cv2.imshow(window_name, display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(f"\n✓ Session ended early. Captures: {capture_count}")
                break
    
    finally:
        release_cameras(video_capture_0, video_capture_1)
    
    print("\n" + "="*50)
    print("SEQUENCE COMPLETE")
    print("="*50 + "\n")

def main():
    # Main loop waiting for commands
    print("="*50)
    print("CAMERA CAPTURE SERVICE WITH YOLO")
    print("="*50)
    
    # Load YOLO model once at startup
    yolo_model = load_yolo_model()
    
    if yolo_model is None:
        print("\nERROR: Cannot start without YOLO model.")
        print("Please ensure yolo11n.pt is in the current directory.")
        return
    
    print("\nCommands:")
    print("  'start' or 's' - Start capture sequence")
    print("  'quit' or 'q'  - Exit program")
    print("\nWaiting for commands...\n")
    
    while True:
        try:
            command = input(">> ").strip().lower()
            
            if command in ['start', 's']:
                capture_sequence(yolo_model)
                print("Ready for next command...\n")
            
            elif command in ['quit', 'q', 'exit']:
                print("Shutting down...")
                break
            
            elif command == '':
                continue
            
            else:
                print(f"Unknown command: '{command}'")
                print("Use 'start' to capture or 'quit' to exit\n")
        
        except KeyboardInterrupt:
            print("\n\nShutting down...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
