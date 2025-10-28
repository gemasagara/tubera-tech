import cv2
import numpy as np
import time
from datetime import datetime
import os
from ultralytics import YOLO
import RPi.GPIO as GPIO

# GPIO Configuration
GPIO_PIN = 17  # GPIO pin for ESP32 communication
GPIO.setmode(GPIO.BCM)
GPIO.setup(GPIO_PIN, GPIO.IN)

# Configuration
CAPTURE_DURATION = 60  # seconds
CAPTURE_INTERVAL = 10  # seconds
CAMERA_0_INDEX = 0
CAMERA_1_INDEX = 2

# YOLO Configuration
YOLO_MODEL_PATH = "yolo11n.pt"

# Define your custom leaf color classes (for future custom model)
LEAF_COLOR_CLASSES = ['dark_green', 'light_green', 'green', 'yellow', 'brown']
WEED_CLASS = 'crop_weed'

# Database configuration
USE_SUPABASE = True

if USE_SUPABASE:
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

def extract_detection_data(results):
    """
    Extract leaf color and weed count from YOLO detection results
    Returns: (leaf_color: str, weed_count: int)
    """
    leaf_color = "none"
    weed_count = 0
    
    try:
        if hasattr(results[0], 'names'):
            class_names = results[0].names
        else:
            return leaf_color, weed_count
        
        if hasattr(results[0], 'boxes') and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = class_names[class_id].lower()
                
                if class_name in LEAF_COLOR_CLASSES:
                    if leaf_color == "none":
                        leaf_color = class_name
                
                elif class_name == WEED_CLASS:
                    weed_count += 1
        
        return leaf_color, weed_count
    
    except Exception as e:
        print(f"  ⚠ Error extracting detection data: {str(e)}")
        return "none", 0

def run_yolo_prediction(model, image_path):
    """
    Run YOLO prediction on an image and save the result
    Returns: (predicted_path, leaf_color, weed_count)
    """
    try:
        results = model(image_path, verbose=False)
        leaf_color, weed_count = extract_detection_data(results)
        
        base_name = os.path.splitext(image_path)[0]
        predicted_path = f"{base_name}_predicted.jpg"
        
        for result in results:
            result.save(filename=predicted_path)
        
        return predicted_path, leaf_color, weed_count
    
    except Exception as e:
        print(f"  ✗ YOLO prediction failed for {image_path}: {str(e)}")
        return None, "none", 0

def initialize_cameras():
    """Initialize both cameras with low resource settings"""
    video_capture_0 = cv2.VideoCapture(CAMERA_0_INDEX)
    video_capture_1 = cv2.VideoCapture(CAMERA_1_INDEX)
    
    if not video_capture_0.isOpened() or not video_capture_1.isOpened():
        print("Error: Could not open cameras")
        return None, None
    
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

def upload_to_supabase(filename, filepath, leaf_color, weed_count):
    """Upload image to Supabase storage with metadata"""
    try:
        with open(filepath, 'rb') as f:
            supabase.storage.from_(BUCKET_NAME).upload(
                file=f,
                path=filename,
                file_options={"content-type": "image/jpeg"}
            )
        
        supabase.table('captures').insert({
            'filename': filename,
            'timestamp': datetime.now().isoformat(),
            'camera': 0 if 'camera_0' in filename else 1,
            'leaf_color': leaf_color,
            'weed_count': weed_count
        }).execute()
        
        print(f"  ✓ Uploaded {filename} to Supabase")
        print(f"    - Leaf color: {leaf_color}")
        print(f"    - Weed count: {weed_count}")
        return True
    except Exception as e:
        print(f"  ✗ Failed to upload {filename}: {str(e)}")
        return False

def save_locally(filename, frame):
    """Save image locally"""
    try:
        cv2.imwrite(filename, frame)
        print(f"  ✓ Saved {filename}")
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

def capture_phase():
    """
    PHASE 1: Just capture images quickly without any processing
    Returns list of captured image filenames
    """
    print("\n" + "="*50)
    print("PHASE 1: CAPTURING IMAGES")
    print("="*50)
    
    video_capture_0, video_capture_1 = initialize_cameras()
    if not video_capture_0 or not video_capture_1:
        return []
    
    window_name = "Dual Camera View"
    cv2.namedWindow(window_name)
    
    start_time = time.time()
    last_capture_time = start_time
    capture_count = 0
    captured_files = []
    
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
            
            if elapsed_time >= CAPTURE_DURATION:
                print(f"\n✓ Capture phase complete! Total captures: {capture_count}")
                break
            
            time_since_last_capture = time.time() - last_capture_time
            if time_since_last_capture >= CAPTURE_INTERVAL:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                filename_0 = f"camera_0_{timestamp}.jpg"
                filename_1 = f"camera_1_{timestamp}.jpg"
                
                save_locally(filename_0, video_frame_0)
                save_locally(filename_1, video_frame_1)
                
                captured_files.append(filename_0)
                captured_files.append(filename_1)
                
                capture_count += 1
                last_capture_time = time.time()
                print(f"Captured pair #{capture_count}")
            
            combined_frame = np.hstack((video_frame_0, video_frame_1))
            display_frame = combined_frame.copy()
            
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
                print(f"\n✓ Capture ended early. Captures: {capture_count}")
                break
    
    finally:
        release_cameras(video_capture_0, video_capture_1)
    
    return captured_files

def processing_phase(yolo_model, captured_files):
    """
    PHASE 2: Process captured images with YOLO, upload, and cleanup
    """
    if not captured_files:
        print("No images to process!")
        return
    
    print("\n" + "="*50)
    print("PHASE 2: PROCESSING & UPLOADING")
    print("="*50)
    print(f"Processing {len(captured_files)} images...\n")
    
    for i, filename in enumerate(captured_files, 1):
        print(f"[{i}/{len(captured_files)}] Processing {filename}...")
        
        if not os.path.exists(filename):
            print(f"  ✗ File not found: {filename}")
            continue
        
        predicted_path, leaf_color, weed_count = run_yolo_prediction(yolo_model, filename)
        
        if predicted_path:
            print(f"  ✓ YOLO prediction created")
            
            if USE_SUPABASE:
                upload_to_supabase(os.path.basename(predicted_path), predicted_path, 
                                 leaf_color, weed_count)
            
            cleanup_files(filename, predicted_path)
        else:
            if USE_SUPABASE:
                upload_to_supabase(filename, filename, "none", 0)
            cleanup_files(filename)
        
        print()
    
    print("="*50)
    print("PROCESSING COMPLETE")
    print("="*50)

def capture_sequence(yolo_model):
    """
    Main sequence: First capture all images, then process them
    """
    if yolo_model is None:
        print("ERROR: YOLO model not loaded. Cannot continue.")
        return
    
    captured_files = capture_phase()
    
    if captured_files:
        processing_phase(yolo_model, captured_files)
    
    print("\n✓ Full sequence complete!\n")

def wait_for_esp32_trigger():
    """
    Wait for ESP32 to send a HIGH signal on GPIO pin
    Returns True when triggered
    """
    print("Waiting for ESP32 trigger...")
    
    # Wait for the pin to go HIGH
    while True:
        if GPIO.input(GPIO_PIN):
            print("✓ ESP32 Request Received!")
            # Small debounce delay
            time.sleep(0.1)
            return True
        time.sleep(0.1)  # Check every 100ms to avoid busy-waiting

def main():
    print("="*50)
    print("CAMERA CAPTURE SERVICE WITH YOLO")
    print("ESP32 GPIO TRIGGER MODE")
    print("="*50)
    
    # Load YOLO model once at startup
    yolo_model = load_yolo_model()
    
    if yolo_model is None:
        print("\nERROR: Cannot start without YOLO model.")
        print("Please ensure yolo11n.pt is in the current directory.")
        GPIO.cleanup()
        return
    
    print(f"\nListening on GPIO pin {GPIO_PIN} for ESP32 trigger...")
    print("Press Ctrl+C to exit\n")
    
    try:
        while True:
            # Wait for ESP32 to trigger
            wait_for_esp32_trigger()
            
            # Run the capture sequence
            capture_sequence(yolo_model)
            
            print("Ready for next ESP32 trigger...\n")
    
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    
    except Exception as e:
        print(f"Error: {str(e)}")
    
    finally:
        # Clean up GPIO
        GPIO.cleanup()
        print("GPIO cleanup complete")

if __name__ == "__main__":
    main()
