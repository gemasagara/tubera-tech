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
    Extract leaf color and weed count from YOLO detection results with confidence scores
    Returns: (leaf_color: str, leaf_confidence: float, weed_count: int, weed_confidences: list)
    """
    leaf_color = "none"
    leaf_confidence = 0.0
    weed_count = 0
    weed_confidences = []
    
    try:
        if hasattr(results[0], 'names'):
            class_names = results[0].names
        else:
            return leaf_color, leaf_confidence, weed_count, weed_confidences
        
        if hasattr(results[0], 'boxes') and results[0].boxes is not None:
            boxes = results[0].boxes
            
            # Track all leaf detections to find the one with highest confidence
            leaf_detections = []
            
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = class_names[class_id].lower()
                confidence = float(box.conf[0])
                
                if class_name in LEAF_COLOR_CLASSES:
                    leaf_detections.append((class_name, confidence))
                
                elif class_name == WEED_CLASS:
                    weed_count += 1
                    weed_confidences.append(confidence)
            
            # Select leaf color with highest confidence
            if leaf_detections:
                leaf_color, leaf_confidence = max(leaf_detections, key=lambda x: x[1])
        
        return leaf_color, leaf_confidence, weed_count, weed_confidences
    
    except Exception as e:
        print(f"  ⚠ Error extracting detection data: {str(e)}")
        return "none", 0.0, 0, []

def run_yolo_prediction(model, image_path):
    """
    Run YOLO prediction on an image and save the result
    Returns: (predicted_path, leaf_color, leaf_confidence, weed_count, weed_confidences)
    """
    try:
        results = model(image_path, verbose=False)
        leaf_color, leaf_confidence, weed_count, weed_confidences = extract_detection_data(results)
        
        base_name = os.path.splitext(image_path)[0]
        predicted_path = f"{base_name}_predicted.jpg"
        
        for result in results:
            result.save(filename=predicted_path)
        
        return predicted_path, leaf_color, leaf_confidence, weed_count, weed_confidences
    
    except Exception as e:
        print(f"  ✗ YOLO prediction failed for {image_path}: {str(e)}")
        return None, "none", 0.0, 0, []

def calculate_image_score(leaf_confidence, weed_confidences, leaf_color):
    """
    Calculate a weighted score for image quality
    Higher score = better representation
    
    Score components:
    - Leaf confidence: 50% weight
    - Average weed confidence: 30% weight
    - Completeness bonus: 20% (detected something meaningful)
    """
    # Base scores
    leaf_score = leaf_confidence * 0.5
    
    # Weed score: average confidence if weeds detected, 0 otherwise
    if weed_confidences:
        weed_score = (sum(weed_confidences) / len(weed_confidences)) * 0.3
    else:
        weed_score = 0.0
    
    # Completeness bonus: reward images that detected a leaf color
    completeness_bonus = 0.2 if leaf_color != "none" else 0.0
    
    total_score = leaf_score + weed_score + completeness_bonus
    
    return total_score

def select_best_image(image_data_list):
    """
    Select the best image from a list of image data
    
    Args:
        image_data_list: List of tuples (filename, predicted_path, leaf_color, 
                         leaf_confidence, weed_count, weed_confidences)
    
    Returns:
        Best image data tuple, or None if list is empty
    """
    if not image_data_list:
        return None
    
    print(f"\n  📊 Analyzing {len(image_data_list)} images to select best representation...")
    
    best_image = None
    best_score = -1
    
    for i, (filename, predicted_path, leaf_color, leaf_conf, weed_count, weed_confs) in enumerate(image_data_list, 1):
        score = calculate_image_score(leaf_conf, weed_confs, leaf_color)
        
        # Calculate average weed confidence for display
        avg_weed_conf = sum(weed_confs) / len(weed_confs) if weed_confs else 0.0
        
        print(f"    Image {i}: score={score:.3f} | leaf={leaf_color}({leaf_conf:.2f}) | "
              f"weeds={weed_count}(avg_conf={avg_weed_conf:.2f})")
        
        if score > best_score:
            best_score = score
            best_image = (filename, predicted_path, leaf_color, leaf_conf, weed_count, weed_confs)
    
    if best_image:
        print(f"  ✓ Selected: {best_image[0]} with score {best_score:.3f}")
    
    return best_image

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
                time.sleep(0.1)
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
                print(f"Captured pair #{capture_count} - {int(remaining_time)}s remaining")
            
            # Small delay to avoid busy-waiting
            time.sleep(0.1)
    
    finally:
        release_cameras(video_capture_0, video_capture_1)
    
    return captured_files

def processing_phase(yolo_model, captured_files):
    """
    PHASE 2: Process captured images with YOLO, select best, upload, and cleanup
    """
    if not captured_files:
        print("No images to process!")
        return
    
    print("\n" + "="*50)
    print("PHASE 2: PROCESSING & SELECTING BEST IMAGES")
    print("="*50)
    print(f"Processing {len(captured_files)} images...\n")
    
    # Separate images by camera
    camera_0_files = [f for f in captured_files if 'camera_0' in f]
    camera_1_files = [f for f in captured_files if 'camera_1' in f]
    
    # Process each camera's images separately
    for camera_name, files in [("Camera 0", camera_0_files), ("Camera 1", camera_1_files)]:
        if not files:
            continue
            
        print(f"\n{'='*50}")
        print(f"Processing {camera_name} - {len(files)} images")
        print('='*50)
        
        # Store all image data for this camera
        image_data_list = []
        
        # Run YOLO on all images
        for i, filename in enumerate(files, 1):
            print(f"[{i}/{len(files)}] Running YOLO on {filename}...")
            
            if not os.path.exists(filename):
                print(f"  ✗ File not found: {filename}")
                continue
            
            predicted_path, leaf_color, leaf_conf, weed_count, weed_confs = \
                run_yolo_prediction(yolo_model, filename)
            
            if predicted_path:
                print(f"  ✓ Detection complete: {leaf_color} (conf={leaf_conf:.2f}), "
                      f"{weed_count} weeds")
                image_data_list.append((filename, predicted_path, leaf_color, 
                                       leaf_conf, weed_count, weed_confs))
            else:
                print(f"  ✗ YOLO failed, skipping this image")
                cleanup_files(filename)
        
        # Select best image from this camera
        best_image = select_best_image(image_data_list)
        
        if best_image:
            filename, predicted_path, leaf_color, _, weed_count, _ = best_image
            
            print(f"\n  📤 Uploading best image from {camera_name}...")
            if USE_SUPABASE:
                upload_to_supabase(os.path.basename(predicted_path), predicted_path, 
                                 leaf_color, weed_count)
            
            # Clean up all images from this camera
            print(f"\n  🧹 Cleaning up {len(image_data_list)} processed images...")
            for img_filename, img_predicted, _, _, _, _ in image_data_list:
                cleanup_files(img_filename, img_predicted)
        else:
            print(f"\n  ⚠ No valid images found for {camera_name}")
    
    print("\n" + "="*50)
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
    print("ESP32 GPIO TRIGGER MODE (HEADLESS)")
    print("WITH BEST IMAGE SELECTION")
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
