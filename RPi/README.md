# Tubera Camera Capture with YOLO (Headless Mode)

### Required Files
- `camera_capture.py` (your main script)
- `yolo11n.pt` (YOLO model file - must be in the same directory)

---

## Configuration

### Camera Settings
Edit these values in the script if needed:
```python
CAPTURE_DURATION = 60   # Total capture time in seconds
CAPTURE_INTERVAL = 10   # Time between captures in seconds
CAMERA_0_INDEX = 0      # First camera device index
CAMERA_1_INDEX = 2      # Second camera device index
```

### GPIO Pin
```python
GPIO_PIN = 17  # GPIO pin for ESP32 communication
```

### Supabase Configuration
Your Supabase credentials are already in the script:
- URL and API key are pre-configured
- Bucket name: `camera-images`
- Make sure your Supabase project has the correct storage bucket and `captures` table

---

## Running the Script

### Option 1: Run Manually (Foreground)
```bash
cd /path/to/your/project
python3 camera_capture.py
```
Press `Ctrl+C` to stop.

### Option 2: Run in Background with Logging
```bash
# Basic background run
python3 camera_capture.py > camera_log.txt 2>&1 &

# Using nohup (keeps running after logout)
nohup python3 camera_capture.py > camera_log.txt 2>&1 &
```

**What this means:**
- `>` redirects standard output to file
- `2>&1` redirects errors to the same file
- `&` runs process in background

**Managing background process:**
```bash
# Check if running
ps aux | grep camera_capture.py

# View live log
tail -f camera_log.txt

# Stop the process (replace PID with actual number)
kill <PID>
```

---

## Auto-Start on Boot (Systemd Service)

### Step 1: Create Service File
```bash
sudo nano /etc/systemd/system/camera-capture.service
```

### Step 2: Add This Content
```ini
[Unit]
Description=Camera Capture with YOLO Detection
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/camera-project
ExecStart=/usr/bin/python3 /home/pi/camera-project/camera_capture.py
Restart=on-failure
RestartSec=10
StandardOutput=append:/home/pi/camera-project/camera_log.txt
StandardError=append:/home/pi/camera-project/camera_log.txt

[Install]
WantedBy=multi-user.target
```

**Important:** Replace `/home/pi/camera-project` with your actual project path!

### Step 3: Enable and Start Service
```bash
# Reload systemd to recognize new service
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable camera-capture.service

# Start service now
sudo systemctl start camera-capture.service
```

### Step 4: Verify Service is Running
```bash
# Check service status
sudo systemctl status camera-capture.service

# View recent logs
journalctl -u camera-capture.service -n 50

# Follow logs in real-time
journalctl -u camera-capture.service -f
```

---

## Managing the Service

### Common Commands
```bash
# Start service
sudo systemctl start camera-capture.service

# Stop service
sudo systemctl stop camera-capture.service

# Restart service
sudo systemctl restart camera-capture.service

# Check status
sudo systemctl status camera-capture.service

# Disable auto-start on boot
sudo systemctl disable camera-capture.service

# Re-enable auto-start
sudo systemctl enable camera-capture.service
```

### View Logs
```bash
# Last 100 lines of log
journalctl -u camera-capture.service -n 100

# Follow live log (Ctrl+C to exit)
journalctl -u camera-capture.service -f

# Log since last boot
journalctl -u camera-capture.service -b

# Log for specific date
journalctl -u camera-capture.service --since "2025-10-28"
```

---

## Troubleshooting

### Service Won't Start
```bash
# Check for errors
sudo systemctl status camera-capture.service
journalctl -u camera-capture.service -n 50

# Common issues:
# 1. Wrong file path in service file
# 2. Missing yolo11n.pt model file
# 3. Camera permission issues
# 4. Python dependencies not installed
```

### Camera Not Found
```bash
# List available cameras
ls -l /dev/video*

# Test camera access
v4l2-ctl --list-devices

# Add user to video group
sudo usermod -a -G video pi
```

### GPIO Permission Issues
```bash
# Add user to gpio group
sudo usermod -a -G gpio pi

# Reboot may be required
sudo reboot
```

### YOLO Model Not Loading
```bash
# Verify model file exists
ls -lh yolo11n.pt

# Check file permissions
chmod 644 yolo11n.pt

# Verify ultralytics installation
pip3 show ultralytics
```

### Service Crashes/Restarts
The service is configured to automatically restart on failure. Check logs:
```bash
journalctl -u camera-capture.service --since "1 hour ago"
```

---

## How It Works

### Workflow
1. **Service starts** and loads YOLO model
2. **Waits for GPIO trigger** from ESP32 (pin goes HIGH)
3. **PHASE 1 - Capture**: Takes photos every 10 seconds for 60 seconds
4. **PHASE 2 - Process**: Runs YOLO detection, uploads to Supabase, cleans up files
5. **Returns to waiting** for next ESP32 trigger

### Output
- Images saved temporarily during capture
- YOLO predictions generated
- Results uploaded to Supabase with metadata:
  - Filename
  - Timestamp
  - Camera number (0 or 1)
  - Detected leaf color
  - Crop Weed count
- Local files deleted after upload

---

## Testing

### Test Without ESP32 Trigger
Modify the `main()` function temporarily to skip GPIO waiting:
```python
# Comment out this line in main():
# wait_for_esp32_trigger()

# Replace with:
time.sleep(5)  # Just wait 5 seconds
```

### Manual GPIO Trigger
```bash
# Install GPIO testing tool
sudo apt-get install python3-gpiozero

# Test GPIO pin 17
python3 -c "from gpiozero import LED; led = LED(17); led.on()"
```

---

## File Structure
```
/home/pi/camera-project/
├── camera_capture.py       # Main script
├── yolo11n.pt             # YOLO model
├── camera_log.txt         # Output log (created automatically)
└── README.md              # This file
```

---

## Performance Tips

1. **Lower camera resolution** for faster processing (already set to 160x120)
2. **Adjust capture intervals** if needed (CAPTURE_INTERVAL setting)
3. **Monitor Raspberry Pi temperature** during extended operation
4. **Use SD card with good write speed** for image operations

---

## Uninstalling

### Remove Service
```bash
# Stop and disable service
sudo systemctl stop camera-capture.service
sudo systemctl disable camera-capture.service

# Remove service file
sudo rm /etc/systemd/system/camera-capture.service

# Reload systemd
sudo systemctl daemon-reload
```

### Remove Files
```bash
cd /home/pi/camera-project
rm camera_capture.py yolo11n.pt camera_log.txt
```

---

## Support

### Check System Status
```bash
# Raspberry Pi info
cat /proc/cpuinfo | grep Model

# Temperature
vcgencmd measure_temp

# Memory usage
free -h

# Disk space
df -h
```

### Debug Mode
Add verbose logging by modifying the script temporarily:
```python
# At the top of the script
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Author
TuberaTech Team

## Version
0.1 - Headless Mode Release
