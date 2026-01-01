
import sys
import os
import numpy as np
import traceback

# Add project root to path
sys.path.append(os.getcwd())

log_file = "verify_log.txt"

def log(msg):
    print(msg)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

if os.path.exists(log_file):
    os.remove(log_file)

try:
    log("Importing video_generator...")
    # Attempt import inside try/catch to log ImportError
    try:
        from interface.utils.video_generator import save_video_mp4
    except Exception as e:
        log(f"Import Error: {e}")
        log(traceback.format_exc())
        sys.exit(1)

    import cv2
    
    filepath = "verify_fix_output.mp4"
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
        except Exception as e:
            log(f"Failed to remove existing file: {e}")

    # Generate dummy frames (RGB)
    frames = [np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8) for _ in range(20)]
    size = (200, 200)
    fps = 5

    log("Calling save_video_mp4...")
    success = save_video_mp4(filepath, frames, fps, size)
    
    if success:
        log("Video generated successfully.")
        
        # Check codec
        if not os.path.exists(filepath):
             log("ERROR: File does not exist after success return.")
             sys.exit(1)
             
        cap = cv2.VideoCapture(filepath)
        if cap.isOpened():
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            log(f"Detected Codec: {codec}")
            
            # Allow avc1 or h264 (case insensitive)
            if codec.lower() in ['avc1', 'h264']:
                log("SUCCESS: Codec is H.264 compatible.")
            else:
                log(f"WARNING: Codec is {codec}, might not work in browser.")
        else:
            log("ERROR: Could not read generated video.")
    else:
        log("ERROR: save_video_mp4 returned False.")

except Exception as e:
    log(f"CRITICAL ERROR: {e}")
    log(traceback.format_exc())
