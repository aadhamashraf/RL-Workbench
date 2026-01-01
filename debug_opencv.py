
import cv2
import numpy as np
import os

t1 = 'test_avc1.mp4'
t2 = 'test_h264.mp4'

size = (100, 100)
fps = 5
frames = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(10)]

def try_codec(name, fourcc_str, path):
    print(f"Testing {name} ({fourcc_str})...")
    try:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        out = cv2.VideoWriter(path, fourcc, fps, size)
        if not out.isOpened():
            print(f"  Failed to open VideoWriter for {name}")
            return False
        
        for f in frames:
            out.write(f)
        out.release()
        
        if os.path.exists(path) and os.path.getsize(path) > 100:
            print(f"  Success! Saved to {path}")
            return True
        else:
            print(f"  Failed: File empty or not created")
            return False
    except Exception as e:
        print(f"  Error: {e}")
        return False

try_codec("AVC1", "avc1", t1)
try_codec("H264", "H264", t2)
