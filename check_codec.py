
import cv2
import os

folder = "outputs/videos"
if not os.path.exists(folder):
    print("No outputs/videos folder found.")
    exit()

files = [f for f in os.listdir(folder) if f.endswith(".mp4")]
if not files:
    print("No MP4 files found.")
    exit()

for f in files:
    path = os.path.join(folder, f)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"File: {f}, Could not open")
        continue

    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    # Decode fourcc
    # This decoding depends on endianness but typically works for ASCII tags
    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    print(f"File: {f}, Codec: {codec}")
