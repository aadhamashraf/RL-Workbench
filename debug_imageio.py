
import imageio
import numpy as np
import os

print("Testing imageio...")
try:
    dummy_frames = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(10)]
    imageio.mimsave('test_debug.mp4', dummy_frames, fps=5, codec='libx264', pixelformat='yuv420p', output_params=['-pix_fmt', 'yuv420p'])
    print("Success! test_debug.mp4 created.")
except Exception as e:
    print(f"Failed: {e}")
    # Inspect if ffmpeg is available
    try:
        import imageio_ffmpeg
        print(f"imageio-ffmpeg version: {imageio_ffmpeg.__version__}")
        print(f"Exe path: {imageio_ffmpeg.get_ffmpeg_exe()}")
    except ImportError:
        print("imageio-ffmpeg not installed")
