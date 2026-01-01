"""Video generator for RL agents using MP4 format"""
import os
import gc
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import imageio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

CUSTOM_ENVS = ['TicTacToe', 'Maze', 'Snake', 'Warehouse', 'TrafficLight']


def generate_inference_video(env, agent, n_eps, fps, env_name, algo, progress_cb=None):
    """Generate MP4 video showing agent inference"""
    
    out_dir = Path("outputs/videos")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = out_dir / f"{env_name}_{algo}_{n_eps}eps.mp4"
    
    frames = []
    stats = []
    
    if env_name == 'Acrobot':
        size = (320, 240)
        method = Image.NEAREST
    elif env_name in ['CartPole', 'MountainCar']:
        size = (400, 300)
        method = Image.BILINEAR
    elif env_name in CUSTOM_ENVS:
        size = (300, 300)
        method = Image.NEAREST
    else:
        size = (500, 350)
        method = Image.BILINEAR
    
    if env_name == 'Acrobot':
        max_steps = 150
        skip = 5
    elif env_name in ['CartPole', 'MountainCar']:
        max_steps = 250
        skip = 3
    elif env_name in CUSTOM_ENVS:
        max_steps = 200
        skip = 4
    else:
        max_steps = 400
        skip = 2
    
    print(f"\nGenerating inference MP4 for {env_name}...")
    print(f"  Settings: {n_eps} episodes, max {max_steps} steps, frame skip={skip}")
    print(f"  Resolution: {size[0]}x{size[1]} (fast mode)")
    
    for ep in range(n_eps):
        s = env.reset()
        done = False
        reward = 0
        steps = 0
        frame_cnt = 0
        
        title = create_title_frame(env_name, algo, ep + 1, n_eps)
        title = resize_frame(title, size, method)
        frames.append(title)
        
        while not done and steps < max_steps:
            if frame_cnt % skip == 0:
                frame = env.render(mode='rgb_array')
                frame = resize_frame(frame, size, method)
                frames.append(frame)
            
            a = agent.get_action(s, explore=False)
            s, r, done, _ = env.step(a)
            reward += r
            steps += 1
            frame_cnt += 1
        
        final = env.render(mode='rgb_array')
        final = resize_frame(final, size, method)
        frames.append(final)
        
        result = create_result_frame(ep + 1, reward, steps)
        result = resize_frame(result, size, method)
        frames.append(result)
        
        stats.append({'reward': reward, 'steps': steps})
        
        print(f"  Episode {ep + 1}/{n_eps}: {steps} steps, reward={reward:.2f}")
        
        if progress_cb:
            progress_cb(int((ep + 1) / n_eps * 100))
    
    print(f"  Encoding video...")
    save_video_mp4(str(filepath), frames, fps, size)
    print(f"  ✓ Saved {len(frames)} frames to {filepath}")
    
    if env_name in ['MountainCar', 'Acrobot']:
        threshold = -100
        success = sum(1 for ep in stats if ep['reward'] > threshold)
    elif env_name in ['CartPole']:
        threshold = 100
        success = sum(1 for ep in stats if ep['steps'] >= threshold)
    else:
        success = sum(1 for ep in stats if ep['reward'] > 0)
    
    summary = {
        'avg_reward': np.mean([ep['reward'] for ep in stats]),
        'avg_steps': np.mean([ep['steps'] for ep in stats]),
        'success_rate': success / n_eps,
        'episodes': stats
    }
    
    del frames
    del stats
    gc.collect()
    
    return str(filepath), summary


def generate_training_video(env, agent, hist, env_name, algo, n_samples=5, fps=5, full=False):
    """Generate MP4 video showing training progress"""
    
    out_dir = Path("outputs/videos")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    suffix = "_full" if full else ""
    filepath = out_dir / f"{env_name}_{algo}_training{suffix}.mp4"
    
    frames = []
    
    if env_name == 'Acrobot':
        size = (320, 240)
        method = Image.NEAREST
    elif env_name in ['CartPole', 'MountainCar']:
        size = (400, 300)
        method = Image.BILINEAR
    elif env_name in CUSTOM_ENVS:
        size = (300, 300)
        method = Image.NEAREST
    else:
        size = (500, 350)
        method = Image.BILINEAR
    
    if env_name == 'Acrobot':
        max_steps = 100
        skip = 6 if not full else 8
    elif env_name in ['CartPole', 'MountainCar']:
        max_steps = 150
        skip = 4 if not full else 6
    elif env_name in CUSTOM_ENVS:
        max_steps = 150
        skip = 5 if not full else 7
    else:
        max_steps = 200
        skip = 3 if not full else 5
    
    mode_txt = "FULL" if full else "SAMPLE"
    print(f"\nGenerating training MP4 for {env_name} ({mode_txt} mode)...")
    print(f"  Settings: {n_samples} episodes, max {max_steps} steps, frame skip={skip}")
    print(f"  Resolution: {size[0]}x{size[1]} (fast mode)")
    
    if 'episode_rewards' in hist:
        total = len(hist['episode_rewards'])
        
        if full:
            indices = list(range(min(n_samples, total)))
        else:
            indices = np.linspace(0, total-1, n_samples, dtype=int)
        
        for idx, ep_num in enumerate(indices):
            progress = create_training_progress_frame(
                env_name, algo, ep_num + 1, total,
                hist['episode_rewards'][ep_num] if ep_num < len(hist['episode_rewards']) else 0
            )
            progress = resize_frame(progress, size, method)
            frames.append(progress)
            
            s = env.reset()
            done = False
            steps = 0
            frame_cnt = 0
            
            while not done and steps < max_steps:
                if frame_cnt % skip == 0:
                    frame = env.render(mode='rgb_array')
                    frame = resize_frame(frame, size, method)
                    frames.append(frame)
                
                explore_rate = max(0.1, 1.0 - (ep_num / total))
                a = agent.get_action(s, explore=(np.random.random() < explore_rate))
                
                s, r, done, _ = env.step(a)
                steps += 1
                frame_cnt += 1
            
            print(f"  Training sample {idx + 1}/{n_samples} (episode {ep_num + 1}): {steps} steps")
    
    else:
        vid_path, _ = generate_inference_video(env, agent, 3, fps, env_name, algo)
        return vid_path
    
    if frames:
        print(f"  Encoding video...")
        save_video_mp4(str(filepath), frames, fps, size)
        print(f"  ✓ Saved {len(frames)} frames to {filepath}")
    
    del frames
    gc.collect()

    return str(filepath)


def save_video_mp4(filepath, frames, fps, size):
    """Save frames as MP4 using imageio with ffmpeg, falling back to OpenCV"""
    
    # Try imageio first (often best quality/compression)
    try:
        frames_u8 = []
        for frame in frames:
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
            frames_u8.append(frame)
        
        # Explicitly request ffmpeg plugin
        imageio.mimsave(
            filepath, frames_u8, fps=fps,
            plugin='ffmpeg',
            codec='libx264',
            pixelformat='yuv420p',
            output_params=['-pix_fmt', 'yuv420p', '-crf', '23']
        )
        print(f"  ✓ Encoded with imageio (H.264)")
        return True
        
    except Exception as e:
        print(f"  Imageio failed: {e}")
        print(f"  Trying OpenCV fallback...")
        
        # Try OpenCV with multiple codec options
        # avc1/H264 are browser friendly (H.264)
        # mp4v is the fallback (MPEG-4 Part 2, widely supported by players but less so by browsers)
        codecs_to_try = [
            ('avc1', 'H.264'), 
            ('H264', 'H.264'), 
            ('mp4v', 'MPEG-4')
        ]
        
        for fourcc_str, codec_name in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                out = cv2.VideoWriter(filepath, fourcc, fps, size)
                
                if not out.isOpened():
                    continue
                
                for frame in frames:
                    # OpenCV expects BGR
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)
                
                out.release()
                
                # Verify file was created and has content
                if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
                    print(f"  ✓ Encoded with OpenCV using {codec_name} ({fourcc_str})")
                    return True
                
            except Exception as cv_e:
                print(f"  Failed with {fourcc_str}: {cv_e}")
                continue
                
        raise RuntimeError("All video encoding methods failed")


def resize_frame(frame, size, method=Image.BILINEAR):
    """Resize frame to target size"""
    img = Image.fromarray(frame)
    img = img.resize(size, method)
    return np.array(img)


def create_title_frame(env_name, algo, ep, total):
    """Create title frame for episode"""
    fig = Figure(figsize=(6, 4), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    ax.text(5, 7, f'{env_name}', ha='center', va='center', 
        fontsize=24, fontweight='bold', color='#667eea')
    ax.text(5, 5, f'{algo} Agent', ha='center', va='center',
        fontsize=20, color='#764ba2')
    ax.text(5, 3, f'Episode {ep}/{total}', ha='center', va='center',
        fontsize=16, color='#333')
    
    canvas.draw()
    w, h = fig.get_size_inches() * fig.get_dpi()
    frame = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    frame = frame.reshape(int(h), int(w), 4)
    frame = frame[:, :, :3]
    plt.close(fig)
    
    return frame


def create_result_frame(ep, reward, steps):
    """Create result frame showing stats"""
    fig = Figure(figsize=(6, 4), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    ax.text(5, 7, 'Episode Complete!', ha='center', va='center',
        fontsize=22, fontweight='bold', color='#2ecc71')
    ax.text(5, 5, f'Total Reward: {reward:.2f}', ha='center', va='center',
        fontsize=18, color='#333')
    ax.text(5, 3, f'Steps: {steps}', ha='center', va='center',
        fontsize=18, color='#333')
    
    canvas.draw()
    w, h = fig.get_size_inches() * fig.get_dpi()
    frame = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    frame = frame.reshape(int(h), int(w), 4)
    frame = frame[:, :, :3]
    plt.close(fig)
    
    return frame


def create_training_progress_frame(env_name, algo, ep, total, reward):
    """Create frame showing training progress"""
    fig = Figure(figsize=(6, 4), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    ax.text(5, 7, 'TRAINING PROGRESS', ha='center', va='center',
           fontsize=22, fontweight='bold', color='#667eea')
    ax.text(5, 5.5, f'{env_name} - {algo}', ha='center', va='center',
           fontsize=16, color='#764ba2')
    ax.text(5, 4, f'Episode {ep}/{total}', ha='center', va='center',
           fontsize=14, color='#333')
    ax.text(5, 2.5, f'Reward: {reward:.2f}', ha='center', va='center',
           fontsize=14, color='#2ecc71')
    
    canvas.draw()
    w, h = fig.get_size_inches() * fig.get_dpi()
    frame = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    frame = frame.reshape(int(h), int(w), 4)
    frame = frame[:, :, :3]
    plt.close(fig)
    
    return frame
