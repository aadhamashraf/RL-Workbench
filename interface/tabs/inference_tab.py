import streamlit as st
from rl import create_environment
from interface.utils.video_generator import generate_inference_video, generate_training_video
from interface.utils.training_history import update_run_videos
import os

def render_video_column(title, caption, video_path, dl_label, speed):
    """Render video column with download"""
    st.markdown(f"#### {title}")
    st.caption(caption)
    
    if isinstance(video_path, tuple):
        video_path = video_path[0]
    
    if video_path and os.path.exists(video_path):
        st.video(video_path, format='video/mp4', start_time=0)
        st.info(f"Tip: Use browser controls or set playback to {speed}x")
        
        with open(video_path, 'rb') as f:
            st.download_button(dl_label, f.read(), os.path.basename(video_path), "video/mp4")
    else:
        st.warning("Video not found. Please regenerate.")

def render_inference_tab():
    """Render inference tab"""
    
    st.markdown("## Agent Visualization")
    
    if not st.session_state.training_complete:
        st.warning("Please train an agent first in the Training tab!")
        return
    
    st.markdown("### Automatic Dual Visualization System (MP4)")
    st.markdown("#### Visualization Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        mode = st.radio(
            "Training Progress Mode:",
            ["Sample Episodes", "Full Training"],
            index=0
        )
    
    with col2:
        if mode == "Full Training":
            total = len(st.session_state.training_history.get('episode_rewards', []))
            
            if total > 100:
                st.warning(f"Full mode: {total} episodes. Large files!")
            
            max_eps = st.slider("Max episodes:", 10, min(total, 1000), min(50, total), 10)
            
            if max_eps > 200:
                st.error(f"WARNING: {max_eps} episodes = 10-30 min, 50-200 MB file!")
        else:
            max_eps = None
    
    if 'training_mode' not in st.session_state:
        st.session_state.training_mode = mode
    if 'max_episodes_full' not in st.session_state:
        st.session_state.max_episodes_full = max_eps
    
    changed = (st.session_state.training_mode != mode or 
               st.session_state.max_episodes_full != max_eps)
    
    if changed:
        st.session_state.training_mode = mode
        st.session_state.max_episodes_full = max_eps
        st.session_state.training_video_path = None
        st.session_state.inference_video_path = None
    
    env_name = st.session_state.selected_environment
    algo_name = st.session_state.selected_algorithm
    
    if env_name == 'Acrobot':
        train_eps = 3 if mode == "Sample Episodes" else "All"
    else:
        train_eps = 5 if mode == "Sample Episodes" else "All"
    
    st.info(f"""
    **Environment:** {env_name}  
    **Algorithm:** {algo_name}  
    **Format:** MP4 (3-5x faster, smaller files)
    
    **Training Video:** {train_eps} episodes ({mode.lower()})  
    **Policy Video:** 1 episode (final performance)
    """)
    
    if 'training_video_path' not in st.session_state:
        st.session_state.training_video_path = None
    if 'inference_video_path' not in st.session_state:
        st.session_state.inference_video_path = None
    if 'last_inference_key' not in st.session_state:
        st.session_state.last_inference_key = None
    
    key = f"{env_name}_{algo_name}"
    
    if (st.session_state.last_inference_key != key or 
        not st.session_state.training_video_path or
        not st.session_state.inference_video_path):
        st.session_state.last_inference_key = key
        generate_both_videos()
    
    if isinstance(st.session_state.get('training_video_path'), tuple):
        st.session_state.training_video_path = st.session_state.training_video_path[0]
    
    if isinstance(st.session_state.get('inference_video_path'), tuple):
        st.session_state.inference_video_path = st.session_state.inference_video_path[0]
    
    if (st.session_state.training_video_path and st.session_state.inference_video_path):
        
        st.markdown("---")
        st.markdown("### Visualization Results")
        st.markdown("#### Playback Controls")
        
        speed = st.select_slider(
            "Video Playback Speed:",
            [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
            value=1.0,
            format_func=lambda x: f"{x}x"
        )
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            txt = st.session_state.get('training_mode', 'Sample Episodes')
            cap = ("Sample episodes showing learning progression" if txt == "Sample Episodes" 
                   else "Full training progression")
            
            render_video_column(
                "Training Progress", cap,
                st.session_state.training_video_path,
                "Download Training Video", speed
            )
        
        with col2:
            render_video_column(
                "Final Learned Policy",
                "Final performance (best behavior)",
                st.session_state.inference_video_path,
                "Download Inference Video", speed
            )
        
        if 'inference_stats' in st.session_state:
            stats = st.session_state.inference_stats
            
            st.markdown("---")
            st.markdown("### Performance Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Reward", f"{stats['avg_reward']:.2f}")
            with col2:
                st.metric("Avg Steps", f"{stats['avg_steps']:.1f}")
            with col3:
                st.metric("Success Rate", f"{stats['success_rate']:.1%}")
            
            with st.expander("Episode Details"):
                for i, ep in enumerate(stats['episodes']):
                    st.markdown(f"**Episode {i+1}:** Reward: {ep['reward']:.2f}, Steps: {ep['steps']}")
        
        st.markdown("---")
        if st.button("Regenerate Videos", use_container_width=True):
            st.session_state.training_video_path = None
            st.session_state.inference_video_path = None
            st.session_state.last_inference_key = None
            st.rerun()

def generate_both_videos():
    """Generate both training and inference videos"""
    
    bar = st.progress(0)
    txt = st.empty()
    
    try:
        env = create_environment(st.session_state.selected_environment)
        agent = st.session_state.trained_agent
        hist = st.session_state.training_history
        
        mode = st.session_state.get('training_mode', 'Sample Episodes')
        
        if mode == "Sample Episodes":
            txt.text("Generating training video (MP4) - Sample mode...")
            bar.progress(25)
            
            n = 3 if st.session_state.selected_environment == 'Acrobot' else 5
            
            train_vid = generate_training_video(
                env, agent, hist,
                st.session_state.selected_environment,
                st.session_state.selected_algorithm,
                n, 5
            )
        else:
            txt.text("Generating training video (MP4) - Full mode...")
            bar.progress(25)
            
            max_eps = st.session_state.get('max_episodes_full', 50)
            total = len(hist.get('episode_rewards', []))
            n = min(max_eps, total)
            
            st.warning(f"Generating {n} episodes. May take several minutes...")
            
            train_vid = generate_training_video(
                env, agent, hist,
                st.session_state.selected_environment,
                st.session_state.selected_algorithm,
                n, 5, True
            )
        
        st.session_state.training_video_path = train_vid
        
        txt.text("Generating inference video (MP4)...")
        bar.progress(60)
        
        inf_vid, stats = generate_inference_video(
            env, agent, 1, 15,
            st.session_state.selected_environment,
            st.session_state.selected_algorithm,
            lambda p: bar.progress(60 + int(p * 0.4))
        )
        st.session_state.inference_video_path = inf_vid
        st.session_state.inference_stats = stats
        
        update_run_videos(
            st.session_state.selected_environment,
            st.session_state.selected_algorithm,
            train_vid, inf_vid
        )
        
        bar.progress(100)
        txt.success("Videos generated successfully! (MP4)")
        st.rerun()
        
    except Exception as e:
        st.error(f"Failed to generate videos: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
