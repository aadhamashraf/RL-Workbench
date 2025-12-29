import streamlit as st
import numpy as np
from datetime import datetime


def save_training_run(environment, algorithm, params, history, agent, 
                     training_video_path=None, inference_video_path=None):
    """Save training run to session state"""
    
    if 'training_runs' not in st.session_state:
        st.session_state.training_runs = []
    
    avg_reward = np.mean(history['episode_rewards'][-100:]) if 'episode_rewards' in history else None
    
    run = {
        'run_id': len(st.session_state.training_runs) + 1,
        'timestamp': datetime.now().timestamp(),
        'environment': environment,
        'algorithm': algorithm,
        'params': params.copy(),
        'history': history,
        'avg_reward': avg_reward,
        'training_time': history.get('training_time', 0),
        'episodes': history.get('episodes'),
        'iterations': history.get('iterations'),
        'training_video_path': training_video_path,
        'inference_video_path': inference_video_path
    }
    
    st.session_state.training_runs.append(run)
    return run['run_id']


def update_run_videos(environment, algorithm, training_video_path, inference_video_path):
    """Update most recent run with video paths"""
    
    if not st.session_state.get('training_runs'):
        return
    
    for run in reversed(st.session_state.training_runs):
        if run['environment'] == environment and run['algorithm'] == algorithm:
            run['training_video_path'] = training_video_path
            run['inference_video_path'] = inference_video_path
            break
