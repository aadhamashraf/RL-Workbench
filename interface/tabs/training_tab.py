import streamlit as st
import time
from rl import create_agent
from rl import create_environment
from interface.utils.visualization_utils import plot_training_metrics
from interface.utils.training_history import save_training_run
import numpy as np

def render_training_tab():
  """Render training tab"""
  
  st.markdown("## Train Your Agent")
  st.markdown("Configure parameters in the sidebar and click **Start Training** to begin.")
  
  col1, col2 = st.columns([2, 1])
  
  with col1:
    st.markdown("### Training Controls")
    
    if not st.session_state.training_complete:
      if st.button("Start Training", use_container_width=True, type="primary"):
        run_training()
    else:
      col_a, col_b = st.columns(2)
      with col_a:
        if st.button("Retrain", use_container_width=True):
          st.session_state.training_complete = False
          st.session_state.trained_agent = None
          st.session_state.training_history = None
          st.rerun()
      with col_b:
        st.button("Training Complete", use_container_width=True, disabled=True)
  
  with col2:
    st.markdown("### Current Setup")
    
    from rl.settings import ALGORITHMS
    algo = ALGORITHMS[st.session_state.selected_algorithm]
    
    info = f"""
    **Environment:** {st.session_state.selected_environment}  
    **Algorithm:** {st.session_state.selected_algorithm}  
    **Type:** {'Planning (Model-Based)' if algo['requires_model'] else 'Learning (Model-Free)'}
    """
    
    if algo['requires_model']:
      st.info(info)
    else:
      st.success(info)
  
  st.markdown("---")
  
  if st.session_state.training_complete and st.session_state.training_history:
    render_training_results()

def run_training():
  """Execute training process"""
  
  env = create_environment(st.session_state.selected_environment)
  params = st.session_state.params.copy()
  
  # Debug: Show parameters being used
  st.markdown("### Parameters Being Used")
  st.json(params)
  
  agent = create_agent(st.session_state.selected_algorithm, env, **params)
  
  st.markdown("### Training Progress")
  progress_bar = st.progress(0)
  status_text = st.empty()
  metrics_placeholder = st.empty()
  
  try:
    history = agent.train(
      progress_callback=lambda p, m: update_training_ui(p, m, progress_bar, status_text, metrics_placeholder)
    )
    
    st.session_state.trained_agent = agent
    st.session_state.training_history = history
    st.session_state.training_complete = True
    
    save_training_run(
      st.session_state.selected_environment,
      st.session_state.selected_algorithm,
      params,
      history,
      agent
    )
    
    progress_bar.progress(100)
    status_text.success("Training completed successfully!")
    
    time.sleep(1)
    st.rerun()
    
  except Exception as e:
    st.error(f"Training failed: {str(e)}")
    import traceback
    st.code(traceback.format_exc())

def update_training_ui(progress, metrics, progress_bar, status_text, metrics_placeholder):
  """Update training UI"""
  progress_bar.progress(progress)
  status_text.text(f"Training... {progress}% complete")
  
  if metrics:
    with metrics_placeholder.container():
      cols = st.columns(len(metrics))
      for idx, (key, val) in enumerate(metrics.items()):
        with cols[idx]:
          st.metric(key, f"{val:.4f}" if isinstance(val, float) else val)

def render_training_results():
  """Display training results"""
  
  st.markdown("### Training Results")
  history = st.session_state.training_history
  col1, col2, col3, col4 = st.columns(4)
  
  with col1:
    if 'episode_rewards' in history:
      avg_reward = np.mean(history['episode_rewards'][-100:])
      st.metric("Avg Reward (Last 100)", f"{avg_reward:.2f}",
               help="Average reward over the last 100 training episodes")
    elif 'iterations' in history:
      st.metric("Iterations", history['iterations'],
               help="Number of iterations until convergence (planning algorithms don't run episodes)")
  
  with col2:
    if 'episode_lengths' in history:
      avg_length = np.mean(history['episode_lengths'][-100:])
      st.metric("Avg Episode Length", f"{avg_length:.1f}",
               help="Average number of steps per episode")
    elif 'delta' in history:
      st.metric("Final Delta", f"{history['delta']:.6f}",
               help="Value function convergence measure (smaller = better convergence)")
  
  with col3:
    if 'training_time' in history:
      st.metric("Training Time", f"{history['training_time']:.2f}s",
               help="Total time spent training")
  
  with col4:
    if 'convergence_episode' in history:
      st.metric("Converged At", f"Episode {history['convergence_episode']}",
               help="Episode where the algorithm converged")
    elif 'converged' in history and history['converged']:
      st.metric("Status", "Converged",
               help="Algorithm successfully converged to optimal policy")
  
  st.markdown("### Training Curves")
  col1, col2, col3 = st.columns([1, 3, 1])
  with col2:
      fig = plot_training_metrics(history, st.session_state.selected_algorithm)
      st.pyplot(fig)
  
  if hasattr(st.session_state.trained_agent, 'V') or hasattr(st.session_state.trained_agent, 'Q'):
    st.markdown("### Learned Value Function")
    
    env = create_environment(st.session_state.selected_environment)
    
    if hasattr(st.session_state.trained_agent, 'V'):
      from interface.utils.visualization_utils import plot_value_function
      col1, col2, col3 = st.columns([1, 2, 1])
      with col2:
          fig = plot_value_function(st.session_state.trained_agent.V, env)
          st.pyplot(fig)
    
    if hasattr(st.session_state.trained_agent, 'Q'):
      from interface.utils.visualization_utils import plot_policy, plot_qvalues
      
      # Plot Policy
      col1, col2, col3 = st.columns([1, 2, 1])
      with col2:
          st.markdown("#### Learned Policy")
          fig = plot_policy(st.session_state.trained_agent.Q, env)
          st.pyplot(fig)
      
      # Plot Q-Values
      st.markdown("#### Action-Value Function Q(s,a)")
      st.markdown("Detailed view of learned values for each action:")
      fig = plot_qvalues(st.session_state.trained_agent.Q, env)
      st.pyplot(fig)

