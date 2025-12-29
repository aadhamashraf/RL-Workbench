import streamlit as st
from rl.settings import ENVIRONMENTS
from rl.settings import ALGORITHMS

def render_sidebar():
  """Render sidebar with environment and algorithm selection"""
  
  with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: white; margin-bottom: 2rem;'>Configuration</h2>", 
                unsafe_allow_html=True)
    
    st.markdown("### Environment")
    env_names = list(ENVIRONMENTS.keys())
    env = st.selectbox(
      "Choose an environment",
      env_names,
      index=env_names.index(st.session_state.selected_environment),
      label_visibility="collapsed"
    )
    
    if env != st.session_state.selected_environment:
      st.session_state.selected_environment = env
      st.session_state.training_complete = False
      st.session_state.trained_agent = None
      st.rerun()
    
    env_info = ENVIRONMENTS[env]
    with st.expander("ℹ Environment Info", expanded=False):
      st.markdown(f"**Description:** {env_info['description']}")
      st.markdown(f"**Type:** {env_info['type']}")
      st.markdown(f"**Difficulty:** {env_info['difficulty']}")
    
    st.markdown("---")
    
    st.markdown("### Algorithm")
    algo_names = list(ALGORITHMS.keys())
    algo = st.selectbox(
      "Choose an algorithm",
      algo_names,
      index=algo_names.index(st.session_state.selected_algorithm),
      label_visibility="collapsed"
    )
    
    if algo != st.session_state.selected_algorithm:
      st.session_state.selected_algorithm = algo
      st.session_state.training_complete = False
      st.session_state.trained_agent = None
      st.rerun()
    
    algo_info = ALGORITHMS[algo]
    with st.expander("ℹ Algorithm Info", expanded=False):
      st.markdown(f"**Type:** {algo_info['type']}")
      st.markdown(f"**Description:** {algo_info['description']}")
      st.markdown("**Key Features:**")
      for feature in algo_info['features']:
        st.markdown(f"- {feature}")
    
    if algo_info['requires_model']:
      st.info("**Planning Algorithm** - Uses environment model")
    else:
      st.success("**Learning Algorithm** - Learns from episodes")
    
    st.markdown("---")
    
    st.markdown("### Hyperparameters")
    render_hyperparameters(algo)
    
    st.markdown("---")
    
    st.markdown("### Quick Actions")
    if st.button("Reset All", use_container_width=True):
      st.session_state.training_complete = False
      st.session_state.trained_agent = None
      st.session_state.training_history = None
      st.rerun()

def render_hyperparameters(algorithm):
  """Render hyperparameter controls"""
  
  algo = ALGORITHMS[algorithm]
  
  if 'params' not in st.session_state:
    st.session_state.params = {}
  
  # Gamma - common to all algorithms
  st.session_state.params['gamma'] = st.slider(
    "Discount Factor (γ)", 0.0, 1.0, 0.99, 0.01,
    help="Importance of future rewards",
    key=f"{algorithm}_gamma"
  )
  
  # Dynamic Programming parameters
  if algo['type'] == 'Dynamic Programming':
    st.session_state.params['theta'] = st.slider(
      "Convergence Threshold (θ)", 0.0001, 0.1, 0.001, 0.0001, format="%.4f",
      help="Convergence criterion for value function",
      key=f"{algorithm}_theta"
    )
    st.session_state.params['max_iterations'] = st.slider(
      "Max Iterations", 10, 1000, 100, 10,
      help="Maximum number of iterations",
      key=f"{algorithm}_max_iterations"
    )
  
  # Monte Carlo parameters
  elif algo['type'] == 'Monte Carlo':
    st.session_state.params['episodes'] = st.slider(
      "Training Episodes", 100, 10000, 1000, 100,
      help="Number of episodes to train",
      key=f"{algorithm}_episodes"
    )
    st.session_state.params['epsilon'] = st.slider(
      "Exploration Rate (ε)", 0.0, 1.0, 0.1, 0.01,
      help="Probability of random action",
      key=f"{algorithm}_epsilon"
    )
    st.session_state.params['epsilon_decay'] = st.slider(
      "Epsilon Decay", 0.9, 1.0, 0.995, 0.001,
      help="Decay rate for exploration",
      key=f"{algorithm}_epsilon_decay"
    )
    st.session_state.params['mc_type'] = st.selectbox(
      "MC Variant", ["FV", "EV"],
      format_func=lambda x: "First-Visit" if x == "FV" else "Every-Visit",
      help="First-Visit or Every-Visit Monte Carlo",
      key=f"{algorithm}_mc_type"
    )
    st.session_state.params['use_alpha'] = st.checkbox(
      "Use Constant Step Size (α)", value=False,
      help="False: sample mean, True: constant alpha",
      key=f"{algorithm}_use_alpha"
    )
    if st.session_state.params['use_alpha']:
      st.session_state.params['alpha'] = st.slider(
        "Step Size (α)", 0.01, 1.0, 0.1, 0.01,
        help="Learning rate for constant step size",
        key=f"{algorithm}_alpha"
      )
  
  # Temporal Difference parameters (TD(0), n-step TD)
  elif algo['type'] == 'Temporal Difference':
    st.session_state.params['episodes'] = st.slider(
      "Training Episodes", 100, 10000, 1000, 100,
      help="Number of episodes to train",
      key=f"{algorithm}_episodes"
    )
    st.session_state.params['alpha'] = st.slider(
      "Learning Rate (α)", 0.01, 1.0, 0.1, 0.01,
      help="Step size for value updates",
      key=f"{algorithm}_alpha"
    )
    st.session_state.params['epsilon'] = st.slider(
      "Exploration Rate (ε)", 0.0, 1.0, 0.1, 0.01,
      help="Probability of random action",
      key=f"{algorithm}_epsilon"
    )
    st.session_state.params['epsilon_decay'] = st.slider(
      "Epsilon Decay", 0.9, 1.0, 0.995, 0.001,
      help="Decay rate for exploration",
      key=f"{algorithm}_epsilon_decay"
    )
    
    # n-step TD specific
    if algorithm == 'n-step TD':
      st.session_state.params['n_steps'] = st.slider(
        "n (steps)", 1, 10, 3, 1,
        help="Number of steps for n-step returns",
        key=f"{algorithm}_n_steps"
      )
  
  # Model-Free Control parameters (SARSA, Q-Learning)
  elif algo['type'] == 'Model-Free Control':
    st.session_state.params['episodes'] = st.slider(
      "Training Episodes", 100, 10000, 1000, 100,
      help="Number of episodes to train",
      key=f"{algorithm}_episodes"
    )
    st.session_state.params['alpha'] = st.slider(
      "Learning Rate (α)", 0.01, 1.0, 0.1, 0.01,
      help="Step size for Q-value updates",
      key=f"{algorithm}_alpha"
    )
    st.session_state.params['epsilon'] = st.slider(
      "Exploration Rate (ε)", 0.0, 1.0, 0.1, 0.01,
      help="Probability of random action",
      key=f"{algorithm}_epsilon"
    )
    st.session_state.params['epsilon_decay'] = st.slider(
      "Epsilon Decay", 0.9, 1.0, 0.995, 0.001,
      help="Decay rate for exploration",
      key=f"{algorithm}_epsilon_decay"
    )


