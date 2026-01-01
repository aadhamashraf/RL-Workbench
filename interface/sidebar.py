import streamlit as st
from rl.settings import ENVIRONMENTS, ALGORITHMS

# Environment grouping configuration
ENVIRONMENT_GROUPS = {
    'GridWorld': {
        'base': 'GridWorld',
        'variations': {
            'Default (10x10)': 'GridWorld',
            'Small (5x5)': 'GridWorld-Small',
            'Medium (10x10)': 'GridWorld-Medium',
            'Large (15x15)': 'GridWorld-Large',
            'Sparse Obstacles': 'GridWorld-Sparse',
            'Dense Obstacles': 'GridWorld-Dense'
        }
    },
    'Maze': {
        'base': 'Maze',
        'variations': {
            'Default (10x10)': 'Maze',
            'Tiny (5x5)': 'Maze-Tiny',
            'Small (7x7)': 'Maze-Small',
            'Medium (10x10)': 'Maze-Medium',
            'Large (15x15)': 'Maze-Large',
            'Huge (20x20)': 'Maze-Huge'
        }
    },
    'FrozenLake': {
        'base': 'FrozenLake',
        'variations': {
            '4x4 (Non-slippery)': 'FrozenLake',
            '8x8 (Non-slippery)': 'FrozenLake-8x8',
            '4x4 (Slippery)': 'FrozenLake-Slippery',
            '8x8 (Slippery)': 'FrozenLake-8x8-Slippery'
        }
    },
    'CartPole': {
        'base': 'CartPole',
        'variations': {
            'Standard (v0)': 'CartPole',
            'Long Episodes (v1)': 'CartPole-Long'
        }
    }
}

# Standalone environments (no variations)
STANDALONE_ENVS = [
    'Taxi', 'CliffWalking', 'Blackjack', 'MountainCar', 'Acrobot',
    'TwoRooms', 'TicTacToe'
]

def render_sidebar():
  """Render sidebar with grouped environment and algorithm selection"""
  
  with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: white; margin-bottom: 2rem;'>Configuration</h2>", 
                unsafe_allow_html=True)
    
    st.markdown("### Environment")
    
    # Get all environment types (groups + standalone)
    env_types = list(ENVIRONMENT_GROUPS.keys()) + STANDALONE_ENVS
    
    # Determine current environment type
    current_env = st.session_state.selected_environment
    current_type = None
    
    # Check if current env is in a group
    for group_name, group_data in ENVIRONMENT_GROUPS.items():
        if current_env in group_data['variations'].values():
            current_type = group_name
            break
    
    # If not in a group, it's standalone
    if current_type is None and current_env in STANDALONE_ENVS:
        current_type = current_env
    
    # Default to first option if not found
    if current_type is None:
        current_type = env_types[0]
    
    # Environment type selector
    env_type = st.selectbox(
      "Environment Type",
      env_types,
      index=env_types.index(current_type) if current_type in env_types else 0,
      key="env_type_selector"
    )
    
    # Variation selector (if applicable)
    if env_type in ENVIRONMENT_GROUPS:
        group = ENVIRONMENT_GROUPS[env_type]
        variations = list(group['variations'].keys())
        
        # Find current variation
        current_variation = None
        for var_name, var_env in group['variations'].items():
            if var_env == current_env:
                current_variation = var_name
                break
        
        if current_variation is None:
            current_variation = variations[0]
        
        variation = st.selectbox(
            "Variation",
            variations,
            index=variations.index(current_variation) if current_variation in variations else 0,
            key="env_variation_selector"
        )
        
        selected_env = group['variations'][variation]
    else:
        # Standalone environment
        selected_env = env_type
    
    # Update session state if environment changed
    if selected_env != st.session_state.selected_environment:
      st.session_state.selected_environment = selected_env
      st.session_state.training_complete = False
      st.session_state.trained_agent = None
      st.rerun()
    
    # Environment info
    env_info = ENVIRONMENTS[selected_env]
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


