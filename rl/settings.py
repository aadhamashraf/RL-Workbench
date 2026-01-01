"""Settings - Configuration for algorithms and environments"""

# Algorithm metadata
ALGORITHMS = {

    'Value Iteration': {
        'type': 'Dynamic Programming',
        'description': 'Updates value function directly to find optimal policy.',
        'features': ['Model-based', 'Combined steps', 'Faster convergence'],
        'complexity': 'Medium',
        'requires_model': True
    },
    'Policy Iteration': {
        'type': 'Dynamic Programming',
        'description': 'Alternates between evaluation and improvement.',
        'features': ['Model-based', 'Guaranteed convergence', 'Full sweeps'],
        'complexity': 'Medium',
        'requires_model': True
    },
    'Monte Carlo': {
        'type': 'Monte Carlo',
        'description': 'Learns from complete episode returns.',
        'features': ['Model-free', 'Episode-based', 'High variance'],
        'complexity': 'Medium',
        'requires_model': False
    },
    'TD(0)': {
        'type': 'Temporal Difference',
        'description': 'Learns values via bootstrapping (prediction only).',
        'features': ['Model-free', 'V-function learning', 'One-step updates'],
        'complexity': 'Low',
        'requires_model': False
    },
    'n-step TD': {
        'type': 'Temporal Difference',
        'description': 'Bootstrap with n-step lookahead.',
        'features': ['Model-free', 'Adjustable n', 'Tunable bias/variance'],
        'complexity': 'High',
        'requires_model': False
    },
    'SARSA': {
        'type': 'Model-Free Control',
        'description': 'On-policy Q-value learning.',
        'features': ['On-policy', 'Conservative', 'Safe learning'],
        'complexity': 'Medium',
        'requires_model': False
    },
    'Q-Learning': {
        'type': 'Model-Free Control',
        'description': 'Off-policy optimal Q-value learning.',
        'features': ['Off-policy', 'Aggressive', 'Max-based updates'],
        'complexity': 'Medium',
        'requires_model': False
    }
}

# Environment metadata
ENVIRONMENTS = {
    # GridWorld Variations (6 total)
    'GridWorld': {
        'description': 'Reach goal while avoiding obstacles (10x10).',
        'type': 'Custom',
        'difficulty': 'Medium',
        'state_space': 'Discrete',
        'action_space': 'Discrete',
        'env_id': None
    },
    'GridWorld-Small': {
        'description': 'Small 5x5 grid navigation.',
        'type': 'Custom',
        'difficulty': 'Easy',
        'state_space': 'Discrete',
        'action_space': 'Discrete',
        'env_id': None
    },
    'GridWorld-Medium': {
        'description': 'Medium 10x10 grid with balanced obstacles.',
        'type': 'Custom',
        'difficulty': 'Medium',
        'state_space': 'Discrete',
        'action_space': 'Discrete',
        'env_id': None
    },
    'GridWorld-Large': {
        'description': 'Large 15x15 grid for complex navigation.',
        'type': 'Custom',
        'difficulty': 'Hard',
        'state_space': 'Discrete',
        'action_space': 'Discrete',
        'env_id': None
    },
    'GridWorld-Sparse': {
        'description': '10x10 grid with few obstacles.',
        'type': 'Custom',
        'difficulty': 'Easy',
        'state_space': 'Discrete',
        'action_space': 'Discrete',
        'env_id': None
    },
    'GridWorld-Dense': {
        'description': '10x10 grid with many obstacles.',
        'type': 'Custom',
        'difficulty': 'Hard',
        'state_space': 'Discrete',
        'action_space': 'Discrete',
        'env_id': None
    },
    
    # Gymnasium Environments
    'FrozenLake': {
        'description': 'Cross the frozen lake without falling in holes.',
        'type': 'Gymnasium',
        'difficulty': 'Easy',
        'state_space': 'Discrete',
        'action_space': 'Discrete',
        'env_id': 'FrozenLake-v1'
    },
    'FrozenLake-8x8': {
        'description': 'Larger frozen lake with more holes (8x8 grid).',
        'type': 'Gymnasium',
        'difficulty': 'Medium',
        'state_space': 'Discrete',
        'action_space': 'Discrete',
        'env_id': 'FrozenLake8x8-v1'
    },
    'FrozenLake-Slippery': {
        'description': '4x4 frozen lake with slippery surface.',
        'type': 'Gymnasium',
        'difficulty': 'Hard',
        'state_space': 'Discrete',
        'action_space': 'Discrete',
        'env_id': 'FrozenLake-v1'
    },
    'FrozenLake-8x8-Slippery': {
        'description': '8x8 frozen lake with slippery surface.',
        'type': 'Gymnasium',
        'difficulty': 'Very Hard',
        'state_space': 'Discrete',
        'action_space': 'Discrete',
        'env_id': 'FrozenLake8x8-v1'
    },
    'Taxi': {
        'description': 'Pickup and delivery task.',
        'type': 'Gymnasium',
        'difficulty': 'Medium',
        'state_space': 'Discrete',
        'action_space': 'Discrete',
        'env_id': 'Taxi-v3'
    },
    'CliffWalking': {
        'description': 'Walk along the edge without falling off.',
        'type': 'Gymnasium',
        'difficulty': 'Medium',
        'state_space': 'Discrete',
        'action_space': 'Discrete',
        'env_id': 'CliffWalking-v0'
    },
    'CartPole': {
        'description': 'Balance a pole on a moving cart.',
        'type': 'Gymnasium',
        'difficulty': 'Medium',
        'state_space': 'Continuous',
        'action_space': 'Discrete',
        'env_id': 'CartPole-v0'
    },
    'CartPole-Long': {
        'description': 'CartPole with longer episode limit.',
        'type': 'Gymnasium',
        'difficulty': 'Medium',
        'state_space': 'Continuous',
        'action_space': 'Discrete',
        'env_id': 'CartPole-v1'
    },
    'MountainCar': {
        'description': 'Drive up a hill using momentum.',
        'type': 'Gymnasium',
        'difficulty': 'Hard',
        'state_space': 'Continuous',
        'action_space': 'Discrete',
        'env_id': 'MountainCar-v0'
    },
    'Blackjack': {
        'description': 'Card game strategy.',
        'type': 'Gymnasium',
        'difficulty': 'Easy',
        'state_space': 'Discrete',
        'action_space': 'Discrete',
        'env_id': 'Blackjack-v1'
    },
    'Acrobot': {
        'description': 'Swing up a two-link robot to reach the goal.',
        'type': 'Gymnasium',
        'difficulty': 'Hard',
        'state_space': 'Continuous',
        'action_space': 'Discrete',
        'env_id': 'Acrobot-v1'
    },
    
    # Maze Variations (6 total)
    'Maze': {
        'description': 'Random maze navigation (10x10).',
        'type': 'Custom',
        'difficulty': 'Medium',
        'state_space': 'Discrete',
        'action_space': 'Discrete',
        'env_id': None
    },
    'Maze-Tiny': {
        'description': 'Tiny 5x5 maze for beginners.',
        'type': 'Custom',
        'difficulty': 'Easy',
        'state_space': 'Discrete',
        'action_space': 'Discrete',
        'env_id': None
    },
    'Maze-Small': {
        'description': 'Small 7x7 maze.',
        'type': 'Custom',
        'difficulty': 'Easy',
        'state_space': 'Discrete',
        'action_space': 'Discrete',
        'env_id': None
    },
    'Maze-Medium': {
        'description': 'Medium 10x10 maze with moderate complexity.',
        'type': 'Custom',
        'difficulty': 'Medium',
        'state_space': 'Discrete',
        'action_space': 'Discrete',
        'env_id': None
    },
    'Maze-Large': {
        'description': 'Large 15x15 maze with high complexity.',
        'type': 'Custom',
        'difficulty': 'Hard',
        'state_space': 'Discrete',
        'action_space': 'Discrete',
        'env_id': None
    },
    'Maze-Huge': {
        'description': 'Huge 20x20 maze for experts.',
        'type': 'Custom',
        'difficulty': 'Very Hard',
        'state_space': 'Discrete',
        'action_space': 'Discrete',
        'env_id': None
    },
    
    # Simple Navigation Environments
    'Corridor': {
        'description': 'Simple 1D corridor navigation.',
        'type': 'Custom',
        'difficulty': 'Easy',
        'state_space': 'Discrete',
        'action_space': 'Discrete',
        'env_id': None
    },
    'TwoRooms': {
        'description': 'Navigate between two rooms through a door.',
        'type': 'Custom',
        'difficulty': 'Medium',
        'state_space': 'Discrete',
        'action_space': 'Discrete',
        'env_id': None
    },
    
    # Games
    'TicTacToe': {
        'description': 'Classic 3x3 game.',
        'type': 'Custom',
        'difficulty': 'Easy',
        'state_space': 'Discrete',
        'action_space': 'Discrete',
        'env_id': None
    }
}

