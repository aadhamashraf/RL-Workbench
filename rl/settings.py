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
    'GridWorld': {
        'description': 'Reach goal while avoiding obstacles.',
        'type': 'Custom',
        'difficulty': 'Easy',
        'state_space': 'Discrete',
        'action_space': 'Discrete',
        'env_id': None
    },
    'FrozenLake': {
        'description': 'Cross the frozen lake without falling in holes.',
        'type': 'Gymnasium',
        'difficulty': 'Easy',
        'state_space': 'Discrete',
        'action_space': 'Discrete',
        'env_id': 'FrozenLake-v1'
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
    'TicTacToe': {
        'description': 'Classic 3x3 game.',
        'type': 'Custom',
        'difficulty': 'Easy',
        'state_space': 'Discrete',
        'action_space': 'Discrete',
        'env_id': None
    },
    'Maze': {
        'description': 'Random maze navigation.',
        'type': 'Custom',
        'difficulty': 'Medium',
        'state_space': 'Discrete',
        'action_space': 'Discrete',
        'env_id': None
    }
}
