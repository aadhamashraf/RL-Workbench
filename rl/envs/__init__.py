"""RL Environments"""

from rl.envs.base import BaseEnvironment
from rl.envs.gridworld import GridWorldEnv
from rl.envs.maze import MazeEnv
from rl.envs.tictactoe import TicTacToeEnv
from rl.envs.corridor import CorridorEnv
from rl.envs.two_rooms import TwoRoomsEnv
from rl.envs.gym_wrapper import GymnasiumEnvWrapper


def create_environment(name, **kwargs):
    """Factory function to create environments"""
    
    # GridWorld variations
    if name == 'GridWorld':
        return GridWorldEnv(size=10, **kwargs)
    elif name == 'GridWorld-Small':
        return GridWorldEnv(size=5, **kwargs)
    elif name == 'GridWorld-Medium':
        return GridWorldEnv(size=10, **kwargs)
    elif name == 'GridWorld-Large':
        return GridWorldEnv(size=15, **kwargs)
    elif name == 'GridWorld-Sparse':
        # Few obstacles
        obstacles = [(2, 2), (7, 7)]
        return GridWorldEnv(size=10, obstacles=obstacles, **kwargs)
    elif name == 'GridWorld-Dense':
        # Many obstacles
        obstacles = [(i, j) for i in range(2, 8) for j in range(2, 8) 
                     if (i + j) % 2 == 0 and (i, j) not in [(0, 0), (9, 9)]]
        return GridWorldEnv(size=10, obstacles=obstacles[:20], **kwargs)
    
    # Maze variations
    elif name == 'Maze':
        return MazeEnv(size=10, difficulty='medium', **kwargs)
    elif name == 'Maze-Tiny':
        return MazeEnv(size=5, difficulty='easy', **kwargs)
    elif name == 'Maze-Small':
        return MazeEnv(size=7, difficulty='easy', **kwargs)
    elif name == 'Maze-Medium':
        return MazeEnv(size=10, difficulty='medium', **kwargs)
    elif name == 'Maze-Large':
        return MazeEnv(size=15, difficulty='hard', **kwargs)
    elif name == 'Maze-Huge':
        return MazeEnv(size=20, difficulty='hard', **kwargs)
    
    # Simple navigation
    elif name == 'Corridor':
        return CorridorEnv(length=10, **kwargs)
    elif name == 'TwoRooms':
        return TwoRoomsEnv(size=10, **kwargs)
    
    # Games
    elif name == 'TicTacToe':
        return TicTacToeEnv(**kwargs)
    
    # Gymnasium environments
    else:
        from rl.settings import ENVIRONMENTS
        if name in ENVIRONMENTS:
            env_id = ENVIRONMENTS[name]['env_id']
            
            # Handle FrozenLake slippery variants
            if name == 'FrozenLake-Slippery':
                return GymnasiumEnvWrapper('FrozenLake-v1', name, is_slippery=True)
            elif name == 'FrozenLake-8x8-Slippery':
                return GymnasiumEnvWrapper('FrozenLake8x8-v1', name, is_slippery=True)
            else:
                return GymnasiumEnvWrapper(env_id, name)
        
        raise ValueError(f"Unknown environment: {name}")


__all__ = [
    'BaseEnvironment',
    'GridWorldEnv',
    'MazeEnv',
    'TicTacToeEnv',
    'CorridorEnv',
    'TwoRoomsEnv',
    'GymnasiumEnvWrapper',
    'create_environment'
]
