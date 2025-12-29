"""RL Environments"""

from rl.envs.base import BaseEnvironment
from rl.envs.gridworld import GridWorldEnv
from rl.envs.maze import MazeEnv
from rl.envs.tictactoe import TicTacToeEnv
from rl.envs.gym_wrapper import GymnasiumEnvWrapper


def create_environment(name, **kwargs):
    """Factory function to create environments"""
    
    ENVS = {
        'GridWorld': GridWorldEnv,
        'Maze': MazeEnv,
        'TicTacToe': TicTacToeEnv,
    }
    
    if name in ENVS:
        return ENVS[name](**kwargs)
    else:
        # Gymnasium environments
        from rl.settings import ENVIRONMENTS
        if name in ENVIRONMENTS:
            env_id = ENVIRONMENTS[name]['env_id']
            return GymnasiumEnvWrapper(env_id, name)
        raise ValueError(f"Unknown environment: {name}")


__all__ = [
    'BaseEnvironment',
    'GridWorldEnv',
    'MazeEnv',
    'TicTacToeEnv',
    'GymnasiumEnvWrapper',
    'create_environment'
]
