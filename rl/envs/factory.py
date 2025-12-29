from rl.settings import ENVIRONMENTS, ENVIRONMENT_CLASSES

def create_environment(name):
  """Create environment instance"""
  
  if name not in ENVIRONMENTS:
    raise ValueError(f"Unknown environment: {name}")
  
  cfg = ENVIRONMENTS[name]
  cls = ENVIRONMENT_CLASSES[name]
  
  if cls == 'GridWorldEnv':
    from rl.envs.gridworld import GridWorldEnv
    return GridWorldEnv()
  
  elif cls == 'GymnasiumEnv':
    from rl.envs.gymnasium_wrapper import GymnasiumEnvWrapper
    return GymnasiumEnvWrapper(cfg['env_id'], name)
  
  elif cls == 'TicTacToeEnv':
    from rl.envs.tictactoe import TicTacToeEnv
    return TicTacToeEnv()
  
  elif cls == 'MazeEnv':
    from rl.envs.maze import MazeEnv
    return MazeEnv(size=10, difficulty='medium')
  
  elif cls == 'SnakeEnv':
    from rl.envs.snake import SnakeEnv
    return SnakeEnv(grid_size=10)
  
  else:
    raise ValueError(f"Unknown environment class: {cls}")

