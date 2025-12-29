import gymnasium as gym
import numpy as np
from rl.envs.base import BaseEnvironment

class GymnasiumEnvWrapper(BaseEnvironment):
  """Wrapper for Gymnasium environments"""
  
  def __init__(self, env_id, name):
    super().__init__()
    self.env_id = env_id
    self.env_name = name
    self.env = gym.make(env_id, render_mode='rgb_array')
    
    self.is_discrete_state = isinstance(self.env.observation_space, gym.spaces.Discrete)
    self.is_tuple_state = isinstance(self.env.observation_space, gym.spaces.Tuple)
    self.is_discrete_action = isinstance(self.env.action_space, gym.spaces.Discrete)
    
    if not self.is_discrete_state and not self.is_tuple_state:
      self.state_bins = self._create_state_bins()
    
    self.reset()
  
  def reset(self):
    """Reset environment"""
    obs, info = self.env.reset()
    self.state = obs
    self.done = False
    
    if self.is_discrete_state:
      return int(obs)
    elif self.is_tuple_state:
      return self._flatten_tuple(obs)
    else:
      return self._discretize(obs)
  
  def step(self, a):
    """Take action"""
    obs, r, terminated, truncated, info = self.env.step(a)
    self.state = obs
    self.done = terminated or truncated
    
    if self.is_discrete_state:
      sid = int(obs)
    elif self.is_tuple_state:
      sid = self._flatten_tuple(obs)
    else:
      sid = self._discretize(obs)
    
    return sid, r, self.done, info
  
  def render(self, mode='rgb_array'):
    """Render environment"""
    try:
      frame = self.env.render()
      
      if frame is None or not isinstance(frame, np.ndarray):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        
        fig = Figure(figsize=(6, 4), dpi=100)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        ax.text(5, 7, f'{self.env_name}', ha='center', va='center',
               fontsize=20, fontweight='bold', color='#667eea')
        ax.text(5, 5, f'State: {self.state}', ha='center', va='center',
               fontsize=14, color='#333')
        ax.text(5, 3, 'Rendering not available', ha='center', va='center',
               fontsize=12, color='#999')
        
        canvas.draw()
        w, h = fig.get_size_inches() * fig.get_dpi()
        frame = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        frame = frame.reshape(int(h), int(w), 4)
        frame = frame[:, :, :3]
        plt.close(fig)
      
      return frame
      
    except Exception as e:
      import matplotlib.pyplot as plt
      from matplotlib.backends.backend_agg import FigureCanvasAgg
      from matplotlib.figure import Figure
      
      fig = Figure(figsize=(6, 4), dpi=100)
      canvas = FigureCanvasAgg(fig)
      ax = fig.add_subplot(111)
      ax.set_xlim(0, 10)
      ax.set_ylim(0, 10)
      ax.axis('off')
      
      ax.text(5, 5, 'Render Error', ha='center', va='center',
             fontsize=16, fontweight='bold', color='red')
      ax.text(5, 3, str(e)[:50], ha='center', va='center',
             fontsize=10, color='#666')
      
      canvas.draw()
      w, h = fig.get_size_inches() * fig.get_dpi()
      frame = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
      frame = frame.reshape(int(h), int(w), 4)
      frame = frame[:, :, :3]
      plt.close(fig)
      
      return frame
  
  def get_state_space_size(self):
    """Get state space size"""
    if self.is_discrete_state:
      return self.env.observation_space.n
    elif self.is_tuple_state:
      return int(np.prod([space.n for space in self.env.observation_space.spaces]))
    else:
      return np.prod([len(bins) + 1 for bins in self.state_bins])
  
  def get_action_space_size(self):
    """Get action space size"""
    if self.is_discrete_action:
      return self.env.action_space.n
    else:
      raise NotImplementedError("Continuous action spaces not supported")
  
  def close(self):
    self.env.close()
  
  def _create_state_bins(self):
    """Create bins for discretizing continuous states"""
    if self.env_name == 'CartPole':
      return [
        np.linspace(-2.4, 2.4, 10),
        np.linspace(-3.0, 3.0, 10),
        np.linspace(-0.3, 0.3, 10),
        np.linspace(-2.0, 2.0, 10)
      ]
    elif self.env_name == 'MountainCar':
      return [
        np.linspace(-1.2, 0.6, 20),
        np.linspace(-0.07, 0.07, 20)
      ]
    elif self.env_name == 'Acrobot':
      return [
        np.linspace(-1, 1, 10),
        np.linspace(-1, 1, 10),
        np.linspace(-1, 1, 10),
        np.linspace(-1, 1, 10),
        np.linspace(-4, 4, 10),
        np.linspace(-9, 9, 10)
      ]
    elif self.env_name == 'LunarLander':
      return [
        np.linspace(-1.0, 1.0, 5),
        np.linspace(-1.0, 1.0, 5),
        np.linspace(-1.0, 1.0, 5),
        np.linspace(-1.0, 1.0, 5),
        np.linspace(-1.0, 1.0, 5),
        np.linspace(-1.0, 1.0, 5),
        np.linspace(0.0, 1.0, 2),
        np.linspace(0.0, 1.0, 2)
      ]
    else:
      low = self.env.observation_space.low
      high = self.env.observation_space.high
      
      bins = []
      for l, h in zip(low, high):
        if np.isinf(l) or np.isinf(h):
           # Fallback for infinite bounds
           l = l if not np.isinf(l) else -10.0
           h = h if not np.isinf(h) else 10.0
        bins.append(np.linspace(l, h, 10))
      return bins
  
  def _discretize(self, s):
    """Convert continuous state to discrete ID"""
    if self.is_discrete_state:
      return int(s)
    
    indices = []
    for i, val in enumerate(s):
      idx = np.digitize(val, self.state_bins[i])
      indices.append(idx)
    
    sid = 0
    mult = 1
    for idx, bins in zip(reversed(indices), reversed(self.state_bins)):
      sid += idx * mult
      mult *= (len(bins) + 1)
    
    return sid
    
  def _flatten_tuple(self, obs):
    """Flatten tuple observation to single integer"""
    sid = 0
    mult = 1
    
    for val, space in zip(reversed(obs), reversed(self.env.observation_space.spaces)):
      sid += int(val) * mult
      mult *= space.n
      
    return sid
  
  def get_transition_prob(self, s, a):
    """Get transition probs (only for specific envs)"""
    env_check = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
    
    if hasattr(env_check, 'P'):
      trans = env_check.P[s][a]
      result = {}
      for prob, next_s, r, done in trans:
        if next_s in result:
          old_prob, old_r = result[next_s]
          result[next_s] = (old_prob + prob, r)
        else:
          result[next_s] = (prob, r)
      return result
    else:
      raise NotImplementedError(
        f"{self.env_name} doesn't support model-based methods"
      )
