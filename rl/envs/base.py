import numpy as np
from abc import ABC, abstractmethod

class BaseEnvironment(ABC):
  """Base class for all RL environments"""
  
  def __init__(self):
    self.state = None
    self.done = False
  
  @abstractmethod
  def reset(self):
    """Reset environment to initial state"""
    pass
  
  @abstractmethod
  def step(self, action):
    """Take action and return (next_state, reward, done, info)"""
    pass
  
  @abstractmethod
  def render(self):
    """Render current state"""
    pass
  
  @abstractmethod
  def get_state_space_size(self):
    """Return number of states"""
    pass
  
  @abstractmethod
  def get_action_space_size(self):
    """Return number of actions"""
    pass
  
  def get_transition_prob(self, s, a):
    """Get transition probs for model-based methods
    Returns: dict of {next_state: (prob, reward)}
    """
    raise NotImplementedError("Environment doesn't support model-based methods")
  
  def close(self):
    """Clean up resources"""
    pass
  
  def __enter__(self):
    return self
  
  def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()
    return False
