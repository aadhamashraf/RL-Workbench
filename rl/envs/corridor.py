import numpy as np
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from rl.envs.base import BaseEnvironment


class CorridorEnv(BaseEnvironment):
    """Simple 1D corridor navigation environment"""
    
    def __init__(self, length=10):
        super().__init__()
        self.length = length
        self.start_pos = 0
        self.goal_pos = length - 1
        self.reset()
    
    def reset(self):
        self.state = self.start_pos
        self.done = False
        return self.state
    
    def step(self, a):
        """Actions: 0=left, 1=right"""
        if self.done:
            return self.state, 0, True, {}
        
        # Move left or right
        if a == 0 and self.state > 0:
            self.state -= 1
        elif a == 1 and self.state < self.length - 1:
            self.state += 1
        
        # Check if reached goal
        reward = 1.0 if self.state == self.goal_pos else -0.01
        self.done = self.state == self.goal_pos
        
        return self.state, reward, self.done, {}
    
    def render(self, mode='rgb_array'):
        if not MATPLOTLIB_AVAILABLE:
            return np.ones((300, 300, 3), dtype=np.uint8) * 200
        
        fig, ax = plt.subplots(figsize=(10, 2))
        
        # Draw corridor
        for i in range(self.length):
            color = 'gold' if i == self.goal_pos else 'white'
            ax.add_patch(Rectangle((i, 0), 1, 1, fc=color, ec='black'))
        
        # Draw agent
        ax.add_patch(plt.Circle((self.state + 0.5, 0.5), 0.3, color='blue'))
        
        ax.set(xlim=(0, self.length), ylim=(0, 1), aspect='equal', title='Corridor')
        ax.axis('off')
        
        if mode == 'rgb_array':
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            img = img[:, :, :3]  # Convert RGBA to RGB
            plt.close(fig)
            return img
        
        return fig
    
    def get_state_space_size(self):
        return self.length
    
    def get_action_space_size(self):
        return 2  # left, right
    
    def get_transition_prob(self, s, a):
        """Get transition probabilities for model-based methods"""
        if s == self.goal_pos:
            return {s: (1.0, 0.0)}
        
        # Determine next state
        if a == 0:  # left
            next_s = max(0, s - 1)
        else:  # right
            next_s = min(self.length - 1, s + 1)
        
        reward = 1.0 if next_s == self.goal_pos else -0.01
        return {next_s: (1.0, reward)}
