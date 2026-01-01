import numpy as np
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from rl.envs.base import BaseEnvironment


class TwoRoomsEnv(BaseEnvironment):
    """Two rooms connected by a door"""
    
    def __init__(self, size=10):
        super().__init__()
        self.size = size
        self.door_pos = (size // 2, size // 2)  # Door in the middle of the wall
        self.start_pos = (0, 0)
        self.goal_pos = (size - 1, size - 1)
        self.action_map = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        self.reset()
    
    def reset(self):
        self.state = self.start_pos
        self.done = False
        return self._get_state_id(self.state)
    
    def step(self, a):
        if self.done:
            return self._get_state_id(self.state), 0, True, {}
        
        y, x = self.state
        dy, dx = self.action_map[a]
        ny, nx = y + dy, x + dx
        
        # Check boundaries
        if 0 <= ny < self.size and 0 <= nx < self.size:
            # Check wall (vertical wall in the middle, except at door)
            wall_x = self.size // 2
            if not (x < wall_x <= nx or nx <= wall_x < x) or (ny, wall_x) == self.door_pos:
                self.state = (ny, nx)
        
        reward = 1.0 if self.state == self.goal_pos else -0.01
        self.done = self.state == self.goal_pos
        
        return self._get_state_id(self.state), reward, self.done, {}
    
    def render(self, mode='rgb_array'):
        if not MATPLOTLIB_AVAILABLE:
            return np.ones((300, 300, 3), dtype=np.uint8) * 200
        
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Draw grid
        for i in range(self.size + 1):
            ax.plot([0, self.size], [i, i], 'k-', linewidth=0.5)
            ax.plot([i, i], [0, self.size], 'k-', linewidth=0.5)
        
        # Draw wall (vertical line in the middle)
        wall_x = self.size // 2
        for y in range(self.size):
            if (y, wall_x) != self.door_pos:
                ax.plot([wall_x, wall_x], [y, y + 1], 'k-', linewidth=3)
        
        # Draw door
        dy, dx = self.door_pos
        ax.add_patch(Rectangle((dx - 0.1, self.size - dy - 1), 0.2, 1, fc='green', ec='black'))
        
        # Draw goal
        gy, gx = self.goal_pos
        ax.add_patch(Rectangle((gx, self.size - gy - 1), 1, 1, fc='gold', ec='black'))
        
        # Draw agent
        y, x = self.state
        ax.add_patch(plt.Circle((x + 0.5, self.size - y - 0.5), 0.3, color='blue'))
        
        ax.set(xlim=(0, self.size), ylim=(0, self.size), aspect='equal', title='Two Rooms')
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
        return self.size ** 2
    
    def get_action_space_size(self):
        return 4
    
    def _get_state_id(self, pos):
        return pos[0] * self.size + pos[1]
    
    def _get_pos_from_state_id(self, sid):
        return divmod(sid, self.size)
    
    def get_transition_prob(self, sid, a):
        """Get transition probabilities for model-based methods"""
        y, x = self._get_pos_from_state_id(sid)
        dy, dx = self.action_map[a]
        ny, nx = y + dy, x + dx
        
        # Check boundaries and wall
        wall_x = self.size // 2
        if 0 <= ny < self.size and 0 <= nx < self.size:
            if not (x < wall_x <= nx or nx <= wall_x < x) or (ny, wall_x) == self.door_pos:
                next_pos = (ny, nx)
            else:
                next_pos = (y, x)
        else:
            next_pos = (y, x)
        
        reward = 1.0 if next_pos == self.goal_pos else -0.01
        return {self._get_state_id(next_pos): (1.0, reward)}
