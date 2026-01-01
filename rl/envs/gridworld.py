import numpy as np
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from rl.envs.base import BaseEnvironment


class GridWorldEnv(BaseEnvironment):

    def __init__(self, size=5, goal_reward=1.0, step_cost=-0.01, obstacles=None):
        super().__init__()
        self.size, self.goal_reward, self.step_cost = size, goal_reward, step_cost
        self.start_pos, self.goal_pos = (0, 0), (size - 1, size - 1)
        self.action_map = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

        if obstacles is None:
            self.obstacles = [
                ((i + 1) % (size - 1), (i * 2 + 1) % (size - 1))
                for i in range(max(2, size // 2))
                if ((i + 1) % (size - 1), (i * 2 + 1) % (size - 1)) not in [(0, 0), self.goal_pos]
            ]
            if not self._has_path():
                self.obstacles = []
        else:
            self.obstacles = obstacles

        self.reset()

    def _has_path(self):
        from collections import deque
        visited = set()
        queue = deque([self.start_pos])
        
        while queue:
            pos = queue.popleft()
            if pos == self.goal_pos:
                return True
            if pos in visited:
                continue
            visited.add(pos)
            
            y, x = pos
            for dy, dx in self.action_map.values():
                ny, nx = y + dy, x + dx
                if (0 <= ny < self.size and 0 <= nx < self.size and 
                    (ny, nx) not in self.obstacles and (ny, nx) not in visited):
                    queue.append((ny, nx))
        
        return False

    def reset(self):
        self.state, self.done = self.start_pos, False
        return self._get_state_id(self.state)

    def step(self, a):
        if self.done:
            return self._get_state_id(self.state), 0, True, {}

        y, x = self.state
        dy, dx = self.action_map[a]
        ny, nx = y + dy, x + dx

        if 0 <= ny < self.size and 0 <= nx < self.size and (ny, nx) not in self.obstacles:
            self.state = (ny, nx)

        reward = self.goal_reward if self.state == self.goal_pos else self.step_cost
        self.done = self.state == self.goal_pos

        return self._get_state_id(self.state), reward, self.done, {}

    def render(self, mode='rgb_array'):
        if not MATPLOTLIB_AVAILABLE:
            return np.ones((300, 300, 3), dtype=np.uint8) * 200

        fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')

        for i in range(self.size + 1):
            ax.plot([0, self.size], [i, i], 'k-')
            ax.plot([i, i], [0, self.size], 'k-')

        for y, x in self.obstacles:
            ax.add_patch(Rectangle((x, self.size - y - 1), 1, 1, fc='gray', ec='black'))

        gy, gx = self.goal_pos
        ax.add_patch(Rectangle((gx, self.size - gy - 1), 1, 1, fc='gold', ec='black'))

        y, x = self.state
        ax.add_patch(plt.Circle((x + 0.5, self.size - y - 0.5), 0.3, color='blue'))

        ax.set(xlim=(0, self.size), ylim=(0, self.size), aspect='equal', title='GridWorld')
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
        y, x = self._get_pos_from_state_id(sid)
        dy, dx = self.action_map[a]
        ny, nx = y + dy, x + dx

        next_pos = (ny, nx) if (0 <= ny < self.size and 0 <= nx < self.size and (ny, nx) not in self.obstacles) else (y, x)

        reward = self.goal_reward if next_pos == self.goal_pos else self.step_cost
        return {self._get_state_id(next_pos): (1.0, reward)}
