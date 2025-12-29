import numpy as np
import time
from rl.core.wrappers.base import Agent
from rl.core import qlearning
class QLearningAgent(Agent):
    
    def __init__(self, env, gamma=0.99, alpha=0.1, epsilon=0.3, epsilon_decay=0.995, epsilon_min=0.01, episodes=1000, **kwargs):
        super().__init__(env, gamma)
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.episodes = episodes
        self.Q = np.zeros((self.n_states, self.n_actions))
        self.policy = None
    
    def train(self, progress_callback=None):
        start = time.time()
        

        state_space = list(range(self.n_states))
        actions = list(range(self.n_actions))
        
        def step_fn(s, a):
            ns, r, done, _ = self.env.step(a)
            return ns, r, done
        
        def reset_fn():
            return self.env.reset()
        

        Q_dict, policy_dict, stats = qlearning.q_learning(
            state_space, actions, step_fn, reset_fn,
            episodes=self.episodes, alpha=self.alpha,
            gamma=self.gamma, epsilon=self.epsilon
        )
        

        for (s, a), val in Q_dict.items():
            self.Q[s, a] = val
        
        
        self.policy = np.array([policy_dict.get(s, 0) for s in range(self.n_states)])
        
        return self.create_training_history(
            start, self.episodes, 
            stats['episode_rewards'], 
            stats['episode_lengths']
        )
    
    def get_action(self, state, explore=False):
        if explore:
            return self.select_epsilon_greedy(self.Q[state], self.epsilon)
        return self.policy[state]
