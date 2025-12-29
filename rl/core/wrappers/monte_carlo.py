import numpy as np
import time
from rl.core.wrappers.base import Agent
from rl.core import mc
class MonteCarloAgent(Agent):
    
    def __init__(self, env, gamma=0.99, epsilon=0.1, epsilon_decay=0.995, epsilon_min=0.01,
                 episodes=1000, mc_type='FV', use_alpha=False, alpha=0.1, **kwargs):
        super().__init__(env, gamma)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.episodes = episodes
        self.mc_type = mc_type
        self.use_alpha = use_alpha
        self.alpha = alpha
        self.Q = np.zeros((self.n_states, self.n_actions))
        self.policy = None
    
    def _generate_episode(self, policy, return_actions=False):
        states = []
        actions_taken = []
        rewards = [0]
        
        state = self.env.reset()
        states.append(state)
        done = False
        max_steps = 1000
        steps = 0
        
        while not done and steps < max_steps:
            action = policy.get(state, np.random.randint(self.n_actions))
            next_state, reward, done, _ = self.env.step(action)
            
            states.append(next_state)
            actions_taken.append(action)
            rewards.append(reward)
            
            state = next_state
            steps += 1
        
        if return_actions:
            return states, rewards, actions_taken
        return states, rewards
    
    def train(self, progress_callback=None):
        start = time.time()
        
        state_space = list(range(self.n_states))
        actions = list(range(self.n_actions))
        
        Q_dict, policy_dict, stats = mc.monte_carlo( state_space, actions, self._generate_episode, episodes=self.episodes, gamma=self.gamma, type=self.mc_type, fixed_alpha=self.use_alpha, alpha=self.alpha)
        
        for (s, a), val in Q_dict.items():
            self.Q[s, a] = val
        
        self.policy = np.array([policy_dict.get(s, 0) for s in range(self.n_states)])
        
        return self.create_training_history( start, self.episodes, stats['episode_rewards'], stats['episode_lengths'] )
    
    def get_action(self, state, explore=False):
        return self.select_epsilon_greedy(self.Q[state], self.epsilon) if explore else self.policy[state]
