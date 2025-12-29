from abc import ABC, abstractmethod
import numpy as np
import time
class Agent(ABC):
    
    def __init__(self, env, gamma=0.99):
        self.env = env
        self.gamma = gamma
        self.n_states = env.get_state_space_size()
        self.n_actions = env.get_action_space_size()
    
    @abstractmethod
    def train(self, progress_callback=None):
        pass
    
    @abstractmethod
    def get_action(self, state, explore=False):
        pass
    
    def select_epsilon_greedy(self, Q_values, epsilon):
        return np.random.randint(self.n_actions) if np.random.random() < epsilon else np.argmax(Q_values)
    
    @staticmethod
    def create_training_history(start_time, episodes, episode_rewards, episode_lengths):
        
        avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
        
        if episode_rewards and len(episode_rewards) > 1:
            if np.std(episode_rewards) < 1e-6:
                success_rate = 50.0
            else:
                threshold = np.percentile(episode_rewards, 75)
                success_rate = (np.sum(np.array(episode_rewards) >= threshold) / len(episode_rewards) * 100)
        else:
            success_rate = 0
  
        return {
            'training_time': time.time() - start_time,
            'episodes': episodes,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'average_reward': avg_reward,
            'success_rate': success_rate
        }
    
    @staticmethod
    def create_dp_history(start_time, stats):
        return {
            'training_time': time.time() - start_time,
            'converged': True,
            'iterations': stats['iterations'],
            'deltas': stats['deltas']
        }
    def select_greedy(self, Q_values):
        return np.argmax(Q_values)
