from typing import List, Callable, Tuple, Dict
import numpy as np
import random
def q_learning(state_space: List[int], actions: List[int], step: Callable, reset: Callable, 
               episodes: int = 1000, alpha: float = 0.1, gamma: float = 1.0, 
               epsilon: float = 0.3) -> Tuple[Dict, Dict, Dict]:
    Q = {(s, a): 0 for s in state_space for a in actions}
    
    episode_rewards = []
    episode_lengths = []
    for _ in range(episodes):
        state = reset()
        total_reward = 0
        steps = 0
        
        max_steps = 1000
        
        done = False
        while not done and steps < max_steps:
            if np.random.rand() < epsilon:
                action = random.choice(actions)
            else:
                action = max(actions, key=lambda a: Q[(state, a)])
            next_state, reward, done = step(state, action)
            
            total_reward += reward
            steps += 1
            
            if done:
                target_val = reward
            else:
                best_next = max(Q[(next_state, a)] for a in actions)
                target_val = reward + gamma * best_next
            Q[(state, action)] += alpha * (target_val - Q[(state, action)])
            state = next_state
            
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
    policy = {s: max(actions, key=lambda a: Q[(s, a)]) for s in state_space}
    return Q, policy, {'episode_rewards': episode_rewards, 'episode_lengths': episode_lengths}
