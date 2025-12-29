import random
from typing import List, Callable, Tuple, Dict
import numpy as np

def monte_carlo(state_space: List[int], actions: List[int], generate_episode: Callable, 
                episodes: int = 1000, gamma: float = 1.0, type: str = "FV", 
                fixed_alpha: bool = False, alpha: float = 0.1) -> Tuple[Dict, Dict, Dict]:

    returns_sum = {}
    returns_count = {}
    Q = {}

    for s in state_space:
        for a in actions:
            Q[(s, a)] = 0
            returns_sum[(s, a)] = 0
            returns_count[(s, a)] = 0

    policy = {s: random.choice(actions) for s in state_space}
    episode_rewards = []
    episode_lengths = []

    for _ in range(episodes):
        states, rewards, actions_taken = generate_episode(policy, return_actions=True)

        episode_rewards.append(sum(rewards))
        episode_lengths.append(len(actions_taken))

        G = 0
        visited = set()

        for t in reversed(range(len(actions_taken))):
            sa = (states[t], actions_taken[t])
            G = rewards[t + 1] + gamma * G
            if not fixed_alpha:
                if type == "FV":
                    if sa not in visited:
                        visited.add(sa)
                        returns_sum[sa] += G
                        returns_count[sa] += 1
                        Q[sa] = returns_sum[sa] / returns_count[sa]
                elif type == "EV":
                    returns_sum[sa] += G
                    returns_count[sa] += 1
                    Q[sa] = returns_sum[sa] / returns_count[sa]
            else:
                if type == "FV":
                    if sa not in visited:
                        visited.add(sa)
                        Q[sa] += alpha * (G - Q[sa])
                elif type == "EV":
                    Q[sa] += alpha * (G - Q[sa])
        for s in state_space:
            policy[s] = max(actions, key=lambda a: Q[(s, a)])
   
    return Q, policy, {'episode_rewards': episode_rewards, 'episode_lengths': episode_lengths}
