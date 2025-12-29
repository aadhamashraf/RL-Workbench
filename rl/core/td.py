import random
from typing import List, Callable, Tuple, Dict, Union
import numpy as np
def td(state_space: List[int], actions: List[int], step: Callable, reset: Callable, 
       episodes: int = 1000, alpha: float = 0.01, gamma: float = 1.0, 
       epsilon: float = 0.3, nstep: bool = False, n: int = 1, 
       q_based: bool = False) -> Tuple[Union[Dict, Dict], Dict, Dict]:
    if not q_based:
        V = {s: 0 for s in state_space}
        policy = {s: random.choice(actions) for s in state_space}
    else:
        Q = {(s, a): 0 for s in state_space for a in actions}
        policy = {s: random.choice(actions) for s in state_space}
    episode_rewards = []
    episode_lengths = []
    for _ in range(episodes):
        state = reset()
        total_reward = 0
        ep_steps = 0
        done = False
        
        if not nstep:
            max_steps = 1000
            steps = 0
            
            while not done and steps < max_steps:
                if np.random.rand() < epsilon:
                    action = random.choice(actions)
                else:
                    action = policy[state]
                next_state, reward, done = step(state, action)
                
                total_reward += reward
                ep_steps += 1
                if not q_based:
                    v_next = V[next_state] if not done else 0
                    V[state] += alpha * (reward + gamma * v_next - V[state])
                else:
                    if not done:
                        if np.random.rand() < epsilon:
                            next_action = random.choice(actions)
                        else:
                            next_action = policy[next_state]
                        target_val = reward + gamma * Q[(next_state, next_action)]
                    else:
                        target_val = reward
                    Q[(state, action)] += alpha * (target_val - Q[(state, action)])
                state = next_state
                steps += 1
                
        else:
            states = [state]
            actions_taken = []
            rewards = []
            t = 0
            T = float('inf')
            max_steps = 1000
            
            while True:
                if t < T:
                    if np.random.rand() < epsilon:
                        action = random.choice(actions)
                    else:
                        action = policy[state]
                    next_state, reward, done = step(state, action)
                    
                    total_reward += reward
                    ep_steps += 1
                    states.append(next_state)
                    actions_taken.append(action)
                    rewards.append(reward)
                    
                    if done or t >= max_steps:
                        T = t + 1
                    
                    state = next_state
                tau = t - n + 1
                if tau >= 0:
                    G = 0
                    for i in range(tau + 1, min(tau + n, T) + 1):
                        if i - 1 < len(rewards):
                            G += (gamma ** (i - tau - 1)) * rewards[i - 1]
                    
                    if tau + n < T:
                        if not q_based:
                            if tau + n < len(states):
                                G += (gamma ** n) * V[states[tau + n]]
                        else:
                            if tau + n < len(states) and tau + n < len(actions_taken):
                                G += (gamma ** n) * Q[(states[tau + n], actions_taken[tau + n])]
                    
                    if not q_based:
                        if tau < len(states):
                            V[states[tau]] += alpha * (G - V[states[tau]])
                    else:
                        if tau < len(states) and tau < len(actions_taken):
                            sa = (states[tau], actions_taken[tau])
                            Q[sa] += alpha * (G - Q[sa])
                
                if tau == T - 1:
                    break
                    
                t += 1
        episode_rewards.append(total_reward)
        episode_lengths.append(ep_steps)
        if not q_based:
            for s in state_space:
                policy[s] = random.choice(actions)
        else:
            for s in state_space:
                policy[s] = max(actions, key=lambda a: Q[(s, a)])
    stats = {'episode_rewards': episode_rewards, 'episode_lengths': episode_lengths}
    return (V, policy, stats) if not q_based else (Q, policy, stats)
