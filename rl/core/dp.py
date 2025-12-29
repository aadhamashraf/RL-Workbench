from typing import Dict, Tuple
import random

def bellman_equation( P: Dict, R: Dict, V: Dict,gamma: float,state: int, mode: str = "VI", policy: Dict = None, action: int = None) -> float:
    
  if state not in P or not P[state]:
        return 0.0
    
    result = 0.0
    best_value = float("-inf")

    for a in P[state]:

        if mode == "Q" and a != action:
            continue
        if mode == "PI" and (policy is None or a not in policy.get(state, {})):
            continue

        q_sa = sum(
            prob * (R[state][a][s_next] + gamma * V[s_next])
            for s_next, prob in P[state][a].items()
        )

        if mode == "VI":
            best_value = max(best_value, q_sa)

        elif mode == "PI":
            result += policy[state][a] * q_sa

        elif mode == "Q":
            return q_sa

    if mode == "VI":
        return best_value if best_value != float("-inf") else 0.0

    return result

def value_iteration(P: Dict, R: Dict, gamma: float, theta: float) -> Tuple[Dict, Dict]:

    V = {state: 0 for state in P}
    
    iteration = 0
    deltas = []
    
    while True:
        iteration += 1
        delta = 0
        for state in P:
            v = V[state]
            V[state] = bellman_equation(P, R, V, gamma, state, mode="VI")
            delta = max(delta, abs(v - V[state]))

        deltas.append(delta)
        
        if delta < theta:
            break
    
    stats = {
        'iterations': iteration,
        'deltas': deltas
    }
    
    return V, stats

def policy_evaluation(policy: Dict, P: Dict, R: Dict, gamma: float, theta: float) -> Dict:

    V = {state: 0 for state in P}

    while True:
        delta = 0

        for state in P:
            v = V[state]
            V[state] = bellman_equation(P, R, V, gamma, state, mode="PI", policy=policy)
            delta = max(delta, abs(v - V[state]))
        if delta < theta:
            break
    return V

def policy_iteration(P: Dict, R: Dict, gamma: float, theta: float) -> Tuple[Dict, Dict, Dict]:

    policy = {state: {random.choice(list(P[state].keys())): 1.0} for state in P if P[state]}
    
    iteration = 0
    deltas = []
    
    while True:
        iteration += 1

        V = policy_evaluation(policy, P, R, gamma, theta)

        policy_stable = True
        max_delta = 0
        
        for state in P:
            if not P[state]:
                continue
            
            old_action = list(policy[state].keys())[0]
            
            best_value = -float('inf')
            best_action = None
            
            for action in P[state]:

                q_value = bellman_equation(P, R, V, gamma, state, mode="Q", action=action)
                if q_value > best_value:
                    best_value = q_value
                    best_action = action
            

            old_value = bellman_equation(P, R, V, gamma, state, mode="Q", action=old_action)
            max_delta = max(max_delta, abs(best_value - old_value))
            
            policy[state] = {best_action: 1.0}
            
            if best_action != old_action:
                policy_stable = False
        
        deltas.append(max_delta)
        
        if policy_stable:
            break
    
    stats = {
        'iterations': iteration,
        'deltas': deltas
    }
    
    return V, policy, stats
