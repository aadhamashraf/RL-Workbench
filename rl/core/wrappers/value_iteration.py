import numpy as np
import time
from rl.core.wrappers.base import Agent
from rl.core import dp
class ValueIterationAgent(Agent):
    
    def __init__(self, env, gamma=0.99, theta=0.001, max_iterations=1000, **kwargs):
        super().__init__(env, gamma)
        self.theta = theta
        self.max_iterations = max_iterations
        self.V = None
        self.policy = None
        self.Q = None
    
    def _build_mdp(self):
        P = {}
        R = {}
        
        for s in range(self.n_states):
            P[s] = {}
            R[s] = {}
            
            for a in range(self.n_actions):
                try:
                    trans = self.env.get_transition_prob(s, a)
                    P[s][a] = {ns: prob for ns, (prob, reward) in trans.items()}
                    R[s][a] = {ns: reward for ns, (prob, reward) in trans.items()}
                except NotImplementedError:
                    raise ValueError("Value Iteration requires model-based environment")
        
        return P, R
    
    def train(self, progress_callback=None):
        start = time.time()
        P, R = self._build_mdp()
        
        V_dict, stats = dp.value_iteration(P, R, self.gamma, self.theta)
        self.V = np.array([V_dict[s] for s in range(self.n_states)])
        

        self.policy = np.zeros(self.n_states, dtype=int)
        self.Q = np.zeros((self.n_states, self.n_actions))
        
        for s in range(self.n_states):
            if P[s]:
                action_vals = []
                for a in range(self.n_actions):
                    if a in P[s]:
                        q_val = dp.bellman_equation(P, R, V_dict, self.gamma, s, mode="Q", action=a)
                        action_vals.append(q_val)
                        self.Q[s, a] = q_val
                    else:
                        action_vals.append(-float('inf'))
                        self.Q[s, a] = -float('inf')
                self.policy[s] = np.argmax(action_vals)
        
        return self.create_dp_history(start, stats)
    
    def get_action(self, state, explore=False):
        return self.select_greedy(self.Q[state])
