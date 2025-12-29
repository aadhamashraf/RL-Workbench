import numpy as np
import time
from rl.core.wrappers.base import Agent
from rl.core import dp
class PolicyEvaluationAgent(Agent):
    
    def __init__(self, env, gamma=0.99, theta=0.001, max_iterations=1000, **kwargs):
        super().__init__(env, gamma)
        self.theta = theta
        self.max_iterations = max_iterations
        self.V = None
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
                    raise ValueError("Policy Evaluation requires model-based environment")
        
        return P, R
    
    def train(self, progress_callback=None):
        start = time.time()
        P, R = self._build_mdp()
        
        policy = {s : {a : 1.0 / len(P[s]) for a in P[s]} for s in P if P[s]}
        
        V_dict = dp.policy_evaluation(policy, P, R, self.gamma, self.theta)
        
        self.V = np.array([V_dict.get(s, 0) for s in range(self.n_states)])
        

        self.Q = np.zeros((self.n_states, self.n_actions))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                if s in P and a in P[s]:
                    self.Q[s, a] = dp.bellman_equation(P, R, V_dict, self.gamma, s, mode="Q", action=a)
        
        return {
            'training_time': time.time() - start,
            'converged': True
        }
    
    def get_action(self, state, explore=False):
        return self.select_greedy(self.Q[state])
