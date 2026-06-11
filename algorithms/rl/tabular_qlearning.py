import numpy as np
import os
import json

class TabularQLearning:
    """
    Enhanced Tabular Q-Learning Baseline.
    Discretizes the continuous scalar observations from the environment Dict space
    into configurable bins. Does not use the history window to remain a true 
    MDP classical baseline.
    """
    def __init__(self, action_dim=2, num_scalar_features=14, bins_per_feature=5, 
                 alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9995, seed=42):
        self.action_dim = action_dim
        self.num_features = num_scalar_features
        self.bins_per_feature = bins_per_feature
        
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # The state space is immense if all 14 features use 5 bins (5^14).
        # We will use a dictionary-based Q-table to lazily instantiate states.
        self.q_table = {}
        
        self.rng = np.random.default_rng(seed)

    def _discretize(self, obs_dict):
        """
        Takes the Dict observation, extracts the 'scalars' array (values [0, 1]),
        and maps each to an integer bin [0, bins_per_feature-1].
        """
        scalars = obs_dict["scalars"]
        
        # scalars are in [0, 1]. Multiply by bins and floor.
        # Clip to ensure 1.0 -> max_bin instead of out of bounds.
        binned = np.clip(np.floor(scalars * self.bins_per_feature), 0, self.bins_per_feature - 1)
        
        # Convert to tuple for dictionary hashing
        return tuple(binned.astype(int))

    def _get_q_values(self, state_tuple):
        if state_tuple not in self.q_table:
            self.q_table[state_tuple] = np.zeros(self.action_dim)
        return self.q_table[state_tuple]

    def predict(self, obs_dict, deterministic=True):
        state_tuple = self._discretize(obs_dict)
        q_vals = self._get_q_values(state_tuple)
        
        if not deterministic and self.rng.random() < self.epsilon:
            action = self.rng.integers(0, self.action_dim)
        else:
            # Break ties purely randomly
            max_val = np.max(q_vals)
            best_actions = np.where(q_vals == max_val)[0]
            action = self.rng.choice(best_actions)
            
        return action, None # None is for state (like RNNs in SB3)

    def learn(self, obs_dict, action, reward, next_obs_dict, done):
        """Standard Q-learning update."""
        state = self._discretize(obs_dict)
        next_state = self._discretize(next_obs_dict)
        
        q_current = self._get_q_values(state)[action]
        q_next_max = np.max(self._get_q_values(next_state)) if not done else 0.0
        
        # Q(s,a) = Q(s,a) + alpha * (r + gamma*max_Q(s',a') - Q(s,a))
        self.q_table[state][action] += self.alpha * (reward + self.gamma * q_next_max - q_current)
        
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        """Saves Q-table and metadata."""
        # Convert tuple keys to strings for JSON
        serializable_q = {str(k): v.tolist() for k, v in self.q_table.items()}
        data = {
            "q_table": serializable_q,
            "epsilon": self.epsilon
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)

    def load(self, filepath):
        """Loads Q-table and metadata."""
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        self.epsilon = data["epsilon"]
        
        # Convert string keys back to tuples of ints
        self.q_table = {}
        for k_str, v_list in data["q_table"].items():
            # Remove parentheses and split by comma
            clean_str = k_str.strip("()").replace(" ", "")
            if clean_str: # avoid empty string case
                parts = clean_str.split(",")
                # filter out empty parts (resulting from trailing comma)
                parts = [p for p in parts if p]
                k_tuple = tuple(int(x) for x in parts)
                self.q_table[k_tuple] = np.array(v_list)
