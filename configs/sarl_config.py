# sarl_config.py
# Centralized configuration for the RL Environment and Training

import os
from configs import config as global_cfg

class RLConfig:
    # ----------------------------------------------------
    # Environment Bounds (for Normalization)
    # ----------------------------------------------------
    MAX_THROUGHPUT_MBPS = global_cfg.PHY_RATE_BPS / 1e6  # Dynamically bounds to phy rate
    MAX_DELAY_MS = 200.0           # Clipping bound for delay
    MAX_QUEUE_OCCUPANCY = float(global_cfg.QMAX)    # Equal to QMAX
    MAX_BACKLOG = 500.0            # Approx max backlog scale
    MAX_SPEED_MPS = float(global_cfg.V_MAX)         # Max V_MAX
    TRAIN_SIM_TIME_S = 10.0        # Duration of one training episode
    DECISION_INTERVAL_S = 1.0      # Time between RL actions (seconds)
    
    # ----------------------------------------------------
    # Observation Features
    # ----------------------------------------------------
    NUM_SCALAR_FEATURES = 14       # MARLtoSARLWrapper produces 14-dim obs vector
    HISTORY_WINDOW_STEPS = 5       # Size of the temporal window
    
    # ----------------------------------------------------
    # Reward Weights (MCA-D3QN)
    # Rt = w_T*T^ - w_D*D^ - w_F*F^ - w_J*J^
    # ----------------------------------------------------
    REWARD_W_THROUGHPUT = 0.35      # MATCH MARL W_THROUGHPUT
    REWARD_W_DELAY = 0.15           # MATCH MARL W_DELAY
    REWARD_W_FAILURES = 0.25        # MATCH MARL W_DROPS
    REWARD_W_JITTER = 0.25          # MATCH MARL W_COLLISIONS
    
    # ----------------------------------------------------
    # Training Hyperparameters Default 
    # ----------------------------------------------------
    TOTAL_TIMESTEPS = 10_000
    N_ENVS = 4
    SEED = 42

    @staticmethod
    def get_results_dir():
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
        os.makedirs(base, exist_ok=True)
        return base
