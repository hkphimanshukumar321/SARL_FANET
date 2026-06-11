# sarl_config.py
# Centralized configuration for the RL Environment and Training

import os
from configs import config as global_cfg
from configs.cluster_config import ClusterConfig as CC

class RLConfig:
    # ----------------------------------------------------
    # Environment Bounds (for Normalization)
    # ----------------------------------------------------
    MAX_THROUGHPUT_MBPS = global_cfg.PHY_RATE_BPS / 1e6
    MAX_DELAY_MS = 200.0
    MAX_QUEUE_OCCUPANCY = float(global_cfg.QMAX)
    MAX_BACKLOG = 500.0
    MAX_SPEED_MPS = float(global_cfg.V_MAX)
    TRAIN_SIM_TIME_S = 10.0
    DECISION_INTERVAL_S = 1.0
    
    # ----------------------------------------------------
    # Observation Features
    # ----------------------------------------------------
    NUM_SCALAR_FEATURES = CC.C_MAX * CC.OBS_DIM_CLUSTER
    HISTORY_WINDOW_STEPS = 5
    
    # ----------------------------------------------------
    # Reward Weights (MCA-D3QN / MCA-PPO)
    # Rt = w_T*T^ - w_D*D^ - w_F*F^ - w_J*J^
    # ----------------------------------------------------
    REWARD_W_THROUGHPUT = 0.35
    REWARD_W_DELAY = 0.15
    REWARD_W_FAILURES = 0.25
    REWARD_W_JITTER = 0.25
    
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
