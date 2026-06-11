from gymnasium import spaces
from stable_baselines3 import A2C, DQN, PPO
from utils.device_manager import resolve_device

def create_sb3_baseline(env, algo_name="dqn", seed=42):
    """
    Creates generic Stable-Baselines3 models for baseline comparison.
    These models will only use the scalar branch of the observation space
    (or automatically flatten the dict obs space into a large 1D vector
    using SB3's default MultiInputPolicy flatten extractor).
    """
    
    # Resolve device from config (respects FORCE_CPU, ENABLE_GPU, TRAIN_ON_GPU)
    device = resolve_device("train")
    
    if algo_name.lower() == "dqn":
        if isinstance(getattr(env, "action_space", None), spaces.MultiDiscrete):
            raise ValueError("SB3 DQN does not support MultiDiscrete joint actions in SARLCentralEnv.")
        model = DQN(
            "MultiInputPolicy",
            env,
            learning_rate=1e-3,
            buffer_size=100000,
            learning_starts=100,
            batch_size=64,
            gamma=0.99,
            target_update_interval=1000,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            exploration_fraction=0.1,
            device=device,
            seed=seed,
            verbose=0
        )
    elif algo_name.lower() == "ppo":
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=3e-4,
            n_steps=128,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            device=device,
            seed=seed,
            verbose=0
        )
    elif algo_name.lower() == "a2c":
        model = A2C(
            "MultiInputPolicy",
            env,
            learning_rate=7e-4,
            n_steps=5,
            gamma=0.99,
            device=device,
            seed=seed,
            verbose=0
        )
    else:
        raise ValueError(f"Unknown SB3 baseline algorithm: {algo_name}")
        
    return model
