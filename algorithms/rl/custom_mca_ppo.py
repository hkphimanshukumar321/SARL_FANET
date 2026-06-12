"""
custom_mca_ppo.py -- MCA-PPO using Stable-Baselines3
=====================================================
Uses standard SB3 PPO to train on the flattened Discrete(2^N)
action space exposed by MARLtoSARLWrapper.
"""

from stable_baselines3 import PPO

def create_mca_ppo(env, seed: int = 42, device: str = "auto", **kwargs):
    """
    Creates an SB3 PPO agent.
    Because MARLtoSARLWrapper flattens the MultiDiscrete space into a
    standard Discrete space, we can simply use the off-the-shelf PPO algorithm.
    """
    # Filter kwargs to only pass what PPO accepts
    valid_keys = {"learning_rate", "n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "clip_range", "ent_coef"}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}

    # Set up defaults
    ppo_kwargs = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "seed": seed,
        "device": device,
        "tensorboard_log": "./results/tensorboard_logs/"
    }
    
    # Override with tuned kwargs
    ppo_kwargs.update(filtered_kwargs)

    model = PPO(
        "MultiInputPolicy",
        env,
        **ppo_kwargs
    )
    return model
