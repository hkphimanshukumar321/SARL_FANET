"""test_sarl_training.py -- Short training smoke tests for SARL models."""

import pytest
import numpy as np


SMOKE_STEPS = 20  # Very short — just verify the training loop runs


class TestTabularTraining:
    def test_tabular_train_loop(self):
        from envs.marl_sarl_wrapper import MARLtoSARLWrapper
        from algorithms.rl.tabular_qlearning import TabularQLearning

        env = MARLtoSARLWrapper(seed=42)
        # Wrapper produces 14-dim SARL obs (not 240-dim SARLCentralEnv obs)
        agent = TabularQLearning(
            action_dim=2,
            num_scalar_features=14,
            seed=42,
        )
        obs, _ = env.reset()
        for _ in range(SMOKE_STEPS):
            action, _ = agent.predict(obs, deterministic=False)
            next_obs, reward, term, trunc, info = env.step(int(action))
            agent.learn(obs, int(action), reward, next_obs, bool(term or trunc))
            obs = next_obs
            if term or trunc:
                obs, _ = env.reset()


class TestMCAD3QNTraining:
    def test_mca_d3qn_train_loop(self):
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
        from envs.marl_sarl_wrapper import MARLtoSARLWrapper
        from algorithms.rl.custom_mca_d3qn import create_mca_d3qn

        env = Monitor(MARLtoSARLWrapper(seed=42))
        vec_env = DummyVecEnv([lambda: env])
        model = create_mca_d3qn(vec_env, seed=42)
        model.learn(total_timesteps=SMOKE_STEPS)

        # Verify predict works
        obs = vec_env.reset()
        action, _ = model.predict(obs, deterministic=True)
        assert action is not None


class TestMCAPPOTraining:
    def test_mca_ppo_train_loop(self):
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
        from envs.marl_sarl_wrapper import MARLtoSARLWrapper
        from algorithms.rl.custom_mca_ppo import create_mca_ppo

        env = Monitor(MARLtoSARLWrapper(seed=42))
        vec_env = DummyVecEnv([lambda: env])
        model = create_mca_ppo(vec_env, seed=42)
        model.learn(total_timesteps=SMOKE_STEPS)

        # Verify predict works
        obs = vec_env.reset()
        action, _ = model.predict(obs, deterministic=True)
        assert action is not None
