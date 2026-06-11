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
        from envs.sarl_central_env import SARLCentralEnv
        from algorithms.rl.custom_mca_d3qn import create_mca_d3qn

        env = SARLCentralEnv(seed=42)
        model = create_mca_d3qn(env, seed=42, device="cpu")
        model.learn(total_timesteps=SMOKE_STEPS)

        # Verify predict works
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        assert action is not None


class TestMCAPPOTraining:
    def test_mca_ppo_train_loop(self):
        from envs.sarl_central_env import SARLCentralEnv
        from algorithms.rl.custom_mca_ppo import create_mca_ppo

        env = SARLCentralEnv(seed=42)
        model = create_mca_ppo(env, seed=42, device="cpu", n_steps=10)
        model.learn(total_timesteps=SMOKE_STEPS)

        # Verify predict works
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        assert action is not None
