"""test_env_wrappers.py -- Test SARL environment wrappers."""

import pytest
import numpy as np


class TestMARLtoSARLWrapper:
    """Test the MARLtoSARLWrapper (Discrete(2) bridge)."""

    def test_reset_returns_valid_obs(self):
        from envs.marl_sarl_wrapper import MARLtoSARLWrapper
        from configs.sarl_config import RLConfig

        env = MARLtoSARLWrapper(seed=42)
        obs, info = env.reset(seed=42)

        assert "scalars" in obs
        assert "history" in obs
        # Wrapper compresses to 14-dim SARL obs, not the 240-dim SARLCentralEnv obs
        assert obs["scalars"].shape == (14,)
        assert obs["history"].shape == (RLConfig.HISTORY_WINDOW_STEPS, 14)
        assert obs["scalars"].dtype == np.float32

    def test_step_returns_valid_transition(self):
        from envs.marl_sarl_wrapper import MARLtoSARLWrapper

        env = MARLtoSARLWrapper(seed=42)
        obs, _ = env.reset(seed=42)
        action = 0  # TDMA
        next_obs, reward, terminated, truncated, info = env.step(action)

        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "scalars" in next_obs

    def test_action_space_discrete(self):
        from envs.marl_sarl_wrapper import MARLtoSARLWrapper
        from gymnasium import spaces

        env = MARLtoSARLWrapper(seed=42)
        assert isinstance(env.action_space, spaces.Discrete)
        assert env.action_space.n == 2


class TestSARLCentralEnv:
    """Test the SARLCentralEnv (MultiDiscrete centralized bridge)."""

    def test_reset_returns_valid_obs(self):
        from envs.sarl_central_env import SARLCentralEnv

        env = SARLCentralEnv(seed=42)
        obs, info = env.reset(seed=42)

        assert "scalars" in obs
        assert "history" in obs

    def test_step_returns_valid_transition(self):
        from envs.sarl_central_env import SARLCentralEnv

        env = SARLCentralEnv(seed=42)
        obs, _ = env.reset(seed=42)
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)

        assert isinstance(reward, float)
        assert isinstance(terminated, bool)

    def test_action_space_multidiscrete(self):
        from envs.sarl_central_env import SARLCentralEnv
        from gymnasium import spaces

        env = SARLCentralEnv(seed=42)
        assert isinstance(env.action_space, spaces.MultiDiscrete)
