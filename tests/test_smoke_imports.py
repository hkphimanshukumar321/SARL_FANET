"""test_smoke_imports.py -- Verify all modules import cleanly."""

import pytest


def test_import_configs():
    from configs import config
    from configs.cluster_config import ClusterConfig
    from configs.sarl_config import RLConfig
    assert hasattr(config, "N")
    assert hasattr(ClusterConfig, "C_MAX")
    assert hasattr(RLConfig, "NUM_SCALAR_FEATURES")


def test_import_algorithms_channel():
    from algorithms.channel.fading import BERCalculator, AWGNChannel, RayleighChannel


def test_import_algorithms_mac():
    from algorithms.mac.baseline import Config, Logger


def test_import_algorithms_mobility():
    from algorithms.mobility.speed import SpeedEngine
    from algorithms.mobility.models import create_mobility_model
    from algorithms.mobility.link import compute_distances


def test_import_algorithms_rl():
    from algorithms.rl.rewards import compute_cluster_reward, compute_team_reward
    from algorithms.rl.features_extractor import MCAFeaturesExtractor
    from algorithms.rl.custom_mca_ppo import create_mca_ppo
    from algorithms.rl.tabular_qlearning import TabularQLearning
    from algorithms.rl.sb3_baselines import create_sb3_baseline


def test_import_envs():
    from envs.marl_mac_env import MARLMacEnv
    from envs.sarl_central_env import SARLCentralEnv
    from envs.sarl_mac_env import AdaptiveMacEnv
    from envs.marl_sarl_wrapper import MARLtoSARLWrapper
    from envs.burst_scheduler import decode_action


def test_import_utils():
    from utils.device_manager import resolve_device


def test_config_sarl_flags():
    """Verify SARL-specific execution flags exist."""
    from configs import config
    assert hasattr(config, "RUN_DQN")
    assert hasattr(config, "RUN_PPO")
    assert hasattr(config, "RUN_A2C")
    assert hasattr(config, "RUN_CUSTOM_RL")
    assert hasattr(config, "RUN_MCA_PPO")
    assert hasattr(config, "RUN_TABULAR_QLEARNING")
