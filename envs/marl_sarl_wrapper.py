"""
MARLtoSARLWrapper — Gymnasium Wrapper that adapts MARLMacEnv (PettingZoo Parallel)
to look like a single-agent Gymnasium environment.

Observation bridging:
  MARL env outputs N×16 per-agent obs → compress to 1×14 SARL obs via mean-pooling + semantic mapping.

Action bridging:
  SARL agent outputs 1 action (0=TDMA, 1=CSMA) → broadcast to all N agents uniformly.

This lets SB3 models (DQN/PPO/A2C), MCA-D3QN, and Tabular Q-Learning
train natively on the MARL environment without code changes.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.marl_mac_env import MARLMacEnv
from configs.sarl_config import RLConfig


class MARLtoSARLWrapper(gym.Env):
    """
    Wraps MARLMacEnv (PettingZoo Parallel) as a single-agent Gymnasium env.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, seed=None, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.marl_env = MARLMacEnv(seed=seed)

        # The wrapper compresses MARL cluster obs to a fixed 14-dim SARL vector
        # (NOT RLConfig.NUM_SCALAR_FEATURES which is C_MAX * OBS_DIM_CLUSTER = 240)
        self.num_scalars = 14
        self.history_len = RLConfig.HISTORY_WINDOW_STEPS  # 5

        self.observation_space = spaces.Dict({
            "scalars": spaces.Box(low=0.0, high=1.0, shape=(self.num_scalars,), dtype=np.float32),
            "history": spaces.Box(low=0.0, high=1.0, shape=(self.history_len, self.num_scalars), dtype=np.float32),
        })
        self.action_space = spaces.Discrete(2)  # 0=TDMA, 1=CSMA

        # History buffer for temporal window
        self.history_buffer = np.zeros((self.history_len, self.num_scalars), dtype=np.float32)
        self.prev_delay = 0.0
        self._prev_action = 0

    def _bridge_obs(self, marl_obs_dict, info=None):
        """
        Compress N×16 MARL observations → 1×14 SARL observation vector.

        MARL features (per agent, 16-dim):
            0: x_norm, 1: y_norm, 2: z_norm,
            3: vx_norm, 4: vy_norm, 5: vz_norm,
            6: dist_to_target, 7: neighbor_count,
            8: mean_neighbor_dist, 9: queue_occupancy,
            10: traffic_load, 11: link_succ_prob,
            12: prev_mac_action, 13: prev_reward,
            14: mean_neighbor_speed, 15: fading_gain

        SARL features (scalar, 14-dim):
            0: load_norm, 1: n_norm, 2: q_norm, 3: b_norm,
            4: thr_norm, 5: del_norm, 6: col_norm, 7: drop_norm,
            8: d_norm, 9: lu_norm, 10: v_norm, 11: v_var_norm,
            12: sp_norm, 13: mac_norm
        """
        if not marl_obs_dict:
            return self._zero_obs()

        all_obs = np.stack(list(marl_obs_dict.values()))  # (N, 16)
        mean_obs = np.mean(all_obs, axis=0)  # (16,)

        # Extract metrics from info if available
        thr_norm = 0.0
        del_norm = 0.0
        col_norm = 0.0
        drop_norm = 0.0
        if info:
            any_info = next(iter(info.values()), {}) if isinstance(info, dict) else {}
            max_thr = RLConfig.MAX_THROUGHPUT_MBPS
            thr_norm = min(any_info.get("throughput", 0.0) / max_thr, 1.0) if max_thr > 0 else 0.0
            del_norm = min(any_info.get("delay_ms", 0.0) / RLConfig.MAX_DELAY_MS, 1.0)
            col_norm = min(any_info.get("collisions", 0) / 5000.0, 1.0)
            drop_norm = min(any_info.get("drops", 0) / 1000.0, 1.0)

        from configs import config as global_cfg
        n_norm = min(global_cfg.N / 150.0, 1.0)

        # Compute speed from velocity components
        speeds = np.sqrt(all_obs[:, 3]**2 + all_obs[:, 4]**2 + all_obs[:, 5]**2) * 30.0  # undo normalization
        mean_speed = np.mean(speeds)
        speed_var = np.var(speeds)

        sarl_obs = np.array([
            float(mean_obs[10]),                                    # 0: load_norm (traffic load)
            float(n_norm),                                          # 1: n_norm (num nodes)
            float(np.clip(mean_obs[9], 0, 1)),                     # 2: q_norm (queue occupancy)
            0.0,                                                    # 3: b_norm (backlog approx)
            float(thr_norm),                                        # 4: thr_norm (throughput)
            float(del_norm),                                        # 5: del_norm (delay)
            float(col_norm),                                        # 6: col_norm (collisions)
            float(drop_norm),                                       # 7: drop_norm (drops)
            float(np.clip(mean_obs[6], 0, 1)),                     # 8: d_norm (avg distance)
            float(np.clip(mean_obs[7], 0, 1)),                     # 9: lu_norm (neighbor density / link up)
            float(min(mean_speed / RLConfig.MAX_SPEED_MPS, 1.0)),  # 10: v_norm (mean speed)
            float(min(speed_var / (RLConfig.MAX_SPEED_MPS**2), 1.0)),  # 11: v_var_norm (speed variance)
            float(np.clip(mean_obs[11], 0, 1)),                    # 12: sp_norm (link succ prob)
            float(self._prev_action),                               # 13: mac_norm (prev action)
        ], dtype=np.float32)

        # Update history buffer
        self.history_buffer = np.roll(self.history_buffer, shift=-1, axis=0)
        self.history_buffer[-1] = sarl_obs

        return {
            "scalars": sarl_obs,
            "history": self.history_buffer.copy(),
        }

    def _zero_obs(self):
        """Return a zeroed observation."""
        return {
            "scalars": np.zeros(self.num_scalars, dtype=np.float32),
            "history": self.history_buffer.copy(),
        }

    def reset(self, seed=None, options=None):
        self.history_buffer.fill(0.0)
        self.prev_delay = 0.0
        self._prev_action = 0

        marl_obs, marl_info = self.marl_env.reset(seed=seed)
        obs = self._bridge_obs(marl_obs)
        return obs, {}

    def step(self, action):
        """Broadcast single action to all MARL agents, step, compress obs."""
        self._prev_action = int(action)
        marl_actions = {a: int(action) for a in self.marl_env.agents}

        marl_obs, rewards, terminations, truncations, infos = self.marl_env.step(marl_actions)

        # Cooperative reward: all agents share the same reward
        reward = float(next(iter(rewards.values()), 0.0)) if rewards else 0.0

        # Terminal check
        any_agent = next(iter(terminations.keys()), None) if terminations else None
        terminated = bool(terminations.get(any_agent, True)) if any_agent else True
        truncated = bool(truncations.get(any_agent, False)) if any_agent else False

        obs = self._bridge_obs(marl_obs, infos)
        info = {"raw_infos": infos}

        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass
