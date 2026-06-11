"""
sarl_central_env.py -- Centralized cluster-level baseline environment.

The agent sees the concatenated cluster observations and issues a joint
MultiDiscrete action vector over the padded C_MAX cluster-head slots.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from algorithms.rl.rewards import compute_team_reward
from configs.cluster_config import ClusterConfig as CC
from configs.sarl_config import RLConfig
from envs.marl_mac_env import MARLMacEnv


class SARLCentralEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, seed=None, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.marl_env = MARLMacEnv(seed=seed)

        self.num_scalars = CC.C_MAX * CC.OBS_DIM_CLUSTER
        self.history_len = RLConfig.HISTORY_WINDOW_STEPS

        self.observation_space = spaces.Dict(
            {
                "scalars": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.num_scalars,),
                    dtype=np.float32,
                ),
                "history": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.history_len, self.num_scalars),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = spaces.MultiDiscrete(np.full(CC.C_MAX, CC.NUM_ACTIONS, dtype=np.int64))

        self.history_buffer = np.zeros((self.history_len, self.num_scalars), dtype=np.float32)

    def _decode_action(self, joint_action) -> dict:
        if np.isscalar(joint_action):
            action_vec = np.full(CC.C_MAX, int(joint_action), dtype=np.int64)
        else:
            action_vec = np.asarray(joint_action, dtype=np.int64).reshape(-1)
            if action_vec.size != CC.C_MAX:
                raise ValueError(f"Expected {CC.C_MAX} central actions, got {action_vec.size}")
        return {f"cluster_{k}": int(action_vec[k]) for k in range(CC.C_MAX)}

    def _bridge_obs(self, marl_obs_dict):
        if not marl_obs_dict:
            return self._zero_obs()

        ordered_obs = []
        for k in range(CC.C_MAX):
            name = f"cluster_{k}"
            ordered_obs.append(
                marl_obs_dict.get(name, np.zeros(CC.OBS_DIM_CLUSTER, dtype=np.float32))
            )

        scalars = np.concatenate(ordered_obs).astype(np.float32)
        self.history_buffer = np.roll(self.history_buffer, shift=-1, axis=0)
        self.history_buffer[-1] = scalars
        return {"scalars": scalars, "history": self.history_buffer.copy()}

    def _zero_obs(self):
        return {
            "scalars": np.zeros(self.num_scalars, dtype=np.float32),
            "history": self.history_buffer.copy(),
        }

    def reset(self, seed=None, options=None):
        self.history_buffer.fill(0.0)
        marl_obs, _ = self.marl_env.reset(seed=seed, options=options)
        return self._bridge_obs(marl_obs), {}

    def step(self, action):
        marl_actions = self._decode_action(action)
        marl_obs, rewards, terminations, truncations, infos = self.marl_env.step(marl_actions)

        active_cids = [k for k in range(CC.C_MAX) if infos.get(f"cluster_{k}", {}).get("alive", False)]
        active_r = [rewards[f"cluster_{k}"] for k in active_cids]
        active_th = [infos[f"cluster_{k}"].get("throughput_mbps", 0.0) for k in active_cids]
        team_reward = compute_team_reward(active_r, active_th)

        terminated = any(terminations.values()) if terminations else True
        truncated = any(truncations.values()) if truncations else False
        obs = self._bridge_obs(marl_obs)
        info = {
            "raw_infos": infos,
            "global_th": infos.get("cluster_0", {}).get("global_th", 0.0),
            "global_drops": infos.get("cluster_0", {}).get("global_drops", 0.0),
            "global_collisions": infos.get("cluster_0", {}).get("global_collisions", 0.0),
            "global_coord_success": infos.get("cluster_0", {}).get("global_coord_success", 0.0),
            "active_clusters": len(active_cids),
        }
        return obs, float(team_reward), bool(terminated), bool(truncated), info

    def render(self):
        return None

    def close(self):
        self.marl_env.close()
