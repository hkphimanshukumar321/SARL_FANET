from __future__ import annotations

import math
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.cluster_config import ClusterConfig as CC


class BranchingReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, act, rew, nobs, done = zip(*batch)
        return list(obs), np.array(act), np.array(rew, dtype=np.float32), list(nobs), np.array(done, dtype=np.float32)

    def __len__(self):
        return len(self.buffer)


class BranchingEncoder(nn.Module):
    def __init__(self, scalar_dim: int, history_len: int, features_dim: int = 256):
        super().__init__()
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.temporal_encoder = nn.GRU(
            input_size=scalar_dim,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
        )
        self.fusion = nn.Sequential(
            nn.Linear(256, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )

    def forward(self, scalars: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        scalar_features = self.scalar_encoder(scalars)
        _, h_n = self.temporal_encoder(history)
        temporal_features = h_n[-1]
        return self.fusion(torch.cat([scalar_features, temporal_features], dim=1))


class BranchingQNetwork(nn.Module):
    def __init__(self, scalar_dim: int, history_len: int, n_branches: int, action_dim: int, features_dim: int = 256):
        super().__init__()
        self.n_branches = n_branches
        self.action_dim = action_dim
        self.encoder = BranchingEncoder(scalar_dim, history_len, features_dim)
        self.value_stream = nn.Sequential(
            nn.Linear(features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.advantage_streams = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(features_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, action_dim),
                )
                for _ in range(n_branches)
            ]
        )

    def forward(self, scalars: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        features = self.encoder(scalars, history)
        value = self.value_stream(features).unsqueeze(1)
        advantages = torch.stack([head(features) for head in self.advantage_streams], dim=1)
        return value + (advantages - advantages.mean(dim=2, keepdim=True))


class MCABranchingD3QNAgent:
    """
    Centralized branching D3QN baseline for factorized MultiDiscrete joint actions.
    """

    def __init__(
        self,
        env,
        learning_rate=1e-3,
        buffer_size=100000,
        batch_size=64,
        gamma=0.99,
        exploration_fraction=0.1,
        target_update_interval=1000,
        seed=42,
        device="cpu",
    ):
        self.env = env
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_interval = target_update_interval
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        self.steps_done = 0
        self.exploration_fraction = exploration_fraction
        self.total_training_steps = 1
        self.eps_start = 1.0
        self.eps_end = 0.05

        scalar_dim = int(env.observation_space["scalars"].shape[0])
        history_len = int(env.observation_space["history"].shape[0])
        self.n_branches = int(env.action_space.nvec.shape[0])
        self.action_dim = int(env.action_space.nvec[0])

        self.q_net = BranchingQNetwork(scalar_dim, history_len, self.n_branches, self.action_dim).to(self.device)
        self.target_net = BranchingQNetwork(scalar_dim, history_len, self.n_branches, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.replay = BranchingReplayBuffer(buffer_size)

    def _epsilon(self):
        progress = min(self.steps_done / max(self.total_training_steps, 1), 1.0)
        return self.eps_end + (self.eps_start - self.eps_end) * max(0.0, 1.0 - progress / max(self.exploration_fraction, 1e-6))

    def _obs_to_tensors(self, obs_batch):
        if isinstance(obs_batch, dict):
            obs_batch = [obs_batch]
        scalars = np.stack([obs["scalars"] for obs in obs_batch]).astype(np.float32)
        history = np.stack([obs["history"] for obs in obs_batch]).astype(np.float32)
        scalars_t = torch.tensor(scalars, dtype=torch.float32, device=self.device)
        history_t = torch.tensor(history, dtype=torch.float32, device=self.device)
        return scalars_t, history_t

    def predict(self, obs, deterministic=True):
        epsilon = 0.0 if deterministic else self._epsilon()
        if self.rng.random() < epsilon:
            action = self.env.action_space.sample()
            return np.asarray(action, dtype=np.int64), None

        with torch.no_grad():
            scalars_t, history_t = self._obs_to_tensors(obs)
            q_values = self.q_net(scalars_t, history_t)[0]
            action = q_values.argmax(dim=1).cpu().numpy().astype(np.int64)
        return action, None

    def _optimize(self):
        if len(self.replay) < self.batch_size:
            return 0.0

        obs_b, act_b, rew_b, nobs_b, done_b = self.replay.sample(self.batch_size)
        scalars_t, history_t = self._obs_to_tensors(obs_b)
        nscalars_t, nhistory_t = self._obs_to_tensors(nobs_b)
        act_t = torch.tensor(act_b, dtype=torch.long, device=self.device)
        rew_t = torch.tensor(rew_b, dtype=torch.float32, device=self.device).unsqueeze(1)
        done_t = torch.tensor(done_b, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.q_net(scalars_t, history_t)
        q_selected = q_values.gather(2, act_t.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_policy_q = self.q_net(nscalars_t, nhistory_t)
            next_actions = next_policy_q.argmax(dim=2, keepdim=True)
            next_target_q = self.target_net(nscalars_t, nhistory_t).gather(2, next_actions).squeeze(-1)
            target = rew_t + self.gamma * next_target_q * (1.0 - done_t)

        loss = F.mse_loss(q_selected, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def learn(self, total_timesteps=10000, callback=None, progress_bar=False):
        self.total_training_steps = max(int(total_timesteps), 1)
        obs, _ = self.env.reset()
        episode_reward = 0.0

        for step in range(total_timesteps):
            self.steps_done += 1
            action, _ = self.predict(obs, deterministic=False)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = bool(terminated or truncated)
            self.replay.push(obs, action, reward, next_obs, done)
            loss = self._optimize()

            if self.steps_done % self.target_update_interval == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

            episode_reward += reward
            if callback is not None:
                callback.locals = {
                    "reward": reward,
                    "done": done,
                    "loss": loss,
                    "info": info,
                    "episode_reward": episode_reward,
                }
                if hasattr(callback, "_on_step"):
                    callback._on_step()

            if done:
                obs, _ = self.env.reset()
                episode_reward = 0.0
            else:
                obs = next_obs

        if callback is not None and hasattr(callback, "_on_training_end"):
            callback._on_training_end()
        return self

    def save(self, path):
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "steps_done": self.steps_done,
            },
            path,
        )

    @classmethod
    def load(cls, path, env, device="cpu"):
        agent = cls(env=env, device=device)
        ckpt = torch.load(path, map_location=device, weights_only=True)
        agent.q_net.load_state_dict(ckpt["q_net"])
        agent.target_net.load_state_dict(ckpt["target_net"])
        agent.optimizer.load_state_dict(ckpt["optimizer"])
        agent.steps_done = int(ckpt.get("steps_done", 0))
        return agent


def create_mca_d3qn(
    env,
    learning_rate=1e-3,
    buffer_size=100000,
    batch_size=64,
    gamma=0.99,
    exploration_fraction=0.1,
    target_update_interval=1000,
    seed=42,
    device="cpu",
):
    return MCABranchingD3QNAgent(
        env=env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        exploration_fraction=exploration_fraction,
        target_update_interval=target_update_interval,
        seed=seed,
        device=device,
    )
