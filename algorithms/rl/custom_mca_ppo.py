"""
custom_mca_ppo.py -- Branching PPO agent for MultiDiscrete joint actions.

Uses the same BranchingEncoder architecture as MCA-D3QN but with an
actor-critic PPO training loop. Each branch produces a policy distribution
over its action dimension, enabling factorized MultiDiscrete control.

Architecture:
    Shared Encoder (GRU + MLP) → BranchingActor (per-branch softmax heads)
                                → Critic (single scalar V(s))

Training: Clipped PPO with GAE advantage estimation and entropy bonus.
"""

from __future__ import annotations

import math
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from configs.cluster_config import ClusterConfig as CC


class BranchingEncoder(nn.Module):
    """Shared encoder — same architecture as MCA-D3QN."""

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


class BranchingActorCritic(nn.Module):
    """Actor-Critic with branching action heads for MultiDiscrete actions."""

    def __init__(
        self,
        scalar_dim: int,
        history_len: int,
        n_branches: int,
        action_dim: int,
        features_dim: int = 256,
    ):
        super().__init__()
        self.n_branches = n_branches
        self.action_dim = action_dim
        self.encoder = BranchingEncoder(scalar_dim, history_len, features_dim)

        # Actor: one head per branch producing logits over action_dim
        self.actor_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(features_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, action_dim),
                )
                for _ in range(n_branches)
            ]
        )

        # Critic: single scalar value
        self.critic = nn.Sequential(
            nn.Linear(features_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, scalars: torch.Tensor, history: torch.Tensor):
        features = self.encoder(scalars, history)
        logits = [head(features) for head in self.actor_heads]
        value = self.critic(features)
        return logits, value

    def get_action_and_value(
        self,
        scalars: torch.Tensor,
        history: torch.Tensor,
        action: torch.Tensor | None = None,
    ):
        """
        Returns:
            action: (batch, n_branches) — sampled or given
            log_prob: (batch,) — sum of log probs across branches
            entropy: (batch,) — sum of entropies across branches
            value: (batch, 1) — state value
        """
        logits_list, value = self.forward(scalars, history)

        all_log_probs = []
        all_entropies = []
        all_actions = []

        for b, logits in enumerate(logits_list):
            dist = Categorical(logits=logits)
            if action is None:
                a = dist.sample()
            else:
                a = action[:, b]
            all_actions.append(a)
            all_log_probs.append(dist.log_prob(a))
            all_entropies.append(dist.entropy())

        actions = torch.stack(all_actions, dim=1)
        log_prob = torch.stack(all_log_probs, dim=1).sum(dim=1)
        entropy = torch.stack(all_entropies, dim=1).sum(dim=1)

        return actions, log_prob, entropy, value


class RolloutBuffer:
    """Simple rollout buffer for PPO on-policy data collection."""

    def __init__(self):
        self.obs_list = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(self, obs, action, log_prob, reward, done, value):
        self.obs_list.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def clear(self):
        self.obs_list.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def __len__(self):
        return len(self.rewards)


class MCABranchingPPOAgent:
    """
    Centralized branching PPO for factorized MultiDiscrete joint actions.
    Mirror of MCA-D3QN but using PPO actor-critic training.
    """

    def __init__(
        self,
        env,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_steps: int = 128,
        n_epochs: int = 4,
        batch_size: int = 64,
        seed: int = 42,
        device: str = "cpu",
    ):
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        self.steps_done = 0

        scalar_dim = int(env.observation_space["scalars"].shape[0])
        history_len = int(env.observation_space["history"].shape[0])
        self.n_branches = int(env.action_space.nvec.shape[0])
        self.action_dim = int(env.action_space.nvec[0])

        self.policy = BranchingActorCritic(
            scalar_dim, history_len, self.n_branches, self.action_dim
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.buffer = RolloutBuffer()

    def _obs_to_tensors(self, obs):
        if isinstance(obs, dict):
            obs = [obs]
        scalars = np.stack([o["scalars"] for o in obs]).astype(np.float32)
        history = np.stack([o["history"] for o in obs]).astype(np.float32)
        return (
            torch.tensor(scalars, dtype=torch.float32, device=self.device),
            torch.tensor(history, dtype=torch.float32, device=self.device),
        )

    def predict(self, obs, deterministic: bool = True):
        with torch.no_grad():
            scalars_t, history_t = self._obs_to_tensors(obs)
            if deterministic:
                logits_list, _ = self.policy(scalars_t, history_t)
                action = torch.stack(
                    [logits.argmax(dim=-1) for logits in logits_list], dim=1
                )
            else:
                action, _, _, _ = self.policy.get_action_and_value(
                    scalars_t, history_t
                )
        return action[0].cpu().numpy().astype(np.int64), None

    def _compute_gae(self, next_value: float):
        """Compute generalized advantage estimation."""
        rewards = self.buffer.rewards
        dones = self.buffer.dones
        values = self.buffer.values

        advantages = []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_v = next_value
            else:
                next_v = values[t + 1]
            delta = rewards[t] + self.gamma * next_v * (1.0 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            advantages.insert(0, gae)
        return advantages

    def _update(self, next_obs):
        """PPO update step."""
        with torch.no_grad():
            s_t, h_t = self._obs_to_tensors(next_obs)
            _, next_value = self.policy(s_t, h_t)
            next_value = float(next_value.item())

        advantages = self._compute_gae(next_value)
        values = self.buffer.values
        returns = [adv + val for adv, val in zip(advantages, values)]

        # Build tensors
        all_scalars = torch.stack(
            [
                torch.tensor(o["scalars"], dtype=torch.float32, device=self.device)
                for o in self.buffer.obs_list
            ]
        )
        all_history = torch.stack(
            [
                torch.tensor(o["history"], dtype=torch.float32, device=self.device)
                for o in self.buffer.obs_list
            ]
        )
        all_actions = torch.tensor(
            np.array(self.buffer.actions), dtype=torch.long, device=self.device
        )
        all_old_log_probs = torch.tensor(
            self.buffer.log_probs, dtype=torch.float32, device=self.device
        )
        all_advantages = torch.tensor(
            advantages, dtype=torch.float32, device=self.device
        )
        all_returns = torch.tensor(
            returns, dtype=torch.float32, device=self.device
        )

        # Normalize advantages
        if len(all_advantages) > 1:
            all_advantages = (all_advantages - all_advantages.mean()) / (
                all_advantages.std() + 1e-8
            )

        dataset_size = len(self.buffer)
        total_loss_sum = 0.0

        for _ in range(self.n_epochs):
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)

            for start in range(0, dataset_size, self.batch_size):
                end = min(start + self.batch_size, dataset_size)
                idx = indices[start:end]

                batch_scalars = all_scalars[idx]
                batch_history = all_history[idx]
                batch_actions = all_actions[idx]
                batch_old_log_probs = all_old_log_probs[idx]
                batch_advantages = all_advantages[idx]
                batch_returns = all_returns[idx]

                _, new_log_probs, entropy, new_values = (
                    self.policy.get_action_and_value(
                        batch_scalars, batch_history, batch_actions
                    )
                )

                # Policy loss (clipped PPO)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(
                        ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
                    )
                    * batch_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(new_values.squeeze(-1), batch_returns)

                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy.mean()
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                total_loss_sum += float(loss.item())

        self.buffer.clear()
        return total_loss_sum

    def learn(self, total_timesteps: int = 10000, callback=None, progress_bar=False):
        obs, _ = self.env.reset()
        episode_reward = 0.0

        for step in range(total_timesteps):
            self.steps_done += 1

            with torch.no_grad():
                s_t, h_t = self._obs_to_tensors(obs)
                action, log_prob, _, value = self.policy.get_action_and_value(
                    s_t, h_t
                )
                action_np = action[0].cpu().numpy().astype(np.int64)
                log_prob_val = float(log_prob.item())
                value_val = float(value.item())

            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            done = bool(terminated or truncated)

            self.buffer.add(obs, action_np, log_prob_val, reward, float(done), value_val)
            episode_reward += reward

            if callback is not None:
                callback.locals = {
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "episode_reward": episode_reward,
                }
                if hasattr(callback, "_on_step"):
                    callback._on_step()

            if len(self.buffer) >= self.n_steps:
                self._update(next_obs)

            if done:
                obs, _ = self.env.reset()
                episode_reward = 0.0
            else:
                obs = next_obs

        # Final update with remaining buffer
        if len(self.buffer) > 0:
            self._update(obs)

        if callback is not None and hasattr(callback, "_on_training_end"):
            callback._on_training_end()
        return self

    def save(self, path):
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "steps_done": self.steps_done,
            },
            path,
        )

    @classmethod
    def load(cls, path, env, device="cpu"):
        agent = cls(env=env, device=device)
        ckpt = torch.load(path, map_location=device, weights_only=True)
        agent.policy.load_state_dict(ckpt["policy"])
        agent.optimizer.load_state_dict(ckpt["optimizer"])
        agent.steps_done = int(ckpt.get("steps_done", 0))
        return agent


def create_mca_ppo(
    env,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_epsilon: float = 0.2,
    entropy_coef: float = 0.01,
    n_steps: int = 128,
    n_epochs: int = 4,
    batch_size: int = 64,
    seed: int = 42,
    device: str = "cpu",
):
    """Factory function for MCA-PPO (mirrors create_mca_d3qn interface)."""
    return MCABranchingPPOAgent(
        env=env,
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_epsilon=clip_epsilon,
        entropy_coef=entropy_coef,
        n_steps=n_steps,
        n_epochs=n_epochs,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )
