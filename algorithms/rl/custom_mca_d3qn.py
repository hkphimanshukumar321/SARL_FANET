import torch
import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MultiInputPolicy
from stable_baselines3.common.policies import BaseModel
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from algorithms.rl.features_extractor import MCAFeaturesExtractor


class DuelingQNetwork(BaseModel):
    """
    Proper Dueling Q-Network head that inherits from BaseModel to ensure 
    compatibility with SB3's internal feature extraction and target network updates.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        features_extractor: nn.Module,
        features_dim: int,
        net_arch=None,
        activation_fn=nn.ReLU,
        normalize_images: bool = True,
        hidden_dim: int = 128,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )
        self.features_dim = features_dim
        n_actions = int(action_space.n)

        # Value stream V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(features_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Advantage stream A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(features_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Predict the q-values by extracting features first.
        """
        features = self.extract_features(obs, self.features_extractor)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        # Mean-subtracted recombination
        return values + (advantages - advantages.mean(dim=1, keepdim=True))

    def _predict(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        q_values = self(observation)
        return q_values.argmax(dim=1).reshape(-1)


class DuelingMultiInputPolicy(MultiInputPolicy):
    """
    Custom policy that uses our DuelingQNetwork.
    """

    def make_q_net(self) -> DuelingQNetwork:
        # Replicate SB3's logic to create/clone the features extractor
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        # Add our custom hidden_dim for the Dueling streams
        net_args["hidden_dim"] = 128
        return DuelingQNetwork(**net_args).to(self.device)


def create_mca_d3qn(env, learning_rate=1e-3, buffer_size=100000,
                    batch_size=64, gamma=0.99, exploration_fraction=0.1,
                    target_update_interval=1000, seed=42, **kwargs):
    """
    Creates the Mobility-Channel-Aware Dueling Double DQN (MCA-D3QN)
    using Stable-Baselines3 DQN with a proper Dueling Q-network head.

    SB3's DQN natively provides:
        - Double Q-learning        (on by default)
        - Replay buffer
        - Epsilon-greedy schedule

    DuelingMultiInputPolicy adds:
        - True V(s) / A(s,a) split via DuelingQNetwork
          (previously approximated by a flat MLP — now fully realised)

    MCAFeaturesExtractor provides:
        - Multi-branch feature extraction (mobility + channel awareness)
        - 256-dim joint embedding fed into both Dueling streams
    """

    policy_kwargs = dict(
        features_extractor_class=MCAFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        # net_arch is intentionally omitted — DuelingQNetwork owns
        # its own hidden layers (128-dim streams), keeping parity
        # with the original net_arch=[128, 128].
    )

    model = DQN(
        DuelingMultiInputPolicy,
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=100,
        batch_size=batch_size,
        tau=1.0,
        gamma=gamma,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        max_grad_norm=10,
        policy_kwargs=policy_kwargs,
        device="auto",
        seed=seed,
        verbose=1,
        tensorboard_log="./results/tensorboard_logs/",
    )

    return model
