import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

class MCAFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom BaseFeaturesExtractor for the Dict observation space.
    - Branch A: Processes the 1D scalar vector (num_scalars) through MLP.
    - Branch B: Processes the 2D history sequence (history_len, num_scalars) through Conv1D.
    - Fuses embeddings and outputs a fixed size feature vector for the RL head.
    """
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        # We assume features_dim is the combined size of the fusion layer output.
        super(MCAFeaturesExtractor, self).__init__(observation_space, features_dim)

        scalar_dim = observation_space.spaces["scalars"].shape[0]
        history_shape = observation_space.spaces["history"].shape # (T, num_scalars)
        
        # Branch A: Scalars Encoder (MLP)
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        # Output dim = 64

        # Branch B: Temporal Sequence Encoder (LSTM for DRQN Architecture)
        # Input history shape: (batch_size, T_history_len, num_scalars)
        input_size = history_shape[1]
        
        self.temporal_encoder = nn.LSTM(
            input_size=input_size, 
            hidden_size=64, 
            num_layers=1, 
            batch_first=True
        )
        
        self.temporal_proj = nn.Sequential(
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        # Output dim = 128
        
        # Fusion dimensionality check
        combined_dim = 64 + 128
        
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU()
        )

    def forward(self, observations) -> torch.Tensor:
        scalars = observations["scalars"]
        history = observations["history"]

        # Branch A
        scalar_features = self.scalar_encoder(scalars)
        
        # Branch B (LSTM Timeline)
        # history input: (batch, T, input_size)
        lstm_out, (h_n, c_n) = self.temporal_encoder(history)
        
        # We only need the final hidden state context from the LSTM tuple
        # h_n shape: (num_layers, batch, hidden_size). 
        # For num_layers=1, it's (1, batch, 64). Squeeze to (batch, 64).
        final_h = h_n[-1] 
        temporal_features = self.temporal_proj(final_h)
        
        # Fusion
        combined = torch.cat([scalar_features, temporal_features], dim=1)
        return self.fusion(combined)
