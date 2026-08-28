import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class DQN(nn.Module):
    """
    Deep Q-Network Model Architecture.
    Maps input state vectors (telemetry observation) to Q-values for each discrete routing action.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = None):
        super(DQN, self).__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128]

        layers = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class DuelingDQN(nn.Module):
    """
    Dueling Deep Q-Network Architecture.
    Separates Value V(s) stream and Advantage A(s, a) stream for superior policy learning.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = None):
        super(DuelingDQN, self).__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128]

        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], action_dim)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        return values + (advantages - advantages.mean(dim=-1, keepdim=True))
