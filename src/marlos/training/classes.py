"""
    Classes required for training
"""

import torch
from gymnasium import Space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    """Minigrid feature extractor for use in training"""

    def __init__(
        self,
        observation_space: Space,
        features_dim: int = 512,
        normalized_image: bool = False,
    ) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            n_flatten = (
                self.cnn(torch.as_tensor(observation_space.sample()[None]))
                .float()
                .shape[1]
            )

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
