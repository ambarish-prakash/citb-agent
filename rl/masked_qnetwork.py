from stable_baselines3.dqn.policies import QNetwork

from typing import List, Optional, Type

import torch
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from utilities import get_invalid_actions

class MaskedQNetwork(QNetwork):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            features_dim=features_dim,
            net_arch=net_arch,
            activation_fn=activation_fn,
            normalize_images=normalize_images,
        )

    def forward(self, obs: PyTorchObs) -> torch.Tensor:
        """
        Predict the q-values.

        :param obs: Observation
        :return: The estimated Q-Value for each action.
        """
        qvals = self.q_net(self.extract_features(obs, self.features_extractor))
        qvals = self.negate_invalid_action_values(qvals, obs)
        return qvals
    
    def negate_invalid_action_values(self, q_values, observations):
        NEGATIVE_VAL = -100
        invalid_actions = get_invalid_actions(observations)
        q_values = torch.where(invalid_actions, NEGATIVE_VAL, q_values)

        return q_values
