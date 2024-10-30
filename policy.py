from typing import Optional, Tuple

import gymnasium as gym
import torch as th
from gymnasium import spaces
from stable_baselines3.common.distributions import (
    Distribution,
)
from stable_baselines3.common.policies import ActorCriticPolicy, BaseFeaturesExtractor
from torch import nn


class MaskedMLPPolicy(ActorCriticPolicy):
    def forward(
        self, obs: th.Tensor, deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        action_mask = get_action_mask(obs)

        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi, action_mask)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]

        return actions, values, log_prob

    def evaluate_actions(
        self, obs, actions: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        action_mask = get_action_mask(obs)
        distribution = self._get_action_dist_from_latent(latent_pi, action_mask)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(self, obs) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        features = super().extract_features(obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features)
        action_mask = get_action_mask(obs)

        return self._get_action_dist_from_latent(latent_pi, action_mask)

    def _get_action_dist_from_latent(
        self, latent_pi: th.Tensor, action_mask: th.Tensor
    ) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)
        mean_actions += action_mask

        return self.action_dist.proba_distribution(action_logits=mean_actions)


def get_action_mask(obs):
    boards = observation_to_grid(obs)  # boards: (batch_n, 4, 4)

    up = is_valid_move_up(boards)  # (batch_n,)
    right = is_valid_move_right(boards)
    down = is_valid_move_down(boards)
    left = is_valid_move_left(boards)

    valid_moves = th.stack([up, right, down, left], dim=1)  # (batch_n, 4)
    action_mask = (1 - valid_moves.int()) * -1e8
    return action_mask


def is_valid_move_left(boards):
    # boards: (batch_n, 4, 4)
    curr_cells = boards[:, :, 1:]  # (batch_n, 4, 3)
    left_cells = boards[:, :, :-1]  # (batch_n, 4, 3)

    not_zero = curr_cells != 0
    left_zero_or_same = (left_cells == 0) | (left_cells == curr_cells)
    valid_moves = not_zero & left_zero_or_same  # (batch_n, 4, 3)

    return valid_moves.any(dim=(1, 2))  # (batch_n,)


def is_valid_move_right(boards):
    curr_cells = boards[:, :, :-1]  # (batch_n, 4, 3)
    right_cells = boards[:, :, 1:]  # (batch_n, 4, 3)

    not_zero = curr_cells != 0
    right_zero_or_same = (right_cells == 0) | (right_cells == curr_cells)
    valid_moves = not_zero & right_zero_or_same  # (batch_n, 4, 3)

    return valid_moves.any(dim=(1, 2))  # (batch_n,)


def is_valid_move_up(boards):
    curr_cells = boards[:, 1:, :]  # (batch_n, 3, 4)
    upper_cells = boards[:, :-1, :]  # (batch_n, 3, 4)

    not_zero = curr_cells != 0
    upper_zero_or_same = (upper_cells == 0) | (upper_cells == curr_cells)
    valid_moves = not_zero & upper_zero_or_same  # (batch_n, 3, 4)

    return valid_moves.any(dim=(1, 2))  # (batch_n,)


def is_valid_move_down(boards):
    curr_cells = boards[:, :-1, :]  # (batch_n, 3, 4)
    lower_cells = boards[:, 1:, :]  # (batch_n, 3, 4)

    not_zero = curr_cells != 0
    lower_zero_or_same = (lower_cells == 0) | (lower_cells == curr_cells)
    valid_moves = not_zero & lower_zero_or_same  # (batch_n, 3, 4)

    return valid_moves.any(dim=(1, 2))  # (batch_n,)


def observation_to_grid(observation):
    powers_of_2 = 2 ** th.arange(1, 17, dtype=th.int32).view(16, 1, 1)
    grid = th.sum(observation * powers_of_2, dim=1)  # (B, 4, 4)
    return grid


class GridCnn(BaseFeaturesExtractor):
    """
    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 64,
        normalized_image: bool = False,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "NatureCNN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)

        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        observations = observations * 0.5
        x = self.cnn(observations)
        return self.linear(x)
