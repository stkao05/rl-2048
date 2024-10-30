from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy

from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)

import collections
import copy
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn




class MaskedMLPPolicy(ActorCriticPolicy):

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
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

    def evaluate_actions(self, obs, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
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


    def _get_action_dist_from_latent(self, latent_pi: th.Tensor, action_mask: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)
        mean_actions += action_mask

        return self.action_dist.proba_distribution(action_logits=mean_actions)


def get_action_mask(obs):
    boards = observation_to_grid(obs)
    batch_n = obs.shape[0]
    action_mask = th.zeros(batch_n, 4)

    for i in range(batch_n):
        action_mask[i] = one_action_mask(boards[i])

    return action_mask


def one_action_mask(board):
    up = is_valid_move_up(board)
    right = is_valid_move_right(board)
    down = is_valid_move_down(board)
    left = is_valid_move_left(board)
    # negative num for invalid action 
    action_mask = (1 - th.tensor([up, right, down, left], dtype=th.int)) * -1e8
    return action_mask


# observation: (B, 16, 4, 4)
def observation_to_grid(observation):
    powers_of_2 = 2 ** th.arange(1, 17, dtype=th.int32).view(16, 1, 1)
    grid = th.sum(observation * powers_of_2, dim=1) # (B, 4, 4)
    return grid


def is_valid_move_left(board):
    for row in board:
        for i in range(1, 4):
            if row[i] != 0 and (row[i - 1] == 0 or row[i - 1] == row[i]):
                return True
    return False

def is_valid_move_right(board):
    for row in board:
        for i in range(2, -1, -1):
            if row[i] != 0 and (row[i + 1] == 0 or row[i + 1] == row[i]):
                return True
    return False

def is_valid_move_up(board):
    for col in range(4):
        for row in range(1, 4):
            if board[row][col] != 0 and (board[row - 1][col] == 0 or board[row - 1][col] == board[row][col]):
                return True
    return False

def is_valid_move_down(board):
    for col in range(4):
        for row in range(2, -1, -1):
            if board[row][col] != 0 and (board[row + 1][col] == 0 or board[row + 1][col] == board[row][col]):
                return True
    return False
