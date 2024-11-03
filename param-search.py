import optuna
from stable_baselines3 import PPO
import warnings
import gymnasium as gym
from gymnasium.envs.registration import register

import wandb

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np
from tqdm import tqdm
import os
from policy import MaskedMLPPolicy, GridCnn
from train import make_train_env, make_eval_env, eval



def objective(trial):
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    gamma = trial.suggest_uniform('gamma', 0.9, 0.9999)
    clip_range = trial.suggest_uniform('clip_range', 0.1, 0.4)
    n_steps = trial.suggest_int('n_steps', 128, 2048, step=128)
    # batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])

    train_env = make_vec_env(make_train_env, n_envs=5, vec_env_cls=SubprocVecEnv)
    eval_env = DummyVecEnv([make_eval_env])
    model = PPO(
        env=train_env,
        policy=MaskedMLPPolicy,
        verbose=0,
        learning_rate=learning_rate,
        gamma=gamma,
        clip_range=clip_range,
        n_steps=n_steps,
        # batch_size=batch_size,
        policy_kwargs={
            "features_extractor_class": GridCnn,
            "net_arch": []
        }
    )

    model.learn(total_timesteps=40000)
    result = eval(eval_env, model, eval_episode_num=50)
    return result["score_mean"]


if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100, n_jobs=1)

    # Print the best hyperparameters
    print("best hyperparameters:", study.best_params)