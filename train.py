import warnings
import gymnasium as gym
from gymnasium.envs.registration import register

import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3 import A2C, DQN, PPO, SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np
from tqdm import tqdm
import os
import time
from eval import evaluation

warnings.filterwarnings("ignore")
register(id="2048-v0", entry_point="envs:My2048Env")
register(id="2048-src", entry_point="envs:Src2048Env")
register(id="2048-eval", entry_point="envs:Eval2048Env")


def make_train_env():
    env = gym.make("2048-v0")
    return env

def make_src_env():
    env = gym.make("2048-src")
    return env

def make_eval_env():
    env = gym.make("2048-eval")
    return env


def eval(env, model, eval_episode_num):
    """Evaluate the model and return avg_score and avg_highest"""
    score = []
    highest = []
    step_count = []
    step_count_illegal = [] # step count due to illegal termination
    step_count_done = [] # step count due to done
    illegal_count = []

    for seed in range(eval_episode_num):
        done = False
        env.seed(seed)  # set seed using old Gym API
        obs = env.reset()
        count = 0

        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            count += 1

        score.append(info[0]["score"])
        highest.append(info[0]["highest"])
        step_count.append(count)
        illegal_count.append(1 if info[0]["illegal_move"] else 0)

        if info[0]["illegal_move"]:
            step_count_illegal.append(count)
        else:
            step_count_done.append(count)

    stats = {
        "score_mean": np.mean(score),
        "score_median": np.median(score),
        "score_max": np.max(score),
        "score_std": np.std(score),
        "score_hist": wandb.Histogram(score),
        "highest_mean": np.mean(highest),
        "highest_median": np.median(highest),
        "highest_max": np.max(highest),
        "highest_std": np.std(highest),
        "highest_hist": wandb.Histogram(highest),
        "step_count_mean": np.mean(step_count),
        "step_count_median": np.median(step_count),
        "step_count_max": np.max(step_count),
        "step_count_std": np.std(step_count),
        "step_count_hist": wandb.Histogram(step_count),
        "step_count_illegal_hist": wandb.Histogram(step_count_illegal),
        "step_count_done_hist": wandb.Histogram(step_count_done),
        "illegal_move": np.sum(illegal_count) / eval_episode_num,
    }

    return stats


def train(eval_env, model, config):
    """Train agent using SB3 algorithm and my_config"""
    current_best = 0
    with tqdm(range(config["epoch_num"]), unit="epoch") as pbar:
        for epoch in pbar:
            model.learn(
                total_timesteps=config["timesteps_per_epoch"],
                reset_num_timesteps=False,
                # callback=WandbCallback(
                #     gradient_save_freq=config["timesteps_per_epoch"],
                #     verbose=2,
                # ),
            )

            stats = eval(eval_env, model, config["eval_episode_num"])
            is_better = current_best < stats["score_mean"]

            wandb.log(stats)
            output = f"epoch: {epoch:<3} | "

            keys = ["score_mean", "highest_max", "step_count_mean", "illegal_move"]
            for key, value in stats.items():
                if key not in keys:
                    continue
                if isinstance(value, wandb.Histogram):
                    continue
                output += f"{key}: {value:4.2f} | "
            pbar.write(output + f" | is_better: {str(is_better)}")

            if is_better and config["save_model"]:
                current_best = stats["score_mean"]
                model_save_path = config["model_save_path"]
                model.save(f"{model_save_path}")


def experiment(config):
    wandb.init(
        project="rl-2048",
        name=config["name"],
        config=config,
        # sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    )

    train_env = make_vec_env(
        make_train_env, n_envs=config["n_envs"], vec_env_cls=SubprocVecEnv
    )
    eval_env = DummyVecEnv([make_eval_env])
    model = config["algorithm"](
        config["policy_network"],
        train_env,
        verbose=0,
        learning_rate=config["learning_rate"],
        policy_kwargs=config["policy_kwargs"] if "policy_kwargs" in config else None,
    )
    train(eval_env, model, config)

    # save model and also the training env
    wandb.save("train.py")
    wandb.save("envs/my2048_env.py")
    if config["save_model"]:
        wandb.save(config["model_save_path"] + ".zip")

    wandb.finish()


if __name__ == "__main__":
    base_config = {
        "algorithm": PPO,
        "policy_network": "MlpPolicy",
        "epoch_num": 200,
        "eval_episode_num": 100,
        "timesteps_per_epoch": 1000,
        "learning_rate": 1e-4,
        "n_envs": 8,
    }

    config = {
        "name": "ppo-reward-scaling-neg-retry",
        "save_model": True,
        "notes": "",
    }
    config.update(base_config)
    config["model_save_path"] = os.path.join("models", config["name"])
    experiment(config)
