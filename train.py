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
register(id="2048-eval", entry_point="envs:Eval2048Env")


def make_env():
    env = gym.make("2048-v0")
    return env


def make_eval_env():
    env = gym.make("2048-eval")
    return env


def eval(env, model, eval_episode_num):
    """Evaluate the model and return avg_score and avg_highest"""
    score = []
    highest = []
    step_count = []
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

    stats = {
        "score_mean": np.mean(score),
        "score_median": np.median(score),
        "score_max": np.max(score),
        "score_std": np.std(score),
        "highest_mean": np.mean(highest),
        "highest_median": np.median(highest),
        "highest_max": np.max(highest),
        "highest_std": np.std(highest),
        "highest_hist": wandb.Histogram(highest),
        "step_count_mean": np.mean(step_count),
        "step_count_median": np.median(step_count),
        "step_count_max": np.max(step_count),
        "step_count_std": np.std(step_count),
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
                #     gradient_save_freq=100,  # unsure how to intepret it
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

            if is_better:
                current_best = stats["score_mean"]
                save_path = config["save_path"]
                model.save(f"{save_path}")


def experiment(config):
    wandb.init(
        project="rl-2048",
        id=config["run_id"],
        config=config,
        save_code=True
        # sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    )

    train_env = make_vec_env(
        make_env, n_envs=config["n_envs"], vec_env_cls=SubprocVecEnv
    )
    eval_env = DummyVecEnv([make_eval_env])
    model = config["algorithm"](
        config["policy_network"],
        train_env,
        verbose=0,
        learning_rate=config["learning_rate"],
    )
    train(eval_env, model, config)


    code_artifact = wandb.Artifact(name="code", type="code")
    code_artifact.add_file("./train.py")
    code_artifact.add_file("./envs/my2048_env.py")
    wandb.log_artifact(code_artifact)

    wandb.save(config["save_path"] + ".zip")
    wandb.finish()


if __name__ == "__main__":
    # config = {
    #     "run_id": "ppo-env8-r10-n0.5-f200-2",
    #     "notes": "np.log2(reward) / 10; negative=-0.5; foul count: 100; use eval env",
    #     "algorithm": PPO,
    #     "policy_network": "MlpPolicy",
    #     "epoch_num": 200,
    #     "eval_episode_num": 100,
    #     "timesteps_per_epoch": 1000,
    #     "learning_rate": 1e-4,
    #     "n_envs": 8,
    # }
    # config["save_path"] = os.path.join("models", config["run_id"])
    # experiment(config)

    base_config = {
        "algorithm": PPO,
        "policy_network": "MlpPolicy",
        "epoch_num": 400,
        "eval_episode_num": 100,
        "timesteps_per_epoch": 1000,
        "learning_rate": 1e-4,
        "n_envs": 8,
    }

    config = {
        "run_id": "ppo-dynamic-penality",
        "notes": "reward = -0.1 * np.log2(self.highest())",
    }
    config.update(base_config)
    config["save_path"] = os.path.join("models", config["run_id"])
    experiment(config)