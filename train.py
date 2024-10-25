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

# os.environ["WANDB_MODE"] = "disabled"
warnings.filterwarnings("ignore")
register(id="2048-v0", entry_point="envs:My2048Env")
register(id="2048-eval", entry_point="envs:Eval2048Env")


my_config = {
    "run_id": "ppo",
    "algorithm": PPO,
    "policy_network": "MlpPolicy",
    "epoch_num": 200,
    "eval_episode_num": 100,
    "timesteps_per_epoch": 1000,
    "learning_rate": 1e-4,
    "n_envs": 1,
}

my_config["save_path"] = os.path.join("models", my_config["run_id"])

# os.environ["WANDB_MODE"] = "offline"
wandb.init(
    project="rl-2048",
    id=my_config["run_id"],
    config=my_config,
    # sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
)


def make_env():
    env = gym.make("2048-v0")
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
                callback=WandbCallback(
                    gradient_save_freq=100,  # unsure how to intepret it
                    verbose=2,
                ),
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


if __name__ == "__main__":
    # logger = configure("/tmp/sb3_log/", ["stdout", "tensorboard"])
    # num_train_envs = 1
    # train_env = DummyVecEnv([make_env for _ in range(num_train_envs)])

    train_env = make_vec_env(
        make_env, n_envs=my_config["n_envs"], vec_env_cls=SubprocVecEnv
    )
    eval_env = DummyVecEnv([make_env])
    model = my_config["algorithm"](
        my_config["policy_network"],
        train_env,
        verbose=0,
        learning_rate=my_config["learning_rate"],
        # tensorboard_log=my_config["run_id"],
    )
    # model.set_logger(logger)
    train(eval_env, model, my_config)

    # run final evaluation on test environment
    test_env = gym.make("2048-eval")
    score, highest = evaluation(test_env, model, True, eval_num=100)
    test_stats = {
        "test_score_mean": np.mean(score),
        "test_score_median": np.median(score),
        "test_score_max": np.max(score),
        "test_score_std": np.std(score),
        "test_highest_mean": np.mean(highest),
        "test_highest_median": np.median(highest),
        "test_highest_max": np.max(highest),
        "test_highest_std": np.std(highest),
    }
    wandb.log(test_stats)
    wandb.save(my_config["save_path"] + ".zip")
    wandb.finish()
