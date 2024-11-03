import gymnasium as gym
import wandb
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO, DQN
from gymnasium.wrappers import RecordVideo
import tqdm



def evaluate(model, env, episodes):
    for episode in range(episodes):
        obs, info = env.reset()
        done, truncate = False, False
        score = 0
        while not done or not truncate:
            # env.render()
            action, _ = model.predict(obs)
            what = env.step(action)
            obs, reward, done, truncate, info = what
            score += reward

    return score

if __name__ == '__main__':
    # train_env = make_vec_env(
    #     make_train_env, n_envs=8, vec_env_cls=SubprocVecEnv
    # )
    wandb.init(
        project="rl-cart",
        name="blah",
    )

    train_env = gym.make('CartPole-v1', render_mode='rgb_array', max_episode_steps=500)
    eval_env = gym.make('CartPole-v1', render_mode='rgb_array', max_episode_steps=500)
    # eval_env = gym.make('CartPole-v1', render_mode='rgb_array')

    epoch_num = 20
    eval_num = 1
    model = DQN("MlpPolicy", train_env, verbose=0)

    for epoch in tqdm.tqdm(range(epoch_num)):
        model.learn(total_timesteps=10000)
        score = evaluate(model, eval_env, eval_num)
        wandb.log({"score": score})
        print(score)
        model.save("ppo_cartpole_model")

    wandb.finish()

    # env = gym.make('CartPole-v1', render_mode='rgb_array')
    # env = RecordVideo(env, video_folder="./videos", disable_logger=False)
    # model = PPO.load("ppo_cartpole_model")
    # evaluate(model, env, 1)
    # env.close()