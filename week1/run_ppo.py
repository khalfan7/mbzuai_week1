import os
import gymnasium as gym
import gymnasium_robotics
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3 import PPO


def make_env():
    return gym.make('FetchPickAndPlace-v2',
                    reward_type='sparse',
                    render_mode=None,
                    max_episode_steps=200)


if __name__ == '__main__':
    gym.register_envs(gymnasium_robotics)

    # Config
    n_envs = 12
    total_timesteps = 1_000_000
    log_dir = 'results/PPO'
    os.makedirs(log_dir, exist_ok=True)

    # Vectorized envs
    env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv, seed=42)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # PPO 
    model = PPO(
        policy='MultiInputPolicy',
        env=env,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.98,
        gae_lambda=0.95,
        ent_coef=0.0,
        device='cuda',
        seed=42,
        verbose=1,
        tensorboard_log=log_dir
    )

    model.learn(total_timesteps=total_timesteps)
    model.save(f'{log_dir}/final_model')
    env.save(f'{log_dir}/vecnormalize.pkl')   
    env.close()