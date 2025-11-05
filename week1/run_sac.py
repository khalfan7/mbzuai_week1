import os
import gymnasium as gym
import gymnasium_robotics
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3 import SAC


def make_env():
    return gym.make('FetchPickAndPlace-v2',
                    reward_type='sparse',
                    render_mode=None)


if __name__ == '__main__':
    gym.register_envs(gymnasium_robotics)

    # Config
    n_envs = 12
    total_timesteps = 1_000_000
    log_dir = 'results/SAC'
    os.makedirs(log_dir, exist_ok=True)

    def make_flattened_env():
        env = make_env()
        env = FlattenObservation(env)
        return env

    env = make_vec_env(make_flattened_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv, seed=42)
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10)

    model = SAC(
        policy='MlpPolicy',
        env=env,
        learning_rate=3e-4,
        buffer_size=1_500_000,
        batch_size=512,
        tau=0.02,
        gamma=0.98,
        learning_starts=5_000,
        device='cuda',
        seed=42,
        verbose=1,
        tensorboard_log=log_dir,
        replay_buffer_kwargs=dict(handle_timeout_termination=True)
    )

    print(f"\nTraining SAC for {total_timesteps:,} steps (sparse reward)…")
    print(f"Environments: {n_envs}")
    model.learn(total_timesteps=total_timesteps)
    model.save(f'{log_dir}/final_model')
    env.save(f'{log_dir}/vecnormalize.pkl')   
    env.close()
    print(f"Saved model + VecNormalize stats to {log_dir}/")
