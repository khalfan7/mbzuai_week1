import gymnasium as gym
import gymnasium_robotics
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO,SAC,HerReplayBuffer
#When you write your code you can pass arguments using argparse
import argparse
import numpy as np #inside these envornments we may use numpy operations
import random
import torch
import os

def set_seed(seed): 
#they start from a seed they feenerate a sequence of operation so for example when u start a thread or not its 
# important how we use submission and addition of netwroks but how the matrix computation is done inside the cpu 
# is something in parallel you dont know which one finish first this is a propblem related to float
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif torch.mps.is_available():
        torch.mps.manual_seed_all(seed)

#we make the environment in a definition so that we can call it multiple times
def make_env():
    env = gym.make('FetchPickAndPlace-v4', 
                   render_mode='None', 
                   max_episode_steps=200,)  
    #'Sparse rewards are only given at the end of a task, 
    #while dense rewards provide frequent feedback at multiple steps.
    env.reset(
    #reset is important because it resets the position of the environment and dont run  into problems with a wrapper
    )
    return env

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='fetchpickandplace-v4', help='Environment Name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--model', type=str, default='PPO', help='Model to use')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.05, help='Target network update rate')
    parser.add_argument('--buffer_size', type=int, default=int(1e6), help='Replay buffer size')
    args = parser.parse_args()
    return args
#after you add all the arguments you can parse them and parse will check the lines of your code in your bash

if __name__ == '__main__':
    args = arg_parse()
    set_seed(args.seed)
    gym.register_envs(gymnasium_robotics)
    env=make_vec_env(make_env, n_envs=5,vec_env_cls=DummyVecEnv)
    if args.model == 'PPO':
        model = PPO(policy='MultiInputPolicy', env=env, verbose=1)
    elif args.model == 'SAC':
        model = SAC(
            policy='MultiInputPolicy',
            env=env,
            buffer_size=args.buffer_size,
            batch_size=512,
            learning_rate=1e-5,
            tau=args.tau,
            gamma=args.gamma,
            device='cpu',
            seed=args.seed,
            verbose=1,
        )

    elif args.model == 'SAC_HER':
        model = SAC(
            policy='MultiInputPolicy',
            env=env,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=4,
                goal_selection_strategy='future',
                online_sampling=True,
                max_episode_length=200,
            ),
            buffer_size=args.buffer_size,
            learning_rate=1e-4,
            tau=args.tau,
            gamma=args.gamma,
            device='cpu',
            seed=args.seed,
            verbose=1,
        )

    model.learn(total_timesteps=1_000_000)
    
    # Create results directory with model name
    results_dir = f'results/{args.model}'
    os.makedirs(results_dir, exist_ok=True)
    
    # Save model in the model-specific directory
    model.save(f'{results_dir}/model.pkl')
#dummy vec env is simply a thread its like a weight data process youre running an app that app will be a process in the system
#ther esources in this system that are allocated to this app will be very high
#threading is also 

#number o threads inside cpu like how many threads you can spawm the thing of threads you can equalize them you can equalize the environment inside the thread like pagination
#something very interesting in computer science

