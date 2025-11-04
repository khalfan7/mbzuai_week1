"""
Run SAC + HER agent for Pick and Place task
"""
import sys
sys.path.insert(0, '.')

from week1 import set_seed, make_env, arg_parse, get_next_test_number
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import SAC, HerReplayBuffer
import os
from datetime import datetime

def main():
    seed = 42
    gamma = 0.99
    tau = 0.05
    buffer_size = int(1e6)
    
    set_seed(seed)
    gym.register_envs(gymnasium_robotics)
    
    # Create logging directory with test naming
    results_dir = 'results/SAC_HER'
    test_num = get_next_test_number(results_dir)
    log_dir = f'{results_dir}/test{test_num}'
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize environment
    env = make_vec_env(make_env, n_envs=32, vec_env_cls=DummyVecEnv)
    
    # Log experiment configuration
    with open(f'{log_dir}/config.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("SAC + HER - Pick and Place (Sparse Reward)\n")
        f.write("="*60 + "\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Algorithm: SAC + HER (Hindsight Experience Replay)\n")
        f.write(f"Total Timesteps: 1,000,000\n")
        f.write(f"Environment: FetchPickAndPlace-v4\n")
        f.write(f"Reward Type: sparse\n")
        f.write(f"Render Mode: None (visualization disabled)\n")
        f.write(f"Device: CUDA (GPU acceleration)\n")
        f.write(f"Max Episode Steps: 200\n")
        f.write(f"Number of Parallel Environments: 32\n")
        f.write(f"Gamma (discount factor): {gamma}\n")
        f.write(f"Tau (target network update): {tau}\n")
        f.write(f"Buffer Size: {buffer_size}\n")
        f.write(f"Learning Rate: 1e-4\n")
        f.write(f"Batch Size: 2048\n")
        f.write(f"HER - n_sampled_goal: 8\n")
        f.write(f"HER - goal_selection_strategy: future\n")
        f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
    
    # Initialize SAC + HER model with optimized hyperparameters
    model = SAC(
        policy='MultiInputPolicy',
        env=env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=8,  # Increased from 4 for more hindsight goals
            goal_selection_strategy='future',
        ),
        buffer_size=buffer_size,
        batch_size=2048,  # Increased to 2048 for better GPU utilization
        learning_rate=1e-4,
        tau=tau,
        gamma=gamma,
        device='cuda',
        seed=seed,
        verbose=1,
        tensorboard_log=log_dir,
        train_freq=1,  # Train every step
        gradient_steps=1,  # Update once per step
        learning_starts=10240  # This is 32 environments * 200 steps * 1.6 (buffer size)
    )
    
    print("="*60)
    print("Training SAC + HER for 1,000,000 timesteps...")
    print(f"Logs saved to: {log_dir}")
    print("="*60 + "\n")
    
    # Train the model
    model.learn(total_timesteps=1_000_000)
    
    # Save the model
    model.save(f'{log_dir}/model.pkl')
    
    # Log completion
    with open(f'{log_dir}/config.txt', 'a') as f:
        f.write(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Training completed successfully!\n")
    
    print(f"\nModel saved to: {log_dir}/model.pkl")
    env.close()

if __name__ == '__main__':
    main()
