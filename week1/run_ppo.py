"""
Run PPO agent for Pick and Place task
"""
import sys
sys.path.insert(0, '.')

from week1 import set_seed, make_env, arg_parse, get_next_test_number
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
import os
from datetime import datetime

def main():
    seed = 42
    set_seed(seed)
    gym.register_envs(gymnasium_robotics)
    
    # Create logging directory with test naming
    results_dir = 'results/PPO'
    test_num = get_next_test_number(results_dir)
    log_dir = f'{results_dir}/test{test_num}'
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize environment
    env = make_vec_env(make_env, n_envs=32, vec_env_cls=DummyVecEnv)
    
    # Log experiment configuration
    with open(f'{log_dir}/config.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("PPO - Pick and Place (Sparse Reward)\n")
        f.write("="*60 + "\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Algorithm: PPO\n")
        f.write(f"Total Timesteps: 1,000,000\n")
        f.write(f"Environment: FetchPickAndPlace-v4\n")
        f.write(f"Reward Type: sparse\n")
        f.write(f"Render Mode: None (visualization disabled)\n")
        f.write(f"Device: CUDA (GPU acceleration)\n")
        f.write(f"Max Episode Steps: 200\n")
        f.write(f"Number of Parallel Environments: 32\n")
        f.write(f"Learning Rate: 1e-3\n")
        f.write(f"N Steps: 4096\n")
        f.write(f"Batch Size: 2048\n")
        f.write(f"Entropy Coefficient: 0.02\n")
        f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
    
    # Initialize PPO model with optimized hyperparameters for better GPU utilization
    model = PPO(
        policy='MultiInputPolicy',
        env=env,
        verbose=1,
        seed=seed,
        device='cuda',
        tensorboard_log=log_dir,
        n_steps=4096,  # Increased from default 2048 for more GPU batch processing
        batch_size=2048,  # Increased to 2048 for better GPU utilization
        learning_rate=1e-3,  # Increased learning rate for faster convergence
        ent_coef=0.02,  # Entropy coefficient for better exploration in sparse reward setting
        vf_coef=0.5,  # Value function coefficient
        max_grad_norm=0.5  # Gradient clipping for stability
    )
    
    print("="*60)
    print("Training PPO for 1,000,000 timesteps...")
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
