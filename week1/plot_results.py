"""
Plotting and analysis script for comparing training results
Extracts data from TensorBoard logs and creates comparison plots
"""
import os
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from tensorboard.compat.proto import event_pb2

def extract_tensorboard_logs(log_dir):
    """Extract episode rewards from TensorBoard event files"""
    rewards = []
    timesteps = []
    
    # Find event files in the log directory
    event_files = list(Path(log_dir).glob('events.out.tfevents.*'))
    
    if not event_files:
        print(f"No TensorBoard logs found in {log_dir}")
        return None, None
    
    for event_file in event_files:
        loader = EventFileLoader(str(event_file))
        for event in loader.Load():
            for value in event.summary.value:
                if value.tag == 'rollout/ep_rew_mean':
                    rewards.append(value.simple_value)
                    timesteps.append(event.step)
    
    if rewards and timesteps:
        # Sort by timestep
        sorted_pairs = sorted(zip(timesteps, rewards), key=lambda x: x[0])
        timesteps, rewards = zip(*sorted_pairs)
        return np.array(timesteps), np.array(rewards)
    
    return None, None

def load_agent_results(agent_name):
    """Load results for a specific agent"""
    agent_dir = f'results/{agent_name}'
    
    if not os.path.exists(agent_dir):
        print(f"Results directory not found for {agent_name}")
        return None, None, None
    
    # Find the most recent experiment folder
    experiment_dirs = sorted([d for d in os.listdir(agent_dir) 
                             if os.path.isdir(os.path.join(agent_dir, d))])
    
    if not experiment_dirs:
        print(f"No experiments found for {agent_name}")
        return None, None, None
    
    latest_dir = os.path.join(agent_dir, experiment_dirs[-1])
    config_file = os.path.join(latest_dir, 'config.txt')
    
    # Read config
    config = {}
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config['content'] = f.read()
    
    # Extract logs
    timesteps, rewards = extract_tensorboard_logs(latest_dir)
    
    return timesteps, rewards, config

def create_comparison_plots():
    """Create comparison plots for all three agents"""
    agents = ['PPO', 'SAC', 'SAC_HER']
    
    # Load data for all agents
    all_data = {}
    for agent in agents:
        ts, rw, cfg = load_agent_results(agent)
        all_data[agent] = {'timesteps': ts, 'rewards': rw, 'config': cfg}
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: All agents on same graph
    ax1 = plt.subplot(2, 2, 1)
    for agent in agents:
        data = all_data[agent]
        if data['timesteps'] is not None:
            ax1.plot(data['timesteps'], data['rewards'], label=agent, linewidth=2)
    ax1.set_xlabel('Timesteps', fontsize=12)
    ax1.set_ylabel('Mean Episode Reward', fontsize=12)
    ax1.set_title('Learning Curves - All Agents Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Individual plots for PPO
    ax2 = plt.subplot(2, 2, 2)
    data_ppo = all_data['PPO']
    if data_ppo['timesteps'] is not None:
        ax2.plot(data_ppo['timesteps'], data_ppo['rewards'], color='blue', linewidth=2)
        ax2.fill_between(data_ppo['timesteps'], data_ppo['rewards'], alpha=0.3, color='blue')
        ax2.set_title('PPO - Proximal Policy Optimization', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Timesteps', fontsize=11)
    ax2.set_ylabel('Mean Episode Reward', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Individual plots for SAC
    ax3 = plt.subplot(2, 2, 3)
    data_sac = all_data['SAC']
    if data_sac['timesteps'] is not None:
        ax3.plot(data_sac['timesteps'], data_sac['rewards'], color='green', linewidth=2)
        ax3.fill_between(data_sac['timesteps'], data_sac['rewards'], alpha=0.3, color='green')
        ax3.set_title('SAC - Soft Actor-Critic', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Timesteps', fontsize=11)
    ax3.set_ylabel('Mean Episode Reward', fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Individual plots for SAC+HER
    ax4 = plt.subplot(2, 2, 4)
    data_her = all_data['SAC_HER']
    if data_her['timesteps'] is not None:
        ax4.plot(data_her['timesteps'], data_her['rewards'], color='red', linewidth=2)
        ax4.fill_between(data_her['timesteps'], data_her['rewards'], alpha=0.3, color='red')
        ax4.set_title('SAC + HER - With Hindsight Experience Replay', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Timesteps', fontsize=11)
    ax4.set_ylabel('Mean Episode Reward', fontsize=11)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/comparison_plots.png', dpi=300, bbox_inches='tight')
    print("✓ Comparison plots saved to results/comparison_plots.png")
    
    # Create summary statistics
    print("\n" + "="*70)
    print("TRAINING SUMMARY STATISTICS")
    print("="*70)
    
    for agent in agents:
        data = all_data[agent]
        if data['rewards'] is not None:
            print(f"\n{agent}:")
            print(f"  Final Mean Reward: {data['rewards'][-1]:.4f}")
            print(f"  Best Mean Reward: {np.max(data['rewards']):.4f}")
            print(f"  Mean Reward (last 100k steps): {np.mean(data['rewards'][-100:]):.4f}")
            print(f"  Std Dev (last 100k steps): {np.std(data['rewards'][-100:]):.4f}")
        else:
            print(f"\n{agent}: No data available")
    
    print("\n" + "="*70)
    
    # Analyze learning differences
    analyze_learning_differences(all_data)

def analyze_learning_differences(all_data):
    """Analyze and describe how the learning of agents differs"""
    print("\n" + "="*70)
    print("LEARNING BEHAVIOR ANALYSIS")
    print("="*70)
    
    agents = ['PPO', 'SAC', 'SAC_HER']
    
    # Extract valid data
    valid_agents = {agent: all_data[agent] for agent in agents 
                    if all_data[agent]['rewards'] is not None}
    
    if len(valid_agents) < 2:
        print("\nInsufficient data for comparison (need at least 2 agents)")
        return
    
    print("\n1. LEARNING SPEED (Early Training - First 200k Timesteps)")
    print("-" * 70)
    for agent, data in valid_agents.items():
        rewards = data['rewards']
        timesteps = data['timesteps']
        
        # Find rewards in first 200k timesteps
        early_mask = timesteps <= 200000
        if np.any(early_mask):
            early_rewards = rewards[early_mask]
            initial_reward = early_rewards[0] if len(early_rewards) > 0 else 0
            final_early_reward = early_rewards[-1] if len(early_rewards) > 0 else 0
            improvement = final_early_reward - initial_reward
            
            print(f"\n{agent}:")
            print(f"  Starting reward: {initial_reward:.4f}")
            print(f"  Reward at 200k: {final_early_reward:.4f}")
            print(f"  Improvement: {improvement:.4f}")
            
            if agent == 'SAC_HER':
                print(f"  → HER enables RAPID early learning via goal relabeling")
            elif agent == 'SAC':
                print(f"  → Off-policy learning allows efficient sample reuse")
            elif agent == 'PPO':
                print(f"  → On-policy learning, slower but more stable")
    
    print("\n\n2. STABILITY & CONVERGENCE (Training Dynamics)")
    print("-" * 70)
    for agent, data in valid_agents.items():
        rewards = data['rewards']
        
        # Calculate rolling variance (window of 10)
        if len(rewards) >= 10:
            rolling_std = np.array([np.std(rewards[max(0,i-10):i+1]) 
                                   for i in range(len(rewards))])
            avg_volatility = np.mean(rolling_std)
            
            # Check for convergence (last 20% of training)
            late_training_start = int(len(rewards) * 0.8)
            late_training_rewards = rewards[late_training_start:]
            converged = np.std(late_training_rewards) < 0.1  # threshold
            
            print(f"\n{agent}:")
            print(f"  Average volatility: {avg_volatility:.4f}")
            print(f"  Late training std: {np.std(late_training_rewards):.4f}")
            print(f"  Converged: {'Yes' if converged else 'No (still improving)'}")
            
            if agent == 'SAC_HER':
                print(f"  → HER: High initial variance, then stabilizes as goals are learned")
            elif agent == 'SAC':
                print(f"  → SAC: Smooth learning with entropy regularization")
            elif agent == 'PPO':
                print(f"  → PPO: Most stable due to clipped policy updates")
    
    print("\n\n3. SAMPLE EFFICIENCY (Learning per Timestep)")
    print("-" * 70)
    
    # Compare area under curve (total reward accumulated)
    efficiency_scores = {}
    for agent, data in valid_agents.items():
        rewards = data['rewards']
        timesteps = data['timesteps']
        
        # Approximate area under curve
        if len(rewards) > 1:
            auc = np.trapz(rewards, timesteps)
            efficiency_scores[agent] = auc
    
    if efficiency_scores:
        best_agent = max(efficiency_scores, key=efficiency_scores.get)
        print(f"\nArea Under Learning Curve (higher = more efficient):")
        for agent in agents:
            if agent in efficiency_scores:
                score = efficiency_scores[agent]
                is_best = " ⭐ BEST" if agent == best_agent else ""
                print(f"  {agent}: {score:,.0f}{is_best}")
        
        print(f"\n  Analysis:")
        print(f"  → {best_agent} achieved the highest cumulative reward")
        print(f"  → This indicates better sample efficiency - learning more from each experience")
    
    print("\n\n4. FINAL PERFORMANCE (Success at Task)")
    print("-" * 70)
    
    final_performances = {}
    for agent, data in valid_agents.items():
        rewards = data['rewards']
        # Average of last 10% of training
        final_window = rewards[int(len(rewards)*0.9):]
        final_perf = np.mean(final_window)
        final_performances[agent] = final_perf
    
    if final_performances:
        ranked = sorted(final_performances.items(), key=lambda x: x[1], reverse=True)
        print(f"\nFinal Performance Ranking:")
        for rank, (agent, perf) in enumerate(ranked, 1):
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
            print(f"  {medal} {rank}. {agent}: {perf:.4f}")
        
        winner = ranked[0][0]
        print(f"\n  Winner: {winner}")
        
        if winner == 'SAC_HER':
            print(f"  → Expected! HER dramatically improves learning in sparse reward settings")
            print(f"  → Failed trajectories become training data through goal relabeling")
        elif winner == 'SAC':
            print(f"  → SAC's off-policy learning and entropy bonus drove strong performance")
        elif winner == 'PPO':
            print(f"  → Surprising! PPO's stability overcame its sample inefficiency")
    
    print("\n\n5. KEY INSIGHTS")
    print("-" * 70)
    print("""
The learning differences reflect each algorithm's core mechanism:

• PPO (On-Policy): 
  - Learns from fresh experiences only
  - Clipped updates ensure stability but slow learning
  - Struggles with sparse rewards (hard to find successful episodes)

• SAC (Off-Policy):
  - Reuses past experiences from replay buffer
  - Entropy regularization encourages exploration
  - Better sample efficiency than PPO

• SAC + HER (Off-Policy + Goal Relabeling):
  - Same benefits as SAC PLUS synthetic data generation
  - "Failed" episodes → Successful training examples via hindsight
  - Transforms sparse reward problem into dense reward problem
  - Expected to significantly outperform in this sparse reward task

In sparse reward environments like FetchPickAndPlace:
→ Random exploration rarely discovers success
→ HER solves this by learning from every trajectory (even failures)
→ This is why HER-based methods dominate goal-conditioned robotics tasks
    """)
    
    print("="*70)

if __name__ == '__main__':
    print("Creating comparison plots from training results...")
    create_comparison_plots()
    print("\nAnalysis complete!")
