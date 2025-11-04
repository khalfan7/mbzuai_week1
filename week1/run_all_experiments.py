"""
Master script to run all three agents sequentially
This ensures reproducible experiments with same random seeds
"""
import subprocess
import sys
import time
from datetime import datetime

def run_experiment(script_name, agent_name):
    """Run a single experiment script"""
    print("\n" + "="*70)
    print(f"Starting {agent_name} training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            cwd='.',
            check=True,
            capture_output=False
        )
        print(f"\n[PASSED] {agent_name} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[FAILED] {agent_name} failed with error!")
        return False

def main():
    print("="*70)
    print("PICK AND PLACE ENVIRONMENT - REPRODUCIBLE EXPERIMENT")
    print("="*70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nThis script will train three RL agents:")
    print("  1. PPO (Proximal Policy Optimization)")
    print("  2. SAC (Soft Actor-Critic)")
    print("  3. SAC + HER (Soft Actor-Critic + Hindsight Experience Replay)")
    print("\nEach agent will train for 1,000,000 timesteps with:")
    print("  - Sparse rewards")
    print("  - Random seed: 42 (for reproducibility)")
    print("  - 32 parallel environments (increased from 16)")
    print("  - Batch size: 2048 (increased from 1024)")
    print("  - Visualization disabled (render_mode=None)")
    print("="*70 + "\n")
    
    experiments = [
        ('run_ppo.py', 'PPO'),
        ('run_sac.py', 'SAC'),
        ('run_sac_her.py', 'SAC + HER')
    ]
    
    results = {}
    
    for script, name in experiments:
        success = run_experiment(script, name)
        results[name] = success
        time.sleep(2)  # Brief pause between experiments
    
    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    for name, success in results.items():
        status = "[PASSED]" if success else "[FAILED]"
        print(f"{name:20} {status}")
    print("="*70)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nAll results are saved in the 'results/' directory")
    print("Check the results/PPO/test1, results/SAC/test1, results/SAC_HER/test1 folders for:")
    print("  - config.txt: Experiment configuration and parameters")
    print("  - model.pkl: Trained model")
    print("  - events.out.tfevents.*: TensorBoard logs")

if __name__ == '__main__':
    main()
