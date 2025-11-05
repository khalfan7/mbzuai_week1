"""
Run experiments sequentially; print only pass/fail for each.
"""
import subprocess
import sys
import time

experiments = [
    ('run_ppo.py', 'PPO'),
    ('run_sac.py', 'SAC'),
    ('run_sac_her.py', 'SAC + HER'),
]


def run_experiment(script_name, agent_name):
    try:
        subprocess.run([sys.executable, script_name], check=True)
        print(f"[PASSED] {agent_name}")
        return True
    except subprocess.CalledProcessError:
        print(f"[FAILED] {agent_name}")
        return False


def main():
    results = {}
    for script, name in experiments:
        results[name] = run_experiment(script, name)
        time.sleep(1)


if __name__ == '__main__':
    main()
