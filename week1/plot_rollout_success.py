#!/usr/bin/env python3
"""
Plot rollout success rate vs timesteps for PPO, SAC, and SAC+HER.

Reads TensorBoard events under:
  <this_script_dir>/results/PPO
  <this_script_dir>/results/SAC
  <this_script_dir>/results/SAC_HER

Looks specifically for the scalar tag: 'rollout/success_rate'.
Saves figure to: <this_script_dir>/results/plot1.png
"""

import os
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# -------- config --------
TAG = "rollout/success_rate"
SMOOTH_WINDOW = 5  # moving-average window (points). Set to 1 to disable.
OUTPUT_FILENAME = "plot1.png"
# ------------------------


def script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def agent_roots() -> Dict[str, str]:
    base_results = os.path.join(script_dir(), "results")
    return {
        "PPO": os.path.join(base_results, "PPO"),
        "SAC": os.path.join(base_results, "SAC"),
        "SAC + HER": os.path.join(base_results, "SAC_HER"),
    }


def find_tb_run_dirs(root: str) -> List[str]:
    """All subdirs containing TensorBoard event files."""
    runs: List[str] = []
    if not os.path.isdir(root):
        return runs
    for cur, _dirs, files in os.walk(root):
        if any(f.startswith("events.out.tfevents") for f in files):
            runs.append(cur)
    return sorted(set(runs))


def load_series_from_run(run_dir: str, tag: str) -> Tuple[List[int], List[float]]:
    """Return (steps, values) for a given TB run dir and tag."""
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 100_000})
    ea.Reload()
    if tag not in set(ea.Tags().get("scalars", [])):
        raise RuntimeError(f"Tag '{tag}' not found in {run_dir}")
    events = ea.Scalars(tag)
    steps = [ev.step for ev in events]
    vals = [float(ev.value) for ev in events]
    return steps, vals


def smooth(vals: List[float], k: int) -> List[float]:
    if k <= 1 or len(vals) == 0:
        return vals
    out: List[float] = []
    s = 0.0
    q: List[float] = []
    for v in vals:
        q.append(v)
        s += v
        if len(q) > k:
            s -= q.pop(0)
        out.append(s / len(q))
    return out


def collect_agent_series(agent: str, root: str, tag: str) -> Tuple[List[int], List[float]]:
    """
    Collect and merge all 'rollout/success_rate' points for an agent.
    If multiple runs exist, we concatenate and sort by step.
    """
    run_dirs = find_tb_run_dirs(root)
    if not run_dirs:
        print(f"[WARN] No TensorBoard runs for {agent} under: {root}")
        return [], []

    all_pairs: List[Tuple[int, float]] = []
    used = 0
    for rd in run_dirs:
        try:
            steps, vals = load_series_from_run(rd, tag)
            print(f"[INFO] {agent}: using {len(steps)} points from {rd}")
            all_pairs.extend(zip(steps, vals))
            used += 1
        except Exception as e:
            print(f"[INFO] {agent}: skipping {rd} ({e})")

    if used == 0:
        return [], []

    all_pairs.sort(key=lambda t: t[0])
    steps = [p[0] for p in all_pairs]
    vals = [p[1] for p in all_pairs]
    return steps, vals


def main():
    roots = agent_roots()

    out_dir = os.path.join(script_dir(), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, OUTPUT_FILENAME)

    plt.figure(figsize=(9, 6))
    plotted = False

    for agent, root in roots.items():
        steps, vals = collect_agent_series(agent, root, TAG)
        if not steps:
            print(f"[INFO] No '{TAG}' data for {agent}.")
            continue
        vals_sm = smooth(vals, SMOOTH_WINDOW)
        plt.plot(steps, vals_sm, label=agent)  # (no explicit colors)

        # mark also the raw final point so you can see the endpoint clearly
        plt.scatter([steps[-1]], [vals[-1]], marker='o')
        plotted = True

    if not plotted:
        print(
            f"No data to plot.\n"
            f"- Make sure training produced TensorBoard logs under results/<AGENT>/ ...\n"
            f"- And that the env sets info['is_success'] so SB3 logs '{TAG}'."
        )
        sys.exit(1)

    plt.xlabel("Timesteps")
    plt.ylabel("Rollout success rate")
    plt.title("FetchPickAndPlace (sparse) — Rollout Success Rate vs Timesteps")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[OK] Saved plot: {out_path}")


if __name__ == "__main__":
    main()
