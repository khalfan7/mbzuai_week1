#!/usr/bin/env bash
set -euo pipefail

# ==== paths / names (as requested) ====
BASE_DIR="/home/khalfan.hableel/ROB7101/ass3"
ENV_DIR="$BASE_DIR/arel"

echo ">> Creating base dir: $BASE_DIR"
mkdir -p "$BASE_DIR"

# ---- sanity check: python version ----
PYVER=$(python3 -c 'import sys;print(".".join(map(str, sys.version_info[:3])))')
echo ">> python3 version: $PYVER"
python3 - <<'PY'
import sys
maj, min = sys.version_info[:2]
assert (maj, min) >= (3,8), f"Need Python >= 3.8 (found {sys.version})"
PY

# ==== create / reuse venv (no sudo needed) ====
if [ -d "$ENV_DIR" ]; then
  echo ">> Reusing existing venv at $ENV_DIR"
else
  echo ">> Creating venv at $ENV_DIR"
  python3 -m venv "$ENV_DIR"
fi

# Activate venv for the rest of the script
# shellcheck source=/dev/null
source "$ENV_DIR/bin/activate"

echo ">> Upgrading pip/setuptools/wheel..."
python -m pip install --upgrade pip setuptools wheel

# ==== PyTorch (CPU by default; CUDA build if drivers are present) ====
if command -v nvidia-smi >/dev/null 2>&1; then
  echo ">> NVIDIA driver detected; installing CUDA PyTorch wheels (cu121)…"
  python -m pip install "torch==2.4.*" --index-url https://download.pytorch.org/whl/cu121
else
  echo ">> No NVIDIA driver found; installing CPU PyTorch…"
  python -m pip install "torch==2.4.*"
fi

# ==== RL stack (versions pinned to play nicely together) ====
echo ">> Installing RL libraries (SB3 + Gymnasium Robotics + MuJoCo)…"
python -m pip install \
  "numpy==1.26.4" \
  "gymnasium==0.29.1" \
  "gymnasium-robotics==1.3.0" \
  "mujoco==3.1.5" \
  "stable-baselines3==2.3.0" \
  "tensorboard>=2.13,<2.18" \
  "protobuf<5" \
  "shimmy>=1.3.0" \
  "rich" "tqdm"

echo
echo "============================================================"
echo " Environment ready at: $ENV_DIR"
echo
echo " Activate it with:"
echo "   source \"$ENV_DIR/bin/activate\""
echo
echo " Quick smoke test (optional):"
echo "   python - <<'PY'"
echo "import gymnasium as gym, gymnasium_robotics"
echo "env = gym.make('FetchPickAndPlace-v4', reward_type='sparse', render_mode=None, max_episode_steps=200)"
echo "print('OK:', env.observation_space, '->', env.action_space)"
echo "env.close()"
echo "PY"
echo
echo " Notes:"
echo " - No sudo used. Training doesn't need display libs (render_mode=None)."
echo " - If you later want on-screen rendering, your admins may need to add GL libs."
echo "============================================================"
