#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_DIR="/home-mscluster/${USER}/projects/HUGs"
DATA_DIR="/datasets/${USER}/HUGs"
ENV_DIR="$PROJECT_DIR/venv"
ANACONDA_MODULE="anaconda/anaconda-2025-12-2"
export APPTAINER_BINDPATH="/datasets${APPTAINER_BINDPATH:+,$APPTAINER_BINDPATH}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Refusing to install on the login node." >&2
  echo "Submit cluster/setup_environment.sbatch through Slurm instead." >&2
  exit 1
fi

[[ -d "$PROJECT_DIR" ]] || {
  echo "Project directory not found: $PROJECT_DIR" >&2
  echo "Run ./sync_to_cluster.sh --once on your Mac first." >&2
  exit 1
}

mkdir -p "$DATA_DIR/venv" "$DATA_DIR/.cache/pip" "$DATA_DIR/.cache/conda"

if ! type module >/dev/null 2>&1; then
  set +u
  source /etc/profile.d/lmod.sh
  set -u
fi
module load "$ANACONDA_MODULE"
export PIP_CACHE_DIR="$DATA_DIR/.cache/pip"
export CONDA_PKGS_DIRS="$DATA_DIR/.cache/conda"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  echo "Creating Python 3.11 environment at $DATA_DIR/venv"
  conda create --yes \
    --override-channels \
    --channel conda-forge \
    --prefix "$ENV_DIR" \
    python=3.11 \
    pip
fi

conda run --no-capture-output --prefix "$ENV_DIR" \
  python -m pip install --upgrade pip setuptools wheel
# The cluster's GTX 1060 nodes are Pascal. cuDNN 9.11+ removed Pascal support.
conda run --no-capture-output --prefix "$ENV_DIR" \
  python -m pip install \
    --editable "$PROJECT_DIR" \
    crafter \
    'nvidia-cudnn-cu12==9.10.2.21'

JAX_PLATFORMS=cpu conda run --no-capture-output --prefix "$ENV_DIR" \
  python - <<'PY'
import gymnasium
import jax
import minigrid
import numpy as np

from embodied.envs.minigrid import Minigrid

print("Environment ready")
print("JAX version:", jax.__version__)
print("Login-node devices:", jax.devices())
print("Gymnasium version:", gymnasium.__version__)
print("MiniGrid version:", minigrid.__version__)

env = Minigrid(
    "Empty-8x8-v0", size=(64, 64), max_episode_steps=1, seed=0)
obs = env.step({"reset": True, "action": np.int32(0)})
assert obs["image"].shape == (64, 64, 3)
assert obs["is_first"] and not obs["is_last"]
obs = env.step({"reset": False, "action": np.int32(0)})
assert obs["is_last"] and not obs["is_terminal"]
env.close()
print("Embodied MiniGrid smoke test: passed")
PY
