#!/usr/bin/env bash
#SBATCH --job-name=hugs_train
#SBATCH --exclude=mscluster48,mscluster44,mscluster51,mscluster65,mscluster68,mscluster75,mscluster76
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-12:00:00
#SBATCH --output=/datasets/%u/HUGs/slurm_logs/%x-%j.out
#SBATCH --error=/datasets/%u/HUGs/slurm_logs/%x-%j.err

set -Eeuo pipefail

PROJECT_DIR="/home-mscluster/${USER}/projects/HUGs"
DATA_DIR="/datasets/${USER}/HUGs"
ENV_DIR="$PROJECT_DIR/venv"
ANACONDA_MODULE="anaconda/anaconda-2025-12-2"
export APPTAINER_BINDPATH="/datasets${APPTAINER_BINDPATH:+,$APPTAINER_BINDPATH}"
export APPTAINERENV_JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"

cd "$PROJECT_DIR"
if ! type module >/dev/null 2>&1; then
  set +u
  source /etc/profile.d/lmod.sh
  set -u
fi
module load "$ANACONDA_MODULE"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  echo "Python environment is missing: $ENV_DIR" >&2
  echo "Submit cluster/setup_environment.sbatch first." >&2
  exit 1
fi

ENV_PYTHON=(
  apptainer exec --nv "$ANACONDA_SIF"
  conda run --no-capture-output --prefix "$ENV_DIR"
  python
)

NVIDIA_LIBS="$("${ENV_PYTHON[@]}" - <<'PY'
import glob
import os
import site

paths = []
for base in site.getsitepackages():
  paths.extend(glob.glob(os.path.join(base, 'nvidia', '*', 'lib')))
print(':'.join(paths))
PY
)"
if [[ -n "$NVIDIA_LIBS" ]]; then
  export APPTAINERENV_LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
XLA_FLAGS_VALUE="--xla_gpu_strict_conv_algorithm_picker=false"
if [[ -d /usr/local/cuda-12.0 ]]; then
  XLA_FLAGS_VALUE+=" --xla_gpu_cuda_data_dir=/usr/local/cuda-12.0"
fi
export APPTAINERENV_XLA_FLAGS="$XLA_FLAGS_VALUE"
export APPTAINERENV_PYTHONUNBUFFERED=1
export APPTAINERENV_OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "HOST=$(hostname)"
echo "JOB_ID=${SLURM_JOB_ID:-unset}"
echo "ENV_DIR=$ENV_DIR"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

"${ENV_PYTHON[@]}" - <<'PY'
import sys

import jax

print("JAX version:", jax.__version__)
try:
  gpu = jax.devices("gpu")
except Exception as exc:
  print("CUDA preflight failed on this bigbatch node.")
  print("Resubmit the job or update the exclusion list.")
  print(f"{type(exc).__name__}: {exc}")
  sys.exit(42)
if not gpu:
  print("CUDA preflight found no GPU devices on this bigbatch node.")
  sys.exit(42)
print("Default devices:", jax.devices())
print("GPU devices:", gpu)
PY

if (($#)); then
  TRAIN_ARGS=("$@")
else
  TRAIN_ARGS=(--configs crafter size1m --script train_eval)
fi

"${ENV_PYTHON[@]}" -u dreamerv3/main.py \
  --logdir "$DATA_DIR/logdir/{timestamp}" \
  "${TRAIN_ARGS[@]}"
