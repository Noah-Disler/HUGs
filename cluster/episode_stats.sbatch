#!/usr/bin/env bash
#SBATCH --job-name=hugs_epstats
#SBATCH --partition=stampede
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=/datasets/%u/HUGs/slurm_logs/%x-%j.out
#SBATCH --error=/datasets/%u/HUGs/slurm_logs/%x-%j.err

# Reports episode structure of the saved replay, to choose a window length.
# CPU only -- no JAX, no GPU, no checkpoint -- so this skips the --nv flag and
# the CUDA setup that train_bigbatch.sbatch and check_latents.sbatch need.

set -Eeuo pipefail

PROJECT_DIR="/home-mscluster/${USER}/projects/HUGs"
ENV_DIR="$PROJECT_DIR/venv"
ANACONDA_MODULE="anaconda/anaconda-2025-12-2"
export APPTAINER_BINDPATH="/datasets${APPTAINER_BINDPATH:+,$APPTAINER_BINDPATH}"
export APPTAINERENV_PYTHONUNBUFFERED=1

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

echo "HOST=$(hostname)"
echo "JOB_ID=${SLURM_JOB_ID:-unset}"

exec apptainer exec "$ANACONDA_SIF" \
  conda run --no-capture-output --prefix "$ENV_DIR" \
  python -u tm/trajectory/scripts/episode_stats.py
