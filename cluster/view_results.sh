#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_DIR="/home-mscluster/${USER}/projects/HUGs"
DATA_DIR="/datasets/${USER}/HUGs"
PORT="${1:-8000}"
ANACONDA_MODULE="anaconda/anaconda-2025-12-2"
export APPTAINER_BINDPATH="/datasets${APPTAINER_BINDPATH:+,$APPTAINER_BINDPATH}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Refusing to start the viewer on the login node." >&2
  echo "Submit cluster/view_results.sbatch through Slurm instead." >&2
  exit 1
fi

if ! type module >/dev/null 2>&1; then
  set +u
  source /etc/profile.d/lmod.sh
  set -u
fi
module load "$ANACONDA_MODULE"

exec apptainer exec "$ANACONDA_SIF" \
  conda run --no-capture-output --prefix "$PROJECT_DIR/venv" \
  scope --basedir "$DATA_DIR/logdir" --port "$PORT"
