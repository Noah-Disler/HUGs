#!/usr/bin/env bash
#SBATCH --job-name=hugs_diag
#SBATCH --partition=biggpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:05:00
#SBATCH --output=/datasets/%u/HUGs/slurm_logs/%x-%j.out
#SBATCH --error=/datasets/%u/HUGs/slurm_logs/%x-%j.err

set -Eeuo pipefail

PROJECT_DIR="/home-mscluster/${USER}/projects/HUGs"
ANACONDA_MODULE="anaconda/anaconda-2025-12-2"
export APPTAINER_BINDPATH="/datasets${APPTAINER_BINDPATH:+,$APPTAINER_BINDPATH}"

if ! type module >/dev/null 2>&1; then
  set +u
  source /etc/profile.d/lmod.sh
  set -u
fi
module load "$ANACONDA_MODULE"

nvidia-smi
apptainer exec --nv "$ANACONDA_SIF" \
  conda run --no-capture-output --prefix "$PROJECT_DIR/venv" \
  python - <<'PY'
import importlib.metadata

import jax
import jax.numpy as jnp
import scope

distribution = importlib.metadata.distribution('scope')
print('JAX devices:', jax.devices())
result = (jnp.ones((128, 128)) @ jnp.ones((128, 128))).block_until_ready()
assert result.shape == (128, 128)
print('JAX GPU compilation: passed')
print('Scope version:', distribution.version)
print('Scope module:', scope.__file__)
print('Scope entry points:')
for entry in distribution.entry_points:
  print(f'  {entry.group}: {entry.name} = {entry.value}')
print('Scope Python files:')
for path in distribution.files or ():
  path = str(path)
  if path.startswith('scope/') and path.endswith('.py'):
    print(f'  {path}')
PY
