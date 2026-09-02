#!/usr/bin/env bash
#SBATCH --job-name=hugs_setup
#SBATCH --partition=stampede
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=/datasets/%u/HUGs/slurm_logs/%x-%j.out
#SBATCH --error=/datasets/%u/HUGs/slurm_logs/%x-%j.err

set -Eeuo pipefail

PROJECT_DIR="/home-mscluster/${USER}/projects/HUGs"
cd "$PROJECT_DIR"
exec ./cluster/setup_environment.sh
