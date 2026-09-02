#!/usr/bin/env bash
#SBATCH --job-name=hugs_viewer
#SBATCH --partition=stampede
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --output=/datasets/%u/HUGs/slurm_logs/%x-%j.out
#SBATCH --error=/datasets/%u/HUGs/slurm_logs/%x-%j.err

set -Eeuo pipefail

PROJECT_DIR="/home-mscluster/${USER}/projects/HUGs"
cd "$PROJECT_DIR"
exec ./cluster/view_results.sh 8000
