#!/bin/bash
#SBATCH --job-name=ab-tok
#SBATCH --account=bfvh-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --gpus-per-node=1
#SBATCH --mem=64g --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -uo pipefail
cd /work/hdd/bfvh/li26/Astrobridge-Train-v6
export HF_HUB_CACHE=/work/hdd/bfvh/li26/hf-cache PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
/u/li26/venvs/astrobridge-v6/bin/python diag/aion_tokens.py
