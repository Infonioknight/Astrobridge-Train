#!/bin/bash
#SBATCH --job-name=ab-diag
#SBATCH --account=bfvh-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -uo pipefail
cd /work/hdd/bfvh/li26/Astrobridge-Train-v6
export HF_HUB_CACHE=/work/hdd/bfvh/li26/hf-cache
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
PY=/u/li26/venvs/astrobridge-v6/bin/python
echo "=========== 1. SPECTRA PADDING TEST ==========="
$PY diag/spec_padding_test.py
echo "=========== 2. QFORMER CAPACITY TEST ==========="
$PY diag/capacity_gpu.py
