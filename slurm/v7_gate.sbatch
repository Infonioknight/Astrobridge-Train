#!/bin/bash
#SBATCH --job-name=ab-v7gate
#SBATCH --account=bfvh-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=32 --gpus-per-node=1
#SBATCH --mem=120g --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -uo pipefail
cd /work/hdd/bfvh/li26/Astrobridge-Train-v6
export HF_HUB_CACHE=/work/hdd/bfvh/li26/hf-cache PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
/u/li26/venvs/astrobridge-v6/bin/python scripts/04_eval.py \
    --checkpoint-dir outputs/checkpoints/stage2/best \
    --lora-dir outputs/checkpoints/stage2/best/lora \
    --split test --out outputs/eval/groundedness_report_v7.json
echo "### DONE"
