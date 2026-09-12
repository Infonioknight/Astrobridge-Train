#!/bin/bash
#SBATCH --job-name=ab-v7evals
#SBATCH --account=bfvh-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=32 --gpus-per-node=1
#SBATCH --mem=200g --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -uo pipefail
cd /work/hdd/bfvh/li26/Astrobridge-Train-v6
export HF_HUB_CACHE=/work/hdd/bfvh/li26/hf-cache PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
PY=/u/li26/venvs/astrobridge-v6/bin/python
REPO=UniverseTBD/astrobridge-model-v7

echo "########## LIGHTCURVE CLASSIFICATION (YSE) ##########"
$PY -m eval.runners.run_lightcurve_eval --repo-id $REPO --backend local \
    --track lightcurve_only --out outputs/eval/v7_yse_lightcurve.json
echo "exit=$?"

echo "########## IMAGE: COLLECT GALAXY10 LABELS ##########"
$PY -m eval.runners.collect_image_labels --repo-id $REPO --backend local \
    --n 150 --seed 0 --out outputs/eval/v7_galaxy10_raw.json
echo "exit=$?"

echo "########## IMAGE: SCORE ##########"
$PY -m eval.runners.score_image_eval --in outputs/eval/v7_galaxy10_raw.json
echo "exit=$?"
echo "### DONE"
