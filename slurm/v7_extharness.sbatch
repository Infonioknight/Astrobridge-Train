#!/bin/bash
#SBATCH --job-name=ab-v7ext
#SBATCH --account=bfvh-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=32 --gpus-per-node=1
#SBATCH --mem=200g --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -uo pipefail
E=/work/hdd/bfvh/li26/Astrobridge-Train-extraction-from-captions
cd $E
export HF_HUB_CACHE=/work/hdd/bfvh/li26/hf-cache PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
# OUR fixed captioner package must win over the clone's stale copy: v7's middle.pt carries
# adapter.out_norm/log_scale, which the old FusionStack has no slots for, and the trim helper
# only exists in ours. `evals` still resolves from the clone.
export PYTHONPATH=/work/hdd/bfvh/li26/Astrobridge-Train-v6/src:$E/src
/u/li26/venvs/astrobridge-v6/bin/python eval_scripts/generate_captions.py \
    --responder eval_configs/caption_responders/astrobridge_v7.yaml \
    --output eval_results/cached_captions/astrobridge_v7_captions.jsonl
echo "### DONE exit=$?"
