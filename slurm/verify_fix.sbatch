#!/bin/bash
# Re-cache spectra with the wavelength fix, retrain stage1 + stage2, and check the prefix
# actually stopped collapsing. See DIAGNOSIS.md.
#SBATCH --job-name=ab-v7
#SBATCH --account=bfvh-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --mem=240g
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -uo pipefail
cd /work/hdd/bfvh/li26/Astrobridge-Train-v6
export HF_HUB_CACHE=/work/hdd/bfvh/li26/hf-cache
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
PY=/u/li26/venvs/astrobridge-v6/bin/python
ACC=/u/li26/venvs/astrobridge-v6/bin/accelerate
NGPU=$(nvidia-smi -L | wc -l)
echo "### $NGPU GPUs on $(hostname)"

# Old spectra cache was built from unsorted wavelength grids — it must not be reused.
# The cache dir is keyed by encoder spec hash, which did NOT change (same encoder, same config),
# so move it aside explicitly rather than relying on the hash to invalidate it.
if [ -d outputs/cache/spectra ] && [ ! -d outputs/cache/spectra.prefix-bug ]; then
  mv outputs/cache/spectra outputs/cache/spectra.prefix-bug
  echo "### moved the pre-fix spectra cache aside"
fi

echo "########## 1. RE-CACHE SPECTRA (skipped if already fixed) ##########"
if [ ! -f outputs/.slurm_stamps/spectra_recached ]; then
  $PY scripts/02_cache_embeddings.py --modality spectra --device cuda || exit 1
  mkdir -p outputs/.slurm_stamps && touch outputs/.slurm_stamps/spectra_recached
else
  echo "### spectra cache already rebuilt with the wavelength fix — skipping"
fi

echo "########## 2. CACHE SANITY ##########"
$PY diag/cache_health.py || true

# Clear the pre-fix checkpoints ONCE. On a resubmit after a walltime timeout this must NOT
# run again: train/loop.py resumes from the newest step_* dir, and wiping them would restart
# stage 1 from scratch every time the 2h ghx4-interactive cap is hit.
if [ ! -f outputs/.slurm_stamps/verify_cleared ]; then
  rm -rf outputs/checkpoints/stage1 outputs/checkpoints/stage2
  mkdir -p outputs/.slurm_stamps && touch outputs/.slurm_stamps/verify_cleared
  echo "### cleared pre-fix checkpoints (first run)"
else
  echo "### resuming: pre-fix checkpoints already cleared on an earlier submit"
fi
echo "########## 3. STAGE 1 ##########"
# Stamp on completion so a resubmit after the 2h cap goes straight to stage 2. Without this,
# run_training would resume stage 1 from its newest step_* checkpoint and train the remaining
# epochs again — wasted time, and worse, it could move stage1/best AFTER stage 2 already
# initialised from it (configs/stage2.yaml: init_from), leaving the two inconsistent.
if [ ! -f outputs/.slurm_stamps/stage1_done ]; then
  $ACC launch --num_processes "$NGPU" --mixed_precision bf16 scripts/03_train_stage1.py || exit 1
  touch outputs/.slurm_stamps/stage1_done
  $PY diag/prefix_health.py outputs/checkpoints/stage1/best
else
  echo "### stage 1 already finished on an earlier submit — skipping"
fi

echo "########## 4. STAGE 2 ##########"
$ACC launch --num_processes "$NGPU" --mixed_precision bf16 scripts/05_train_stage2.py || exit 1
$PY diag/prefix_health.py outputs/checkpoints/stage2/best

echo "########## 5. CAPTION DIVERSITY ##########"
$PY diag/caption_diversity.py outputs/checkpoints/stage2/best outputs/checkpoints/stage2/best/lora
echo "### DONE"
