#!/bin/bash
# =============================================================================
#  AstroBridge captioner — full README pipeline as one batch job.
#
#  Runs README steps 1 through "Sharing the trained model", in order:
#    1. install                                   (uv venv + editable install)
#    2. HF auth check                             (non-interactive: token must already exist)
#    3. make check-access
#    4. make test
#    5. make manifest / make captions / lightcurve embedding cache (CPU)
#    6. make cache / stage1 / eval / stage2 / eval                 (GPU)
#    7. collect metrics + groundedness reports into outputs/publish_files/
#    8. make publish  +  upload those metrics files alongside the model
#
#  Submit from the repo root — #SBATCH --output paths are relative to the submission
#  directory, not to PROJECT_DIR:
#      sbatch slurm/run_pipeline.sbatch
#  or override the upload target without editing this file:
#      sbatch --export=ALL,HF_REPO_ID=UniverseTBD/astrobridge-model-v4_qwen \
#             slurm/run_pipeline.sbatch
#
#  The job is resumable: each completed step drops a stamp in
#  outputs/.slurm_stamps/, so re-submitting after a timeout skips finished work
#  and stage1/stage2 pick up from their own latest checkpoint (see
#  src/captioner/train/loop.py). Delete a stamp to force that step to rerun.
# =============================================================================
#SBATCH --job-name=astrobridge-v6
#SBATCH --account=bfvh-dtai-gh
#SBATCH --partition=ghx4-interactive
# 1 node. The 2-node/8-rank run (3131676) measured ~1.9 min/epoch vs v5's ~3.5 on 4 GPUs
# -- 1.8x for 2x the hardware -- and the doubled effective batch pushed stage-1 val_loss to
# ~1.12 against v5's 1.0589. It also waits longer in the queue. Set nodes=2 again only to
# exercise the multi-node path in slurm/_accel_node.sh, which is kept and still works.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --mem=240g
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
#  ALLOCATION: 4 GPUs on ghx4-interactive. That partition caps walltime at 2h (MaxTime
#  02:00:00), which is the binding constraint here, not the GPU count. Reference from the
#  1-GPU v4 run: 3h11m end to end — stage1 1h44m, stage2 26m, and the two groundedness evals
#  27m each. stage1/stage2 shard across GPUs via accelerate (num_processes is set from
#  `nvidia-smi -L` below), so on 4 GPUs they should fall to roughly 30m and 7m; the evals do
#  NOT shard and stay ~27m each. That puts the full pipeline near 1h40m — inside 2h, but not
#  by much, and the two evals are over half of what is left.
#
#  If it does hit the wall: run_step stamps in outputs/.slurm_stamps make a resubmit resume at
#  STAGE granularity, so completed stages are skipped. Note the granularity is the stage, not
#  the epoch — a timeout part-way through stage1 loses that stage's progress. Submitting with
#  --export=ALL,SKIP_EVAL=1 drops both groundedness gates and brings the run comfortably inside
#  the cap, at the cost of the groundedness report in the model card.

set -euo pipefail

# =============================================================================
#  SET ME — where the trained model + metrics get uploaded
# =============================================================================
# The HF Hub repo id to publish to, i.e. everything after huggingface.co/.
# Defaults to this branch's target (UniverseTBD/astrobridge-model-v4_qwen); override with
# `sbatch --export=ALL,HF_REPO_ID=...`, or set it empty to run everything but skip the upload.
HF_REPO_ID="${HF_REPO_ID:-UniverseTBD/astrobridge-model-v6}"

# Publish privately (default) or publicly.
HF_PUBLIC="${HF_PUBLIC:-0}"

# -----------------------------------------------------------------------------
#  Optional knobs
# -----------------------------------------------------------------------------
PROJECT_DIR="${PROJECT_DIR:-/work/hdd/bfvh/li26/Astrobridge-Train-v6}"
# The venv goes on /u, NOT in $PROJECT_DIR on /work. Measured on this node: `import torch`
# takes 69s from a venv on /work (Lustre) versus 5-12s from /u — Lustre is pathological for
# the thousands-of-small-files access pattern Python imports generate. The full test suite
# could not finish in 6 minutes from /work and runs in 96s from /u. That penalty is paid by
# every stage AND by every accelerate worker, so on 4 GPUs it is 4x per training launch —
# real money against a 2h walltime cap. Model/data blobs stay on /work (HF_HUB_CACHE below):
# those are few and large, which Lustre handles fine.
VENV_DIR="${VENV_DIR:-/u/li26/venvs/astrobridge-v6}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.11.9}"

# Model/dataset downloads land here, NOT on /u. This is load-bearing, not tidiness: /u/$USER has
# a 103G hard quota with ~35G free, while Qwen3.5-9B (~18G) and gemma-4-12B (~24G) together need
# ~42G — running both branches' jobs against the $HOME cache would hit the quota mid-download.
# /work/hdd/bfvh has ~800G free. Only the *cache* moves; the HF token stays in $HF_HOME, so auth
# keeps working. Already seeded with the datasets + AION, so nothing re-downloads unnecessarily.
export HF_HUB_CACHE="${HF_HUB_CACHE:-/work/hdd/bfvh/$USER/hf-cache}"

# =============================================================================
#  Environment
# =============================================================================
cd "$PROJECT_DIR"

# `module` isn't always defined in a non-interactive shell; source Lmod first if so.
if ! command -v module >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "${LMOD_PKG:-/opt/cray/pe/lmod/lmod}/init/bash"
fi
set +u                      # Lmod's shell function trips over `set -u`
module load "$PYTHON_MODULE"
set -u

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

STAMP_DIR="outputs/.slurm_stamps"
PUBLISH_FILES="outputs/publish_files"
mkdir -p "$STAMP_DIR" "$PUBLISH_FILES" outputs/eval

# run_step <stamp-name> <command...>
# Stamps only on success, so a step killed by the wall clock reruns next submit.
run_step() {
    local name="$1"; shift
    if [[ -f "$STAMP_DIR/$name" ]]; then
        echo "=== [skip] $name — already completed"
        return 0
    fi
    echo "=== [run ] $name  ($(date '+%F %T'))"
    "$@"
    touch "$STAMP_DIR/$name"
    echo "=== [done] $name  ($(date '+%F %T'))"
}

JOB_ID="${SLURM_JOB_ID:-local}"
echo "Job $JOB_ID on $(hostname)"
nvidia-smi -L || true

# =============================================================================
#  README step 1 — install
# =============================================================================
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "=== creating venv at $VENV_DIR"
    uv venv --python "$(command -v python3)" "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Deliberately NOT stamped. It takes ~6s and is idempotent, whereas a stamp would skip the
# reinstall after a dependency changes in pyproject.toml — leaving a re-submitted job running
# against a stale environment and failing later for a reason the log doesn't explain.
uv pip install -e ".[dev]"

# =============================================================================
#  README step 2 — Hugging Face auth
#  `hf auth login` is interactive and can't run in a batch job. The token must
#  already be cached (it is, under ~/.cache/huggingface/token) or passed in as
#  HF_TOKEN. This just fails fast and loudly if it isn't.
# =============================================================================
python - <<'PY'
import sys
from huggingface_hub import whoami
try:
    print(f"Hugging Face auth OK — logged in as {whoami()['name']}")
except Exception as e:
    sys.exit(
        f"Hugging Face auth FAILED: {type(e).__name__}: {e}\n"
        "Run `hf auth login` on the login node before submitting, or submit with\n"
        "  sbatch --export=ALL,HF_TOKEN=hf_xxx slurm/run_pipeline.sbatch"
    )
PY

# =============================================================================
#  README step 3 — check access
# =============================================================================
run_step 03_check_access make check-access

# =============================================================================
#  README step 4 — tests (expects "120 passed"; nothing below is trustworthy
#  until this is green, so the `set -e` abort here is deliberate)
# =============================================================================
run_step 04_test make test

# =============================================================================
#  README step 5 — build the data (no GPU needed)
# =============================================================================
run_step 05a_manifest make manifest
run_step 05b_captions make captions
run_step 05c_cache_lightcurve python scripts/02_cache_embeddings.py --modality lightcurve --device cpu

# =============================================================================
#  README step 6 — GPU work
# =============================================================================
# Match accelerate's process count to the GPUs actually allocated to this job
# instead of the 4 hardcoded in configs/accelerate_ddp.yaml. The Makefile reads
# ACCELERATE_CONFIG, so stage1/stage2 pick this up with no other changes.
NGPU="$(nvidia-smi -L | wc -l)"                    # GPUs on THIS node
NNODES="${SLURM_JOB_NUM_NODES:-1}"
WORLD=$(( NGPU * NNODES ))                         # total ranks across the allocation
ACCELERATE_CONFIG="outputs/accelerate_job_${JOB_ID}.yaml"
sed "s|^num_processes:.*|num_processes: ${WORLD}|" configs/accelerate_ddp.yaml > "$ACCELERATE_CONFIG"
export ACCELERATE_CONFIG

# run_training <stamp> <script.py>
# One node: `accelerate launch` directly, exactly as before.
# Several nodes: srun one launcher PER NODE, each with its own --machine_rank, all pointed at
# node 0 for rendezvous. accelerate then owns the NGPU local ranks on each node, so the world is
# NNODES x NGPU. The training code is already rank-aware (is_main_process guards and
# wait_for_everyone barriers in train/loop.py), so nothing above the launcher changes.
if [ "$NNODES" -gt 1 ]; then
    export ACCEL_NUM_PROCESSES="$WORLD"
    export ACCEL_NUM_MACHINES="$NNODES"
    export ACCEL_MAIN_IP="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
    # Derive the port from the job id so two of these can never collide on a shared node.
    export ACCEL_MAIN_PORT=$(( 29500 + (${SLURM_JOB_ID:-0} % 1000) ))
    export PROJECT_DIR
    run_training() {
        run_step "$1" srun --ntasks="$NNODES" --ntasks-per-node=1 \
            "$PROJECT_DIR/slurm/_accel_node.sh" "$2"
    }
    echo "=== accelerate: ${NNODES} node(s) x ${NGPU} GPU = ${WORLD} ranks, rendezvous ${ACCEL_MAIN_IP}:${ACCEL_MAIN_PORT}"
else
    run_training() { run_step "$1" make "${2}" ACCELERATE_CONFIG="$ACCELERATE_CONFIG"; }
    echo "=== accelerate configured for ${NGPU} GPU(s) on one node: $ACCELERATE_CONFIG"
fi

run_step 06a_cache  make cache
if [ "$NNODES" -gt 1 ]; then run_training 06b_stage1 scripts/03_train_stage1.py
else run_training 06b_stage1 stage1; fi

# scripts/04_eval.py rather than `make eval`: the make target can't pass --out,
# and both stages would otherwise overwrite the same
# outputs/eval/groundedness_report.json. Stage 2's eval keeps that default path
# because scripts/06_publish_model.py reads exactly it when building the card.
#
# Neither eval passes --lora-dir, matching the README. (--lora-dir expects a
# PEFT-format directory; the stage2 checkpoint's lora/ holds a raw adapter.pt,
# so pointing it there would fail. Publishing converts it to PEFT format.)
run_step 06c_eval_stage1 python scripts/04_eval.py \
    --checkpoint-dir outputs/checkpoints/stage1/best \
    --out outputs/eval/groundedness_report_stage1.json

if [ "$NNODES" -gt 1 ]; then run_training 06d_stage2 scripts/05_train_stage2.py
else run_training 06d_stage2 stage2; fi

run_step 06e_eval_stage2 python scripts/04_eval.py \
    --checkpoint-dir outputs/checkpoints/stage2/best \
    --out outputs/eval/groundedness_report.json

# =============================================================================
#  Step 7 — collect metrics + groundedness reports
#
#  Filenames mirror UniverseTBD/astrobridge-model-v3_qwen, which carries
#  metrics_stage1.jsonl, metrics_stage2.jsonl and groundedness_report.json
#  next to adapter_model.safetensors / middle.pt / README.md.
#
#  metrics.jsonl is written per-run by src/captioner/train/loop.py: one JSON
#  line per optimizer step (split="train") plus a "train_epoch"/"val" pair per
#  epoch. Read with pd.read_json(path, lines=True).
# =============================================================================
collect_metrics() {
    cp outputs/checkpoints/stage1/metrics.jsonl "$PUBLISH_FILES/metrics_stage1.jsonl"
    cp outputs/checkpoints/stage2/metrics.jsonl "$PUBLISH_FILES/metrics_stage2.jsonl"
    # Copy only what exists: a bare cp of a missing report aborts the run under `set -e` at
    # the last step before publishing.
    for r in groundedness_report.json groundedness_report_stage1.json; do
        if [[ -f "outputs/eval/$r" ]]; then
            cp "outputs/eval/$r" "$PUBLISH_FILES/$r"
        else
            echo "    (no outputs/eval/$r — skipping)"
        fi
    done
    echo "Collected into $PUBLISH_FILES:"
    ls -la "$PUBLISH_FILES"
}
# Not stamped — cheap, and always reflects the newest run.
echo "=== [run ] 07_collect_metrics"
collect_metrics

# =============================================================================
#  Step 8 — publish (README "Sharing the trained model")
# =============================================================================
if [[ -z "$HF_REPO_ID" ]]; then
    echo "=== HF_REPO_ID is empty — skipping upload."
    echo "    Everything is on disk; publish later with:"
    echo "      make publish CKPT=outputs/checkpoints/stage2/best REPO=<org>/<name>"
    echo "      hf upload <org>/<name> $PUBLISH_FILES/metrics_stage1.jsonl metrics_stage1.jsonl"
    exit 0
fi

PUBLISH_ARGS=(--checkpoint-dir outputs/checkpoints/stage2/best --repo-id "$HF_REPO_ID")
if [[ "$HF_PUBLIC" == "1" ]]; then
    PUBLISH_ARGS+=(--public)
fi

# scripts/06_publish_model.py directly rather than `make publish`, so --public
# can be threaded through; identical otherwise.
run_step 08a_publish python scripts/06_publish_model.py "${PUBLISH_ARGS[@]}"

# The publish script uploads the adapter + middle.pt + model card. The metrics
# files aren't part of it, so they go up in a second pass.
echo "=== [run ] 08b_upload_metrics"
# `hf upload` raises FileNotFoundError on a missing path, which under `set -e` fails the run
# AFTER the model itself has already uploaded. Upload only what exists.
for f in metrics_stage1.jsonl metrics_stage2.jsonl groundedness_report.json groundedness_report_stage1.json; do
    if [[ ! -f "$PUBLISH_FILES/$f" ]]; then
        echo "    (no $f — skipping)"
        continue
    fi
    echo "    uploading $f"
    hf upload "$HF_REPO_ID" "$PUBLISH_FILES/$f" "$f"
done

echo "=== Pipeline complete: https://huggingface.co/$HF_REPO_ID"
