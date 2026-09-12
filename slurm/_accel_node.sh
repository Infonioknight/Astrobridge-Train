#!/usr/bin/env bash
# One `accelerate launch` per NODE, started by srun --ntasks-per-node=1 from run_pipeline.sbatch.
#
# Exists as its own file rather than an inline `srun bash -c '...'` because --machine_rank must
# be $SLURM_NODEID expanded INSIDE the task. Expand it in the launching shell and every node
# launches as rank 0, which does not error — it hangs in NCCL rendezvous waiting for peers that
# never identify themselves. A file reads the variable itself and cannot get that wrong.
#
# Usage (via srun): _accel_node.sh <script.py> [args...]
# Environment, set by the caller:
#   ACCELERATE_CONFIG    base yaml (num_processes/num_machines are overridden on the CLI below)
#   ACCEL_NUM_PROCESSES  total ranks across ALL nodes (nodes x gpus-per-node)
#   ACCEL_NUM_MACHINES   node count
#   ACCEL_MAIN_IP        rendezvous host — node 0's hostname
#   ACCEL_MAIN_PORT      rendezvous port
set -euo pipefail

cd "${PROJECT_DIR:?PROJECT_DIR must be set}"

RANK="${SLURM_NODEID:-0}"
echo "[accel] node=$(hostname -s) machine_rank=${RANK}/${ACCEL_NUM_MACHINES} " \
     "world=${ACCEL_NUM_PROCESSES} rendezvous=${ACCEL_MAIN_IP}:${ACCEL_MAIN_PORT}" >&2

# CLI flags beat the yaml, so the committed configs/accelerate_ddp.yaml stays a valid
# single-node config and is not rewritten per run.
exec accelerate launch \
    --config_file "$ACCELERATE_CONFIG" \
    --num_processes "$ACCEL_NUM_PROCESSES" \
    --num_machines "$ACCEL_NUM_MACHINES" \
    --machine_rank "$RANK" \
    --main_process_ip "$ACCEL_MAIN_IP" \
    --main_process_port "$ACCEL_MAIN_PORT" \
    "$@"
