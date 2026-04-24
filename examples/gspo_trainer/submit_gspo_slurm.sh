#!/usr/bin/env bash
# ===========================================================================
# Slurm submission script for GSPO training (Qwen3.5-9B)
#
# Usage:
#   sbatch examples/gspo_trainer/submit_gspo_slurm.sh
#
# Optional overrides via env vars:
#   TOTAL_TRAINING_STEPS=5 sbatch examples/gspo_trainer/submit_gspo_slurm.sh
# ===========================================================================
#SBATCH --job-name=gspo-qwen3_5-9b
#SBATCH --partition=condo-haiphan
#SBATCH --account=PCON0781
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=6-00:00:00
#SBATCH --output=/fs/ess/PCON0781/data/khoa/verl/logs/gspo-%j.out
#SBATCH --error=/fs/ess/PCON0781/data/khoa/verl/logs/gspo-%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
VERL_DIR=/fs/ess/PCON0781/data/khoa/verl
PYTHON_BIN=/users/PCON0781/nkhoa2410/anaconda3/envs/verl/bin/python3
export PATH="/users/PCON0781/nkhoa2410/anaconda3/envs/verl/bin:$PATH"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
export MAX_JOBS=4
export NVCC_THREADS=2
export HYDRA_FULL_ERROR=1
export RAY_LOGGING_LEVEL=INFO
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
# Unset ROCm visibility variable in the batch shell — worker nodes have it
# set by Slurm and it conflicts with CUDA_VISIBLE_DEVICES that Ray injects
# into each actor. We also strip it inside every srun (see below) because
# Slurm re-adds it per node. Do NOT set RAY_EXPERIMENTAL_NOSET_*; we want
# Ray to assign CUDA_VISIBLE_DEVICES per actor based on its slot.
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES

# ---------------------------------------------------------------------------
# Create log dir
# ---------------------------------------------------------------------------
mkdir -p /fs/ess/PCON0781/data/khoa/verl/logs

echo "=========================================================="
echo "Job ID     : $SLURM_JOB_ID"
echo "Nodes      : $SLURM_JOB_NODELIST"
echo "GPUs/node  : $SLURM_GPUS_PER_NODE"
echo "Num nodes  : $SLURM_NNODES"
echo "=========================================================="

# ---------------------------------------------------------------------------
# Discover head node IP
# ---------------------------------------------------------------------------
nodes_array=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
head_node="${nodes_array[0]}"
head_node_ip=$(srun --nodes=1 --ntasks=1 --mem=0 --export=ALL -w "$head_node" hostname --ip-address)

# Prefer IPv4 if multiple addresses returned
if [[ "$head_node_ip" == *" "* ]]; then
    IFS=' ' read -ra ADDR <<<"$head_node_ip"
    for addr in "${ADDR[@]}"; do
        if [[ "$addr" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
            head_node_ip="$addr"
            break
        fi
    done
fi

RAY_PORT=6379
RAY_HEAD_ADDR="${head_node_ip}:${RAY_PORT}"
export RAY_HEAD_ADDR
echo "Ray head: $RAY_HEAD_ADDR on node $head_node"

# ---------------------------------------------------------------------------
# Start Ray head
# ---------------------------------------------------------------------------
echo "Starting Ray HEAD on $head_node ..."
srun --nodes=1 --ntasks=1 --mem=0 --export=ALL -w "$head_node" \
    bash -c "unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES; exec '$PYTHON_BIN' -m ray.scripts.scripts start \
        --head \
        --node-ip-address='$head_node_ip' \
        --port='$RAY_PORT' \
        --num-cpus='${SLURM_CPUS_PER_TASK}' \
        --num-gpus='${SLURM_GPUS_PER_NODE}' \
        --block" &

sleep 15

# ---------------------------------------------------------------------------
# Start Ray workers on remaining nodes
# ---------------------------------------------------------------------------
worker_num=$(( SLURM_JOB_NUM_NODES - 1 ))
for (( i=1; i<=worker_num; i++ )); do
    node_i="${nodes_array[$i]}"
    echo "Starting Ray WORKER $i on $node_i ..."
    srun --nodes=1 --ntasks=1 --mem=0 --export=ALL -w "$node_i" \
        bash -c "unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES; exec '$PYTHON_BIN' -m ray.scripts.scripts start \
            --address='$RAY_HEAD_ADDR' \
            --num-cpus='${SLURM_CPUS_PER_TASK}' \
            --num-gpus='${SLURM_GPUS_PER_NODE}' \
            --block" &
    sleep 5
done

# Give workers time to register
sleep 20

echo "Ray cluster status:"
srun --overlap --nodes=1 --ntasks=1 --mem=0 --export=ALL -w "$head_node" \
    "$PYTHON_BIN" -m ray.scripts.scripts status || true

# ---------------------------------------------------------------------------
# Launch training from the head node
# ---------------------------------------------------------------------------
echo "Launching GSPO training ..."
cd "$VERL_DIR"

# Workarounds for vLLM hybrid (Mamba/FLA) Qwen3.5 + sleep-mode incompatibility:
# vLLM EngineCore crashes on step 2 with "CUDA error: invalid argument" inside
# the recurrent-state cache copy after waking from sleep. Keep sleep disabled.
# With sleep off + LoRA merge, FSDP must summon the full unsharded base model
# (~26 GB) on GPU 0 BEFORE pushing weights to vLLM, so vLLM cannot exceed ~55%
# of the 93 GB H100 (=51 GB) without OOMing during merge. 0.50 leaves headroom
# for activation spikes; 0.55 is the maximum safe value.
export ROLLOUT_ENABLE_SLEEP_MODE=false
export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.50}

# Forward the FULL Slurm allocation to the run script. The inner `srun --nodes=1`
# below would otherwise reset SLURM_NNODES=1 in the child environment, causing
# verl to see only 1 node even when the Ray cluster spans all of them.
export NNODES="${NNODES:-${SLURM_JOB_NUM_NODES:-1}}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-${SLURM_GPUS_PER_NODE:-4}}"

PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks=1 --mem=0 --export=ALL -w "$head_node" \
    bash -c "unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES; exec bash examples/gspo_trainer/run_qwen3_5_0_8b_gspo.sh \
        trainer.use_legacy_worker_impl=disable" \
        2>&1 | tee "/fs/ess/PCON0781/data/khoa/verl/logs/gspo-${SLURM_JOB_ID}-train.log"

echo "Training finished."
