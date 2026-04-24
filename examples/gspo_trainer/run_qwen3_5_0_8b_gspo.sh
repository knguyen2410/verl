#!/usr/bin/env bash
set -xeuo pipefail

# ---------------------------
# Quick-start GSPO with Qwen3.5-0.8B
# ---------------------------
# Example:
#   bash examples/gspo_trainer/run_qwen3_5_0_8b_gspo.sh
#
# Optional overrides:
#   MODEL_PATH=Qwen/Qwen3.5-0.8B-Instruct \
#   DATA_ROOT=$PWD/data \
#   CKPTS_DIR=$PWD/checkpoints/qwen3_5_0_8b_gspo \
#   GPUS_PER_NODE=4 \
#   NNODES=1 \
#   bash examples/gspo_trainer/run_qwen3_5_0_8b_gspo.sh
export MAX_JOBS=2
export NVCC_THREADS=1


export HYDRA_FULL_ERROR=1
export RAY_LOGGING_LEVEL=INFO
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}
PYTHON_BIN=${PYTHON_BIN:-python3}
PYTHON_BIN_RESOLVED=$(command -v "$PYTHON_BIN" || true)
if [[ -n "$PYTHON_BIN_RESOLVED" ]]; then
    PYTHON_BIN_DIR=$(dirname "$PYTHON_BIN_RESOLVED")
    # Keep the selected Python env's bin path visible to all spawned workers.
    if [[ ":$PATH:" != *":$PYTHON_BIN_DIR:"* ]]; then
        export PATH="$PYTHON_BIN_DIR:$PATH"
    fi
    if [[ -x "$PYTHON_BIN_DIR/ninja" ]]; then
        export NINJA="$PYTHON_BIN_DIR/ninja"
        export CMAKE_MAKE_PROGRAM="$PYTHON_BIN_DIR/ninja"
    fi
fi
if [[ -n "${CUDA_PATH:-}" ]] && [[ -n "${CUDA_HOME:-}" ]] && [[ "$CUDA_HOME" != "$CUDA_PATH" ]]; then
    echo "CUDA_HOME and CUDA_PATH differ; setting CUDA_HOME to CUDA_PATH for consistency."
    export CUDA_HOME="$CUDA_PATH"
fi
if [[ "${PYTORCH_CUDA_ALLOC_CONF}" == *"expandable_segments:True"* ]]; then
    echo "PYTORCH_CUDA_ALLOC_CONF contains expandable_segments:True; overriding for vLLM compatibility."
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
fi

# Avoid Ray worker crash from mixed ROCm + CUDA/HIP visibility variables.
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]] && [[ -n "${CUDA_VISIBLE_DEVICES:-}${HIP_VISIBLE_DEVICES:-}" ]]; then
    echo "ROCR_VISIBLE_DEVICES conflicts with CUDA/HIP visibility; unsetting ROCR_VISIBLE_DEVICES."
    unset ROCR_VISIBLE_DEVICES
fi

# MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.5-0.8B}
MODEL_PATH=${MODEL_PATH:-"/fs/ess/PCON0781/data/khoa/Qwen3.5-9B-Phase4.5-step5400"}
DATA_ROOT=${DATA_ROOT:-$PWD/data/gsm8k}
CKPTS_DIR=${CKPTS_DIR:-$PWD/checkpoints/qwen3_5_9b_gspo}
# IMPORTANT: leave TOKENIZER_PATH / VOCAB_DP_PATH unset by default. The base model
# at MODEL_PATH ships its own tokenizer; the previous tokenizerDP path uses a
# DIFFERENT vocabulary (e.g. <|im_end|> id 22202 vs the model's 248044), which
# causes the model to receive garbage token ids and produce garbage outputs
# (reward = 0). Only set them if you really know they match the base checkpoint.
# TOKENIZER_PATH=/fs/ess/PCON0781/data/khoa/tokenizerDP/Qwen3.5-9B
# VOCAB_DP_PATH=/fs/ess/PCON0781/data/khoa/Tensor/Qwen3.5-9B/Qwen3.5-9B-eps1.0.pt
TOKENIZER_PATH=${TOKENIZER_PATH:-}
VOCAB_DP_PATH=${VOCAB_DP_PATH:-}
# Hybrid tuning config:
# - FULL_FINETUNE_FIRST_LAYERS / FULL_FINETUNE_LAST_LAYERS are fully trained
# - Middle layers are LoRA layers (set by LORA_* settings)
FULL_FINETUNE_FIRST_LAYERS=${FULL_FINETUNE_FIRST_LAYERS:-4}
FULL_FINETUNE_LAST_LAYERS=${FULL_FINETUNE_LAST_LAYERS:-4}
# LoRA hyperparameters MUST match the SFT adapter when LORA_ADAPTER_PATH is set
# (verl will load adapter_config.json from that path and ignore LORA_RANK/LORA_ALPHA).
# These defaults match Training/ SFT pipeline (r=64, alpha=128).
LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-128}
# Optional: continue from a pre-trained PEFT LoRA adapter (e.g. an SFT checkpoint).
# When set, verl uses ``PeftModel.from_pretrained`` (preserves the SFT LoRA weights
# instead of randomly initializing) and auto-loads ``pgcode_trainable_params.bin`` /
# ``trainable_parameters.bin`` from the same directory if present.
# Leave empty to train GSPO directly on the base model with a freshly initialized
# LoRA adapter (current default).
LORA_ADAPTER_PATH=${LORA_ADAPTER_PATH:-}
# Explicit LM target_modules (used as a fallback when ``layers_to_transform`` is set
# and the user-facing ``target_modules`` would otherwise be the string "all-linear").
# Mirrors the SFT adapter_config.json so PEFT applies LoRA to LM blocks (not the
# visual encoder).
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-'[in_proj_b,k_proj,o_proj,v_proj,down_proj,gate_proj,q_proj,in_proj_a,in_proj_z,in_proj_qkv,out_proj,up_proj]'}
# Workaround for vLLM LoRA adapter crash with sparse layers_to_transform.
# Keep as true for hybrid first/last full + middle LoRA setups.
LORA_MERGE_FOR_ROLLOUT=${LORA_MERGE_FOR_ROLLOUT:-true}

# Use bf16 model init for FSDP actor/ref to avoid fp32 FlashAttention fallback and long startup stalls.
MODEL_DTYPE=${MODEL_DTYPE:-bf16}

# Stabilize startup and first-step latency in mixed environments.
# These can be overridden from shell if you want max performance.
ACTOR_USE_TORCH_COMPILE=${ACTOR_USE_TORCH_COMPILE:-false}
# CUDA graphs in vLLM give ~10-15% gen speedup. Set to true now that startup is stable.
ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-false}
# Sleep mode frees KV cache before weight sync, which is required for large models
# (e.g. 9B+) on colocated setups where vLLM and the FSDP actor share the same GPUs.
# Without sleep, FSDP summon_full_params can OOM because vLLM holds ~75 GB.
ROLLOUT_ENABLE_SLEEP_MODE=${ROLLOUT_ENABLE_SLEEP_MODE:-true}

PROJECT_NAME=${PROJECT_NAME:-RL-GSPO}
EXP_NAME=${EXP_NAME:-gspo-qwen3_5-9b-first}
RESUME_MODE=${RESUME_MODE:-auto}

# Auto-detect Slurm allocation so a single sbatch -N3 -G16 fully utilises every GPU
# without manual NNODES / GPUS_PER_NODE overrides.
GPUS_PER_NODE=${GPUS_PER_NODE:-${SLURM_GPUS_PER_NODE:-4}}
NNODES=${NNODES:-${SLURM_NNODES:-1}}
NCCL_SINGLE_NODE_SAFE_MODE=${NCCL_SINGLE_NODE_SAFE_MODE:-true}
FORCE_DISABLE_LEGACY_WORKER_MULTI_GPU=${FORCE_DISABLE_LEGACY_WORKER_MULTI_GPU:-true}

# Safety mode to reach first training step on memory-constrained nodes.
# Values: auto | true | false
FIRST_STEP_SAFE_MODE=${FIRST_STEP_SAFE_MODE:-auto}
if [[ "$FIRST_STEP_SAFE_MODE" == "auto" ]]; then
    if [[ "$GPUS_PER_NODE" -eq 1 && "$NNODES" -eq 1 ]]; then
        FIRST_STEP_SAFE_MODE=true
    else
        FIRST_STEP_SAFE_MODE=false
    fi
fi

# Throughput-focused defaults. With 12 GPUs (3x4), TRAIN_BATCH_SIZE=12 yields
# 12*ROLLOUT_N=96 trajectories per step, ~8/GPU — keeps vLLM decode lanes full
# and lets short responses overlap the slowest sequences.
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-12}
# GRPO/GSPO compute group-relative advantages, so ROLLOUT_N must be >= 2.
ROLLOUT_N=${ROLLOUT_N:-8}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1536}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
# In the engine_workers (use_legacy_worker_impl=disable) path, ray_trainer multiplies
# this by rollout.n internally → final mini_batch_size = PPO_MINI_BATCH_SIZE * ROLLOUT_N
# sequences. Setting this == TRAIN_BATCH_SIZE means "one PPO update per rollout" (the
# canonical GRPO/GSPO setting). Setting it larger than TRAIN_BATCH_SIZE makes the
# internal mini-batch exceed the available data and crashes with "X % Y != 0".
# Must be divisible by world_size (NNODES*GPUS_PER_NODE).
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-12}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
# Must be >= MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH so single sequences fit.
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}
# For hybrid FLA models (Qwen3.5), max_num_seqs directly controls recurrent state memory:
# 1024 seqs × 24 FLA layers × 16 heads × 128×128 × 2 bytes = ~12 GB persistent allocation.
# Keep at 64 to reduce FLA states to ~750 MB and leave room during vLLM init profiling.
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-64}
# Limit context window to actual training needs (max_prompt + max_response = 2560).
# Without this, vLLM defaults to model's max_position_embeddings (262144), causing
# unnecessarily large KV cache allocation and profiling overhead.
# Must be >= MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH.
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-8192}
# Reduced from 0.85 to leave GPU headroom for FSDP summon_full_params during weight sync.
# With sleep mode + max_num_seqs=64: model (18 GB) + FLA states (0.75 GB) + KV cache fits
# in 0.50 × 93 = 46.5 GB pool. FSDP summon during update: 18 GB vLLM (sleep) + 18 GB FSDP
# summon = 36 GB, well within 93 GB per GPU.
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.65}
# Must exceed the largest single tensor chunk for async vLLM weight update.
# For 9B models, embed_tokens is 248320×4096×bf16 ≈ 1940 MB — needs bucket > 1940 MB.
UPDATE_WEIGHTS_BUCKET_MB=${UPDATE_WEIGHTS_BUCKET_MB:-2048}
ROLLOUT_UPDATE_INTERVAL=${ROLLOUT_UPDATE_INTERVAL:-4}
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-true}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-true}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-true}
REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-1}
# Default to null = unlimited (driven by trainer.total_epochs over the full dataset).
# Set TOTAL_TRAINING_STEPS=<int> to cap the run.
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-null}

if [[ "$FIRST_STEP_SAFE_MODE" == "true" ]]; then
    echo "FIRST_STEP_SAFE_MODE enabled: applying low-memory launch defaults"
    TRAIN_BATCH_SIZE=1
    ROLLOUT_N=1
    MAX_PROMPT_LENGTH=128
    MAX_RESPONSE_LENGTH=64
    PPO_MINI_BATCH_SIZE=1
    PPO_MICRO_BATCH_SIZE_PER_GPU=1
    PPO_MAX_TOKEN_LEN_PER_GPU=256
    # vLLM (Mamba cache align mode) requires this to be >= internal block size (e.g. 544).
    # Keep it modest but above that threshold for first-step stability.
    ROLLOUT_MAX_NUM_BATCHED_TOKENS=1024
    ROLLOUT_MAX_NUM_SEQS=16
    ROLLOUT_MAX_MODEL_LEN=2048
    ROLLOUT_GPU_MEMORY_UTILIZATION=0.10
    AGENT_NUM_WORKERS=1
    REWARD_NUM_WORKERS=1
    TOTAL_TRAINING_STEPS=1
    if [[ "$RESUME_MODE" == "auto" ]]; then
        RESUME_MODE=disable
    fi
fi

TOTAL_GPUS=$((GPUS_PER_NODE * NNODES))

# Safer NCCL defaults for single-node multi-GPU runs.
# Can be overridden from the shell if you need to tune transport behavior.
if [[ "$NCCL_SINGLE_NODE_SAFE_MODE" == "true" ]] && [[ "$NNODES" -eq 1 ]] && [[ "$TOTAL_GPUS" -gt 1 ]]; then
    export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
    export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
    export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-1}
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
fi

AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-$TOTAL_GPUS}
if [[ "$AGENT_NUM_WORKERS" -lt 1 ]]; then
    AGENT_NUM_WORKERS=1
fi
if [[ "$REWARD_NUM_WORKERS" -lt 1 ]]; then
    REWARD_NUM_WORKERS=1
fi

# FSDP worker normalizes PPO mini batch by world size.
# Keep PPO_MINI_BATCH_SIZE large enough so normalized value stays > 0.
NORMALIZATION_WORLD_SIZE=$TOTAL_GPUS
if [[ "$PPO_MINI_BATCH_SIZE" -lt "$NORMALIZATION_WORLD_SIZE" ]]; then
    echo "Raising PPO_MINI_BATCH_SIZE from $PPO_MINI_BATCH_SIZE to $NORMALIZATION_WORLD_SIZE for multi-GPU normalization safety"
    PPO_MINI_BATCH_SIZE=$NORMALIZATION_WORLD_SIZE
fi

# Actor update dispatch chunks by data-parallel world size; if train batch is
# smaller than TOTAL_GPUS, one mini-batch can reach update_actor with len < dp_size
# and fail with "expecting td with length divisible by chunks".
if [[ "$TOTAL_GPUS" -gt 1 ]] && [[ "$TRAIN_BATCH_SIZE" -lt "$TOTAL_GPUS" ]]; then
    echo "Raising TRAIN_BATCH_SIZE from $TRAIN_BATCH_SIZE to $TOTAL_GPUS to satisfy multi-GPU actor update chunking"
    TRAIN_BATCH_SIZE=$TOTAL_GPUS
fi

# Optional clamp for legacy tiny-batch behavior; keep disabled by default so
# multi-GPU runs can use all workers/GPUs.
CLAMP_AGENT_WORKERS_TO_TRAIN_BATCH=${CLAMP_AGENT_WORKERS_TO_TRAIN_BATCH:-false}
if [[ "$CLAMP_AGENT_WORKERS_TO_TRAIN_BATCH" == "true" ]] && [[ "$TRAIN_BATCH_SIZE" -gt 0 && "$AGENT_NUM_WORKERS" -gt "$TRAIN_BATCH_SIZE" ]]; then
    echo "Clamping AGENT_NUM_WORKERS from $AGENT_NUM_WORKERS to TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE"
    AGENT_NUM_WORKERS=$TRAIN_BATCH_SIZE
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && "$GPUS_PER_NODE" -ge 1 ]]; then
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPUS_PER_NODE - 1)))
    export CUDA_VISIBLE_DEVICES
    echo "CUDA_VISIBLE_DEVICES is unset; defaulting to $CUDA_VISIBLE_DEVICES"
fi

# Set PREP_DATASET=true for first run if parquet files do not exist.
PREP_DATASET=${PREP_DATASET:-false}
ENABLE_PREFLIGHT=${ENABLE_PREFLIGHT:-true}

# ---------------------------
# Dataset selection
# ---------------------------
# By default, train on the union of all available datasets under data/ and
# validate on the union of their eval splits. Override TRAIN_FILES / VAL_FILES
# (space-separated absolute paths) to use a custom mix.
# Default to the templated dataset (prompts wrapped via
# scripts/wrap_prompts_with_template.py). Override DATA_DIR=$PWD/data to use the
# raw prompts.
DATA_DIR=${DATA_DIR:-$PWD/data_templated}
DEFAULT_TRAIN_FILES=(
    "$DATA_DIR/gsm8k/train.parquet"
    "$DATA_DIR/math/train.parquet"
)
DEFAULT_VAL_FILES=(
    "$DATA_DIR/gsm8k/test.parquet"
    "$DATA_DIR/math/test.parquet"
)

if [[ -n "${TRAIN_FILES:-}" ]]; then
    # shellcheck disable=SC2206
    TRAIN_FILES_ARR=($TRAIN_FILES)
else
    TRAIN_FILES_ARR=("${DEFAULT_TRAIN_FILES[@]}")
fi
if [[ -n "${VAL_FILES:-}" ]]; then
    # shellcheck disable=SC2206
    VAL_FILES_ARR=($VAL_FILES)
else
    VAL_FILES_ARR=("${DEFAULT_VAL_FILES[@]}")
fi

# Backward-compat: if TRAIN_FILE / TEST_FILE are explicitly set, they win.
if [[ -n "${TRAIN_FILE:-}" ]]; then
    TRAIN_FILES_ARR=("$TRAIN_FILE")
fi
if [[ -n "${TEST_FILE:-}" ]]; then
    VAL_FILES_ARR=("$TEST_FILE")
fi

if [[ "$PREP_DATASET" == "true" ]]; then
    mkdir -p "$DATA_ROOT"
    "$PYTHON_BIN" examples/data_preprocess/gsm8k.py --local_save_dir "$DATA_ROOT"
fi

for f in "${TRAIN_FILES_ARR[@]}" "${VAL_FILES_ARR[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "Missing dataset file: $f"
        echo "Override TRAIN_FILES / VAL_FILES (space-separated) or PREP_DATASET=true to regenerate GSM8K."
        exit 1
    fi
done

format_hydra_list() {
    local out="["
    local first=1
    for p in "$@"; do
        if [[ $first -eq 1 ]]; then
            out+="'$p'"
            first=0
        else
            out+=",'$p'"
        fi
    done
    out+="]"
    echo "$out"
}

# Preflight: ensure dependencies and CUDA are usable in the selected Python env before launching Ray workers.
if [[ "$ENABLE_PREFLIGHT" == "true" ]]; then
if command -v timeout >/dev/null 2>&1; then
    timeout 45s "$PYTHON_BIN" - <<'PY'
import os
import sys

try:
    import torch
except Exception as e:
    print(f"[preflight] Failed to import torch: {e}")
    sys.exit(1)

try:
    import ray  # noqa: F401
except Exception as e:
    print(f"[preflight] Failed to import ray: {e}")
    print("[preflight] Activate/install the runtime env that has ray before launching.")
    sys.exit(1)

cuda_ver = torch.version.cuda
is_available = torch.cuda.is_available()
device_count = torch.cuda.device_count()

if not is_available:
    print("[preflight] torch.cuda.is_available() is False.")
    print(f"[preflight] torch={torch.__version__}, torch.version.cuda={cuda_ver}, device_count={device_count}")
    print(f"[preflight] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print("[preflight] This environment cannot initialize CUDA, so NCCL workers will fail.")
    print("[preflight] Please use a torch build compatible with your host CUDA driver.")
    sys.exit(1)

if device_count < 1:
    print("[preflight] No visible CUDA devices in this process.")
    print(f"[preflight] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    sys.exit(1)
PY
    preflight_status=$?
    if [[ $preflight_status -eq 124 ]]; then
        echo "[preflight] Timed out while probing CUDA; continuing because ENABLE_PREFLIGHT=true but timed probe expired."
    elif [[ $preflight_status -ne 0 ]]; then
        exit $preflight_status
    fi
else
    "$PYTHON_BIN" - <<'PY'
import os
import sys

try:
    import torch
except Exception as e:
    print(f"[preflight] Failed to import torch: {e}")
    sys.exit(1)

try:
    import ray  # noqa: F401
except Exception as e:
    print(f"[preflight] Failed to import ray: {e}")
    print("[preflight] Activate/install the runtime env that has ray before launching.")
    sys.exit(1)

cuda_ver = torch.version.cuda
is_available = torch.cuda.is_available()
device_count = torch.cuda.device_count()

if not is_available:
    print("[preflight] torch.cuda.is_available() is False.")
    print(f"[preflight] torch={torch.__version__}, torch.version.cuda={cuda_ver}, device_count={device_count}")
    print(f"[preflight] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print("[preflight] This environment cannot initialize CUDA, so NCCL workers will fail.")
    print("[preflight] Please use a torch build compatible with your host CUDA driver.")
    sys.exit(1)

if device_count < 1:
    print("[preflight] No visible CUDA devices in this process.")
    print(f"[preflight] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    sys.exit(1)
PY
fi
fi

mkdir -p "$CKPTS_DIR"

# Guard auto-resume when checkpoint shard world size does not match current run.
if [[ "$RESUME_MODE" == "auto" ]]; then
    latest_step_dir=$(ls -d "$CKPTS_DIR"/global_step_* 2>/dev/null | sort -V | tail -n 1 || true)
    if [[ -n "$latest_step_dir" ]]; then
        missing_shard=false
        for ((rank=0; rank<TOTAL_GPUS; rank++)); do
            expected_actor_shard="$latest_step_dir/actor/model_world_size_${TOTAL_GPUS}_rank_${rank}.pt"
            if [[ ! -f "$expected_actor_shard" ]]; then
                missing_shard=true
                break
            fi
        done
        if [[ "$missing_shard" == "true" ]]; then
            echo "Found checkpoint at $latest_step_dir but actor shards are incomplete for world_size=$TOTAL_GPUS"
            echo "Disabling auto resume for this run to avoid incompatible checkpoint load."
            RESUME_MODE=disable
        fi
    fi
fi

train_files=$(format_hydra_list "${TRAIN_FILES_ARR[@]}")
test_files=$(format_hydra_list "${VAL_FILES_ARR[@]}")
echo "[dataset] train_files=$train_files"
echo "[dataset] val_files=$test_files"

extra_model_args=(
    "actor_rollout_ref.model.full_finetune_first_layers=$FULL_FINETUNE_FIRST_LAYERS"
    "actor_rollout_ref.model.full_finetune_last_layers=$FULL_FINETUNE_LAST_LAYERS"
    "actor_rollout_ref.model.lora_rank=$LORA_RANK"
    "actor_rollout_ref.model.lora_alpha=$LORA_ALPHA"
    "actor_rollout_ref.model.lora.merge=$LORA_MERGE_FOR_ROLLOUT"
    "+actor_rollout_ref.model.lora.target_modules=$LORA_TARGET_MODULES"
)

if [[ -n "$TOKENIZER_PATH" ]]; then
    extra_model_args+=("actor_rollout_ref.model.tokenizer_path=$TOKENIZER_PATH")
fi

# Auto-detect: if MODEL_PATH itself is a PEFT adapter directory, treat it as the
# LoRA adapter source and pass BASE_MODEL_PATH as the actual model path. This keeps
# the user-facing knob (MODEL_PATH) pointing at the checkpoint folder while still
# loading the clean base model under the hood.
MODEL_PATH_FOR_VERL="$MODEL_PATH"
if [[ -f "$MODEL_PATH/adapter_config.json" ]]; then
    if [[ -z "$BASE_MODEL_PATH" ]]; then
        echo "ERROR: MODEL_PATH ($MODEL_PATH) is a PEFT adapter directory but BASE_MODEL_PATH is not set." >&2
        exit 1
    fi
    if [[ ! -f "$BASE_MODEL_PATH/config.json" ]]; then
        echo "ERROR: BASE_MODEL_PATH ($BASE_MODEL_PATH) does not contain config.json." >&2
        exit 1
    fi
    echo "[model] MODEL_PATH is a PEFT adapter dir."
    echo "[model]   base model      -> $BASE_MODEL_PATH"
    echo "[model]   lora adapter    -> $MODEL_PATH"
    MODEL_PATH_FOR_VERL="$BASE_MODEL_PATH"
    LORA_ADAPTER_PATH="$MODEL_PATH"
fi

# Auto-detect: if MODEL_PATH itself is a PEFT adapter directory, treat it as the
# LoRA adapter source and pass BASE_MODEL_PATH as the actual model path. This keeps
# the user-facing knob (MODEL_PATH) pointing at the checkpoint folder while still
# loading the clean base model under the hood.
MODEL_PATH_FOR_VERL="$MODEL_PATH"
if [[ -f "$MODEL_PATH/adapter_config.json" ]]; then
    if [[ -z "$BASE_MODEL_PATH" ]]; then
        echo "ERROR: MODEL_PATH ($MODEL_PATH) is a PEFT adapter directory but BASE_MODEL_PATH is not set." >&2
        exit 1
    fi
    if [[ ! -f "$BASE_MODEL_PATH/config.json" ]]; then
        echo "ERROR: BASE_MODEL_PATH ($BASE_MODEL_PATH) does not contain config.json." >&2
        exit 1
    fi
    echo "[model] MODEL_PATH is a PEFT adapter dir."
    echo "[model]   base model      -> $BASE_MODEL_PATH"
    echo "[model]   lora adapter    -> $MODEL_PATH"
    MODEL_PATH_FOR_VERL="$BASE_MODEL_PATH"
    LORA_ADAPTER_PATH="$MODEL_PATH"
fi

if [[ -n "$LORA_ADAPTER_PATH" ]]; then
    extra_model_args+=("+actor_rollout_ref.model.lora_adapter_path=$LORA_ADAPTER_PATH")
fi

if [[ -n "$VOCAB_DP_PATH" ]]; then
    extra_model_args+=("actor_rollout_ref.model.vocab_dp_path=$VOCAB_DP_PATH")
fi

# Multi-GPU async rollout is more stable with the new worker implementation.
final_cli_args=("$@")
if [[ "$FORCE_DISABLE_LEGACY_WORKER_MULTI_GPU" == "true" ]] && [[ "$TOTAL_GPUS" -gt 1 ]]; then
    filtered_cli_args=()
    for arg in "${final_cli_args[@]}"; do
        if [[ "$arg" == "trainer.use_legacy_worker_impl=enable" ]]; then
            echo "Removing trainer.use_legacy_worker_impl=enable for multi-GPU stability"
            continue
        fi
        filtered_cli_args+=("$arg")
    done
    final_cli_args=("trainer.use_legacy_worker_impl=disable" "+actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend=triton" "+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_flashinfer_autotune=false" "+actor_rollout_ref.rollout.engine_kwargs.vllm.distributed_executor_backend=uni" "${filtered_cli_args[@]}")
fi

"$PYTHON_BIN" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.prompt_key=prompt \
    data.return_raw_chat=True \
    data.truncation='error' \
    data.filter_overlong_prompts=True \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.use_torch_compile=$ACTOR_USE_TORCH_COMPILE \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=3e-4 \
    actor_rollout_ref.actor.clip_ratio_high=4e-4 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=$ACTOR_PARAM_OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$ACTOR_OPTIMIZER_OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.model_dtype=$MODEL_DTYPE \
    actor_rollout_ref.ref.fsdp_config.param_offload=$REF_PARAM_OFFLOAD \
    actor_rollout_ref.ref.fsdp_config.model_dtype=$MODEL_DTYPE \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.agent.num_workers=$AGENT_NUM_WORKERS \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=$ROLLOUT_ENFORCE_EAGER \
    +actor_rollout_ref.rollout.enable_sleep_mode=$ROLLOUT_ENABLE_SLEEP_MODE \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=$UPDATE_WEIGHTS_BUCKET_MB \
    actor_rollout_ref.rollout.max_num_batched_tokens=$ROLLOUT_MAX_NUM_BATCHED_TOKENS \
    actor_rollout_ref.rollout.max_num_seqs=$ROLLOUT_MAX_NUM_SEQS \
    actor_rollout_ref.rollout.max_model_len=$ROLLOUT_MAX_MODEL_LEN \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    reward.reward_manager.name=dapo \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=False \
    +reward.reward_kwargs.overlong_buffer_cfg.len=4096 \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward.reward_kwargs.max_resp_len=4096 \
    reward.num_workers=$REWARD_NUM_WORKERS \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.n_gpus_per_node="$GPUS_PER_NODE" \
    trainer.nnodes="$NNODES" \
    trainer.val_before_train=False \
    trainer.test_freq=100 \
    trainer.save_freq=100 \
    trainer.total_epochs=3 \
    trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
    trainer.rollout_update_interval=$ROLLOUT_UPDATE_INTERVAL \
    trainer.default_local_dir="$CKPTS_DIR" \
    trainer.resume_mode="$RESUME_MODE" \
    trainer.log_val_generations=8 \
    "${extra_model_args[@]}" \
    "${final_cli_args[@]}"