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

export HYDRA_FULL_ERROR=1
export RAY_LOGGING_LEVEL=INFO
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}
if [[ "${PYTORCH_CUDA_ALLOC_CONF}" == *"expandable_segments:True"* ]]; then
    echo "PYTORCH_CUDA_ALLOC_CONF contains expandable_segments:True; overriding for vLLM compatibility."
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
fi

# Avoid Ray worker crash from mixed ROCm + CUDA/HIP visibility variables.
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]] && [[ -n "${CUDA_VISIBLE_DEVICES:-}${HIP_VISIBLE_DEVICES:-}" ]]; then
    echo "ROCR_VISIBLE_DEVICES conflicts with CUDA/HIP visibility; unsetting ROCR_VISIBLE_DEVICES."
    unset ROCR_VISIBLE_DEVICES
fi

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.5-0.8B}
DATA_ROOT=${DATA_ROOT:-$PWD/data/gsm8k}
CKPTS_DIR=${CKPTS_DIR:-$PWD/checkpoints/qwen3_5_0_8b_gspo}
TOKENIZER_PATH=/project/phan/codellama/tokenizerDP/Qwen3.5-0.8B
VOCAB_DP_PATH=/project/phan/codellama/Tensor/Qwen3.5-0.8B/Qwen3.5-0.8B-pertubed.pt

# Hybrid tuning config:
# - FULL_FINETUNE_FIRST_LAYERS / FULL_FINETUNE_LAST_LAYERS are fully trained
# - Middle layers are LoRA layers (set by LORA_* settings)
TOKENIZER_PATH=${TOKENIZER_PATH:-$TOKENIZER_PATH}
VOCAB_DP_PATH=${VOCAB_DP_PATH:-$VOCAB_DP_PATH}
FULL_FINETUNE_FIRST_LAYERS=${FULL_FINETUNE_FIRST_LAYERS:-4}
FULL_FINETUNE_LAST_LAYERS=${FULL_FINETUNE_LAST_LAYERS:-4}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-16}
# Workaround for vLLM LoRA adapter crash with sparse layers_to_transform.
# Keep as true for hybrid first/last full + middle LoRA setups.
LORA_MERGE_FOR_ROLLOUT=${LORA_MERGE_FOR_ROLLOUT:-true}

# Stabilize startup and first-step latency in mixed environments.
# These can be overridden from shell if you want max performance.
ACTOR_USE_TORCH_COMPILE=${ACTOR_USE_TORCH_COMPILE:-false}
ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-true}
ROLLOUT_ENABLE_SLEEP_MODE=${ROLLOUT_ENABLE_SLEEP_MODE:-false}

PROJECT_NAME=${PROJECT_NAME:-RL-GSPO}
EXP_NAME=${EXP_NAME:-gspo-qwen3_5-0_8b-first}

GPUS_PER_NODE=${GPUS_PER_NODE:-2}
NNODES=${NNODES:-1}

# Memory-safe defaults (override from shell for higher throughput).
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
ROLLOUT_N=${ROLLOUT_N:-1}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1536}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-2}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-4096}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-2048}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.15}
UPDATE_WEIGHTS_BUCKET_MB=${UPDATE_WEIGHTS_BUCKET_MB:-256}
ROLLOUT_UPDATE_INTERVAL=${ROLLOUT_UPDATE_INTERVAL:-4}
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-true}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-true}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-true}

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && "$GPUS_PER_NODE" -ge 2 ]]; then
    export CUDA_VISIBLE_DEVICES=0,1
    echo "CUDA_VISIBLE_DEVICES is unset; defaulting to $CUDA_VISIBLE_DEVICES"
fi

# Set PREP_DATASET=true for first run if parquet files do not exist.
PREP_DATASET=${PREP_DATASET:-false}

TRAIN_FILE=${TRAIN_FILE:-$DATA_ROOT/train.parquet}
TEST_FILE=${TEST_FILE:-$DATA_ROOT/test.parquet}

if [[ "$PREP_DATASET" == "true" ]]; then
    mkdir -p "$DATA_ROOT"
    python examples/data_preprocess/gsm8k.py --local_save_dir "$DATA_ROOT"
fi

if [[ ! -f "$TRAIN_FILE" || ! -f "$TEST_FILE" ]]; then
    echo "Missing dataset files."
    echo "Expected: $TRAIN_FILE and $TEST_FILE"
    echo "Set PREP_DATASET=true to auto-generate GSM8K parquet files."
    exit 1
fi

# Preflight: ensure CUDA is usable in the current Python env before launching Ray workers.
python3 - <<'PY'
import os
import sys

try:
    import torch
except Exception as e:
    print(f"[preflight] Failed to import torch: {e}")
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

mkdir -p "$CKPTS_DIR"

train_files="['$TRAIN_FILE']"
test_files="['$TEST_FILE']"

extra_model_args=(
    "actor_rollout_ref.model.tokenizer_path=$TOKENIZER_PATH"
    "actor_rollout_ref.model.full_finetune_first_layers=$FULL_FINETUNE_FIRST_LAYERS"
    "actor_rollout_ref.model.full_finetune_last_layers=$FULL_FINETUNE_LAST_LAYERS"
    "actor_rollout_ref.model.lora_rank=$LORA_RANK"
    "actor_rollout_ref.model.lora_alpha=$LORA_ALPHA"
    "actor_rollout_ref.model.lora.merge=$LORA_MERGE_FOR_ROLLOUT"
)

if [[ -n "$VOCAB_DP_PATH" ]]; then
    extra_model_args+=("actor_rollout_ref.model.vocab_dp_path=$VOCAB_DP_PATH")
fi

python3 -m verl.trainer.main_ppo \
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
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
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
    actor_rollout_ref.ref.fsdp_config.param_offload=$REF_PARAM_OFFLOAD \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=$ROLLOUT_ENFORCE_EAGER \
    +actor_rollout_ref.rollout.enable_sleep_mode=$ROLLOUT_ENABLE_SLEEP_MODE \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=$UPDATE_WEIGHTS_BUCKET_MB \
    actor_rollout_ref.rollout.max_num_batched_tokens=$ROLLOUT_MAX_NUM_BATCHED_TOKENS \
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
    trainer.logger='["console","wandb"]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.n_gpus_per_node="$GPUS_PER_NODE" \
    trainer.nnodes="$NNODES" \
    trainer.val_before_train=False \
    trainer.test_freq=10 \
    trainer.save_freq=20 \
    trainer.total_epochs=3 \
    trainer.total_training_steps=100 \
    trainer.rollout_update_interval=$ROLLOUT_UPDATE_INTERVAL \
    trainer.default_local_dir="$CKPTS_DIR" \
    trainer.resume_mode=auto \
    trainer.log_val_generations=2 \
    "${extra_model_args[@]}" \
    "$@"