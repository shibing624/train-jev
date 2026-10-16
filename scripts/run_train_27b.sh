#!/usr/bin/env bash
# JEV recipe on Qwen3.8-27B (the backbone of autotrust/JEV-27B / JEV-27B-VL). Same code and hyper-parameters as 4B;
# only memory settings differ. LoRA 108.8M + head 123k trainable (identical to the JEV-27B card).
#
# Completed reference: 4x H20, gradient checkpointing, micro_tokens 16384,
# 5,123 steps, 23.24 s/step, 74.9 GB peak per rank, 33.07 h including periodic validation.
#
#   GPUS=0,1,2,3 bash scripts/run_train_27b.sh
set -euo pipefail
cd "$(dirname "$0")/.."

GPUS=${GPUS:-0,1,2,3}
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)
BASE=${BASE:-models/Qwen3.8-27B}
DATA=${DATA:-data/jev-distill-corpus-v3}
OUT=${OUT:-runs/jev-27b}
MAX_STEPS=${MAX_STEPS:-0}
MASTER_PORT=${MASTER_PORT:-29550}
EVAL_EVERY=${EVAL_EVERY:-500}
VAL_N=${VAL_N:-4000}
SEED=${SEED:-0}
STEPS=${STEPS:-0}
MICRO_TOKENS=${MICRO_TOKENS:-16384}
SAVE_STATE_EVERY=${SAVE_STATE_EVERY:-100}
resume_args=()
if [[ "${RESUME:-0}" == "1" ]]; then
    if [[ ! -f "$OUT/state/meta.json" && ! -f "$OUT/state.old/meta.json" ]]; then
        echo "No resumable training state in $OUT/state or $OUT/state.old" >&2
        exit 2
    fi
    resume_args=(--resume)
fi

mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=$GPUS PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node "$NPROC" --master_port "$MASTER_PORT" -m trainjev.train \
    --base "$BASE" \
    --data "$DATA" \
    --out "$OUT" \
    --batch 128 \
    --lora_r 16 \
    --lr 1e-4 \
    --head_lr 2e-4 \
    --rps_w 0.5 \
    --perm 0.3 \
    --placeholder_w 0.05 \
    --max_len 2048 \
    --micro_tokens "$MICRO_TOKENS" \
    --eval_every "$EVAL_EVERY" \
    --val_n "$VAL_N" \
    --max_steps "$MAX_STEPS" \
    --steps "$STEPS" --seed "$SEED" --save_state_every "$SAVE_STATE_EVERY" \
    "${resume_args[@]}" "$@" 2>&1 | tee -a "$OUT/train.log"
