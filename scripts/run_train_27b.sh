#!/usr/bin/env bash
# JEV recipe on Qwen3.8-27B (the backbone of autotrust/JEV-27B / JEV-27B-VL). Same code and hyper-parameters as 4B;
# only memory settings differ. LoRA 108.8M + head 123k trainable (identical to the JEV-27B card).
#
# Measured on 1x H20 96GB, gradient checkpointing on, --micro_tokens 16384: ~50 s/step, 75 GB peak, ~70 h/epoch.
# Each rank processes batch/world rows; on 4B, 3 GPUs gave 2.3x over 1 GPU, so 8 GPUs should take roughly 10-12 h
# per epoch (estimate, not measured).
#
#   GPUS=0,1,2,3,4,5,6,7 bash scripts/run_train_27b.sh
set -euo pipefail
cd "$(dirname "$0")/.."

GPUS=${GPUS:-0,1,2,3,4,5,6,7}
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)
BASE=${BASE:-models/Qwen3.8-27B}
DATA=${DATA:-data/jev-distill-corpus-v3}
OUT=${OUT:-runs/jev-27b}
MAX_STEPS=${MAX_STEPS:-0}
MASTER_PORT=${MASTER_PORT:-29550}

mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=$GPUS PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
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
    --micro_tokens 16384 \
    --eval_every 500 \
    --val_n 4000 \
    --max_steps "$MAX_STEPS" \
    "$@" 2>&1 | tee "$OUT/train.log"
