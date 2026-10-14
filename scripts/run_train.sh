#!/usr/bin/env bash
# Train the JEV-recipe System 1 model (LoRA + 24-slot decision head) on Qwen3.5-4B.
# Reference run: 3x H20 (96 GB), 5,123 steps (one epoch over 655,806 rows), ~2.8 h, 48.8 GB peak per GPU.
#
#   GPUS=0,1,2 bash scripts/run_train.sh
#   GPUS=0 MAX_STEPS=20 OUT=runs/smoke bash scripts/run_train.sh     # smoke test
set -euo pipefail
cd "$(dirname "$0")/.."

GPUS=${GPUS:-0,1,2}
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)
BASE=${BASE:-models/Qwen3.5-4B}
DATA=${DATA:-data/jev-distill-corpus-v3}
OUT=${OUT:-runs/jev-4b}
MAX_STEPS=${MAX_STEPS:-0}
EVAL_EVERY=${EVAL_EVERY:-500}
VAL_N=${VAL_N:-4000}
MASTER_PORT=${MASTER_PORT:-29540}

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
    --micro_tokens 4096 \
    --no_gc \
    --eval_every "$EVAL_EVERY" \
    --val_n "$VAL_N" \
    --max_steps "$MAX_STEPS" \
    "$@" 2>&1 | tee "$OUT/train.log"
