#!/usr/bin/env bash
# Train the JEV-recipe System 1 model (LoRA + 24-slot decision head) on Qwen3.5-4B.
# Reference run: 3x H20 (96 GB), 5,123 steps (one epoch-equivalent draw budget), ~2.8 h.
#
#   GPUS=0,1,2 bash scripts/run_train.sh
#   GPUS=0 MAX_STEPS=20 OUT=runs/smoke bash scripts/run_train.sh     # smoke test
set -euo pipefail
cd "$(dirname "$0")/.."

GPUS=${GPUS:-0,1,2}
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)
BASE=${BASE:-Qwen/Qwen3.5-4B}
DATA=${DATA:-data/jev-distill-corpus-v3}
OUT=${OUT:-runs/jev-4b}
MAX_STEPS=${MAX_STEPS:-0}
EVAL_EVERY=${EVAL_EVERY:-500}
VAL_N=${VAL_N:-4000}
MASTER_PORT=${MASTER_PORT:-29540}
SEED=${SEED:-0}
STEPS=${STEPS:-0}
MICRO_TOKENS=${MICRO_TOKENS:-4096}
SAVE_STATE_EVERY=${SAVE_STATE_EVERY:-100}
LR=${LR:-1e-4}
HEAD_LR=${HEAD_LR:-2e-4}
data_args=()
[[ -z "${INIT_ADAPTER:-}" ]] || data_args+=(--init_adapter "$INIT_ADAPTER")
[[ -z "${MIX:-}" ]] || data_args+=(--mix "$MIX")
[[ -z "${MIX_FRACTION:-}" ]] || data_args+=(--mix_fraction "$MIX_FRACTION")
[[ -z "${KIND_COUNTS:-}" ]] || data_args+=(--kind_counts "$KIND_COUNTS")
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
    --lr "$LR" \
    --head_lr "$HEAD_LR" \
    --rps_w 0.5 \
    --perm 0.3 \
    --placeholder_w 0.05 \
    --max_len 2048 \
    --micro_tokens "$MICRO_TOKENS" \
    --no_gc \
    --eval_every "$EVAL_EVERY" \
    --val_n "$VAL_N" \
    --max_steps "$MAX_STEPS" \
    --steps "$STEPS" --seed "$SEED" --save_state_every "$SAVE_STATE_EVERY" \
    "${data_args[@]}" "${resume_args[@]}" "$@" 2>&1 | tee -a "$OUT/train.log"
