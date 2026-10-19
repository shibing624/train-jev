#!/usr/bin/env bash
# Stage 3 AutoThink: stay causal so the model can generate, and keep one-pass rows by dropping
# half of the supervised thoughts (override with THOUGHT_DROP).
#
#   INIT_ADAPTER=runs/jev-4b/best MIX=data/precision-mix/train.jsonl MIX_FRACTION=0.25 \
#   LR=2e-5 HEAD_LR=4e-5 STEPS=2048 MAX_STEPS=512 OUT=runs/jev-4b-autothink \
#   bash scripts/run_train_autothink.sh
set -euo pipefail
cd "$(dirname "$0")/.."
ATTENTION=causal THOUGHT_DROP="${THOUGHT_DROP:-0.5}" bash scripts/run_train.sh "$@"
