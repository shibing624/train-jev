#!/usr/bin/env bash
# Stage 2 Instant: continue a causal checkpoint with bidirectional full-attention layers.
# Linear attention stays causal. Do not flip this flag only at eval time.
#
#   INIT_ADAPTER=runs/jev-4b/best MIX=data/precision-mix/train.jsonl MIX_FRACTION=0.25 \
#   LR=2e-5 HEAD_LR=4e-5 STEPS=2048 MAX_STEPS=512 OUT=runs/jev-4b-instant \
#   bash scripts/run_train_instant.sh
set -euo pipefail
cd "$(dirname "$0")/.."
ATTENTION=bidirectional bash scripts/run_train.sh "$@"
