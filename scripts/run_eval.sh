#!/usr/bin/env bash
# Fit per-kind temperatures and evaluate on jev-distill test_set_30k (6,000 rows) and ood (3,000 rows).
#   CKPT=runs/jev-4b/best bash scripts/run_eval.sh
#   BASE_ONLY=1 bash scripts/run_eval.sh          # untrained reference
set -euo pipefail
cd "$(dirname "$0")/.."

BASE=${BASE:-models/Qwen3.5-4B}
DATA=${DATA:-data/jev-distill-corpus-v3}
CKPT=${CKPT:-runs/jev-4b/best}

if [[ "${BASE_ONLY:-0}" == "1" ]]; then
    OUT=${OUT:-eval/base}
    python -m trainjev.evaluate --base "$BASE" --data "$DATA" --base_only --out "$OUT"
else
    OUT=${OUT:-eval/$(basename "$(dirname "$CKPT")")}
    python -m trainjev.evaluate --base "$BASE" --data "$DATA" --adapter "$CKPT" --fit_calibration --out "$OUT"
fi
python scripts/compare_corpus.py "$(basename "$OUT")=$OUT/summary.json"
