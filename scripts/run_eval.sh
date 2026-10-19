#!/usr/bin/env bash
# Evaluate on jev-distill using the published model's calibration by default.
#   bash scripts/run_eval.sh
#   CKPT=runs/jev-4b/best FIT_CALIBRATION=1 bash scripts/run_eval.sh
#   BASE_ONLY=1 bash scripts/run_eval.sh          # untrained reference
set -euo pipefail
cd "$(dirname "$0")/.."

BASE=${BASE:-Qwen/Qwen3.5-4B}
DATA=${DATA:-data/jev-distill-corpus-v3}
CKPT=${CKPT:-shibing624/train-jev-4b-lora}
extra_args=()
if [[ "${FIT_CALIBRATION:-0}" == "1" ]]; then
    extra_args+=(--fit_calibration)
fi
if [[ -n "${ADAPTER_REVISION:-}" ]]; then
    extra_args+=(--adapter_revision "$ADAPTER_REVISION")
fi

if [[ "${BASE_ONLY:-0}" == "1" ]]; then
    OUT=${OUT:-eval/base}
    python -m trainjev.evaluate --base "$BASE" --data "$DATA" --base_only --out "$OUT"
else
    OUT=${OUT:-eval/$(basename "$CKPT")}
    python -m trainjev.evaluate --base "$BASE" --data "$DATA" --adapter "$CKPT" \
        "${extra_args[@]}" --out "$OUT"
fi
python scripts/compare_corpus.py "$(basename "$OUT")=$OUT/summary.json"
