#!/usr/bin/env bash
# Serve a checkpoint behind the TypeSafe System One API.
#   bash scripts/run_serve.sh
#   BASE=Qwen/Qwen3.8-27B CKPT=shibing624/train-jev-27b-lora bash scripts/run_serve.sh
#   VISION=1 bash scripts/run_serve.sh          # TypeSafe image assets
#   STRATEGY=tournament ...   # choices with > 16 options: single (default) | tournament | permute
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=${BASE:-Qwen/Qwen3.5-4B}
CKPT=${CKPT:-shibing624/train-jev-4b-lora}
PORT=${PORT:-8020}
HOST=${HOST:-127.0.0.1}
revision_args=()
if [[ -n "${ADAPTER_REVISION:-}" ]]; then
    revision_args=(--adapter_revision "$ADAPTER_REVISION")
fi
if [[ "${VISION:-0}" == "1" ]]; then
    revision_args+=(--vision)
fi
variant_args=()
[[ -z "${VARIANT:-}" ]] || variant_args+=(--variant "$VARIANT")
python -m trainjev.serve --base "$BASE" --adapter "$CKPT" --host "$HOST" --port "$PORT" \
    --name "${NAME:-$(basename "$CKPT")}" --device "${DEVICE:-cuda}" --strategy "${STRATEGY:-single}" \
    "${revision_args[@]}" "${variant_args[@]}" "$@"
