#!/usr/bin/env bash
# Serve a checkpoint behind the TypeSafe System One API.
#   CKPT=runs/jev-4b/best PORT=8020 bash scripts/run_serve.sh
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=${BASE:-models/Qwen3.5-4B}
CKPT=${CKPT:-runs/jev-4b/best}
PORT=${PORT:-8020}
HOST=${HOST:-127.0.0.1}
python -m trainjev.serve --base "$BASE" --adapter "$CKPT" --host "$HOST" --port "$PORT" --name "${NAME:-jev-4b}"
