#!/usr/bin/env bash
# Wide-choice eval (17-256 options): CLINC150 / BANKING77 / MASSIVE en, read with each > 16-option strategy.
# Start a train-jev server first (CKPT=runs/jev-4b/best PORT=8020 bash scripts/run_serve.sh), then
#   URL=http://127.0.0.1:8020 NAME=jev-4b bash scripts/run_wide.sh
# Optional reference on the same rows (JEV-27B-VL behind the System One API, its own default strategy):
#   python -m trainjev.serve --proxy http://127.0.0.1:8011 --port 8021 --name JEV-27B-VL &
#   REF_URL=http://127.0.0.1:8021 URL=... bash scripts/run_wide.sh
set -euo pipefail
cd "$(dirname "$0")/.."
URL=${URL:-http://127.0.0.1:8020}
NAME=${NAME:-jev-4b}
SUITE=${SUITE:-data/wide}
LIMIT=${LIMIT:-100}
STRATEGIES=${STRATEGIES:-"single tournament permute"}
mkdir -p eval
[ -f "$SUITE/suite.json" ] || python scripts/wide_build.py --out "$SUITE"
runs=()
for s in $STRATEGIES; do
  out=eval/wide_${NAME}_${s}.jsonl
  python scripts/pk_benchmark.py run --url "$URL" --suite "$SUITE" --limit "$LIMIT" --strategy "$s" --out "$out"
  runs+=("$s=$out")
done
if [ -n "${REF_URL:-}" ]; then
  python scripts/pk_benchmark.py run --url "$REF_URL" --suite "$SUITE" --limit "$LIMIT" --out eval/wide_reference.jsonl
  runs+=("reference=eval/wide_reference.jsonl")
fi
python scripts/pk_benchmark.py report --seen '' "${runs[@]}"
