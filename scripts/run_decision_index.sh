#!/usr/bin/env bash
# Decision Index 0.3 (public benchmarks) on a 6,000-row stratified sample, scored with the kit's own scorer.
#
# One-time setup (needs network; ~30 GB of public datasets):
#   git clone https://github.com/apolinario/decision-index third_party/decision-index
#   (cd third_party/decision-index && git apply ../../patches/decision-index-resumable-download.patch && pip install -e .)
#   (cd third_party/decision-index && python -m decision_index suite rebuild --work work && \
#      python -m decision_index suite import \
#        --rows work/artifacts/benchmark-suite/release-v2-rebuilt/selected-rows.jsonl.gz \
#        --added-rows work/artifacts/benchmark-suite/release-v2-rebuilt/added-rows.jsonl.gz \
#        --gsm8k-rows work/artifacts/benchmark-suite/release-v3-rebuilt/gsm8k-rows.jsonl.gz)
#   python scripts/di_sample.py --di_root third_party/decision-index --n 6000 \
#     --rows_out data/di_sample6000.jsonl.gz --suite_out third_party/decision-index/suite-0.3-sample6000
#
# Then, with a server running (python -m trainjev.serve --adapter runs/jev-4b/best --port 8020):
#   URL=http://127.0.0.1:8020 NAME=jev-4b bash scripts/run_decision_index.sh
set -euo pipefail
cd "$(dirname "$0")/.."

DI=${DI:-third_party/decision-index}
URL=${URL:-http://127.0.0.1:8020}
NAME=${NAME:-jev-4b}
ROWS=${ROWS:-data/di_sample6000.jsonl.gz}
SUITE=${SUITE:-$DI/suite-0.3-sample6000}
OUT=${OUT:-eval/di_$NAME}

python -m decision_index run --engine http --option base_url="$URL" --option model="$NAME" \
    --rows "$ROWS" --out "$OUT"
python -m decision_index score --edition 0.3 --suite-dir "$SUITE" --results "$OUT/results.jsonl" --out "$OUT/score_s"
python scripts/di_index.py --di_root "$DI" "$NAME=$OUT/score_s/index.json" ${BOARD:+--board "$BOARD"}
