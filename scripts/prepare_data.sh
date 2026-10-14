#!/usr/bin/env bash
# Download the training data and the base model.
#   bash scripts/prepare_data.sh            # -> data/jev-distill-corpus-v3, models/Qwen3.5-4B
# Set HF_ENDPOINT (e.g. https://hf-mirror.com) if the Hub is not reachable directly.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR=${DATA_DIR:-data/jev-distill-corpus-v3}
BASE_DIR=${BASE_DIR:-models/Qwen3.5-4B}

huggingface-cli download --repo-type dataset SargeDev/jev-distill-corpus-v3 --local-dir "$DATA_DIR"
huggingface-cli download Qwen/Qwen3.5-4B --local-dir "$BASE_DIR"

python - "$DATA_DIR" <<'EOF'
import sys, collections, json, os
d = sys.argv[1]
for split in ("train", "validation", "calibration", "test_set_30k", "ood"):
    p = os.path.join(d, f"{split}.jsonl")
    n, kinds = 0, collections.Counter()
    with open(p) as f:
        for line in f:
            n += 1; kinds[json.loads(line)["kind"]] += 1
    print(f"{split:13s} {n:8d}  {dict(kinds)}")
EOF
