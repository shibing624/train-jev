#!/usr/bin/env bash
# Download the training data and the base model.
#   bash scripts/prepare_data.sh            # -> data/jev-distill-corpus-v3, models/Qwen3.5-4B
# Set HF_ENDPOINT (e.g. https://hf-mirror.com) if the Hub is not reachable directly.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR=${DATA_DIR:-data/jev-distill-corpus-v3}
BASE_DIR=${BASE_DIR:-models/Qwen3.5-4B}
BASE_REPO=${BASE_REPO:-Qwen/Qwen3.5-4B}
DATA_REPO=${DATA_REPO:-SargeDev/jev-distill-corpus-v3}
DATA_REVISION=${DATA_REVISION:-75111a13a8455032f0435050bfc7e0c9026ecfe8}
if [[ -z "${BASE_REVISION:-}" ]]; then
    case "$BASE_REPO" in
        Qwen/Qwen3.5-4B) BASE_REVISION=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a ;;
        Qwen/Qwen3.8-27B) BASE_REVISION=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 ;;
        *) echo "Set BASE_REVISION explicitly for reproducible downloads" >&2; exit 2 ;;
    esac
fi

hf download --repo-type dataset "$DATA_REPO" --revision "$DATA_REVISION" --local-dir "$DATA_DIR"
hf download "$BASE_REPO" --revision "$BASE_REVISION" --local-dir "$BASE_DIR"

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
