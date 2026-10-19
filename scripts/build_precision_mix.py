# -*- coding: utf-8 -*-
"""Write a local contrast-pair mix. This is not DecisionPivot and not a Decision Index score.

    python scripts/build_precision_mix.py --pairs 50 --out data/precision-mix
"""
import argparse
import json
from pathlib import Path

from trainjev.precision import audit_rows, build_rows, write_jsonl


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=50, help="contrast pairs per family")
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--out", default="data/precision-mix")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for split, seed in (("train", args.seed), ("dev", args.seed + 1)):
        rows = build_rows(args.pairs, seed)
        audit = audit_rows(rows)
        if audit["mismatches"]:
            raise SystemExit("\n".join(audit["mismatches"][:20]))
        write_jsonl(out / f"{split}.jsonl", rows)
        report[split] = {key: audit[key] for key in ("rows", "pairs", "sha256")}
    (out / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
