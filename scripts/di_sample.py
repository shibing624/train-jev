# -*- coding: utf-8 -*-
"""Build the fixed DI sample and its scoring suite; default excludes HLE/HoVer for the published 35-panel result."""
import argparse
import collections
import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path


class FilteredSuite:
    def __init__(self, suite, dropped):
        self.suite, self.dropped = suite, set(dropped)

    def rows(self, **kwargs):
        for row in self.suite.rows(**kwargs):
            if row["_evaluation"]["catalog_id"] not in self.dropped:
                yield row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--di_root", required=True)
    parser.add_argument("--source_suite", help="Defaults to <di_root>/suite-<edition>")
    parser.add_argument("--edition", default="0.3")
    parser.add_argument("--n", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--drop", default="45,61",
                        help="Dropped catalog IDs; use an empty string for a 37-panel sample")
    parser.add_argument("--rows_out", required=True)
    parser.add_argument("--suite_out", required=True)
    a = parser.parse_args(argv)
    if a.n <= 0:
        parser.error("--n must be positive")
    source = Path(a.source_suite) if a.source_suite else Path(a.di_root) / f"suite-{a.edition}"
    rows_out, destination = Path(a.rows_out), Path(a.suite_out)
    if rows_out.exists() or destination.exists():
        raise FileExistsError("Refusing to overwrite existing sample inputs/suite; use a new versioned output")
    dropped = {int(i) for i in a.drop.split(",") if i}
    sys.path.insert(0, a.di_root)
    from decision_index import editions
    from decision_index.suite.io import Suite
    from decision_index.suite.sample import stratified_sample

    suite = FilteredSuite(Suite(str(source), a.edition), dropped)
    counts = collections.Counter(r["_evaluation"]["catalog_id"] for r in suite.rows(apply_exclusions=True))
    rows_out.parent.mkdir(parents=True, exist_ok=True)
    sample = stratified_sample(suite, a.n, str(rows_out), seed=a.seed)
    opener = gzip.open if rows_out.suffix == ".gz" else open
    with opener(rows_out, "rt") as f:
        selected = [json.loads(line) for line in f]
    ids = {r["_evaluation"]["run_id"] for r in selected}
    destination.mkdir(parents=True)
    for name in (editions.ROWS_FILE, editions.ADDED_FILE, editions.GSM8K_FILE):
        if not (source / name).exists():
            continue
        with gzip.open(source / name, "rt") as f, gzip.open(destination / name, "wt") as g:
            for line in f:
                if json.loads(line)["_evaluation"]["run_id"] in ids:
                    g.write(line)
    for name in (editions.EXCLUSIONS_FILE, editions.MANIFEST_FILE):
        if (source / name).exists():
            shutil.copy(source / name, destination / name)
    payload = "".join(json.dumps({"state": r["state"], "questions": r["questions"]},
                               ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for r in selected)
    manifest = {"edition": a.edition, "seed": a.seed, "requested_rows": a.n, "actual_rows": len(selected),
                "dropped_catalog_ids": sorted(dropped), "source_catalog_counts": dict(counts),
                "sample_catalog_counts": dict(collections.Counter(r["_evaluation"]["catalog_id"] for r in selected)),
                "run_id_order_sha256": hashlib.sha256(
                    "\n".join(r["_evaluation"]["run_id"] for r in selected).encode()).hexdigest(),
                "model_inputs_sha256": hashlib.sha256(payload.encode()).hexdigest(),
                "sample_only_scoring_suite": True, "official_full_index": False}
    (destination / "sampling_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"sample": sample, "manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()
