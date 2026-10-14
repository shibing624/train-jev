"""Build a stratified Decision Index sample and a matching scoring suite.

The full Decision Index 0.3 public suite is ~140k requests. For fast iteration we score every model on the same
stratified sample (round-robin over benchmarks, whole groups, fixed seed), and build a sample-only suite directory so
the official scorer computes coverage over the sample instead of counting unsampled rows as wrong.

Run inside a decision-index checkout after ``suite rebuild`` + ``suite import``:

    python scripts/di_sample.py --di_root third_party/decision-index --n 6000 \
        --rows_out data/di_sample6000.jsonl.gz --suite_out third_party/decision-index/suite-0.3-sample6000
"""
import argparse
import collections
import gzip
import json
import shutil
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--di_root", required=True)
ap.add_argument("--edition", default="0.3")
ap.add_argument("--n", type=int, default=6000)
ap.add_argument("--rows_out", required=True)
ap.add_argument("--suite_out", required=True)
a = ap.parse_args()
sys.path.insert(0, a.di_root)
from decision_index import editions  # noqa: E402
from decision_index.suite.io import Suite  # noqa: E402
from decision_index.suite.sample import stratified_sample  # noqa: E402

src = Path(a.di_root) / f"suite-{a.edition}"
suite = Suite(str(src), a.edition)
c = collections.Counter(r["_evaluation"]["catalog_id"] for r in suite.rows(apply_exclusions=True))
print(f"full suite: {len(c)} benchmarks, {sum(c.values())} rows")
print(stratified_sample(suite, a.n, a.rows_out))

ids = {json.loads(line)["_evaluation"]["run_id"] for line in gzip.open(a.rows_out, "rt")}
dst = Path(a.suite_out)
dst.mkdir(parents=True, exist_ok=True)
for name in (editions.ROWS_FILE, editions.ADDED_FILE, editions.GSM8K_FILE):
    if not (src / name).exists():
        continue
    n = 0
    with gzip.open(src / name, "rt") as f, gzip.open(dst / name, "wt") as g:
        for line in f:
            if json.loads(line)["_evaluation"]["run_id"] in ids:
                g.write(line)
                n += 1
    print(f"{name}: {n} rows")
for name in (editions.EXCLUSIONS_FILE, editions.MANIFEST_FILE):
    if (src / name).exists():
        shutil.copy(src / name, dst / name)
print(f"sample suite -> {dst}")
