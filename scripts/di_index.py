"""Decision Index 0.3 public-index math on a benchmark panel with some benchmarks dropped.

Two benchmarks of the 0.3 public suite could not be rebuilt in our environment: HLE (catalog id 45, gated on the HF
Hub) and HoVer (61, a 2 GB Wikipedia dump). This script removes them from their areas (gold weights and fixed area
weights as in ``index-0.3.json``; the remaining benchmarks of each area renormalised) and, given the board's own
``di_index.json``, re-scores published models on the same panel so the numbers are comparable.

    python scripts/di_index.py --di_root third_party/decision-index \
        jev-4b=eval/di_jev-4b/score_s/index.json --board data/di_index.json
"""
import argparse
import json
import os

DEFAULT_REFS = ["Kev 4B r10", "JEV-27B", "vLLM-SR d3-mini (4B)", "Sieve 4B", "ezjev 4B s2", "Kev 27B"]

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="*", help="name=path/to/score/index.json")
ap.add_argument("--di_root", required=True)
ap.add_argument("--edition", default="0.3")
ap.add_argument("--drop", default="45,61", help="catalog ids removed from the panel")
ap.add_argument("--board", default="", help="the board's di_index.json (HF space multimodalart/jev-decision-index)")
ap.add_argument("--refs", default=",".join(DEFAULT_REFS))
a = ap.parse_args()

with open(os.path.join(a.di_root, "decision_index", "data", f"index-{a.edition}.json")) as f:
    spec = json.load(f)
DROP = {int(x) for x in a.drop.split(",") if x}
W = spec["area_sizing"]["fixed"]
GOLD = {int(k): v for k, v in spec["gold"].items()}


def index(skill):
    """skill: ``{benchmark id: chance-corrected skill in [0, 1]}`` -> (index 0-100, {area: skill})."""
    areas = {}
    for ar in spec["areas"]:
        ids = [n for n in ar["benchmarks"] if n not in DROP]
        w = [GOLD.get(n, 1.0) for n in ids]
        areas[ar["id"]] = sum(wi * skill.get(n, 0.0) for wi, n in zip(w, ids)) / sum(w)
    return 100 * sum(W[k] * v for k, v in areas.items()), areas


rows = []
for arg in a.runs:
    name, path = arg.split("=", 1)
    with open(path) as f:
        b = json.load(f)["benchmarks"]
    rows.append((name, {int(k): v["skill"] for k, v in b.items() if v.get("in_index")}))
if a.board:
    with open(a.board) as f:
        board = json.load(f)
    refs = {"Jev 1.13 (board)": board["jev"]}
    want = set(a.refs.split(","))
    for m in board["models"]:
        if m["name"] in want:
            refs[m["name"] + " (board)"] = m
    for name, m in refs.items():
        bb = m.get("benchmarks") or {}
        if bb:
            rows.append((name, {int(k): v.get("skill") or 0.0 for k, v in bb.items()}))

n_bench = sum(len([n for n in ar["benchmarks"] if n not in DROP]) for ar in spec["areas"])
print(f"{'model':30s} {'index' + str(n_bench):>8s} " + " ".join(f"{ar['id'][:9]:>9s}" for ar in spec["areas"]))
for name, sk in rows:
    ix, ar = index(sk)
    print(f"{name:30s} {ix:8.2f} " + " ".join(f"{ar[x['id']]:9.3f}" for x in spec["areas"]))
