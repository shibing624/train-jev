# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: 把 trainjev.evaluate 的 summary 排成对照表。

Side-by-side table of ``trainjev.evaluate`` summaries.

    python scripts/compare_corpus.py base=eval/base/summary.json jev-4b=eval/jev-4b/summary.json
"""
import json
import sys


def fmt(x):
    return "  -   " if x is None else f"{x:.4f}"


def row(name, path):
    with open(path) as f:
        s = json.load(f)
    for sp in ("test_set_30k", "ood"):
        if sp not in s:
            continue
        r = s[sp]
        m = r.get("calibrated", r["raw"])
        j = r.get("calibrated_jev_labelled", r.get("raw_jev_labelled"))

        def g(k, f):
            return (m.get(k) or {}).get(f)

        print(f"{name:14s} {sp:13s} n={m['all']['n']:5d} KL {fmt(m['all']['kl'])}  "
              f"KL(Jev-labelled) {fmt(j['all']['kl']) if j else '  -   '}  "
              f"noul top1 {fmt(g('noul', 'top1'))} auroc {fmt(g('noul', 'auroc'))} | "
              f"choice top1 {fmt(g('choice', 'top1'))} | score top1 {fmt(g('score', 'top1'))} "
              f"mae {fmt(g('score', 'score_mae'))}")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        n, p = arg.split("=", 1)
        row(n, p)
