# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: 构建 >16 选项的宽选项意图分类题库（CLINC150 / BANKING77 / MASSIVE）。

Build the wide-choice suite: zero-shot intent classification with every intent offered in one ``choice`` question,
the setting of JEV-27B-VL's "Choice questions with up to 256 options" table.

  clinc150     150 options   ``clinc_oos`` (plus) test, in-scope rows only
  banking77     77 options   ``legacy-datasets/banking77`` test
  massive_en    59 options   ``mteb/amazon_massive_intent`` test/en

Label names are used as the options (``_`` -> space in ``pk_benchmark.py``): CLINC150 sorted, BANKING77 and MASSIVE
exactly as in ``pk_build.py`` (so those two scenarios are byte-identical to the PK ones). Every scenario is shuffled
with ``random.Random(0)`` and capped at 600 rows, in the same file layout as ``pk_build.py``, so
``pk_benchmark.py run / report`` work on it unchanged.

    python scripts/wide_build.py --out data/wide
    python scripts/pk_benchmark.py run --url http://127.0.0.1:8020 --suite data/wide --strategy single \\
        --out eval/wide_jev-4b_single.jsonl
"""
import argparse
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pk_build import fetch, save  # noqa: E402


def build(out):
    from datasets import load_dataset

    metas = []
    d = load_dataset("clinc_oos", "plus", split="test")
    names = d.features["intent"].names
    rows = [(r["text"], names[r["intent"]]) for r in d if names[r["intent"]] != "oos"]
    metas.append(save(out, "clinc150", rows, sorted(set(names) - {"oos"}), "choice", "en", "clinc_oos:plus/test"))

    d = load_dataset("legacy-datasets/banking77", split="test")
    names = d.features["label"].names
    metas.append(save(out, "banking77", [(r["text"], names[r["label"]]) for r in d], names, "choice", "en",
                      "legacy-datasets/banking77:test"))

    raw = gzip.decompress(fetch("mteb/amazon_massive_intent", "test/en.json.gz")).decode()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    metas.append(save(out, "massive_en", [(r["text"], r["label_text"]) for r in rows],
                      sorted({r["label_text"] for r in rows}), "choice", "en", "mteb/amazon_massive_intent:test/en"))

    with open(f"{out}/suite.json", "w") as f:
        json.dump(metas, f, ensure_ascii=False, indent=1)
    print(f"{len(metas)} scenarios -> {out}/suite.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="data/wide")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    build(a.out)
