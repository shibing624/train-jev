"""Build the 13-dataset classification PK suite from public Hugging Face datasets.

13 scenarios that span all three primitives and two languages, each with an independent published reference:

  choice   AG News (4), DAIR Emotion (6), BANKING77 (77), MASSIVE intent en / zh (59), XNLI en / zh (3), TNEWS (15)
  score    SST-5 (5 ordered levels), Yelp review stars (5)
  boolean  BoolQ, deepset prompt-injections, Civil Comments toxicity (balanced)

Every scenario is shuffled with ``random.Random(0)`` and capped at 600 rows; the benchmark uses the first 100 of each
(1,300 rows). Output: ``<out>/<name>.jsonl`` (``{"text", "label"}``), ``<out>/<name>.meta.json``, ``<out>/suite.json``.

    python scripts/pk_build.py --out data/pk
"""
import argparse
import gzip
import io
import json
import os
import random
import urllib.request

N_MAX = 600
SEED = 0
HF = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
# CLUE TNEWS label codes -> Chinese category names (newer `datasets` releases expose only the codes);
# names are sorted to give a stable option order.
TNEWS = {"100": "民生故事", "101": "文化", "102": "娱乐", "103": "体育", "104": "财经", "106": "房产", "107": "汽车",
         "108": "教育", "109": "科技", "110": "军事", "112": "旅游", "113": "国际", "114": "证券股票", "115": "农业",
         "116": "电子游戏"}


def save(out, name, rows, labels, primitive, lang, source):
    rows = list(rows)
    random.Random(SEED).shuffle(rows)
    rows = rows[:N_MAX]
    with open(f"{out}/{name}.jsonl", "w") as f:
        for t, lab in rows:
            f.write(json.dumps({"text": t, "label": lab}, ensure_ascii=False) + "\n")
    meta = {"name": name, "labels": labels, "primitive": primitive, "lang": lang, "n": len(rows),
            "random": 1.0 / len(labels), "source": source}
    with open(f"{out}/{name}.meta.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"  {name:18s} n={len(rows):4d} labels={len(labels):3d} {primitive:8s} {lang}  <- {source}")
    return meta


def fetch(repo, path):
    with urllib.request.urlopen(f"{HF}/datasets/{repo}/resolve/main/{path}", timeout=300) as r:
        return r.read()


def build(out):
    from datasets import load_dataset

    metas = []

    d = load_dataset("fancyzhx/ag_news", split="test")
    names = ["World", "Sports", "Business", "Sci/Tech"]
    metas.append(save(out, "ag_news", [(r["text"], names[r["label"]]) for r in d], names, "choice", "en",
                      "fancyzhx/ag_news:test"))

    d = load_dataset("dair-ai/emotion", split="test")
    names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    metas.append(save(out, "dair_emotion", [(r["text"], names[r["label"]]) for r in d], names, "choice", "en",
                      "dair-ai/emotion:test"))

    d = load_dataset("legacy-datasets/banking77", split="test")
    names = d.features["label"].names
    metas.append(save(out, "banking77", [(r["text"], names[r["label"]]) for r in d], names, "choice", "en",
                      "legacy-datasets/banking77:test"))

    for lang, tag in (("en", "en"), ("zh-CN", "zh")):
        raw = gzip.decompress(fetch("mteb/amazon_massive_intent", f"test/{lang}.json.gz")).decode()
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        labels = sorted({r["label_text"] for r in rows})
        metas.append(save(out, f"massive_{tag}", [(r["text"], r["label_text"]) for r in rows], labels, "choice", tag,
                          f"mteb/amazon_massive_intent:test/{lang}"))

    import pandas as pd

    nm = {0: "entailment", 1: "neutral", 2: "contradiction"}
    for tag in ("en", "zh"):
        d = pd.read_parquet(io.BytesIO(fetch("facebook/xnli", f"{tag}/validation-00000-of-00001.parquet")))
        rows = [(f"Premise: {a}\nHypothesis: {b}", nm[int(c)])
                for a, b, c in zip(d["premise"], d["hypothesis"], d["label"]) if int(c) in nm]
        metas.append(save(out, f"xnli_{tag}", rows, list(nm.values()), "choice", tag,
                          f"facebook/xnli:{tag}/validation"))

    d = load_dataset("clue/clue", "tnews", split="validation")
    names = d.features["label"].names
    to_name = [TNEWS.get(n, n) for n in names]
    metas.append(save(out, "tnews_zh", [(r["sentence"], to_name[r["label"]]) for r in d], sorted(to_name),
                      "choice", "zh", "clue/clue:tnews/validation"))

    d = load_dataset("SetFit/sst5", split="test")
    names = ["very negative", "negative", "neutral", "positive", "very positive"]
    metas.append(save(out, "sst5", [(r["text"], names[r["label"]]) for r in d], names, "score", "en",
                      "SetFit/sst5:test"))

    d = load_dataset("Yelp/yelp_review_full", split="test")
    names = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]
    metas.append(save(out, "yelp_stars", [(r["text"][:1500], names[r["label"]]) for r in d], names, "score", "en",
                      "Yelp/yelp_review_full:test"))

    d = load_dataset("google/boolq", split="validation")
    metas.append(save(out, "boolq", [(f"Passage: {r['passage'][:1200]}\nQuestion: {r['question']}",
                                      "true" if r["answer"] else "false") for r in d],
                      ["false", "true"], "boolean", "en", "google/boolq:validation"))

    d = load_dataset("deepset/prompt-injections", split="test")
    metas.append(save(out, "prompt_injection", [(r["text"], "true" if r["label"] == 1 else "false") for r in d],
                      ["false", "true"], "boolean", "en", "deepset/prompt-injections:test"))

    d = load_dataset("google/civil_comments", split="test")
    rows = [(r["text"][:1200], "true" if r["toxicity"] >= 0.5 else "false") for r in d.select(range(6000))]
    pos = [r for r in rows if r[1] == "true"][:300]
    neg = [r for r in rows if r[1] == "false"][:300]
    metas.append(save(out, "toxicity", pos + neg, ["false", "true"], "boolean", "en",
                      "google/civil_comments:test[:6000], toxicity>=0.5, 300+300"))

    with open(f"{out}/suite.json", "w") as f:
        json.dump(metas, f, ensure_ascii=False, indent=1)
    print(f"{len(metas)} scenarios -> {out}/suite.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="data/pk")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    build(a.out)
