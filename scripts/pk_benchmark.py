"""Run the 13-dataset PK suite against a System One endpoint and/or score saved runs.

Every system gets the same input: the text (first 4,000 characters), the dataset's label set and one fixed question
per primitive (no per-task prompt engineering):

  choice  -> a ``choice`` question over the label names ("_" -> " "), "Which label best describes this input?"
  score   -> a ``choice`` question over the ordered level names, "Rate this input on the given scale."
  boolean -> a ``noul`` question, "Is the statement or property described true for this input?"

Run a server (100 rows per scenario = 1,300 rows; resumable)::

    python scripts/pk_benchmark.py run --url http://127.0.0.1:8020 --suite data/pk --out eval/pk_jev-4b.jsonl

Compare runs on the rows they share (the first one is the McNemar reference)::

    python scripts/pk_benchmark.py report jev-4b=eval/pk_jev-4b.jsonl kev-4b=eval/pk_kev-4b.jsonl

Five sources overlap with Kev's training data (``--seen ag_news,banking77,sst5,yelp_stars,boolq``); ``report``
prints them separately so systems trained on them can be compared fairly.
"""
import argparse
import json
import math
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

CHOICE_Q = "Which label best describes this input?"
SCORE_Q = "Rate this input on the given scale."
NOUL_Q = "Is the statement or property described true for this input?"
KEV_SEEN = "ag_news,banking77,sst5,yelp_stars,boolq"


def load_suite(path):
    with open(f"{path}/suite.json") as f:
        metas = json.load(f)
    out = []
    for m in metas:
        with open(f"{path}/{m['name']}.jsonl") as f:
            out.append((m, [json.loads(line) for line in f if line.strip()]))
    return out


def post(url, body):
    req = urllib.request.Request(url + "/v1/systemone", json.dumps(body).encode(), {"content-type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read()), time.perf_counter() - t


def one(url, model, meta, i, row):
    text, labels = row["text"][:4000], meta["labels"]
    rec = {"scenario": meta["name"], "index": i, "primitive": meta["primitive"], "lang": meta["lang"],
           "gold": row["label"]}
    if meta["primitive"] == "boolean":
        out, dt = post(url, {"model": model, "state": text,
                             "questions": {"q": {"type": "noul", "instructions": meta.get("question") or NOUL_Q}}})
        p = out["answers"]["q"]["noul"]
        rec.update(pred="true" if p >= 0.5 else "false", conf=max(p, 1 - p), probs=[1 - p, p])
    else:
        opts = [lab.replace("_", " ") for lab in labels]
        q = SCORE_Q if meta["primitive"] == "score" else CHOICE_Q
        out, dt = post(url, {"model": model, "state": text, "questions": {
            "q": {"type": "choice", "instructions": q, "criteria": {o: None for o in opts}}}})
        pr = out["answers"]["q"]["probabilities"]
        ps = [pr[o] for o in opts]
        k = max(range(len(ps)), key=ps.__getitem__)
        rec.update(pred=labels[k], conf=ps[k], probs=ps)
    rec.update(latency_s=dt, usage=out.get("usage"))
    return rec


def cmd_run(a):
    done = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            done = {(r["scenario"], r["index"]) for r in map(json.loads, f)}
    jobs = [(m, i, r) for m, rows in load_suite(a.suite) for i, r in enumerate(rows[: a.limit])
            if (m["name"], i) not in done]
    print(f"resume {len(done)}, todo {len(jobs)} -> {a.out}", flush=True)
    t0 = time.time()
    with open(a.out, "a", encoding="utf-8") as fh, ThreadPoolExecutor(a.workers) as ex:
        for n, rec in enumerate(ex.map(lambda j: one(a.url, a.model, *j), jobs), 1):
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if n % 100 == 0:
                fh.flush()
                print(f"  {n}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    print(f"done in {time.time() - t0:.0f}s")


# ---------------------------------------------------------------------------- report
def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar(a, b):
    """Exact two-sided McNemar test: (wins of a over b, losses, p)."""
    w = sum(x and not y for x, y in zip(a, b))
    lo = sum(y and not x for x, y in zip(a, b))
    n = w + lo
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(w, lo) + 1)) / 2 ** n) if n else 1.0
    return w, lo, p


def ece(conf, correct, bins=10):
    e = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if lo < c <= hi]
        if idx:
            gap = sum(conf[i] for i in idx) / len(idx) - sum(correct[i] for i in idx) / len(idx)
            e += len(idx) / len(conf) * abs(gap)
    return e


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return (xs[n // 2] + xs[(n - 1) // 2]) / 2


def cmd_report(a):
    arms = dict(x.split("=", 1) for x in a.runs)
    data = {}
    for n, p in arms.items():
        with open(p) as f:
            data[n] = {(r["scenario"], r["index"]): r for r in map(json.loads, f)}
    keys = sorted(set.intersection(*[set(v) for v in data.values()]))
    names = list(arms)
    first = data[names[0]]
    for k in keys:
        assert len({data[n][k]["gold"] for n in names}) == 1, f"gold differs at {k}"
    seen = set(a.seen.split(",")) if a.seen else set()

    def corr(n, ks):
        return [data[n][k]["pred"] == data[n][k]["gold"] for k in ks]

    def acc(n, ks):
        return sum(corr(n, ks)) / len(ks) if ks else float("nan")

    scen = []
    for k in keys:
        if k[0] not in scen:
            scen.append(k[0])
    w = max(17, max(len(n) for n in names) + 2)
    print(f"paired rows {len(keys)}  ({len(scen)} scenarios)")
    print(f"{'scenario':18s}{'prim':>8s}" + "".join(f"{n:>{w}s}" for n in names))
    for s in scen:
        ks = [k for k in keys if k[0] == s]
        tag = "*" if s in seen else " "
        print(f"{s + tag:18s}{first[ks[0]]['primitive']:>8s}" + "".join(f"{acc(n, ks):{w}.3f}" for n in names))
    print("-" * (26 + w * len(names)))
    rows = [("accuracy", lambda n: f"{acc(n, keys):.3f}"),
            ("95% CI", lambda n: "[%.3f,%.3f]" % wilson(sum(corr(n, keys)), len(keys))),
            ("ECE", lambda n: f"{ece([data[n][k]['conf'] for k in keys], corr(n, keys)):.3f}"),
            (f"vs {names[0][:10]} W/L p", lambda n: "-" if n == names[0] else "%d/%d p=%.2g" % mcnemar(
                corr(n, keys), corr(names[0], keys))),
            ("p50 latency ms", lambda n: f"{1000 * median([data[n][k]['latency_s'] for k in keys]):.0f}")]
    for lab, f in rows:
        print(f"{lab:26s}" + "".join(f"{f(n):>{w}s}" for n in names))
    for prim in ("choice", "score", "boolean"):
        ks = [k for k in keys if first[k]["primitive"] == prim]
        if ks:
            print(f"{prim + f' (n={len(ks)})':26s}" + "".join(f"{acc(n, ks):{w}.3f}" for n in names))
    if seen:
        for lab, ks in (("Kev-seen sources *", [k for k in keys if k[0] in seen]),
                        ("Kev-unseen sources", [k for k in keys if k[0] not in seen])):
            print(f"{lab + f' (n={len(ks)})':26s}" + "".join(f"{acc(n, ks):{w}.3f}" for n in names))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url", required=True)
    r.add_argument("--model", default="jev-4b")
    r.add_argument("--suite", default="data/pk")
    r.add_argument("--limit", type=int, default=100, help="rows per scenario")
    r.add_argument("--workers", type=int, default=1, help="1 = serial, clean latency")
    r.add_argument("--out", required=True)
    p = sub.add_parser("report")
    p.add_argument("runs", nargs="+", help="name=path.jsonl")
    p.add_argument("--seen", default=KEV_SEEN, help="scenarios in Kev's training sources ('' to disable)")
    a = ap.parse_args(argv)
    {"run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
