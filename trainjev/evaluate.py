# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: 在蒸馏语料上评测，并按问题类型拟合温度。

Evaluate a System 1 model on jev-distill-corpus-v3 splits and fit per-kind temperatures.

Backends (same rows, same metrics for each):
  --adapter DIR        a train-jev checkpoint (LoRA + head.pt), local GPU
  --base_only          the untrained reference: base model + head initialised from lm_head
  --decide_url URL     JEV-27B-VL's own ``serve_decide.py`` server (its native /v1/decide route), for comparison
  --systemone_url URL  any TypeSafe-compatible /v1/systemone endpoint (e.g. a Kev server)

``--fit_calibration`` fits a per-kind temperature on the calibration split (placeholder rows excluded) and, for
``--adapter``, writes ``<adapter>/calibration.json``. Writes ``<out>/<split>.jsonl`` (row id, logits) and
``<out>/summary.json`` (metrics raw and at the fitted temperatures; ``*_jev_labelled`` = rows labelled by Jev 1.13).

    python -m trainjev.evaluate --base Qwen/Qwen3.5-4B --adapter runs/jev-4b/best --fit_calibration --out eval/jev-4b
"""
import argparse
import json
import math
import os
import time

from trainjev.attention import add_attention_argument
from trainjev.data import encode_many, is_placeholder, load_rows, micro_batches
from trainjev.metrics import metrics
from trainjev.modeling import resolve_checkpoint


def softmax_T(z, T):
    m = max(z)
    e = [math.exp((x - m) / T) for x in z]
    s = sum(e)
    return [x / s for x in e]


def fit_temperatures(rows, zs):
    """Per-kind T minimising mean cross-entropy to the soft targets (golden-section search on log T in [0.3, 5])."""
    temps = {}
    for kind in ("noul", "choice", "score"):
        idx = [i for i, r in enumerate(rows) if r["kind"] == kind]
        if not idx:
            continue

        def loss(lt):
            T = math.exp(lt)
            s = 0.0
            for i in idx:
                p = softmax_T(zs[i], T)
                s -= sum(t * math.log(max(q, 1e-12)) for t, q in zip(rows[i]["target"], p))
            return s / len(idx)

        lo, hi = math.log(0.3), math.log(5.0)
        g = (math.sqrt(5) - 1) / 2
        for _ in range(40):
            c, d = hi - g * (hi - lo), lo + g * (hi - lo)
            if loss(c) < loss(d):
                hi = d
            else:
                lo = c
        temps[kind] = math.exp((lo + hi) / 2)
    return temps


def make_backend(a):
    """-> logits_of(rows) returning one logit (or log-prob) vector per row."""
    if a.systemone_url or a.decide_url:
        if getattr(a, "experimental_prompt_bidirectional", False):
            raise ValueError("Experimental prompt attention cannot be selected for a remote backend")
        from concurrent.futures import ThreadPoolExecutor

        import httpx

        cli = httpx.Client(base_url=a.systemone_url or a.decide_url, timeout=600)

        def one_systemone(r):
            st = r["state"] if isinstance(r["state"], str) else json.dumps(r["state"], ensure_ascii=False)
            if r["kind"] == "noul":
                q = {"type": "noul", "instructions": r["question"]}
            elif r["kind"] == "choice":
                q = {"type": "choice", "instructions": r["question"],
                     "criteria": {f"{i}": o for i, o in enumerate(r["options"])}}
            else:
                q = {"type": "score", "instructions": r["question"], "criteria": [str(i) for i in range(6)]}
            body = {"model": a.model_name, "state": st, "questions": {"q": q}}
            ans = cli.post("/v1/systemone", json=body).json()["answers"]["q"]
            if r["kind"] == "noul":
                p = [1 - ans["noul"], ans["noul"]]
            else:
                p = [ans["probabilities"][str(i)] for i in range(len(r["options"]))]
            return [math.log(max(x, 1e-30)) for x in p]

        def one_decide(r):
            j = cli.post("/v1/decide", json={"kind": r["kind"], "state": r["state"], "question": r["question"],
                                              "options": r["options"] if r["kind"] == "choice" else None,
                                              "thinking": "off"}).json()
            # serve_decide applies its own temperatures; log-probs make any further fit relative to them
            return [math.log(max(p, 1e-30)) for p in j["probabilities"]]

        one = one_systemone if a.systemone_url else one_decide

        def logits_of(rows):
            with ThreadPoolExecutor(a.workers) as ex:
                return list(ex.map(one, rows))

        return logits_of

    import torch

    from trainjev.modeling import last_hidden, load_model

    enabled = getattr(a, "experimental_prompt_bidirectional", False)
    tok, body, head = load_model(a.base, "" if a.base_only else a.adapter,
                                experimental_prompt_bidirectional=enabled)

    @torch.no_grad()
    def logits_of(rows):
        out = [None] * len(rows)
        ids = encode_many(tok, rows)
        items = [(len(e), i) for i, e in enumerate(ids)]
        for mb in micro_batches(items, a.batch_tokens):
            h = last_hidden(body, [ids[i] for i in mb], 0, "cuda")
            for i, hh in zip(mb, h):
                out[i] = head.logits(hh, rows[i]["kind"], len(rows[i]["options"])).tolist()
        return out

    return logits_of


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate on jev-distill-corpus-v3 and fit temperatures")
    ap.add_argument("--base", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--adapter_revision", default=None, help="HF adapter revision/tag")
    ap.add_argument("--base_only", action="store_true")
    ap.add_argument("--decide_url", default="", help="JEV-27B-VL serve_decide.py server, for comparison")
    ap.add_argument("--systemone_url", default="")
    ap.add_argument("--model_name", default="kev-latest", help="'model' field sent to --systemone_url")
    ap.add_argument("--data", default="data/jev-distill-corpus-v3")
    ap.add_argument("--splits", default="test_set_30k:6000,ood:3000", help="name:n_rows (fixed seed-7 subset)")
    ap.add_argument("--fit_calibration", action="store_true")
    ap.add_argument("--temps", default="", help="calibration.json to apply (default: <adapter>/calibration.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch_tokens", type=int, default=24576)
    add_attention_argument(ap)
    a = ap.parse_args(argv)
    adapter_source = a.adapter
    if a.adapter:
        a.adapter = resolve_checkpoint(a.adapter, a.adapter_revision)
    if a.fit_calibration and adapter_source and a.adapter != adapter_source and not os.path.isdir(adapter_source):
        raise ValueError("For refitting calibration, download the HF adapter to a writable local directory first")
    os.makedirs(a.out, exist_ok=True)
    logits_of = make_backend(a)

    summary = {"model": adapter_source or ("base_only" if a.base_only else (a.decide_url or a.systemone_url)),
               "experimental_prompt_bidirectional": a.experimental_prompt_bidirectional}
    temps = None
    if a.fit_calibration:
        cal = [r for r in load_rows(f"{a.data}/calibration.jsonl") if not is_placeholder(r)]
        t = time.time()
        zs = logits_of(cal)
        temps = fit_temperatures(cal, zs)
        summary["calibration"] = {"n": len(cal), "per_kind": temps, "seconds": time.time() - t}
        print("fitted temperatures", temps, flush=True)
        if a.adapter:
            with open(os.path.join(a.adapter, "calibration.json"), "w") as f:
                json.dump({"per_kind": temps, "fit_rows": len(cal)}, f, indent=1)
    elif a.temps or (a.adapter and os.path.exists(os.path.join(a.adapter, "calibration.json"))):
        with open(a.temps or os.path.join(a.adapter, "calibration.json")) as f:
            temps = json.load(f)["per_kind"]

    for spec in a.splits.split(","):
        name, n = spec.split(":")
        rows = load_rows(f"{a.data}/{name}.jsonl", int(n), seed=7)
        rows = [r for r in rows if not (r["kind"] == "choice" and len(r["options"]) > 16)]
        t = time.time()
        zs = logits_of(rows)
        dt = time.time() - t
        raw = [softmax_T(z, 1.0) for z in zs]
        res = {"n": len(rows), "seconds": dt, "raw": metrics(rows, raw)}
        jev = [i for i, r in enumerate(rows) if r.get("source") == "yuri_v3"]
        if jev:
            res["raw_jev_labelled"] = metrics([rows[i] for i in jev], [raw[i] for i in jev])
        if temps:
            cal = [softmax_T(z, temps.get(r["kind"], 1.0)) for z, r in zip(zs, rows)]
            res["calibrated"] = metrics(rows, cal)
            if jev:
                res["calibrated_jev_labelled"] = metrics([rows[i] for i in jev], [cal[i] for i in jev])
        summary[name] = res
        with open(os.path.join(a.out, f"{name}.jsonl"), "w") as f:
            for r, z in zip(rows, zs):
                f.write(json.dumps({"id": r["id"], "kind": r["kind"], "logits": z}) + "\n")
        print(name, json.dumps({k: v for k, v in res.items() if not k.endswith("jev_labelled")})[:900], flush=True)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
