# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: 兼容 TypeSafe System One 的服务，同时提供 /v1/decide。

TypeSafe System One compatible server (``POST /v1/systemone``) plus a ``serve_decide.py``-compatible ``/v1/decide``.

Backends
  --adapter DIR   a train-jev checkpoint (LoRA merged in memory + 24-slot head), HF transformers, bf16
  --base_only     the untrained reference (head initialised from lm_head)
  --proxy URL     forward to a JEV-27B-VL ``serve_decide.py`` /v1/decide endpoint (one question per call)

Request mapping, identical for every backend and identical to what the model saw in training:
  noul   -> kind noul,   options [false, true]                       answer p(true)
  choice -> kind choice, options "name" or "name: description"       > 16 options: serve_decide.py's tournament
  score  -> 6 levels: kind score with the level legend; other level counts: kind choice over the level texts
Questions of one request are scored independently (each sees the state and its own question only), batched by length.
Images in ``state`` are supported by ``--proxy`` only; for local image decisions see ``trainjev.eval_vision``.

    python -m trainjev.serve --base Qwen/Qwen3.5-4B --adapter runs/jev-4b/best --port 8020
"""
import argparse
import asyncio
import json
import math
import os
import time

import torch

from trainjev.data import encode, micro_batches
from trainjev.modeling import last_hidden, load_model, load_temperatures

S = {}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="System One compatible server for train-jev checkpoints")
    ap.add_argument("--base", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--base_only", action="store_true")
    ap.add_argument("--proxy", default="")
    ap.add_argument("--calibration", default="", help="calibration.json (default: <adapter>/calibration.json)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8020)
    ap.add_argument("--max_len", type=int, default=32768)
    ap.add_argument("--batch_tokens", type=int, default=32768)
    ap.add_argument("--name", default="jev-4b")
    return ap.parse_args(argv)


def setup(a):
    S["args"] = a
    if a.proxy:
        import httpx

        S["client"] = httpx.AsyncClient(base_url=a.proxy, timeout=900)
        return
    adapter = "" if a.base_only else a.adapter
    tok, body, head = load_model(a.base, adapter)
    cal = a.calibration or (os.path.join(adapter, "calibration.json") if adapter else "")
    S.update(tok=tok, body=body, head=head, temps=load_temperatures(cal), lock=asyncio.Lock())
    print("temperatures", S["temps"], flush=True)


# ---------------------------------------------------------------------- inference
def _state_text(state):
    if isinstance(state, str):
        return state
    if isinstance(state, list) and any(isinstance(p, dict) and ("image" in p or p.get("type") == "image_url")
                                       for p in state):
        raise ValueError("images are not supported by this backend")
    return json.dumps(state, ensure_ascii=False)


@torch.no_grad()
def _hf_many(items):
    """items: ``[(kind, state, question, options)]`` (<= 16 options each) -> ``[(probs, n_tokens)]``."""
    tok, body, head, temps, a = S["tok"], S["body"], S["head"], S["temps"], S["args"]
    enc = [encode(tok, {"kind": k, "state": s, "question": q, "options": o}, a.max_len) for k, s, q, o in items]
    out = [None] * len(items)
    for mb in micro_batches([(len(e), i) for i, e in enumerate(enc)], a.batch_tokens):
        h = last_hidden(body, [enc[i] for i in mb], 0, "cuda")
        for i, hh in zip(mb, h):
            k, _, _, o = items[i]
            z = head.logits(hh, k, len(o)) / temps.get(k, 1.0)
            out[i] = (torch.softmax(z, -1).tolist(), len(enc[i]))
    return out


def _hf_dist(kind, state, question, options):
    return _hf_many([(kind, state, question, options)])[0]


def _groups(n, k=16):
    g = math.ceil(n / k)
    base, extra = divmod(n, g)
    out, i = [], 0
    for j in range(g):
        size = base + (1 if j < extra else 0)
        out.append(list(range(i, i + size)))
        i += size
    return out


def tournament(dist, opts):
    """``serve_decide.py``'s > 16-option tournament over a ``dist(options) -> probs`` callable.

    Options are split into balanced groups of <= 16; each group is scored, the group winners plus the best runners-up
    (up to 16) are scored again in a final, and the final distribution is spread back over all options.
    """
    n = len(opts)
    groups = _groups(n)
    in_group = {}
    for g in groups:
        for o, p in zip(g, dist([opts[i] for i in g])):
            in_group[o] = p
    chosen = [max(g, key=lambda o: (in_group[o], -o)) for g in groups]
    rest = sorted((o for g in groups for o in g if o not in set(chosen)), key=lambda o: (-in_group[o], o))
    fin = sorted(chosen + rest[: max(0, 16 - len(chosen))])
    final = dict(zip(fin, dist([opts[i] for i in fin])))
    group_of = {o: gi for gi, g in enumerate(groups) for o in g}
    share, cap = [0.0] * len(groups), [0.0] * len(groups)
    for f in fin:
        share[group_of[f]] += final[f]
        cap[group_of[f]] += in_group[f]
    among = sum(x * y for x, y in zip(share, cap))
    p = [final[o] * among if o in final else share[group_of[o]] * in_group[o] for o in range(n)]
    s = sum(p)
    return [x / s for x in p]


# ---------------------------------------------------------------------- request mapping
def _render(v):
    if v is None or v == "":
        return ""
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def map_question(q):
    """System One question -> ``(kind, question text, options, answer keys)``."""
    instr = _render(q.get("instructions")) or ""
    if q["type"] == "noul":
        c = q.get("criteria") or {}
        extra = "; ".join(f"{k}: {_render(v)}" for k, v in c.items() if _render(v))
        return "noul", instr + (f" ({extra})" if extra else ""), ["false", "true"], ["false", "true"]
    if q["type"] == "choice":
        keys = list(q["criteria"])
        opts = [k if not _render(v) or _render(v) == k else f"{k}: {_render(v)}" for k, v in q["criteria"].items()]
        return "choice", instr, opts, keys
    if q["type"] == "score":
        levels = [_render(x) for x in q["criteria"]]
        keys = [str(i) for i in range(len(levels))]
        if len(levels) == 6:
            legend = "; ".join(f"{i} = {lv}" for i, lv in enumerate(levels))
            return "score", f"{instr} (levels: {legend})", keys, keys
        return "choice", instr, levels, keys
    raise ValueError(f"unsupported question type {q['type']}")


async def _proxy_dist(kind, state, question, options):
    r = await S["client"].post("/v1/decide", json={"kind": kind, "state": state, "question": question,
                                                     "options": options if kind == "choice" else None,
                                                     "thinking": "off"})
    if r.status_code != 200:
        raise ValueError(r.text[:500])
    j = r.json()
    return j["probabilities"], j.get("usage", {}).get("prompt_tokens", 0)


def build_app():
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    app = FastAPI(title="train-jev System One server")

    @app.post("/v1/systemone")
    async def systemone(req: Request):
        body = await req.json()
        t0 = time.perf_counter()
        answers, ntok = {}, 0
        try:
            parsed = {qid: map_question(q) for qid, q in body["questions"].items()}
            for kind, _, opts, _ in parsed.values():
                if kind == "choice" and not 2 <= len(opts) <= 256:
                    return JSONResponse({"error": "choice needs 2-256 options per choice"}, status_code=422)
            res = {}
            if S["args"].proxy:
                outs = await asyncio.gather(*(_proxy_dist(k, body["state"], qt, o) for k, qt, o, _ in parsed.values()))
                res = dict(zip(parsed, outs))
            else:
                st = _state_text(body["state"])
                small = [qid for qid, v in parsed.items() if not (v[0] == "choice" and len(v[2]) > 16)]
                async with S["lock"]:
                    for qid, r in zip(small, _hf_many([(parsed[q][0], st, parsed[q][1], parsed[q][2]) for q in small])):
                        res[qid] = r
                    for qid, (kind, qtext, opts, keys) in parsed.items():
                        if qid in res:
                            continue
                        cnt = [0]

                        def d(o, qtext=qtext, cnt=cnt):
                            pr, n = _hf_dist("choice", st, qtext, o)
                            cnt[0] += n
                            return pr

                        res[qid] = (tournament(d, opts), cnt[0])
            for qid, q in body["questions"].items():
                _, _, _, keys = parsed[qid]
                p, n = res[qid]
                ntok += n
                if q["type"] == "noul":
                    answers[qid] = {"type": "noul", "noul": float(p[1])}
                elif q["type"] == "choice":
                    pr = dict(zip(keys, map(float, p)))
                    answers[qid] = {"type": "choice", "choice": max(pr, key=pr.get), "probabilities": pr}
                else:
                    pr = dict(zip(keys, map(float, p)))
                    answers[qid] = {"type": "score", "score": sum(i * v for i, v in enumerate(p)), "probabilities": pr}
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=422)
        return {"model": S["args"].name, "answers": answers, "usage": {"input_tokens": ntok},
                "latency_ms": 1000 * (time.perf_counter() - t0)}

    @app.get("/health")
    async def health():
        return {"ok": True, "model": S["args"].name}

    @app.post("/v1/decide")
    async def decide(req: Request):
        """``serve_decide.py``-compatible System 1 route: ``{kind, state, question, options}`` -> probabilities."""
        b = await req.json()
        t0 = time.perf_counter()
        kind = b["kind"]
        opts = b.get("options") or (["false", "true"] if kind == "noul" else [str(i) for i in range(6)])
        if S["args"].proxy:
            return JSONResponse({"error": "use the upstream /v1/decide"}, status_code=400)
        try:
            st = _state_text(b["state"])
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=422)
        async with S["lock"]:
            if kind == "choice" and len(opts) > 16:
                cnt = [0]

                def d(o):
                    pr, n = _hf_dist("choice", st, b["question"], o)
                    cnt[0] += n
                    return pr

                p, n = tournament(d, opts), cnt[0]
            else:
                p, n = _hf_dist(kind, st, b["question"], opts)
        k = max(range(len(p)), key=p.__getitem__)
        return {"kind": kind, "options": opts, "probabilities": p, "choice_index": k, "choice": opts[k],
                "adaptation": "native" if kind != "choice" or len(opts) <= 16 else "tournament",
                "model": S["args"].name, "usage": {"prompt_tokens": n}, "elapsed_seconds": time.perf_counter() - t0}

    return app


def main(argv=None):
    import uvicorn

    a = parse_args(argv)
    setup(a)
    uvicorn.run(build_app(), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
