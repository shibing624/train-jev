# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: 兼容 TypeSafe System One API（/v1/systemone）的推理服务。

TypeSafe System One compatible server: ``POST /v1/systemone`` (and ``GET /health``).

Backends
  --adapter DIR   a train-jev checkpoint (LoRA merged in memory + 24-slot head), HF transformers, bf16
  --base_only     the untrained reference (head initialised from lm_head)
  --proxy URL     put JEV-27B-VL behind the same API, for side-by-side comparison: each question is forwarded to its
                  ``serve_decide.py`` server (native ``/v1/decide`` route, one question per call)

Request mapping, identical for every backend and identical to what the model saw in training:
  noul   -> kind noul,   options [false, true]                       answer p(true)
  choice -> kind choice, options "name" or "name: description"       > 16 options: see "Wide choices" below
  score  -> 6 levels: kind score with the level legend; other level counts: kind choice over the level texts
Questions of one request are scored independently (each sees the state and its own question only), batched by length.
Image assets (TypeSafe base64 JSON or multipart) are supported with ``--vision`` or ``--proxy``.
The text-only loading path remains the default; image requests do not require an extra opt-in header.

Wide choices (17-256 options), ``--strategy`` (the three strategies of JEV-27B-VL's ``serve_decide.py``):
  single      (default, as serve_decide.py on Qwen) one pass: labels A-P (trained head rows), then Q-Z, AA, AB, ...
              read from the backbone's own lm_head rows with zero bias
  tournament  (serve_decide.py's default on Gemma) groups of <= 16 + a final of 16; only the trained A-P labels are
              ever read, at the cost of ceil(n/16) + 1 passes
  permute     single over 4 option orders (the given one + 3 seeded shuffles), probabilities averaged

    python -m trainjev.serve --base Qwen/Qwen3.5-4B --adapter runs/jev-4b/best --port 8020
"""
import argparse
import asyncio
import json
import math
import os
import random
import time
import zlib
from contextlib import ExitStack

import torch
from PIL import Image

from trainjev.assets import AssetBundle, AssetError, prepare_assets, read_request
from trainjev.data import encode, micro_batches, prompt
from trainjev.modeling import last_hidden, load_model, load_temperatures, load_vision_model, resolve_checkpoint

S = {}
STRATEGIES = ("single", "tournament", "permute")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="System One compatible server for train-jev checkpoints")
    ap.add_argument("--base", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--adapter_revision", default=None, help="HF adapter revision/tag; unused for local paths")
    ap.add_argument("--base_only", action="store_true")
    ap.add_argument("--vision", action="store_true", help="load the full VL backbone for TypeSafe image assets")
    ap.add_argument("--proxy", default="")
    ap.add_argument("--calibration", default="", help="calibration.json (default: <adapter>/calibration.json)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8020)
    ap.add_argument("--max_len", type=int, default=32768)
    ap.add_argument("--batch_tokens", type=int, default=32768)
    ap.add_argument("--name", default="jev-4b")
    ap.add_argument("--strategy", default="single", choices=STRATEGIES, help="choices with > 16 options")
    ap.add_argument("--device", default="cuda")
    return ap.parse_args(argv)


def setup(a):
    S["args"] = a
    if a.proxy:
        import httpx

        S["client"] = httpx.AsyncClient(base_url=a.proxy, timeout=900)
        return
    adapter = "" if a.base_only else resolve_checkpoint(a.adapter, a.adapter_revision)
    a.adapter = adapter
    dtype = torch.bfloat16 if torch.device(a.device).type == "cuda" else torch.float32
    if a.vision:
        tok, body, head, processor, vision = load_vision_model(a.base, adapter, dtype=dtype, device=a.device)
    else:
        tok, body, head = load_model(a.base, adapter, dtype=dtype, device=a.device)
        processor = vision = None
    cal = a.calibration or (os.path.join(adapter, "calibration.json") if adapter else "")
    S.update(tok=tok, body=body, head=head, processor=processor, vision=vision,
             temps=load_temperatures(cal), lock=asyncio.Lock())
    print("temperatures", S["temps"], "strategy", a.strategy, flush=True)


# ---------------------------------------------------------------------- inference
def _state_text(state):
    if isinstance(state, str):
        return state
    if isinstance(state, list) and any(isinstance(p, dict) and ("image" in p or p.get("type") == "image_url")
                                       for p in state):
        raise ValueError("images are not supported by this backend")
    return json.dumps(state, ensure_ascii=False)


@torch.no_grad()
def _hf_image(kind, state, question, options, assets):
    if S.get("vision") is None or S.get("processor") is None:
        raise ValueError("Local image requests require starting the server with --vision")
    a, tok = S["args"], S["tok"]
    text, images = assets.render(prompt(kind, state, question, options, tok.choice_labels))
    image_processor = getattr(S["processor"], "image_processor", None)
    factor = (getattr(image_processor, "patch_size", 0) or 0) * (getattr(image_processor, "merge_size", 0) or 0)
    size = getattr(image_processor, "size", None)
    max_pixels = size.get("longest_edge") if isinstance(size, dict) else getattr(size, "longest_edge", None)
    with ExitStack() as resources:
        prepared = []
        for image in images:
            aspect = max(image.size) / min(image.size)
            if factor and min(image.size) < factor and aspect <= 200:
                scale = factor / min(image.size)
                image = image.resize((math.ceil(image.width * scale), math.ceil(image.height * scale)),
                                     Image.Resampling.LANCZOS)
                resources.callback(image.close)
            elif factor and aspect > 200:
                # Pad only inputs outside Qwen's grid/aspect assumptions; never crop content.
                working = image.copy()
                resources.callback(working.close)
                side = max(factor, math.isqrt(int(max_pixels or 16_777_216)))
                working.thumbnail((side, side), Image.Resampling.LANCZOS)
                width = max(working.width, factor, math.ceil(working.height / 200))
                height = max(working.height, factor, math.ceil(working.width / 200))
                canvas = Image.new("RGB", (width, height), "white")
                resources.callback(canvas.close)
                canvas.paste(working, ((width - working.width) // 2, (height - working.height) // 2))
                image = canvas
            prepared.append(image)
        # Reject guaranteed-overlong repeated images before allocating their patch tensors.
        if factor and max_pixels and all(math.ceil(im.width / factor) * math.ceil(im.height / factor)
                                        * factor * factor <= max_pixels for im in prepared):
            minimum_tokens = sum((im.width // factor) * (im.height // factor) for im in prepared)
            if minimum_tokens > a.max_len:
                raise ValueError("Image prompt exceeds --max_len; images and text are never silently truncated")
        inputs = S["processor"](text=[text], images=prepared, return_tensors="pt", truncation=False)
    tokens = int(inputs["input_ids"].shape[-1])
    if tokens > a.max_len:
        raise ValueError("Image prompt exceeds --max_len; images and text are never silently truncated")
    inputs = inputs.to(a.device)
    h = S["vision"](**inputs).last_hidden_state[0, -1]
    z = S["head"].logits(h, kind, len(options)) / S["temps"].get(kind, 1.0)
    return torch.softmax(z, -1).tolist(), tokens


@torch.no_grad()
def _hf_many(items, assets=None):
    """items: ``[(kind, state, question, options)]`` (<= 256 options each) -> ``[(probs, n_tokens)]``."""
    tok, body, head, temps, a = S["tok"], S["body"], S["head"], S["temps"], S["args"]
    if assets and assets.markers:
        out, text_items, text_indices = [None] * len(items), [], []
        for index, item in enumerate(items):
            if assets.has_images(prompt(*item, tok.choice_labels)):
                out[index] = _hf_image(*item, assets)
            else:
                text_indices.append(index)
                text_items.append(item)
        for index, result in zip(text_indices, _hf_many(text_items)):
            out[index] = result
        return out
    enc = [encode(tok, {"kind": k, "state": s, "question": q, "options": o}, a.max_len, tok.choice_labels)
           for k, s, q, o in items]
    out = [None] * len(items)
    for mb in micro_batches([(len(e), i) for i, e in enumerate(enc)], a.batch_tokens):
        h = last_hidden(body, [enc[i] for i in mb], 0, a.device)
        for i, hh in zip(mb, h):
            k, _, _, o = items[i]
            z = head.logits(hh, k, len(o)) / temps.get(k, 1.0)
            out[i] = (torch.softmax(z, -1).tolist(), len(enc[i]))
    return out


def _groups(n, k=16):
    g = math.ceil(n / k)
    base, extra = divmod(n, g)
    out, i = [], 0
    for j in range(g):
        size = base + (1 if j < extra else 0)
        out.append(list(range(i, i + size)))
        i += size
    return out


def tournament(dist, opts, dist_many=None):
    """``serve_decide.py``'s > 16-option tournament over a ``dist(options) -> probs`` callable.

    Options are split into balanced groups of <= 16; each group is scored, the group winners plus the best runners-up
    (up to 16) are scored again in a final, and the final distribution is spread back over all options.
    ``dist_many([options, ...]) -> [probs, ...]``, when given, scores all groups in one batch.
    """
    n = len(opts)
    groups = _groups(n)
    scored = dist_many([[opts[i] for i in g] for g in groups]) if dist_many else [dist([opts[i] for i in g])
                                                                                  for g in groups]
    in_group = {}
    for g, ps in zip(groups, scored):
        for o, p in zip(g, ps):
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


def permute_orders(question, n, k=4):
    """The given order plus ``k - 1`` shuffles seeded by the question text (as in ``serve_decide.py``)."""
    rng = random.Random(zlib.crc32(question.encode()))
    return [list(range(n))] + [rng.sample(range(n), n) for _ in range(k - 1)]


def permute(dist_many, question, opts, k=4):
    """Average of ``dist`` over :func:`permute_orders`; ``dist_many([options, ...]) -> [probs, ...]``."""
    orders = permute_orders(question, len(opts), k)
    res = dist_many([[opts[i] for i in o] for o in orders])
    p = [0.0] * len(opts)
    for o, r in zip(orders, res):
        for i, v in zip(o, r):
            p[i] += v / len(orders)
    return p


def wide_dist(strategy, kind, state, question, opts, assets=None):
    """-> ``(probs, n_tokens)`` for any question; ``strategy`` only matters for choices with > 16 options."""
    cnt = [0]

    def many(option_lists):
        items = [(kind, state, question, o) for o in option_lists]
        r = _hf_many(items, assets) if assets and assets.markers else _hf_many(items)
        cnt[0] += sum(n for _, n in r)
        return [p for p, _ in r]

    if kind != "choice" or len(opts) <= 16 or strategy == "single":
        p = many([opts])[0]
    elif strategy == "permute":
        p = permute(many, question, opts)
    elif strategy == "tournament":
        p = tournament(lambda o: many([o])[0], opts, many)
    else:
        raise ValueError(f"unknown strategy {strategy}")
    return p, cnt[0]


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


def _proxy_image_fields(state, question, options, assets):
    """Native /decide accepts images only in state: label question/criterion images there."""
    parts = [part if isinstance(part, str) else {"image": part.data_url}
             for part in assets.parts(_state_text(state))]
    count = 0

    def reference_images(text, location):
        nonlocal count
        result = []
        for part in assets.parts(text):
            if isinstance(part, str):
                result.append(part)
            else:
                count += 1
                label = f"{location} image {count}"
                result.append(f"[{label} in state]")
                parts.extend([f"\n[{label}]\n", {"image": part.data_url}])
        return "".join(result)

    question = reference_images(question, "question")
    options = [reference_images(option, f"option {index}") for index, option in enumerate(options)]
    return parts, question, options


async def _proxy_dist(kind, state, question, options, assets=None, strategy=None):
    if assets and assets.markers:
        state, question, options = _proxy_image_fields(state, question, options, assets)
    payload = {"kind": kind, "state": state, "question": question,
               "options": options if kind == "choice" else None, "thinking": "off"}
    if strategy is not None:
        payload["strategy"] = strategy
    r = await S["client"].post("/v1/decide", json=payload)
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
        t0 = time.perf_counter()
        answers, ntok = {}, 0
        assets = AssetBundle()
        try:
            original, uploads = await read_request(req)
            if original.get("assets") and not S["args"].proxy and S.get("vision") is None:
                raise ValueError("Local image requests require starting the server with --vision")
            body, assets = await asyncio.to_thread(prepare_assets, original, uploads)
            if "state" not in body or not isinstance(body.get("questions"), dict) or not body["questions"]:
                raise ValueError("state and a nonempty questions object are required")
            for q in body["questions"].values():
                if not isinstance(q, dict) or q.get("type") not in ("noul", "choice", "score"):
                    raise ValueError("questions must contain noul, choice, or score objects")
                if q["type"] == "choice" and not isinstance(q.get("criteria"), dict):
                    raise ValueError("choice criteria must be an object")
                if q["type"] == "noul" and q.get("criteria") is not None and not isinstance(q["criteria"], dict):
                    raise ValueError("noul criteria must be an object")
                if q["type"] == "score" and (not isinstance(q.get("criteria"), list)
                                             or not 2 <= len(q["criteria"]) <= 256):
                    raise ValueError("score criteria must contain 2-256 levels")
            strategy = body.get("strategy") or S["args"].strategy
            if strategy not in STRATEGIES:
                raise ValueError(f"strategy must be one of {STRATEGIES}")
            parsed = {qid: map_question(q) for qid, q in body["questions"].items()}
            for kind, _, opts, _ in parsed.values():
                if kind == "choice" and not 2 <= len(opts) <= 256:
                    return JSONResponse({"error": "choice needs 2-256 options per choice"}, status_code=422)
            res = {}
            if S["args"].proxy:
                outs = await asyncio.gather(*(_proxy_dist(k, body["state"], qt, o, assets, strategy)
                                             for k, qt, o, _ in parsed.values()))
                res = dict(zip(parsed, outs))
            else:
                st = _state_text(body["state"])
                if assets.markers and S.get("vision") is None:
                    raise ValueError("Local image requests require starting the server with --vision")
                one_pass = strategy == "single"
                small = [qid for qid, v in parsed.items() if one_pass or not (v[0] == "choice" and len(v[2]) > 16)]
                async with S["lock"]:
                    items = [(parsed[q][0], st, parsed[q][1], parsed[q][2]) for q in small]
                    outputs = _hf_many(items, assets) if assets.markers else _hf_many(items)
                    for qid, r in zip(small, outputs):
                        res[qid] = r
                    for qid, (kind, qtext, opts, _) in parsed.items():
                        if qid not in res:
                            res[qid] = wide_dist(strategy, kind, st, qtext, opts, assets)
            for qid, q in body["questions"].items():
                _, _, _, keys = parsed[qid]
                p, n = res[qid]
                ntok += n
                if q["type"] == "noul":
                    answers[qid] = {"type": "noul", "noul": float(p[1])}
                elif q["type"] == "choice":
                    pr = dict(zip(keys, map(float, p)))
                    choice = max(pr, key=pr.__getitem__)
                    answers[qid] = {"type": "choice", "choice": choice, "confidence": pr[choice],
                                    "probabilities": pr}
                else:
                    pr = dict(zip(keys, map(float, p)))
                    answers[qid] = {"type": "score", "score": sum(i * v for i, v in enumerate(p)),
                                    "confidence": max(pr.values()),
                                    "legend": {str(i): criterion for i, criterion in enumerate(q["criteria"])},
                                    "probabilities": pr}
        except AssetError as e:
            return JSONResponse({"error": str(e)}, status_code=e.status_code)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=422)
        except torch.OutOfMemoryError:
            return JSONResponse({"error": "Insufficient inference memory; reduce image count or context"},
                                status_code=503)
        finally:
            assets.close()
        return {"model": S["args"].name, "answers": answers, "usage": {"input_tokens": ntok},
                "latency_ms": 1000 * (time.perf_counter() - t0)}

    @app.get("/health")
    async def health():
        return {"ok": True, "model": S["args"].name,
                "images": bool(S["args"].proxy or getattr(S["args"], "vision", False))}

    return app


def main(argv=None):
    import uvicorn

    a = parse_args(argv)
    setup(a)
    uvicorn.run(build_app(), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
