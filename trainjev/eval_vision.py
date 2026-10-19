# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: 在 Decision Index 视觉集上做零样本图像决策评测。

Zero-shot image decisions on the Decision Index public vision sets.

The adapter and head are trained on text only; the Qwen3.5 vision tower is untouched. This script loads the full VL
model, merges the LoRA into its language model, and reads the decision head at the last prompt token of
``serve_decide.py``'s image prompt::

    [kind] choice\n[state] <images>\n[question] {q}\n[options]\nA) ..\n[decision]:

Requires a checkout of https://github.com/apolinario/decision-index with the vision suite rebuilt
(``python -m decision_index suite rebuild --vision --work work --sets ...``); pass it as ``--di_root``.

    python -m trainjev.eval_vision --base Qwen/Qwen3.5-4B --adapter runs/jev-4b/best \
        --di_root third_party/decision-index --out eval/vision-jev-4b
"""
import argparse
import base64
import collections
import hashlib
import io
import json
import os
import sys
import time

from trainjev.attention import apply_prompt_attention
from trainjev.modeling import resolve_checkpoint

DEFAULT_SETS = "full_cvbench,full_blink,full_realworldqa,full_charxiv,full_infovqa"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Zero-shot image decisions (Decision Index vision sets)")
    ap.add_argument("--base", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--adapter_revision", default=None, help="HF adapter revision/tag")
    ap.add_argument("--base_only", action="store_true")
    ap.add_argument("--decide_url", default="", help="JEV-27B-VL serve_decide.py server, for comparison")
    ap.add_argument("--di_root", required=True, help="decision-index checkout with work/vision rebuilt")
    ap.add_argument("--sets", default=DEFAULT_SETS)
    ap.add_argument("--per_set", type=int, default=300, help="first N items per set by sha256(id)")
    ap.add_argument("--max_pixels", type=int, default=0, help="0 = the kit's default")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.adapter:
        a.adapter = resolve_checkpoint(a.adapter, a.adapter_revision)
    os.makedirs(a.out, exist_ok=True)
    sys.path.insert(0, a.di_root)
    from decision_index.vision import images as img
    from decision_index.vision import rows as vrows

    max_pixels = a.max_pixels or img.MAX_PIXELS
    root = os.path.join(a.di_root, "work", "vision")

    items = []
    for s in a.sets.split(","):
        its = vrows.select(s, vrows.load_items(root, s))
        items += sorted(its, key=lambda it: hashlib.sha256(it["id"].encode()).hexdigest())[: a.per_set]
    print(f"{len(items)} items from {a.sets}", flush=True)

    def question_of(it):
        q = vrows.request(it)["answer"]
        return q["instructions"], list(q["criteria"].values()), list(q["criteria"])

    def pil(p):
        return img.pil(p, root, max_pixels).convert("RGB")

    if a.decide_url:
        from concurrent.futures import ThreadPoolExecutor

        import httpx

        cli = httpx.Client(base_url=a.decide_url, timeout=900)

        def data_url(p):
            b = io.BytesIO()
            pil(p).save(b, format="PNG")
            return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()

        def one(it):
            q, opts, _ = question_of(it)
            t = time.perf_counter()
            j = cli.post("/v1/decide", json={"kind": "choice", "state": [{"image": data_url(p)} for p in it["images"]],
                                             "question": q, "options": opts, "thinking": "off"}).json()
            return j["probabilities"], 1000 * (time.perf_counter() - t), j.get("usage", {}).get("prompt_tokens")

        with ThreadPoolExecutor(a.workers) as ex:
            res = list(ex.map(one, items))
    else:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        from trainjev.data import LABELS
        from trainjev.modeling import VERBALIZER_IDS, DecisionHead, load_temperatures, load_trained

        proc = AutoProcessor.from_pretrained(a.base)
        m = AutoModelForImageTextToText.from_pretrained(a.base, dtype=torch.bfloat16)
        head = DecisionHead(m.get_output_embeddings().weight[VERBALIZER_IDS])
        adapter = "" if a.base_only else a.adapter
        if adapter:
            m.model.language_model = load_trained(m.model.language_model, head, adapter)
        apply_prompt_attention(m.model.language_model, head.attention_mode)
        temps = load_temperatures(os.path.join(adapter, "calibration.json") if adapter else "")
        m.cuda().eval()
        head.cuda()
        vis = "<|vision_start|><|image_pad|><|vision_end|>"

        @torch.no_grad()
        def one(it):
            q, opts, _ = question_of(it)
            text = ("[kind] choice\n[state] " + vis * len(it["images"]) + f"\n[question] {q}\n[options]\n"
                    + "\n".join(f"{LABELS[i]}) {o}" for i, o in enumerate(opts)) + "\n[decision]:")
            x = proc(text=[text], images=[pil(p) for p in it["images"]], return_tensors="pt").to("cuda")
            t = time.perf_counter()
            h = m.model(**x).last_hidden_state[0, -1]
            z = head.logits(h, "choice", len(opts)) / temps.get("choice", 1.0)
            torch.cuda.synchronize()
            return torch.softmax(z, -1).tolist(), 1000 * (time.perf_counter() - t), int(x["input_ids"].shape[1])

        res = [one(it) for it in items]

    by = collections.defaultdict(list)
    with open(os.path.join(a.out, "rows.jsonl"), "w") as f:
        for it, (p, ms, ntok) in zip(items, res):
            _, opts, keys = question_of(it)
            pred = keys[max(range(len(p)), key=p.__getitem__)]
            ok = int(pred == it["gold"])
            by[it["bench"]].append((ok, len(opts), max(p), ms))
            f.write(json.dumps({"id": it["id"], "bench": it["bench"], "gold": it["gold"], "pred": pred, "probs": p,
                                "ms": ms, "tokens": ntok}) + "\n")
    summ = {}
    for b, v in by.items():
        acc = sum(x[0] for x in v) / len(v)
        chance = sum(1 / x[1] for x in v) / len(v)
        summ[b] = {"n": len(v), "acc": acc, "chance": chance, "skill": max(0.0, (acc - chance) / (1 - chance)),
                   "p50_ms": sorted(x[3] for x in v)[len(v) // 2]}
    summ["_mean"] = {k: sum(s[k] for s in summ.values()) / len(by) for k in ("acc", "skill")}
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
