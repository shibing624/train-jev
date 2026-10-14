# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: 训练前检查：未训练的决策头必须等于底座在 verbalizer 上的 next-token 读出。

Sanity gate before training: the untrained head must equal the base model's restricted next-token readout.

At initialisation the 24 head rows are copies of lm_head's verbalizer rows, so softmax over the head slots must match
softmax over lm_head logits restricted to the same tokens (|dp| < 1e-5 in fp32). Then prints the untrained reference
metrics on a validation subset.

    python scripts/check_init.py --base Qwen/Qwen3.5-4B --data data/jev-distill-corpus-v3
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trainjev import (  # noqa: E402
    RANGES,
    VERBALIZER_IDS,
    DecisionHead,
    check_verbalizers,
    encode,
    last_hidden,
    load_rows,
    metrics,
)

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="Qwen/Qwen3.5-4B")
ap.add_argument("--data", default="data/jev-distill-corpus-v3")
ap.add_argument("--n", type=int, default=96, help="rows for the equivalence check")
ap.add_argument("--n_metric", type=int, default=2000, help="rows for the untrained reference metrics")
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.base)
check_verbalizers(tok)
m = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.float32).cuda().eval()
head = DecisionHead(m.get_output_embeddings().weight[VERBALIZER_IDS]).cuda()
rows = load_rows(f"{a.data}/validation.jsonl", a.n_metric, seed=1)
worst = 0.0
with torch.no_grad():
    for r in rows[: a.n]:
        x = torch.tensor([encode(tok, r)]).cuda()
        logits = m(x).logits[0, -1].float()
        kind, n = r["kind"], len(r["options"])
        lo, _ = RANGES[kind]
        ref = torch.softmax(logits[VERBALIZER_IDS[lo: lo + n]], -1)
        got = torch.softmax(head.logits(m.model(input_ids=x).last_hidden_state[0, -1], kind, n), -1)
        worst = max(worst, (ref - got).abs().max().item())
print(f"equivalence gate: rows {a.n}  max|dp| = {worst:.3e}  -> {'PASS' if worst < 1e-5 else 'FAIL'}")

m = m.to(torch.bfloat16)
probs = []
t = time.time()
with torch.no_grad():
    for i in range(0, len(rows), 32):
        b = rows[i: i + 32]
        h = last_hidden(m.model, [encode(tok, r) for r in b], tok.pad_token_id or 0, "cuda")
        for r, hh in zip(b, h):
            probs.append(torch.softmax(head.logits(hh, r["kind"], len(r["options"])), -1).tolist())
print(f"untrained reference on {len(rows)} validation rows ({time.time() - t:.0f}s):")
print(json.dumps(metrics(rows, probs), indent=1))
