# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: 24 槽位决策头，以及从底座 lm_head 读出 verbalizer 行做初始化。

Decision head and backbone readout.

A JEV System 1 model is a causal LM backbone (here Qwen3.5) with a LoRA adapter and a 24-slot fp32 decision head
that replaces ``lm_head`` on the decision path:

    slot  0-1   noul    false, true
    slot  2-7   score   0 .. 5
    slot  8-23  choice  A .. P

The head rows are initialised from the backbone's ``lm_head`` rows of those verbalizer tokens, so at step 0 the head
reproduces the pretrained model's restricted next-token distribution exactly (see ``scripts/check_init.py``).
"""
import torch
import torch.nn as nn

from trainjev.data import LABELS

VERBALIZERS = ["false", "true", "0", "1", "2", "3", "4", "5"] + LABELS
# Qwen3.5 tokenizer ids of the verbalizers; checked at load time by ``check_verbalizers``.
VERBALIZER_IDS = [3721, 1802, 15, 16, 17, 18, 19, 20] + list(range(32, 48))
RANGES = {"noul": (0, 2), "score": (2, 8), "choice": (8, 24)}
# Every projection of both layer types of Qwen3.5: Gated DeltaNet (in_proj_qkv / in_proj_z / out_proj),
# full attention (q / k / v / o_proj) and the MLP.
LORA_TARGETS = ["in_proj_qkv", "in_proj_z", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]


def check_verbalizers(tok):
    """Every verbalizer must be exactly one token, and each choice label must stay one token inside an option line."""
    got = [tok.encode(v, add_special_tokens=False) for v in VERBALIZERS]
    assert got == [[i] for i in VERBALIZER_IDS], got
    for lab, t in zip(LABELS, VERBALIZER_IDS[8:]):
        assert t in tok.encode(f"x\n{lab}) y", add_special_tokens=False), lab


class DecisionHead(nn.Module):
    """24-slot fp32 head: ``logits[slot] = W[slot] . h + b[slot]``, sliced to the question's kind and option count."""

    def __init__(self, init_rows):
        super().__init__()
        self.weight = nn.Parameter(init_rows.detach().float().clone())  # [24, d]
        self.bias = nn.Parameter(torch.zeros(init_rows.shape[0]))

    def logits(self, h, kind, n):
        lo, _ = RANGES[kind]
        return h.float() @ self.weight[lo: lo + n].T + self.bias[lo: lo + n]


def last_hidden(body, batch_ids, pad_id, device):
    """Right-padded batch -> final-norm hidden state at each row's last real token.

    Right padding keeps every real token's causal context exact for both the attention and the Gated DeltaNet
    (recurrent) layers, so no attention mask is needed.
    """
    L = max(len(x) for x in batch_ids)
    x = torch.full((len(batch_ids), L), pad_id, dtype=torch.long)
    for i, s in enumerate(batch_ids):
        x[i, : len(s)] = torch.tensor(s)
    x = x.to(device)
    h = body(input_ids=x).last_hidden_state
    idx = torch.tensor([len(s) - 1 for s in batch_ids], device=device)
    return h[torch.arange(len(batch_ids), device=device), idx]


def load_model(base, adapter="", dtype=torch.bfloat16, device="cuda"):
    """Load ``(tokenizer, body, head)`` for inference.

    ``adapter=""`` gives the untrained reference (head initialised from ``lm_head``); otherwise the LoRA is merged into
    the text backbone and the trained head is loaded from ``<adapter>/head.pt``.
    """
    import os

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base)
    check_verbalizers(tok)
    m = AutoModelForCausalLM.from_pretrained(base, dtype=dtype)
    head = DecisionHead(m.get_output_embeddings().weight[VERBALIZER_IDS])
    body = m.model
    if adapter:
        from peft import PeftModel

        body = PeftModel.from_pretrained(body, adapter).merge_and_unload()
        sd = torch.load(os.path.join(adapter, "head.pt"), map_location="cpu")
        head.weight.data.copy_(sd["weight"])
        head.bias.data.copy_(sd["bias"])
    del m.lm_head
    body.to(device).eval()
    head.to(device)
    return tok, body, head


def load_temperatures(path):
    """Per-kind temperatures from a ``calibration.json``; 1.0 for every kind when the file is absent."""
    import json
    import os

    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)["per_kind"]
    return {"noul": 1.0, "choice": 1.0, "score": 1.0}
