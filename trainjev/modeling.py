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

Choices with more than 16 options (``single`` / ``permute`` strategies in ``trainjev.serve``) read the extra labels
(``Q`` .. ``Z``, ``AA``, ...) from the backbone's own ``lm_head`` rows with zero bias, as JEV-27B-VL's
``serve_decide.py`` does; these rows are never trained and are not part of ``head.pt``.
"""
import string
from pathlib import Path

import torch
import torch.nn as nn

from trainjev.data import LABELS

VERBALIZERS = ["false", "true", "0", "1", "2", "3", "4", "5"] + LABELS
# Qwen3.5 tokenizer ids of the verbalizers; checked at load time by ``check_verbalizers``.
VERBALIZER_IDS = [3721, 1802, 15, 16, 17, 18, 19, 20] + list(range(32, 48))
RANGES = {"noul": (0, 2), "score": (2, 8), "choice": (8, 24)}
MAX_OPTIONS = 256
# Every projection of both layer types of Qwen3.5: Gated DeltaNet (in_proj_qkv / in_proj_z / out_proj),
# full attention (q / k / v / o_proj) and the MLP.
LORA_TARGETS = ["in_proj_qkv", "in_proj_z", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]


def resolve_checkpoint(checkpoint, revision=None):
    """Resolve a local inference directory or an HF adapter repo, including its custom head and temperatures."""
    if not checkpoint:
        return ""
    path = Path(checkpoint).expanduser()
    if path.is_dir():
        return str(path)
    if path.exists() or str(checkpoint).startswith(("/", "./", "../", "~")) or len(path.parts) > 2:
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=str(checkpoint), revision=revision,
                             allow_patterns=["adapter_config.json", "adapter_model.safetensors", "model.safetensors",
                                             "head.pt", "calibration.json", "training_recipe.json",
                                             "release_manifest.json"])


def check_verbalizers(tok):
    """Every verbalizer must be exactly one token, and each choice label must stay one token inside an option line."""
    got = [tok.encode(v, add_special_tokens=False) for v in VERBALIZERS]
    assert got == [[i] for i in VERBALIZER_IDS], got
    for lab, t in zip(LABELS, VERBALIZER_IDS[8:]):
        assert t in tok.encode(f"x\n{lab}) y", add_special_tokens=False), lab


def choice_labels(tok, n=MAX_OPTIONS):
    """``serve_decide.py``'s option labels: ``A`` .. ``Z`` then ``AA`` .. ``ZZ``, keeping those that are one token on
    their own and inside an option line; the first 16 are the trained ``A`` .. ``P``. -> ``(labels, token ids)``."""
    labs, ids = [], []
    for lab in list(string.ascii_uppercase) + [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]:
        t = tok.encode(lab, add_special_tokens=False)
        if len(t) == 1 and t[0] in tok.encode(f"x\n{lab}) y", add_special_tokens=False):
            labs.append(lab)
            ids.append(t[0])
        if len(labs) == n:
            break
    assert labs[:16] == LABELS and ids[:16] == VERBALIZER_IDS[8:], "first labels must be the trained A-P head"
    return labs, ids


class DecisionHead(nn.Module):
    """24-slot fp32 head: ``logits[slot] = W[slot] . h + b[slot]``, sliced to the question's kind and option count.

    ``wide`` (optional, untrained, not saved) holds the ``lm_head`` rows of choice labels 17+ for wide choices.
    """

    def __init__(self, init_rows):
        super().__init__()
        self.weight = nn.Parameter(init_rows.detach().float().clone())  # [24, d]
        self.bias = nn.Parameter(torch.zeros(init_rows.shape[0]))
        self.register_buffer("wide", torch.zeros(0, init_rows.shape[1]), persistent=False)
        self.attention_mode = "causal"
        self.lm_head = None

    def logits(self, h, kind, n):
        lo, _ = RANGES[kind]
        if kind != "choice" or n <= 16:
            return h.float() @ self.weight[lo: lo + n].T + self.bias[lo: lo + n]
        if n - 16 > self.wide.shape[0]:
            raise ValueError(f"{n} options need wide label rows (set head.wide, as load_model does)")
        w = torch.cat([self.weight[lo: lo + 16], self.wide[: n - 16]])
        b = torch.cat([self.bias[lo: lo + 16], self.bias.new_zeros(n - 16)])
        return h.float() @ w.T + b

    def slot_logits(self, h):
        """``[B, d]`` -> ``[B, 24]`` logits over every slot (training path; see ``losses.kl_rps_batch``)."""
        return h.float() @ self.weight.T + self.bias


def last_hidden(body, batch_ids, pad_id, device, attention="causal"):
    """Final-norm hidden state at each row's last real token.

    ``causal`` right-pads and leaves every layer causal, which is the historical System 1 path.
    ``bidirectional`` runs one unpadded forward per exact length, with full-attention layers attending over the
    whole row and linear-attention layers left causal. See ``trainjev.attention``.
    """
    from trainjev.attention import apply_prompt_attention, grouped_hidden

    if attention not in ("causal", "bidirectional"):
        raise ValueError(f"attention must be causal or bidirectional, got {attention}")
    apply_prompt_attention(body, attention)
    if attention == "bidirectional":
        return grouped_hidden(body, batch_ids, device)
    L = max(len(x) for x in batch_ids)
    x = torch.full((len(batch_ids), L), pad_id, dtype=torch.long)
    for i, s in enumerate(batch_ids):
        x[i, : len(s)] = torch.tensor(s)
    x = x.to(device)
    h = body(input_ids=x).last_hidden_state
    idx = torch.tensor([len(s) - 1 for s in batch_ids], device=device)
    return h[torch.arange(len(batch_ids), device=device), idx]


def is_full_checkpoint(ckpt):
    """A ``--full_ft`` checkpoint holds the text-model weights (``model.safetensors``) instead of a LoRA adapter."""
    import os

    return bool(ckpt) and os.path.exists(os.path.join(ckpt, "model.safetensors")) and not os.path.exists(
        os.path.join(ckpt, "adapter_config.json"))


def load_trained(text_model, head, ckpt, *, trainable=False):
    """Apply a train-jev checkpoint to ``text_model`` (a Qwen3.5 text model) and ``head``; returns the text model.

    LoRA checkpoints are merged into the weights; full checkpoints replace every weight (strict key match).
    """
    import os

    ckpt = resolve_checkpoint(ckpt)
    if is_full_checkpoint(ckpt):
        if trainable:
            raise ValueError("A trainable LoRA initialization requires an adapter checkpoint, not full weights")
        from safetensors.torch import load_file

        sd = load_file(os.path.join(ckpt, "model.safetensors"))
        dtype = next(text_model.parameters()).dtype
        missing, unexpected = text_model.load_state_dict({k: v.to(dtype) for k, v in sd.items()}, strict=False)
        if missing or unexpected:
            raise ValueError(f"full checkpoint does not match the base: missing {missing[:5]} "
                             f"unexpected {unexpected[:5]}")
    else:
        from peft import PeftModel

        if trainable:
            text_model = PeftModel.from_pretrained(text_model, ckpt, is_trainable=True)
        else:
            text_model = PeftModel.from_pretrained(text_model, ckpt).merge_and_unload()
    sd = torch.load(os.path.join(ckpt, "head.pt"), map_location="cpu", weights_only=True)
    if tuple(sd["weight"].shape) != tuple(head.weight.shape) or tuple(sd["bias"].shape) != tuple(head.bias.shape):
        raise ValueError("Decision head dimensions do not match the selected backbone")
    if "verbalizer_ids" in sd and list(sd["verbalizer_ids"]) != VERBALIZER_IDS:
        raise ValueError("Decision head verbalizer ids do not match the tokenizer protocol")
    if not torch.isfinite(sd["weight"]).all() or not torch.isfinite(sd["bias"]).all():
        raise ValueError("Decision head contains nonfinite values")
    head.weight.data.copy_(sd["weight"])
    head.bias.data.copy_(sd["bias"])
    if "attention" in sd:
        if sd["attention"] not in ("causal", "bidirectional"):
            raise ValueError(f"checkpoint attention must be causal or bidirectional, got {sd['attention']}")
        head.attention_mode = sd["attention"]
    return text_model


def checkpoint_attention(checkpoint):
    """Attention mode recorded in ``head.pt``. Checkpoints written before that field are causal."""
    import os

    path = resolve_checkpoint(checkpoint) if checkpoint else ""
    if not path:
        return "causal"
    head_path = os.path.join(path, "head.pt")
    if not os.path.exists(head_path):
        return "causal"
    sd = torch.load(head_path, map_location="cpu", weights_only=True)
    if "attention" not in sd:
        return "causal"
    if sd["attention"] not in ("causal", "bidirectional"):
        raise ValueError(f"checkpoint attention must be causal or bidirectional, got {sd['attention']}")
    return sd["attention"]


def load_model(base, adapter="", dtype=torch.bfloat16, device="cuda", keep_lm_head=False):
    """Load ``(tokenizer, body, head)`` for inference.

    ``adapter=""`` gives the untrained reference (head initialised from ``lm_head``); otherwise ``adapter`` is a
    train-jev checkpoint directory (LoRA or full-parameter, see ``load_trained``). Bidirectional checkpoints load
    SDPA attention. ``keep_lm_head`` retains the backbone output projection on ``head.lm_head`` for AutoThink.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from trainjev.attention import apply_prompt_attention

    mode = checkpoint_attention(adapter)
    load = {"dtype": dtype}
    if mode == "bidirectional":
        load["attn_implementation"] = "sdpa"
    adapter = resolve_checkpoint(adapter)
    tok = AutoTokenizer.from_pretrained(base)
    check_verbalizers(tok)
    m = AutoModelForCausalLM.from_pretrained(base, **load)
    head = DecisionHead(m.get_output_embeddings().weight[VERBALIZER_IDS])
    tok.choice_labels, wide_ids = choice_labels(tok)  # labels 17-256: untrained lm_head rows, see module docstring
    head.wide = m.get_output_embeddings().weight[wide_ids[16:]].detach().float().clone()
    body = m.model
    if adapter:
        body = load_trained(body, head, adapter)
    if keep_lm_head:
        head.lm_head = m.lm_head
    else:
        del m.lm_head
    body.to(device).eval()
    head.to(device)
    if head.lm_head is not None:
        head.lm_head.to(device).eval()
    apply_prompt_attention(body, head.attention_mode)
    return tok, body, head


def load_vision_model(base, adapter="", dtype=torch.bfloat16, device="cuda", keep_lm_head=False):
    """Load the full Qwen VL model; share its trained text body with the ordinary text path."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from trainjev.attention import apply_prompt_attention

    mode = checkpoint_attention(adapter)
    load = {"dtype": dtype}
    if mode == "bidirectional":
        load["attn_implementation"] = "sdpa"
    adapter = resolve_checkpoint(adapter)
    processor = AutoProcessor.from_pretrained(base)
    tok = processor.tokenizer
    check_verbalizers(tok)
    model = AutoModelForImageTextToText.from_pretrained(base, **load)
    if not hasattr(model.model, "language_model"):
        raise ValueError("--vision requires a Qwen3.5 / Qwen3.8 multimodal backbone")
    head = DecisionHead(model.get_output_embeddings().weight[VERBALIZER_IDS])
    tok.choice_labels, wide_ids = choice_labels(tok)
    head.wide = model.get_output_embeddings().weight[wide_ids[16:]].detach().float().clone()
    if adapter:
        model.model.language_model = load_trained(model.model.language_model, head, adapter)
    if keep_lm_head:
        head.lm_head = model.lm_head
    else:
        del model.lm_head
    model.to(device).eval()
    head.to(device)
    if head.lm_head is not None:
        head.lm_head.to(device).eval()
    apply_prompt_attention(model.model.language_model, head.attention_mode)
    return tok, model.model.language_model, head, processor, model.model


def load_temperatures(path):
    """Per-kind temperatures from a ``calibration.json``; 1.0 for every kind when the file is absent."""
    import json
    import os

    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)["per_kind"]
    return {"noul": 1.0, "choice": 1.0, "score": 1.0}
