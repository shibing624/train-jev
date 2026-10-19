# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: 全注意力层的 prompt 双向注意力。线性注意力（Gated DeltaNet）保持因果。

Prompt-bidirectional attention for Qwen3.5 full-attention layers.

Instant decision readouts see the whole prompt before the answer token, so those layers may attend to later
question and option tokens. The linear-attention layers stay causal: their recurrence is unchanged, and they keep
whatever mask the stock forward builds. This matches the StartLux-Decision Instant design (arXiv 2610.17309):
train with the modified mask; do not flip a causally trained checkpoint only at inference.

``create_causal_mask`` returns ``None`` on an unpadded row and SDPA then uses ``module.is_causal``. Turning
bidirectional attention on therefore does two things: full-attention modules get ``is_causal = False``, and the
Qwen mask builder returns no causal mask (or a padding mask that is symmetric inside each row).
"""
import torch

_STATE = {"on": False, "installed": False, "original": None}


def _install():
    if _STATE["installed"]:
        return
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen

    original = qwen.create_causal_mask

    def create_causal_mask(**kwargs):
        if not _STATE["on"]:
            return original(**kwargs)
        attention_mask = kwargs.get("attention_mask")
        embeds = kwargs["inputs_embeds"]
        if attention_mask is None or bool((attention_mask == 1).all()):
            return None
        real = attention_mask.to(dtype=torch.bool)
        allowed = real[:, :, None] & real[:, None, :]
        length = real.shape[-1]
        eye = torch.eye(length, dtype=torch.bool, device=real.device)
        allowed = allowed | (~real)[:, :, None] & eye.unsqueeze(0)
        mask = embeds.new_zeros(real.shape[0], 1, length, length)
        mask.masked_fill_(~allowed[:, None], torch.finfo(embeds.dtype).min)
        return mask

    qwen.create_causal_mask = create_causal_mask
    _STATE["installed"] = True
    _STATE["original"] = original


def full_attention_modules(body):
    """``self_attn`` modules of the first decoder that declares ``layer_types``. Empty for non-Qwen stubs."""
    if not hasattr(body, "modules"):
        return []
    for module in body.modules():
        if not hasattr(module, "config") or not hasattr(module, "layers"):
            continue
        layer_types = module.config.layer_types if hasattr(module.config, "layer_types") else None
        if not layer_types:
            continue
        return [module.layers[i].self_attn for i, kind in enumerate(layer_types) if kind == "full_attention"]
    return []


def apply_prompt_attention(body, attention):
    """Make full-attention layers bidirectional or restore the stock causal mask. Linear layers are not edited."""
    if attention not in ("causal", "bidirectional"):
        raise ValueError(f"attention must be causal or bidirectional, got {attention}")
    _install()
    _STATE["on"] = attention == "bidirectional"
    for attn in full_attention_modules(body):
        attn.is_causal = attention == "causal"


def grouped_hidden(body, batch_ids, device):
    """Last-token states, one forward per exact length so no padding token enters a bidirectional row."""
    groups = {}
    for index, ids in enumerate(batch_ids):
        groups.setdefault(len(ids), []).append(index)
    hidden = [None] * len(batch_ids)
    for length, indexes in groups.items():
        tokens = torch.tensor([batch_ids[i] for i in indexes], dtype=torch.long, device=device)
        states = body(input_ids=tokens, use_cache=False).last_hidden_state
        for row, index in enumerate(indexes):
            hidden[index] = states[row, length - 1]
    return torch.stack(hidden)
