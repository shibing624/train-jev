# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: 对同一注意力模式的 LoRA 和决策头做均匀权重平均。
"""
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file

from trainjev.modeling import VERBALIZER_IDS


def _load_head(path):
    state = torch.load(os.path.join(path, "head.pt"), map_location="cpu", weights_only=True)
    if list(state["verbalizer_ids"]) != VERBALIZER_IDS:
        raise ValueError(f"{path} verbalizer ids do not match this tree")
    mode = state["attention"] if "attention" in state else "causal"
    if mode not in ("causal", "bidirectional"):
        raise ValueError(f"{path} has attention {mode}")
    return state, mode


def average_checkpoints(paths, out):
    """Write the uniform average of LoRA adapters and decision heads. Configs and attention modes must match."""
    if len(paths) < 2:
        raise ValueError("averaging needs at least two checkpoints")
    heads, modes = [], []
    for path in paths:
        state, mode = _load_head(path)
        heads.append(state)
        modes.append(mode)
    if len(set(modes)) != 1:
        raise ValueError(f"attention modes differ: {modes}")
    configs, tensors = [], []
    for path in paths:
        with open(os.path.join(path, "adapter_config.json"), encoding="utf-8") as stream:
            configs.append(json.load(stream))
        name = os.path.join(path, "adapter_model.safetensors")
        if not os.path.exists(name):
            raise FileNotFoundError(f"{path} has no adapter_model.safetensors")
        tensors.append(load_file(name))
    signature = [(cfg["r"], cfg["lora_alpha"], tuple(cfg["target_modules"])) for cfg in configs]
    if len(set(signature)) != 1:
        raise ValueError("adapter configs differ in r, alpha or target modules")
    keys = set(tensors[0])
    if any(set(item) != keys for item in tensors[1:]):
        raise ValueError("adapter tensors do not share keys")
    averaged = {key: torch.stack([item[key].float() for item in tensors]).mean(0).to(tensors[0][key].dtype)
                for key in keys}
    weight = torch.stack([state["weight"].float() for state in heads]).mean(0)
    bias = torch.stack([state["bias"].float() for state in heads]).mean(0)
    os.makedirs(out, exist_ok=True)
    save_file(averaged, os.path.join(out, "adapter_model.safetensors"))
    shutil.copyfile(os.path.join(paths[0], "adapter_config.json"), os.path.join(out, "adapter_config.json"))
    base = heads[0]["base"] if "base" in heads[0] else ""
    torch.save({"weight": weight, "bias": bias, "verbalizer_ids": VERBALIZER_IDS, "attention": modes[0],
                "base": base, "averaged_from": list(paths)}, os.path.join(out, "head.pt"))
    return {"checkpoints": len(paths), "attention": modes[0], "tensors": len(keys)}
