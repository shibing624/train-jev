# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: 断点续训：模型和优化器状态用 torch.distributed.checkpoint 按分片保存，另存 step、历史和 RNG。

Resumable training state, shared by the LoRA and the full-parameter trainers.

``<dir>/`` holds the trainable model weights and the optimiser state written with ``torch.distributed.checkpoint``
(sharded for FSDP2 parameters, written once for replicated tensors; frozen weights are skipped), ``meta.json`` (next
step, best, history) and ``rng_<rank>.pt``. Data order needs no state: sampling and augmentation are seeded by the
step. A new state is written to ``<dir>.tmp`` and swapped in only when complete.
"""
import json
import os
import random
import shutil

import torch
import torch.distributed as dist
import torch.nn as nn

from trainjev.attention import attention_metadata, validate_attention_metadata


def _options(strict=True):
    from torch.distributed.checkpoint.state_dict import StateDictOptions

    # frozen (LoRA base) weights are not saved, so loading must not insist on them
    return StateDictOptions(ignore_frozen_params=True, strict=strict)


def _rng_state():
    return {"py": random.getstate(), "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def _set_rng_state(st):
    random.setstate(st["py"])
    torch.set_rng_state(st["cpu"])
    if st["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(st["cuda"])


def _find(path):
    """The last complete state: ``path``, or ``path.old`` if a swap was interrupted."""
    for p in (path, path + ".old"):
        if os.path.exists(os.path.join(p, "meta.json")):
            return p
    return None


def require_state(path):
    """Fail before model loading when --resume was requested without a completed training state."""
    found = _find(path)
    if found is None:
        raise FileNotFoundError(f"No completed training state at {path} or {path}.old; "
                                "an adapter/head inference checkpoint cannot resume optimizer/RNG state")
    return found


def save_state(path, body, head, opt, meta):
    """Collective: every rank must call it."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict

    rank, tmp = dist.get_rank(), path + ".tmp"
    if rank == 0 and os.path.exists(tmp):
        shutil.rmtree(tmp)
    dist.barrier()
    msd, osd = get_state_dict(nn.ModuleDict({"body": body, "head": head}), opt, options=_options())
    dcp.save({"model": msd, "optim": osd}, checkpoint_id=tmp)
    torch.save(_rng_state(), os.path.join(tmp, f"rng_{rank}.pt"))
    dist.barrier()
    if rank == 0:
        with open(os.path.join(tmp, "meta.json"), "w") as f:
            json.dump(dict(meta, world=dist.get_world_size(), attention_config=attention_metadata(body)), f)
        if os.path.exists(path):
            os.replace(path, path + ".old")
        os.replace(tmp, path)
        shutil.rmtree(path + ".old", ignore_errors=True)
    dist.barrier()


def load_state(path, body, head, opt):
    """Collective. Restores weights, optimiser and RNG in place; -> the saved ``meta`` dict, or ``None`` if there is
    no state at ``path``."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    p = _find(path)
    if p is None:
        return None
    with open(os.path.join(p, "meta.json")) as f:
        meta = json.load(f)
    enabled = attention_metadata(body)["mode"] != "causal"
    validate_attention_metadata(meta.get("attention_config"), enabled, strict=True)
    if meta["world"] != dist.get_world_size():
        raise ValueError(f"{p} was written by {meta['world']} ranks, now {dist.get_world_size()}: per-rank data "
                         "slices and RNG would not match")
    mods = nn.ModuleDict({"body": body, "head": head})
    msd, osd = get_state_dict(mods, opt, options=_options())
    dcp.load({"model": msd, "optim": osd}, checkpoint_id=p)
    res = set_state_dict(mods, opt, model_state_dict=msd, optim_state_dict=osd, options=_options(strict=False))
    frozen = {n for n, q in mods.named_parameters() if not q.requires_grad}
    missing = [k for k in res.missing_keys if k not in frozen]
    if missing or res.unexpected_keys:
        raise ValueError(f"{p} does not match the model: missing {missing[:5]} unexpected {res.unexpected_keys[:5]}")
    _set_rng_state(torch.load(os.path.join(p, f"rng_{dist.get_rank()}.pt"), weights_only=False))
    return meta
