# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: 训练损失：选项上的 KL(teacher || model)，score 再加序数 RPS。

``kl_rps`` scores one row; ``kl_rps_batch`` scores a micro-batch at once on padded ``[B, 24]`` slot tensors (the
training path), with the same value per row.
"""
import torch
import torch.nn.functional as F

from trainjev.modeling import RANGES

NUM_SLOTS = 24
KINDS = ("noul", "score", "choice")


def kl_rps(z, target, kind, rps_w, brier_w=0.0):
    """KL(target || softmax(z)) (+ ``rps_w`` * ranked probability score for score rows). ``z``, ``target``: ``[n]``."""
    logp = F.log_softmax(z, -1)
    t = target.clamp_min(0)
    kl = (t * (torch.log(t.clamp_min(1e-12)) - logp)).sum()
    if kind == "score" and rps_w:
        cp, ct = torch.cumsum(logp.exp(), 0), torch.cumsum(t, 0)
        kl = kl + rps_w * ((cp - ct) ** 2)[:-1].sum() / (len(t) - 1)
    if brier_w:
        kl = kl + brier_w * ((logp.exp() - t) ** 2).sum()
    return kl


def pack_targets(rows):
    """Rows -> ``(slot_lo [B], n [B], target [B, 24], is_score [B], weight [B])`` CPU tensors for ``kl_rps_batch``.

    Each row's target is normalised to sum 1 and written into its kind's slot range, zero elsewhere.
    """
    B = len(rows)
    lo, n, score, w = torch.zeros(B, dtype=torch.long), torch.zeros(B, dtype=torch.long), torch.zeros(B), torch.zeros(B)
    tgt = torch.zeros(B, NUM_SLOTS)
    for i, r in enumerate(rows):
        a, b = RANGES[r["kind"]]
        k = len(r["options"])
        if k > b - a:
            raise ValueError(f"{r['kind']} row with {k} options does not fit the {b - a} head slots")
        t = torch.tensor(r["target"], dtype=torch.float32)
        tgt[i, a: a + k] = t / t.sum()
        lo[i], n[i], score[i], w[i] = a, k, float(r["kind"] == "score"), r["weight"]
    return lo, n, tgt, score, w


def kl_rps_batch(z, lo, n, target, is_score, rps_w, brier_w=0.0):
    """Per-row ``kl_rps`` for a micro-batch. ``z``: ``[B, 24]`` head logits over all slots; the other arguments come
    from :func:`pack_targets` (on ``z``'s device). Only slots ``lo .. lo + n`` of each row take part. -> ``[B]``."""
    slot = torch.arange(z.shape[1], device=z.device)
    valid = (slot >= lo[:, None]) & (slot < (lo + n)[:, None])
    logp = F.log_softmax(z.float().masked_fill(~valid, float("-inf")), -1)
    t = target.clamp_min(0)
    logp0 = logp.masked_fill(~valid, 0.0)  # -inf * 0 would be nan
    kl = (t * (torch.log(t.clamp_min(1e-12)) - logp0)).sum(-1)
    if rps_w:
        a, b = RANGES["score"]
        p, ts = logp0[:, a:b].exp(), t[:, a:b]
        cp, ct = torch.cumsum(p, -1), torch.cumsum(ts, -1)
        # score rows always use all 6 levels; cumulative gaps over the first n-1 levels, divided by (n - 1)
        k = (slot[: b - a] < (n - 1)[:, None]).float()
        rps = (((cp - ct) ** 2) * k).sum(-1) / (n - 1).clamp_min(1)
        kl = kl + rps_w * rps * is_score
    if brier_w:
        kl = kl + brier_w * ((logp.exp() - t) ** 2).sum(-1)
    return kl
