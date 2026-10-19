# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: AutoThink 门控、两次读出的混合，以及 RLCD² 的奖励和组内优势。

AutoThink controls and the RLCD² reward from StartLux-Decision (arXiv 2610.17309).

A causal checkpoint scores every question once. ``auto`` writes a thought only when the top probability is below
``gate`` (0.6 by default; the published Decision Index / DecisionPivot numbers use 0.9) and mixes
``(1 - mix) * p0 + mix * p1`` with ``mix`` 0.75. Instant (bidirectional) checkpoints do not think.

RLCD² rewards the change in a decision-weighted proper score after the thought, minus ``lambda`` per token past a
soft budget. Traces shorter than 8 tokens get reward 0. Group advantages are the centered rewards clipped to
[-5, 5], without a standard-deviation normalisation. This module is the objective; it does not run a policy loop.
"""
import math
from dataclasses import dataclass

import torch

THINK_MODES = ("auto", "never", "always")
_MODE_ALIASES = {"auto": "auto", "never": "never", "always": "always", "hybrid": "auto", "plain": "never",
                 "think": "always"}
# w(c) = 0.5 * Uniform(0, 1) + 0.5 * Beta(8, 2). Closed forms of the integrals in the paper's equation (3).


@dataclass
class ThinkControls:
    think_mode: str = "never"
    gate: float = 0.6
    budget: int = 4096
    mix: float = 0.75
    details: bool = False
    thought_text: bool = False


def _fraction(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) \
            or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a number from 0 to 1")
    return float(value)


def parse_controls(body, defaults, can_think):
    """Request fields on top of the server defaults. Thinking is refused when ``can_think`` is false."""
    mode = defaults.think_mode
    if body.get("think_mode") is not None:
        raw = body["think_mode"]
        if raw not in _MODE_ALIASES:
            raise ValueError("think_mode must be auto, never, or always")
        mode = _MODE_ALIASES[raw]
    if not can_think and mode != "never":
        raise ValueError("this server answers in one pass; leave think_mode out or set it to \"never\"")
    gate = _fraction(body["gate"], "gate") if body.get("gate") is not None else defaults.gate
    mix = _fraction(body["mix"], "mix") if body.get("mix") is not None else defaults.mix
    budget = defaults.budget
    if body.get("think_budget") is not None:
        raw = body["think_budget"]
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
            raise ValueError("think_budget must be an integer of at least 1")
        budget = min(raw, defaults.budget)
    details = defaults.details
    thought_text = defaults.thought_text
    if body.get("return_details") is not None:
        if not isinstance(body["return_details"], bool):
            raise ValueError("return_details must be true or false")
        details = body["return_details"]
    if body.get("return_thought") is not None:
        if not isinstance(body["return_thought"], bool):
            raise ValueError("return_thought must be true or false")
        thought_text = body["return_thought"]
    return ThinkControls(mode, gate, budget, mix, details or thought_text, thought_text)


def should_think(probabilities, controls):
    """``auto`` thinks when the top probability is strictly below ``gate``."""
    if controls.think_mode == "never":
        return False
    if controls.think_mode == "always":
        return True
    return max(probabilities) < controls.gate


def mix_probabilities(first, after, mix):
    """``mix`` is the weight of the readout after the thought."""
    if len(first) != len(after):
        raise ValueError("the two readouts must cover the same options")
    return [(1 - mix) * a + mix * b for a, b in zip(first, after)]


def _beta_cdf(q):
    """CDF of Beta(8, 2) on [0, 1]: 9 q^8 - 8 q^9."""
    return 9 * q ** 8 - 8 * q ** 9


def weight_integral(q):
    """W(q) = ∫_0^q w(c) dc."""
    q = min(1.0, max(0.0, q))
    return 0.5 * q + 0.5 * _beta_cdf(q)


def cost_integral(q):
    """C(q) = ∫_0^q c w(c) dc."""
    q = min(1.0, max(0.0, q))
    return 0.25 * q ** 2 + 4 * q ** 9 - 3.6 * q ** 10


def binary_score(q, outcome):
    """s(q, z) = z W(q) - C(q) for one reported probability."""
    return outcome * weight_integral(q) - cost_integral(q)


def decision_score(probabilities, target, kind):
    """Decision-weighted proper score. Ordered ``score`` questions use cumulative probabilities, except the last."""
    if len(probabilities) != len(target):
        raise ValueError("probabilities and target must have the same length")
    if kind == "score":
        total_p = total_t = 0.0
        score = 0.0
        for index, (p, t) in enumerate(zip(probabilities, target)):
            total_p += p
            total_t += t
            if index < len(probabilities) - 1:
                score += binary_score(total_p, total_t)
        return score
    return sum(binary_score(p, t) for p, t in zip(probabilities, target))


def rlcd2_reward(first, after, target, n_tokens, kind, budget=3072, lam=1e-4, min_tokens=8):
    """Paper equation (5). A trace that stops before ``min_tokens`` is empty and scores 0."""
    if n_tokens < min_tokens:
        return 0.0
    gain = decision_score(after, target, kind) - decision_score(first, target, kind)
    return gain - lam * max(0, n_tokens - budget)


def group_advantages(rewards):
    """Centered rewards clipped to [-5, 5]. Empty traces stay in the group with reward 0."""
    if not rewards:
        return []
    center = sum(rewards) / len(rewards)
    return [min(5.0, max(-5.0, reward - center)) for reward in rewards]


def clipped_policy_loss(ratios, advantages, eps=0.2):
    """The token-summed surrogate of paper equation (6), averaged and ready to minimize. KL is separate."""
    if len(ratios) != len(advantages) or not ratios:
        raise ValueError("ratios and advantages must be the same non-empty length")
    total = 0.0
    for ratio, advantage in zip(ratios, advantages):
        clipped = min(max(ratio, 1 - eps), 1 + eps)
        total -= min(ratio * advantage, clipped * advantage)
    return total / len(ratios)


def k3_kl(log_policy, log_reference):
    """Schulman k3 estimator π_ref/π - log(π_ref/π) - 1, averaged over the given token log-probabilities."""
    if len(log_policy) != len(log_reference) or not log_policy:
        raise ValueError("log probabilities must be the same non-empty length")
    total = 0.0
    for log_p, log_ref in zip(log_policy, log_reference):
        gap = log_ref - log_p
        total += math.exp(gap) - gap - 1
    return total / len(log_policy)


def repeating(tokens, block=16):
    """True when the newest ``block`` tokens copy the block before them."""
    return len(tokens) >= 2 * block and tokens[-block:] == tokens[-2 * block:-block]


def thought_text(decode, tokens):
    """Decode generated tokens and drop a trailing ``[decision]`` marker, including a partial one at the end."""
    text = decode(tokens)
    cut = text.find("[decision]")
    if cut >= 0:
        text = text[:cut]
    return text.strip()


def sample_token(logits, temperature, top_k, top_p, generator=None):
    """One token id. ``temperature <= 0`` is greedy. ``top_k`` 20 and ``top_p`` 0.95 match the published sampler."""
    scores = logits.float().reshape(-1)
    if temperature <= 0:
        return int(torch.argmax(scores).item())
    scores = scores / temperature
    if top_k > 0 and top_k < scores.numel():
        values, indexes = torch.topk(scores, top_k)
        kept = torch.full_like(scores, float("-inf"))
        kept.scatter_(0, indexes, values)
        scores = kept
    probs = torch.softmax(scores, -1)
    if top_p < 1:
        ordered, indexes = torch.sort(probs, descending=True)
        keep = torch.cumsum(ordered, 0) - ordered <= top_p
        keep[0] = True
        ordered = ordered * keep.to(ordered.dtype)
        ordered = ordered / ordered.sum()
        pick = int(torch.multinomial(ordered, 1, generator=generator).item())
        return int(indexes[pick].item())
    return int(torch.multinomial(probs, 1, generator=generator).item())


@torch.no_grad()
def generate_thought(body, lm_head, prompt_ids, decode, eos, max_new, temperature, top_k, top_p, generator=None):
    """Continue ``prompt_ids`` (1-D, on device) causally. Returns the thought text and the number of new tokens."""
    if max_new < 1:
        raise ValueError("max_new must be at least 1")
    device = prompt_ids.device
    ids = prompt_ids
    past = None
    produced = []
    for _ in range(max_new):
        step = ids.view(1, -1) if past is None else ids[-1:].view(1, -1)
        out = body(input_ids=step, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = sample_token(lm_head(out.last_hidden_state[:, -1, :]), temperature, top_k, top_p, generator)
        if nxt in eos:
            break
        produced.append(nxt)
        ids = torch.cat([ids, torch.tensor([nxt], dtype=torch.long, device=device)])
        if repeating(produced):
            produced = produced[:-16]
            break
        if "[decision]" in decode(produced):
            break
    return thought_text(decode, produced), len(produced)
