# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: prompt 模板、分词和按长度组 batch。

Prompt template, tokenisation and batching.

The prompt follows JEV-27B-VL's ``serve_decide.py`` byte for byte (template ``bare-v1``)::

    [kind] {kind}
    [state] {state}
    [question] {question}
    [options]
    {lines}
    [decision]:

``choice`` lines are ``"A) option"``; ``noul`` / ``score`` lines are the option strings themselves
(``false`` / ``true`` and ``0`` .. ``5``). The decision is read at the last prompt token.
"""
import json
import random

LABELS = list("ABCDEFGHIJKLMNOP")


def prompt(kind, state, question, options):
    if not isinstance(state, str):
        state = json.dumps(state, ensure_ascii=False)
    lines = [f"{LABELS[i]}) {o}" for i, o in enumerate(options)] if kind == "choice" else list(options)
    return f"[kind] {kind}\n[state] {state}\n[question] {question}\n[options]\n" + "\n".join(lines) + "\n[decision]:"


def encode(tok, row, max_len=2048):
    """-> token ids. An over-long state is cut in the middle; the question and options are never cut."""
    ids = tok(prompt(row["kind"], row["state"], row["question"], row["options"]), add_special_tokens=False).input_ids
    if len(ids) <= max_len:
        return ids
    st = tok(row["state"] if isinstance(row["state"], str) else json.dumps(row["state"], ensure_ascii=False),
             add_special_tokens=False).input_ids
    keep = max(64, len(st) - (len(ids) - max_len) - 8)
    cut = tok.decode(st[: keep // 2]) + " … " + tok.decode(st[-(keep - keep // 2):])
    r = dict(row, state=cut)
    return tok(prompt(r["kind"], r["state"], r["question"], r["options"]), add_special_tokens=False).input_ids[:max_len]


def load_rows(path, limit=None, seed=None):
    """Read a jsonl file; optionally a fixed random subset of ``limit`` rows."""
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    if limit and limit < len(rows):
        rows = random.Random(seed or 0).sample(rows, limit)
    return rows


def is_placeholder(row):
    """jev-distill ``yuri_v1`` rows, or any 2-way row with a uniform target, carry no decision signal."""
    uniform_pair = len(row["target"]) == 2 and len(set(round(x, 6) for x in row["target"])) == 1
    return row.get("source") == "yuri_v1" or uniform_pair


def micro_batches(items, max_tokens):
    """items: ``[(length, payload)]`` -> length-sorted micro-batches with ``rows * max_len <= max_tokens``."""
    items = sorted(items, key=lambda x: x[0])
    out, cur, cur_max = [], [], 0
    for n, p in items:
        m = max(cur_max, n)
        if cur and m * (len(cur) + 1) > max_tokens:
            out.append(cur)
            cur, m = [], n
        cur.append(p)
        cur_max = m
    if cur:
        out.append(cur)
    return out
