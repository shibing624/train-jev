# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: 本地精确对比对。一对题只改一个会翻转答案的数字，用来训练和检查 System 1 容易失手的边界。

Local contrast pairs for precision-sensitive decisions.

This is not DecisionPivot and it is not a Decision Index score. Each pair differs by one answer-changing integer.
A pair is correct only when both members are correct. The state contains every number the label depends on; the
optional ``thought`` restates the comparison so a second readout can be trained on a known trace.
"""
import hashlib
import json
import random
import re

FAMILIES = ("refund", "deadline", "count", "sum")


def pair_accuracy(rows):
    """``rows`` items need ``pair_id`` and ``correct``. Returns ``(accuracy, n_pairs)``."""
    groups = {}
    for row in rows:
        groups.setdefault(row["pair_id"], []).append(bool(row["correct"]))
    if not groups:
        return 0.0, 0
    return sum(all(flags) for flags in groups.values()) / len(groups), len(groups)


def _noul(case, state, question, label, thought, pair_id, family):
    return {"kind": "noul", "state": state, "question": question, "options": ["false", "true"],
            "target": [1 - label, label], "weight": 1.0, "thought": thought, "pair_id": pair_id,
            "domain": f"precision-{family}", "source": "precision-mix", "case": case}


def _choice(case, state, question, options, label, thought, pair_id, family):
    target = [0.0] * len(options)
    target[label] = 1.0
    return {"kind": "choice", "state": state, "question": question, "options": options, "target": target,
            "weight": 1.0, "thought": thought, "pair_id": pair_id, "domain": f"precision-{family}",
            "source": "precision-mix", "case": case}


def _refund(rng, seed, index):
    cap = rng.randint(20, 400)
    ok = rng.randint(0, cap)
    bad = cap + rng.randint(1, 50)
    pair = f"refund-{seed}-{index}"
    question = ("Should this charge be auto-refunded? Auto-refund only when the charged amount is less than "
                "or equal to the cap.")

    def one(charged, label):
        case = f"{pair}:{charged}"
        state = f"[precision-refund case={case}] Policy cap {cap} USD. Amount charged {charged} USD."
        thought = f"charged {charged}; cap {cap}; {charged} <= {cap} is {bool(label)}."
        return _noul(case, state, question, label, thought, pair, "refund")

    return [one(ok, 1), one(bad, 0)]


def _deadline(rng, seed, index):
    deadline = rng.randint(10, 300)
    early = rng.randint(0, deadline)
    late = deadline + rng.randint(1, 30)
    pair = f"deadline-{seed}-{index}"
    question = "Is the event on time? On time means the event day number is less than or equal to the deadline day."

    def one(event, label):
        case = f"{pair}:{event}"
        state = f"[precision-deadline case={case}] Deadline day {deadline}. Event day {event}."
        thought = f"event {event}; deadline {deadline}; {event} <= {deadline} is {bool(label)}."
        return _noul(case, state, question, label, thought, pair, "deadline")

    return [one(early, 1), one(late, 0)]


def _count(rng, seed, index):
    threshold = rng.randint(2, 12)
    pair = f"count-{seed}-{index}"
    question = "How does the item count compare with the threshold?"
    options = ["under", "exact", "over"]

    def one(count, label):
        case = f"{pair}:{count}"
        items = " ".join(f"item{n}" for n in range(count)) or "(none)"
        state = f"[precision-count case={case}] Threshold {threshold}. Items: {items}."
        names = ["under", "exact", "over"]
        thought = f"count {count}; threshold {threshold}; relation {names[label]}."
        return _choice(case, state, question, options, label, thought, pair, "count")

    return [one(threshold, 1), one(threshold + 1, 2)]


def _sum(rng, seed, index):
    left = rng.randint(0, 40)
    right = rng.randint(0, 40)
    pair = f"sum-{seed}-{index}"
    question = "How does left + right compare with the pivot?"
    options = ["less", "equal", "greater"]

    def one(a, label):
        case = f"{pair}:{a}"
        state = f"[precision-sum case={case}] Left {a}. Right {right}. Pivot {left + right}."
        names = ["less", "equal", "greater"]
        thought = f"{a} + {right} = {a + right}; pivot {left + right}; relation {names[label]}."
        return _choice(case, state, question, options, label, thought, pair, "sum")

    return [one(left, 1), one(left + 1, 2)]


_BUILDERS = {"refund": _refund, "deadline": _deadline, "count": _count, "sum": _sum}
_RULES = {
    "refund": (re.compile(r"Policy cap (\d+) USD\. Amount charged (\d+) USD\."),
               lambda cap, charged: int(int(charged) <= int(cap)), "noul"),
    "deadline": (re.compile(r"Deadline day (\d+)\. Event day (\d+)\."),
                 lambda deadline, event: int(int(event) <= int(deadline)), "noul"),
    "count": (re.compile(r"Threshold (\d+)\. Items: (.*)\."),
              lambda threshold, items: (0 if items == "(none)" else len(items.split())) - int(threshold), "count"),
    "sum": (re.compile(r"Left (\d+)\. Right (\d+)\. Pivot (\d+)\."),
            lambda left, right, pivot: (int(left) + int(right)) - int(pivot), "sum"),
}


def build_rows(pairs_per_family, seed):
    """``pairs_per_family`` contrast pairs in each of the four families. Deterministic in ``seed``."""
    if pairs_per_family < 1:
        raise ValueError("pairs_per_family must be at least 1")
    rng = random.Random(seed)
    rows = []
    for index in range(pairs_per_family):
        for family in FAMILIES:
            rows.extend(_BUILDERS[family](rng, seed, index))
    return rows


def _expected_choice(family, numbers):
    if family == "count":
        delta = _RULES[family][1](*numbers)
    else:
        delta = _RULES[family][1](*numbers)
    if delta < 0:
        return 0
    if delta == 0:
        return 1
    return 2


def audit_rows(rows):
    """Check that every pair flips the label and that the label matches the integers in the state."""
    groups = {}
    for row in rows:
        groups.setdefault(row["pair_id"], []).append(row)
    mismatches = []
    for pair_id, members in groups.items():
        if len(members) != 2:
            mismatches.append(f"{pair_id}: expected 2 members, got {len(members)}")
            continue
        labels = []
        for row in members:
            family = row["domain"].split("-", 1)[1]
            pattern, rule, kind = _RULES[family]
            found = pattern.search(row["state"])
            if found is None:
                mismatches.append(f"{row['case']}: state does not match the {family} template")
                continue
            if kind == "noul":
                label = rule(*found.groups())
            elif family == "count":
                label = _expected_choice(family, found.groups())
            else:
                label = _expected_choice(family, found.groups())
            labels.append(label)
            got = row["target"].index(max(row["target"]))
            if got != label:
                mismatches.append(f"{row['case']}: target index {got} != rule {label}")
        if len(labels) == 2 and labels[0] == labels[1]:
            mismatches.append(f"{pair_id}: the two members share a label")
    digest = hashlib.sha256("\n".join(json.dumps(row, sort_keys=True) for row in rows).encode()).hexdigest()
    return {"rows": len(rows), "pairs": len(groups), "mismatches": mismatches, "sha256": digest}


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
