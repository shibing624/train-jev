import math
import os

import pytest
import torch

from trainjev import (
    RANGES,
    VERBALIZER_IDS,
    VERBALIZERS,
    DecisionHead,
    ece,
    is_placeholder,
    kl_rps,
    last_hidden,
    metrics,
    micro_batches,
    prompt,
)
from trainjev.serve import map_question, tournament


# ----------------------------------------------------------------------------- prompt
def test_prompt_choice_matches_serve_decide_template():
    p = prompt("choice", "s", "q?", ["red", "blue"])
    assert p == "[kind] choice\n[state] s\n[question] q?\n[options]\nA) red\nB) blue\n[decision]:"


def test_prompt_noul_and_score_list_raw_options():
    assert prompt("noul", "s", "q", ["false", "true"]).endswith("[options]\nfalse\ntrue\n[decision]:")
    assert prompt("score", "s", "q", [str(i) for i in range(6)]).endswith("0\n1\n2\n3\n4\n5\n[decision]:")


def test_prompt_serialises_structured_state():
    assert '[state] {"a": 1}' in prompt("noul", {"a": 1}, "q", ["false", "true"])


# ----------------------------------------------------------------------------- head
def test_slot_layout():
    assert len(VERBALIZERS) == len(VERBALIZER_IDS) == 24
    assert RANGES == {"noul": (0, 2), "score": (2, 8), "choice": (8, 24)}


def test_untrained_head_equals_restricted_lm_head_readout():
    torch.manual_seed(0)
    lm_head = torch.randn(64, 16)
    ids = list(range(24))
    head = DecisionHead(lm_head[ids])
    h = torch.randn(16)
    for kind, n in (("noul", 2), ("score", 6), ("choice", 5)):
        lo, _ = RANGES[kind]
        ref = torch.softmax(lm_head[ids[lo: lo + n]] @ h, -1)
        got = torch.softmax(head.logits(h, kind, n), -1)
        assert torch.allclose(ref, got, atol=1e-6)


def test_last_hidden_picks_last_real_token_under_right_padding():
    class Body(torch.nn.Module):
        def forward(self, input_ids):
            class Out:
                pass

            o = Out()
            o.last_hidden_state = input_ids.float().unsqueeze(-1).repeat(1, 1, 3)
            return o

    h = last_hidden(Body(), [[5, 6, 7], [9]], 0, "cpu")
    assert h[:, 0].tolist() == [7.0, 9.0]


# ----------------------------------------------------------------------------- loss
def test_kl_is_zero_at_target_and_positive_elsewhere():
    t = torch.tensor([0.2, 0.5, 0.3])
    assert kl_rps(torch.log(t), t, "choice", 0.5).item() == pytest.approx(0.0, abs=1e-6)
    assert kl_rps(torch.zeros(3), t, "choice", 0.5).item() > 0


def test_rps_only_applies_to_score():
    t = torch.tensor([1.0, 0, 0, 0, 0, 0])
    z = torch.zeros(6)
    base = kl_rps(z, t, "choice", 0.5).item()
    with_rps = kl_rps(z, t, "score", 0.5).item()
    cp = torch.cumsum(torch.full((6,), 1 / 6), 0)
    rps = ((cp - 1.0) ** 2)[:-1].sum().item() / 5
    assert with_rps == pytest.approx(base + 0.5 * rps, rel=1e-5)


# ----------------------------------------------------------------------------- data utils
def test_micro_batches_respect_token_budget():
    items = [(n, i) for i, n in enumerate([5, 100, 30, 30, 60, 7, 1])]
    lens = dict((i, n) for n, i in items)
    mbs = micro_batches(items, 120)
    assert sorted(i for mb in mbs for i in mb) == list(range(7))
    for mb in mbs:
        assert len(mb) == 1 or max(lens[i] for i in mb) * len(mb) <= 120


def test_is_placeholder():
    assert is_placeholder({"source": "yuri_v1", "target": [0.3, 0.7]})
    assert is_placeholder({"source": "x", "target": [0.5, 0.5]})
    assert not is_placeholder({"source": "x", "target": [0.4, 0.6]})
    assert not is_placeholder({"source": "x", "target": [1 / 3] * 3})


# ----------------------------------------------------------------------------- metrics
def test_metrics_perfect_predictions():
    rows = [{"kind": "noul", "target": [0.1, 0.9]}, {"kind": "noul", "target": [0.8, 0.2]},
            {"kind": "score", "target": [0, 0, 1, 0, 0, 0]}]
    m = metrics(rows, [r["target"] for r in rows])
    assert m["all"]["kl"] == pytest.approx(0.0, abs=1e-9)
    assert m["noul"]["top1"] == 1.0 and m["noul"]["auroc"] == 1.0
    assert m["score"]["score_mae"] == pytest.approx(0.0, abs=1e-9)


def test_ece_bounds():
    assert ece([1.0, 1.0], [1, 1]) == pytest.approx(0.0)
    assert ece([0.9, 0.9], [0, 0]) == pytest.approx(0.9)


# ----------------------------------------------------------------------------- serving
def test_map_question_types():
    assert map_question({"type": "noul", "instructions": "ok?"})[0:3] == ("noul", "ok?", ["false", "true"])
    kind, q, opts, keys = map_question({"type": "choice", "instructions": "pick",
                                        "criteria": {"a": "first", "b": None, "c": "c"}})
    assert (kind, opts, keys) == ("choice", ["a: first", "b", "c"], ["a", "b", "c"])
    kind, q, opts, _ = map_question({"type": "score", "instructions": "rate", "criteria": list("uvwxyz")})
    assert kind == "score" and opts == [str(i) for i in range(6)] and "0 = u; 1 = v" in q
    kind, _, opts, _ = map_question({"type": "score", "instructions": "rate", "criteria": ["lo", "mid", "hi"]})
    assert kind == "choice" and opts == ["lo", "mid", "hi"]


def test_tournament_is_a_distribution_and_finds_the_best_option():
    n = 40
    scores = {f"opt{i}": math.exp(-abs(i - 27)) for i in range(n)}

    def dist(opts):
        s = sum(scores[o] for o in opts)
        return [scores[o] / s for o in opts]

    p = tournament(dist, list(scores))
    assert len(p) == n and sum(p) == pytest.approx(1.0)
    assert max(range(n), key=p.__getitem__) == 27


# ----------------------------------------------------------------------------- tokenizer (optional)
TOKENIZER = os.environ.get("TRAINJEV_TOKENIZER", "")


@pytest.mark.skipif(not TOKENIZER, reason="set TRAINJEV_TOKENIZER to a Qwen3.5 checkpoint to run")
def test_qwen35_verbalizers_and_truncation():
    from transformers import AutoTokenizer

    from trainjev import check_verbalizers, encode

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    check_verbalizers(tok)
    row = {"kind": "choice", "state": "word " * 5000, "question": "Which?", "options": ["x", "y"]}
    ids = encode(tok, row, max_len=512)
    assert len(ids) <= 512
    assert tok.decode(ids).endswith("[question] Which?\n[options]\nA) x\nB) y\n[decision]:")
