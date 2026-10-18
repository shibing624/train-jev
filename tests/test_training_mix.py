import copy
import json
import random
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model

from trainjev.losses import kl_rps, kl_rps_batch, pack_targets
from trainjev.modeling import RANGES, VERBALIZER_IDS, DecisionHead, load_trained
from trainjev.train import parse_args, setup_data


def corpus(tmp_path):
    rows = []
    for kind, n in (("noul", 2), ("score", 6), ("choice", 4)):
        for i in range(20):
            rows.append({"id": f"base:{kind}:{i}", "kind": kind, "state": str(i), "question": "q",
                         "options": [str(j) for j in range(n)], "target": [1.0] + [0.0] * (n - 1),
                         "source": "yuri_v1" if i % 4 == 0 else "teacher"})
    extra = [dict(r, id=r["id"].replace("base", "gold"), source="gold", target=r["target"][::-1])
             for r in rows if r["kind"] == "choice"]
    for name in ("train", "validation"):
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    mix = tmp_path / "gold.jsonl"
    mix.write_text("".join(json.dumps(r) + "\n" for r in extra))
    return rows, mix


def args(tmp_path, mix="", fraction=None, counts=""):
    return SimpleNamespace(data=str(tmp_path), mix=str(mix), mix_fraction=fraction, kind_counts=counts,
                           placeholder_w=0.05, val_n=100, steps=20, batch=12, seed=17, perm=0.3)


def test_legacy_sampling_preserved(tmp_path):
    rows, _ = corpus(tmp_path)
    _, _, _, draw, _ = setup_data(args(tmp_path), lambda *x: None)
    for s in range(10):
        rng = random.Random(17 * 1000003 + s)
        idx = [i for k in ("noul", "score", "choice")
               for i in rng.sample([i for i, r in enumerate(rows) if r["kind"] == k], 2)]
        idx += rng.sample(range(len(rows)), 6)
        assert [r["id"] for r in draw(s)] == [rows[i]["id"] for i in idx]


def test_exact_mix_keeps_kind_and_base_row_weights(tmp_path):
    _, mix = corpus(tmp_path)
    _, _, _, pure, _ = setup_data(args(tmp_path, mix, 0, "choice:6,noul:3,score:3"), lambda *x: None)
    _, _, _, mixed, _ = setup_data(args(tmp_path, mix, 0.25, "choice:6,noul:3,score:3"), lambda *x: None)
    for s in range(10):
        a, b = pure(s), mixed(s)
        assert sum(r["_mixed"] for r in b) == 3
        assert [r["kind"] for r in a] == [r["kind"] for r in b]
        assert [r["weight"] for r in a] == [r["weight"] for r in b]
        assert [r["id"] for r, q in zip(a, b) if not q["_mixed"]] == [q["id"] for q in b if not q["_mixed"]]


def test_invalid_mix_and_kind_counts_fail(tmp_path):
    _, mix = corpus(tmp_path)
    with pytest.raises(ValueError):
        setup_data(args(tmp_path, mix, 1.1), print)
    with pytest.raises(ValueError):
        setup_data(args(tmp_path, mix, 0.2, "choice:3,noul:3,score:3"), print)
    with pytest.raises(ValueError):
        setup_data(args(tmp_path, "", 0.2), print)
    _, _, _, draw, _ = setup_data(args(tmp_path, mix, 0.75, "choice:3,noul:3,score:6"), print)
    with pytest.raises(ValueError):
        draw(0)


@pytest.mark.parametrize("coefficient", [0.0, 0.2, 1.0])
def test_vector_brier_values_and_gradients_match_scalar(coefficient):
    torch.manual_seed(3)
    rows = [{"kind": k, "options": [str(j) for j in range(n)], "target": [0.2] + [0.8 / (n - 1)] * (n - 1),
             "weight": 1.0} for k, n in (("noul", 2), ("score", 6), ("choice", 4), ("choice", 16))]
    a = torch.randn(4, 24, requires_grad=True)
    b = a.detach().clone().requires_grad_(True)
    lo, n, tgt, score, _ = pack_targets(rows)
    vector = kl_rps_batch(a, lo, n, tgt, score, 0.5, coefficient)
    scalar = torch.stack([kl_rps(b[i, RANGES[r["kind"]][0]:RANGES[r["kind"]][0] + len(r["options"])],
                               torch.tensor(r["target"]), r["kind"], 0.5, coefficient) for i, r in enumerate(rows)])
    assert torch.allclose(vector, scalar, atol=1e-6)
    vector.sum().backward()
    scalar.sum().backward()
    assert torch.allclose(a.grad, b.grad, atol=1e-6)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(8, 8, bias=False)

    def forward(self, x):
        return self.q_proj(x)


def test_trainable_load_keeps_adapter_and_head(tmp_path):
    torch.manual_seed(0)
    base = Tiny()
    fresh = copy.deepcopy(base)
    reference_base = copy.deepcopy(base)
    model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, lora_dropout=0, target_modules=["q_proj"]))
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.data.fill_(0.03)
    model.eval().save_pretrained(tmp_path)
    weight, bias = torch.full((24, 8), 0.17), torch.arange(24).float() / 100
    torch.save({"weight": weight, "bias": bias, "verbalizer_ids": VERBALIZER_IDS}, tmp_path / "head.pt")
    head = DecisionHead(torch.zeros(24, 8))
    loaded = load_trained(fresh, head, str(tmp_path), trainable=True)
    assert isinstance(loaded, PeftModel)
    assert all(p.requires_grad for n, p in loaded.named_parameters() if "lora_" in n)
    assert torch.equal(head.weight, weight) and torch.equal(head.bias, bias)
    x = torch.randn(2, 8)
    with torch.no_grad():
        assert torch.allclose(loaded.eval()(x), model(x), atol=1e-6)
        merged = load_trained(reference_base, head, str(tmp_path))
        assert not isinstance(merged, PeftModel)
        assert torch.allclose(merged(x), model(x), atol=1e-6)


def test_training_defaults_do_not_enable_new_objectives():
    a = parse_args(["--out", "unused"])
    assert a.mix_fraction is None and a.brier_w == 0 and a.init_adapter == "" and a.kind_counts == ""
