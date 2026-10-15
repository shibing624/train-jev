# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: 训练路径优化的等价性测试：向量化 loss、批量分词、meta 建模 + rank0 广播、断点续训。
"""
import os
import random

import pytest
import torch

from trainjev.data import encode, encode_many
from trainjev.losses import kl_rps, kl_rps_batch, pack_targets
from trainjev.modeling import RANGES, DecisionHead


def _rows(n, seed=0):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        kind = ("noul", "score", "choice")[i % 3]
        k = {"noul": 2, "score": 6, "choice": rng.randint(2, 16)}[kind]
        t = [rng.random() ** 3 for _ in range(k)]
        if i % 7 == 0:
            t[rng.randrange(k)] = 0.0  # zero mass on a slot
        if i % 11 == 0:
            t = [0.0] * k
            t[rng.randrange(k)] = 5.0  # one-hot, un-normalised
        rows.append({"kind": kind, "options": [f"o{j}" for j in range(k)], "target": t,
                     "weight": rng.choice([1.0, 0.05, 2.5])})
    return rows


@pytest.mark.parametrize("rps_w", [0.5, 0.0])
def test_vectorised_loss_equals_per_row_kl_rps(rps_w):
    torch.manual_seed(0)
    head = DecisionHead(torch.randn(24, 32))
    head.bias.data.normal_()
    rows = _rows(97)
    h = torch.randn(len(rows), 32) * 3
    ref = []
    for r, hh in zip(rows, h):
        t = torch.tensor(r["target"])
        ref.append(kl_rps(head.logits(hh, r["kind"], len(r["options"])), t / t.sum(), r["kind"], rps_w) * r["weight"])
    lo, n, tgt, sc, w = pack_targets(rows)
    got = kl_rps_batch(head.slot_logits(h), lo, n, tgt, sc, rps_w) * w
    assert torch.allclose(got, torch.stack(ref), atol=1e-5, rtol=1e-5)


def test_vectorised_loss_gradients_match():
    torch.manual_seed(1)
    rows = _rows(40, seed=1)
    h = torch.randn(len(rows), 16)
    grads = []
    for vec in (False, True):
        head = DecisionHead(torch.randn(24, 16, generator=torch.Generator().manual_seed(2)))
        if vec:
            lo, n, tgt, sc, w = pack_targets(rows)
            loss = (kl_rps_batch(head.slot_logits(h), lo, n, tgt, sc, 0.5) * w).sum()
        else:
            loss = sum(kl_rps(head.logits(hh, r["kind"], len(r["options"])),
                              torch.tensor(r["target"]) / sum(r["target"]), r["kind"], 0.5) * r["weight"]
                       for r, hh in zip(rows, h))
        loss.backward()
        grads.append((head.weight.grad.clone(), head.bias.grad.clone()))
    assert torch.allclose(grads[0][0], grads[1][0], atol=1e-5) and torch.allclose(grads[0][1], grads[1][1], atol=1e-5)


def test_pack_targets_layout_and_rejects_wide_rows():
    lo, n, tgt, sc, w = pack_targets([{"kind": "choice", "options": ["a", "b", "c"], "target": [1, 1, 2],
                                       "weight": 0.5}])
    a, _ = RANGES["choice"]
    assert (lo.item(), n.item(), sc.item(), w.item()) == (a, 3, 0.0, 0.5)
    assert tgt[0, a: a + 3].tolist() == [0.25, 0.25, 0.5] and tgt.sum().item() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        pack_targets([{"kind": "choice", "options": list("abcdefghijklmnopq"), "target": [1] * 17, "weight": 1}])


class _CharTok:
    """Tiny deterministic tokenizer with the HF call / decode interface used by ``encode``."""

    class _Out:
        def __init__(self, ids):
            self.input_ids = ids

    def __call__(self, text, add_special_tokens=False):
        if isinstance(text, list):
            return self._Out([[ord(c) for c in t] for t in text])
        return self._Out([ord(c) for c in text])

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def test_encode_many_equals_encode_including_truncation():
    rows = [{"kind": k, "state": "s" * n, "question": "q?", "options": ["x", "y", "z"] if k == "choice" else
             (["false", "true"] if k == "noul" else [str(i) for i in range(6)])}
            for k, n in [("choice", 10), ("noul", 900), ("score", 3), ("choice", 2000), ("noul", 0)]]
    rows.append({"kind": "noul", "state": {"a": [1, 2]}, "question": "q", "options": ["false", "true"]})
    tok = _CharTok()
    assert encode_many(tok, rows, 512) == [encode(tok, r, 512) for r in rows]
    assert encode_many(tok, [], 512) == []


TOKENIZERS = [t for t in os.environ.get("TRAINJEV_TOKENIZER", "").split(",") if t]


@pytest.mark.skipif(not TOKENIZERS, reason="set TRAINJEV_TOKENIZER to Qwen3.5 / 3.8 checkpoints to run")
@pytest.mark.parametrize("path", TOKENIZERS or ["-"])
def test_encode_many_equals_encode_on_qwen(path):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    rng = random.Random(0)
    words = ["退款", "refund", " ", "\n", "订单#4821", "😀", "Ticket", "{", "}", "A)", "…", "état"]
    rows = []
    for i in range(300):
        state = "".join(rng.choice(words) for _ in range(rng.choice([3, 50, 1500])))
        rows.append({"kind": "choice", "state": state if i % 5 else {"k": state}, "question": "Which?",
                     "options": [rng.choice(words) + str(j) for j in range(rng.randint(2, 16))]})
    assert encode_many(tok, rows, 1024) == [encode(tok, r, 1024) for r in rows]


# ------------------------------------------------------------------ distributed paths, 2 ranks on gloo (CPU)
def _tiny_qwen(path):
    from transformers import AutoModelForCausalLM, Qwen3_5TextConfig

    tc = Qwen3_5TextConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                           layer_types=["linear_attention"] * 3 + ["full_attention"], num_attention_heads=2,
                           num_key_value_heads=1, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=2,
                           linear_key_head_dim=8, linear_value_head_dim=8, tie_word_embeddings=False)
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_config(tc, dtype=torch.float32)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.1)  # avoid all-ones norms etc. so a wrong load shows up
    m.save_pretrained(path)


def _worker_meta_load(rank, world, port, path, out):
    import inspect

    import torch.distributed as dist
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
    from torch.distributed.fsdp import fully_shard
    from transformers import AutoModelForCausalLM

    from trainjev.modeling import VERBALIZER_IDS
    from trainjev.train_full import load_sharded

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    ids = [i % 128 for i in VERBALIZER_IDS]
    import trainjev.train_full as tf

    tf.VERBALIZER_IDS = ids  # the tiny vocab has 128 tokens

    # This fixture runs on CPU, even when the environment has CUDA-only FLA kernels installed.
    qwen.torch_chunk_gated_delta_rule = inspect.unwrap(qwen.torch_chunk_gated_delta_rule)
    qwen.torch_recurrent_gated_delta_rule = inspect.unwrap(qwen.torch_recurrent_gated_delta_rule)

    def shard(body):
        for layer in body.layers:
            fully_shard(layer)
        fully_shard(body)

    body, rows = load_sharded(path, rank, "cpu", shard)
    full = get_model_state_dict(body, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    x = torch.randint(0, 128, (2, 9), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        h = body.eval()(input_ids=x).last_hidden_state
    if rank == 0:
        ref = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
        rsd = ref.model.state_dict()
        with torch.no_grad():
            want = ref.model.eval()(input_ids=x).last_hidden_state
        torch.save({"w": max((full[k] - rsd[k]).abs().max().item() for k in rsd), "keys": set(full) == set(rsd),
                    "rows": (rows - ref.get_output_embeddings().weight[ids]).abs().max().item(),
                    "h": (h - want).abs().max().item()}, out)
    else:
        torch.save({"rows1": rows.clone(), "h1": h}, out + ".1")
    dist.destroy_process_group()


def _spawn(fn, *args):
    import torch.multiprocessing as mp

    port = random.Random(os.getpid()).randint(20000, 40000)
    mp.spawn(fn, args=(2, port, *args), nprocs=2, join=True)


def test_meta_build_and_rank0_broadcast_reproduce_the_checkpoint(tmp_path):
    pytest.importorskip("transformers")
    path, out = str(tmp_path / "tiny"), str(tmp_path / "res.pt")
    _tiny_qwen(path)
    _spawn(_worker_meta_load, path, out)
    r = torch.load(out, weights_only=False)
    assert r["keys"] and r["w"] == 0.0 and r["rows"] == 0.0
    assert r["h"] < 1e-5  # rotary buffers re-initialised: same forward as a normal load
    r1 = torch.load(out + ".1", weights_only=False)
    assert r1["h1"].shape == (2, 9, 32)


def _worker_resume(rank, world, port, d, out):
    import torch.distributed as dist
    import torch.nn as nn

    from trainjev.resume import load_state, save_state

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)

    def make():
        torch.manual_seed(0)
        body = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
        body[0].weight.requires_grad_(False)  # a frozen weight, as in LoRA
        head = DecisionHead(torch.randn(24, 4))
        opt = torch.optim.AdamW([p for p in [*body.parameters(), *head.parameters()] if p.requires_grad], lr=0.1)
        return body, head, opt

    def step(body, head, opt, s):
        g = torch.Generator().manual_seed(100 * s + rank)
        loss = head.slot_logits(body(torch.randn(5, 8, generator=g))).pow(2).mean() + torch.rand(()) * 0  # uses RNG
        loss.backward()
        for p in opt.param_groups[0]["params"]:  # replicated training, as in trainjev.train
            dist.all_reduce(p.grad)
            p.grad /= world
        opt.step()
        opt.zero_grad()

    b1, h1, o1 = make()
    for s in range(4):
        step(b1, h1, o1, s)
    b2, h2, o2 = make()
    for s in range(2):
        step(b2, h2, o2, s)
    save_state(d, b2, h2, o2, {"step": 2, "best": [1.0, 2], "hist": [{"step": 2}]})
    b3, h3, o3 = make()
    meta = load_state(d, b3, h3, o3)
    for s in range(2, 4):
        step(b3, h3, o3, s)
    if rank == 0:
        diff = max((p - q).abs().max().item() for p, q in zip([*b1.parameters(), *h1.parameters()],
                                                              [*b3.parameters(), *h3.parameters()]))
        torch.save({"diff": diff, "meta": meta, "fresh": load_state(d + "_none", b3, h3, o3)}, out)
    else:
        load_state(d + "_none", b3, h3, o3)
    dist.destroy_process_group()


def test_resume_reproduces_uninterrupted_training(tmp_path):
    pytest.importorskip("torch.distributed.checkpoint")
    d, out = str(tmp_path / "state"), str(tmp_path / "res.pt")
    _spawn(_worker_resume, d, out)
    r = torch.load(out, weights_only=False)
    assert r["diff"] == 0.0 and r["fresh"] is None
    assert r["meta"]["step"] == 2 and r["meta"]["best"] == [1.0, 2] and r["meta"]["world"] == 2
    assert {"meta.json", "rng_0.pt", "rng_1.pt", ".metadata"} <= set(os.listdir(d))
    assert not os.path.exists(d + ".tmp") and not os.path.exists(d + ".old")
