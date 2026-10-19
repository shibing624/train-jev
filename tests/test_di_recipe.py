# -*- coding: utf-8 -*-
"""Instant attention, AutoThink controls, RLCD² scores and local contrast pairs."""
import asyncio
import inspect
import os

import pytest
import torch
from fastapi.testclient import TestClient

import trainjev.serve as srv
from trainjev.attention import apply_prompt_attention
from trainjev.autothink import (
    ThinkControls,
    clipped_policy_loss,
    decision_score,
    generate_thought,
    group_advantages,
    k3_kl,
    mix_probabilities,
    parse_controls,
    rlcd2_reward,
    sample_token,
    should_think,
)
from trainjev.average import average_checkpoints
from trainjev.data import prompt
from trainjev.modeling import DecisionHead, last_hidden
from trainjev.precision import audit_rows, build_rows, pair_accuracy


def test_prompt_without_a_thought_is_unchanged():
    assert prompt("choice", "s", "q?", ["red", "blue"]) == (
        "[kind] choice\n[state] s\n[question] q?\n[options]\nA) red\nB) blue\n[decision]:")
    text = prompt("noul", "s", "q", ["false", "true"], thought="charged 3; cap 2; 3 <= 2 is False.")
    assert text.endswith("[thought]\ncharged 3; cap 2; 3 <= 2 is False.\n[decision]:")
    assert text.index("[thought]") < text.index("[decision]")


def test_autothink_gate_and_mix():
    controls = ThinkControls("auto", 0.6, 32, 0.75)
    assert should_think([0.4, 0.6], controls) is False
    assert should_think([0.41, 0.59], controls) is True
    assert should_think([0.1, 0.9], ThinkControls("always", 0.6, 32, 0.75)) is True
    assert should_think([0.1, 0.9], ThinkControls("never", 0.6, 32, 0.75)) is False
    # 0.25 * p0 + 0.75 * p1. A swapped mix would yield [0.4, 0.6].
    assert mix_probabilities([0.2, 0.8], [1.0, 0.0], 0.75) == pytest.approx([0.8, 0.2])
    parsed = parse_controls({"think_mode": "hybrid", "gate": 0.9, "mix": 1, "think_budget": 10},
                            ThinkControls("never", 0.6, 8, 0.75), True)
    assert (parsed.think_mode, parsed.gate, parsed.budget, parsed.mix) == ("auto", 0.9, 8, 1.0)
    with pytest.raises(ValueError):
        parse_controls({"think_mode": "always"}, ThinkControls(), False)


def test_rlcd2_reward_peaks_at_the_target_and_clips_advantages():
    target = [0.0, 1.0]
    perfect = decision_score([0.0, 1.0], target, "noul")
    unsure = decision_score([0.5, 0.5], target, "noul")
    wrong = decision_score([1.0, 0.0], target, "noul")
    assert perfect > unsure > wrong
    assert rlcd2_reward([0.5, 0.5], [0.0, 1.0], target, 7, "noul") == 0.0
    gain = rlcd2_reward([0.5, 0.5], [0.0, 1.0], target, 8, "noul", budget=8, lam=1e-4)
    penalised = rlcd2_reward([0.5, 0.5], [0.0, 1.0], target, 18, "noul", budget=8, lam=1e-4)
    assert gain == pytest.approx(perfect - unsure)
    assert penalised == pytest.approx(gain - 1e-3)
    assert group_advantages([1.0, 3.0]) == pytest.approx([-1.0, 1.0])
    assert group_advantages([100.0, 0.0]) == [5.0, -5.0]
    assert clipped_policy_loss([1.0, 1.0], [1.0, -1.0]) == pytest.approx(0.0)
    assert k3_kl([0.0], [0.0]) == pytest.approx(0.0)
    assert k3_kl([0.0], [-1.0]) > 0


def test_precision_pairs_flip_one_detail_and_match_the_rule():
    rows = build_rows(3, 11)
    audit = audit_rows(rows)
    assert audit["mismatches"] == []
    assert audit["pairs"] == 12 and audit["rows"] == 24
    scored = [{"pair_id": row["pair_id"], "correct": True} for row in rows]
    assert pair_accuracy(scored) == (1.0, 12)
    scored[0]["correct"] = False
    accuracy, n_pairs = pair_accuracy(scored)
    assert n_pairs == 12 and accuracy == pytest.approx(11 / 12)


def _cpu_qwen():
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen
    from transformers import AutoModelForCausalLM, Qwen3_5TextConfig

    qwen.torch_chunk_gated_delta_rule = inspect.unwrap(qwen.torch_chunk_gated_delta_rule)
    qwen.torch_recurrent_gated_delta_rule = inspect.unwrap(qwen.torch_recurrent_gated_delta_rule)
    config = Qwen3_5TextConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        layer_types=["full_attention", "linear_attention", "full_attention"], num_attention_heads=2,
        num_key_value_heads=1, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=8, linear_value_head_dim=8, tie_word_embeddings=False)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    model.eval()
    return model


def test_bidirectional_full_attention_sees_later_tokens_and_causal_restores():
    model = _cpu_qwen()
    body = model.model
    ids = torch.randint(1, 128, (2, 10), generator=torch.Generator().manual_seed(1))
    flipped = ids.clone()
    flipped[:, -1] = (flipped[:, -1] + 9) % 128
    sequences = [row.tolist() for row in ids]
    causal = last_hidden(body, sequences, 0, "cpu", "causal")
    bidirectional = last_hidden(body, sequences, 0, "cpu", "bidirectional")
    restored = last_hidden(body, sequences, 0, "cpu", "causal")
    assert torch.allclose(causal, restored, atol=1e-5)
    assert not torch.allclose(causal, bidirectional, atol=1e-3)

    def states(tokens, attention):
        apply_prompt_attention(body, attention)
        with torch.no_grad():
            return body(input_ids=tokens, use_cache=False).last_hidden_state

    with torch.no_grad():
        before = states(ids, "causal")
        after = states(flipped, "causal")
        assert torch.allclose(before[:, 0], after[:, 0], atol=1e-6)
        apply_prompt_attention(body, "bidirectional")
        b0 = states(ids, "bidirectional")
        b1 = states(flipped, "bidirectional")
        assert (b0[:, 0] - b1[:, 0]).abs().max() > 1e-3
    apply_prompt_attention(body, "causal")
    short = [[1, 2, 3, 4], [5, 6, 7], [8, 9, 10, 11]]
    mixed = last_hidden(body, short, 0, "cpu", "bidirectional")
    alone = last_hidden(body, [short[0]], 0, "cpu", "bidirectional")
    assert torch.allclose(mixed[0], alone[0], atol=1e-5)
    apply_prompt_attention(body, "causal")


def test_bidirectional_readout_backpropagates():
    model = _cpu_qwen()
    model.train()
    ids = [[1, 2, 3, 4, 5], [6, 7, 8]]
    hidden = last_hidden(model.model, ids, 0, "cpu", "bidirectional")
    hidden.sum().backward()
    assert any(param.grad is not None and float(param.grad.abs().sum()) > 0 for param in model.parameters())
    apply_prompt_attention(model.model, "causal")


def test_generate_thought_stops_on_the_decision_marker():
    model = _cpu_qwen()

    class Tok:
        def decode(self, tokens):
            return "note [decision] more" if len(tokens) >= 2 else "note"

    text, produced = generate_thought(
        model.model, model.lm_head, torch.tensor([4, 5, 6]), Tok().decode, set(), 8, 0.0, 20, 0.95)
    assert produced >= 2 and "[decision]" not in text and text == "note"
    logits = torch.tensor([0.1, 5.0, 0.2])
    assert sample_token(logits, 0.0, 20, 0.95) == 1


def test_autothink_request_mixes_the_second_readout_and_decisions_alias():
    torch.manual_seed(0)

    class TextBody:
        def __call__(self, input_ids):
            return type("Out", (), {"last_hidden_state": torch.zeros(*input_ids.shape, 1)})()

    class Head:
        lm_head = None
        attention_mode = "causal"

        def logits(self, h, kind, n):
            return torch.zeros(n)

    class Tok:
        choice_labels = list("ABCDEFGHIJKLMNOP")
        eos_token_id = None

        def __call__(self, text, add_special_tokens=False):
            return type("E", (), {"input_ids": [1, 2, 3]})()

    srv.S.clear()
    srv.S.update(args=srv.parse_args(["--variant", "autothink", "--device", "cpu", "--mix", "0.75", "--gate", "0.6"]),
                 tok=Tok(), body=TextBody(), head=Head(),
                 temps={"noul": 1.0, "choice": 1.0, "score": 1.0}, lock=asyncio.Lock(), attention="causal")

    def write(*args):
        return "event 4; deadline 3; 4 <= 3 is False.", 5

    def read(*args):
        return [0.1, 0.9], 9

    original_write, original_read = srv._write_thought, srv._read_thought
    srv._write_thought = write
    srv._read_thought = read
    payload = {"state": "Deadline day 3. Event day 4.",
               "questions": {"on_time": {"type": "noul", "instructions": "Is the event on time?"}},
               "return_details": True, "return_thought": True}
    try:
        with TestClient(srv.build_app()) as client:
            always = client.post("/v1/systemone", json={**payload, "think_mode": "always"})
            response = client.post("/v1/decisions", json=payload)
            assert response.status_code == 200, response.text
            answer = response.json()["answers"]["on_time"]
            # first readout is uniform 0.5/0.5, second is 0.1/0.9, mix 0.75 on the second
            assert answer["noul"] == pytest.approx(0.25 * 0.5 + 0.75 * 0.9)
            assert answer["thought"].startswith("event 4")
            assert response.json()["usage"]["output_tokens"] == 5
            assert always.status_code == 200
    finally:
        srv._write_thought, srv._read_thought = original_write, original_read
        srv.S.clear()

    srv.S.clear()
    srv.S.update(args=srv.parse_args(["--device", "cpu"]),
                 tok=Tok(),
                 body=TextBody(), head=Head(), temps={"noul": 1.0, "choice": 1.0, "score": 1.0},
                 lock=asyncio.Lock(), attention="causal")
    with TestClient(srv.build_app()) as client:
        denied = client.post("/v1/systemone", json={**payload, "think_mode": "always"})
        assert denied.status_code == 422
        kept = client.post("/v1/systemone", json={"state": "s", "questions": payload["questions"]})
        assert kept.status_code == 200
        assert kept.json()["answers"]["on_time"]["noul"] == pytest.approx(0.5)
        assert "output_tokens" not in kept.json()["usage"]
    srv.S.clear()


def test_average_lora_checkpoints(tmp_path):
    from safetensors.torch import load_file, save_file

    from trainjev.modeling import VERBALIZER_IDS

    def write(path, value, mode):
        os.makedirs(path, exist_ok=True)
        torch.save({"weight": torch.full((24, 4), value), "bias": torch.full((24,), value),
                    "verbalizer_ids": VERBALIZER_IDS, "attention": mode}, os.path.join(path, "head.pt"))
        save_file({"lora": torch.full((2, 2), value)}, os.path.join(path, "adapter_model.safetensors"))
        with open(os.path.join(path, "adapter_config.json"), "w", encoding="utf-8") as stream:
            stream.write('{"r": 2, "lora_alpha": 4, "target_modules": ["q_proj"]}\n')

    write(tmp_path / "a", 1.0, "bidirectional")
    write(tmp_path / "b", 3.0, "bidirectional")
    out = str(tmp_path / "avg")
    report = average_checkpoints([str(tmp_path / "a"), str(tmp_path / "b")], out)
    assert report["attention"] == "bidirectional"
    head = torch.load(os.path.join(out, "head.pt"), weights_only=True)
    assert torch.equal(head["weight"], torch.full((24, 4), 2.0))
    assert load_file(os.path.join(out, "adapter_model.safetensors"))["lora"].equal(torch.full((2, 2), 2.0))
    write(tmp_path / "c", 1.0, "causal")
    with pytest.raises(ValueError):
        average_checkpoints([str(tmp_path / "a"), str(tmp_path / "c")], str(tmp_path / "bad"))


def test_decision_head_declares_attention_mode():
    head = DecisionHead(torch.zeros(24, 4))
    assert head.attention_mode == "causal" and head.lm_head is None
