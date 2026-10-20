"""CPU-only tiny random Qwen evidence for the opt-in prompt-attention implementation; no weights/downloads."""
import copy
import inspect
import json

import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen3_5TextConfig

from trainjev.attention import (
    BIDIRECTIONAL,
    FLAG,
    attention_metadata,
    check_checkpoint_attention,
    configure_prompt_attention,
    save_attention_metadata,
    validate_attention_metadata,
)
from trainjev.modeling import DecisionHead, last_hidden, load_model, load_trained, load_vision_model
from trainjev.train import parse_args


@pytest.fixture(autouse=True)
def cpu_delta_kernels(monkeypatch):
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen

    # Exercise the stock PyTorch CPU DeltaNet implementation, not an installed CUDA-only FLA dispatch.
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
        monkeypatch.setattr(qwen, name, inspect.unwrap(getattr(qwen, name)))


def tiny():
    config = Qwen3_5TextConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, linear_num_key_heads=2,
        linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8,
        rope_parameters={"rope_type": "default", "rope_theta": 10000, "partial_rotary_factor": 1.0,
                         "mrope_section": [3, 3, 2]},
    )
    torch.manual_seed(17)
    return AutoModelForCausalLM.from_config(config, dtype=torch.float32, attn_implementation="sdpa").model.eval()


def hidden(body, ids, mask=None):
    return body(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state


def test_default_is_exact_stock_causal_and_cli_opt_in():
    body = tiny()
    ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    original_forward = body.forward
    with torch.no_grad():
        refs = [hidden(body, ids, m) for m in (None, mask)]
        configure_prompt_attention(body)
        assert body.forward == original_forward and not hasattr(body.config, FLAG)
        for ref, m in zip(refs, (None, mask)):
            torch.testing.assert_close(hidden(body, ids, m), ref, rtol=0, atol=0)
        torch.testing.assert_close(last_hidden(body, [[1, 2, 3, 4], [5, 6]], 0, "cpu"),
                                   torch.stack([refs[0][0, -1], hidden(body, ids[1:2, :2])[0, -1]]),
                                   rtol=1e-5, atol=1e-6)
    assert not getattr(parse_args(["--out", "unused"]), FLAG)
    assert getattr(parse_args(["--out", "unused", f"--{FLAG}"]), FLAG)


def test_late_token_reaches_early_positions_only_with_bidi_and_linear_mixer_stays_causal():
    causal = tiny()
    bidi = configure_prompt_attention(copy.deepcopy(causal), True)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    late = ids.clone()
    late[0, -1] = 9
    causal_linear, bidi_linear = [], []
    hooks = [causal.layers[0].register_forward_hook(lambda m, a, o: causal_linear.append(o.detach().clone())),
             bidi.layers[0].register_forward_hook(lambda m, a, o: bidi_linear.append(o.detach().clone()))]
    with torch.no_grad():
        c1, c2 = hidden(causal, ids), hidden(causal, late)
        b1, b2 = hidden(bidi, ids), hidden(bidi, late)
    for hook in hooks:
        hook.remove()
    torch.testing.assert_close(c1[:, :-1], c2[:, :-1], rtol=0, atol=0)
    assert (b1[0, 0] - b2[0, 0]).abs().max() > 1e-5
    torch.testing.assert_close(causal_linear[0][:, :-1], causal_linear[1][:, :-1], rtol=0, atol=0)
    for c, b in zip(causal_linear, bidi_linear):
        torch.testing.assert_close(c, b, rtol=0, atol=0)
    assert [i for i, layer in enumerate(bidi.layers) if hasattr(layer, "self_attn")] == [1, 3]
    assert bidi.layers[1].self_attn.is_causal  # module default remains untouched; only the full-layer mask changes
    # A separate causal model is not affected by enabling this model.
    torch.testing.assert_close(hidden(causal, ids), c1, rtol=0, atol=0)


def test_padding_contents_lengths_batchmates_and_readout_are_invariant():
    body = configure_prompt_attention(tiny(), True)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 0, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]])
    alternate = ids.clone()
    alternate[1, 3:] = torch.tensor([80, 81, 82])
    other_row = ids.clone()
    other_row[0] = torch.tensor([90, 91, 92, 93, 94, 95])
    with torch.no_grad():
        reference = hidden(body, ids, mask)
        torch.testing.assert_close(hidden(body, alternate, mask)[:, :3], reference[:, :3], rtol=0, atol=0)
        torch.testing.assert_close(hidden(body, other_row, mask)[1, :3], reference[1, :3], rtol=0, atol=0)
        torch.testing.assert_close(hidden(body, ids[1:2, :3])[0], reference[1, :3], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(hidden(body, ids[:1])[0], reference[0], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(last_hidden(body, [ids[0].tolist(), ids[1, :3].tolist()], 0, "cpu"),
                                   torch.stack([reference[0, -1], reference[1, 2]]), rtol=0, atol=0)


@pytest.mark.parametrize("lora", [False, True])
def test_finite_nonzero_grads_and_gradient_checkpoint_recomputation_match(lora):
    plain = configure_prompt_attention(tiny(), True)
    if lora:
        from peft import LoraConfig, get_peft_model

        plain = get_peft_model(plain, LoraConfig(r=2, lora_alpha=4, lora_dropout=0,
                                               target_modules=["q_proj", "v_proj", "in_proj_qkv"]))
    checkpointed = copy.deepcopy(plain)
    checkpointed.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if lora:
        checkpointed.enable_input_require_grads()
    head = DecisionHead(torch.randn(24, 32))
    checked_head = copy.deepcopy(head)
    calls = []
    decoder = checkpointed.get_base_model() if lora else checkpointed
    assert all(layer.gradient_checkpointing for layer in decoder.layers)
    hook = decoder.layers[1].register_forward_pre_hook(lambda m, a: calls.append(1))
    outputs = []
    for body, decision_head in [(plain, head), (checkpointed, checked_head)]:
        body.train()
        h = last_hidden(body, [[1, 2, 3, 4, 5], [6, 7, 8]], 0, "cpu")
        loss = decision_head.slot_logits(h).square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        outputs.append(h.detach())
        grads = [p.grad for p in body.parameters() if p.requires_grad and p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads) > 0
        assert torch.isfinite(decision_head.weight.grad).all() and decision_head.weight.grad.abs().sum() > 0
        if not lora:
            assert body.embed_tokens.weight.grad[1:9].abs().sum() > 0
            assert body.layers[1].self_attn.q_proj.weight.grad.abs().sum() > 0
            assert body.layers[0].linear_attn.in_proj_qkv.weight.grad.abs().sum() > 0
        else:
            assert any(p.grad is not None and p.grad.abs().sum() > 0
                       for n, p in body.named_parameters() if "lora_B" in n)
    hook.remove()
    assert len(calls) >= 2  # the full-attention layer really ran again during backward
    torch.testing.assert_close(*outputs, rtol=0, atol=0)
    for (name, p), (other_name, q) in zip(plain.named_parameters(), checkpointed.named_parameters()):
        assert name == other_name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("kwargs", [
    {"use_cache": True}, {"past_key_values": object()}, {"position_ids": torch.zeros(3, 1, 4, dtype=torch.long)},
    {"position_ids": torch.tensor([[4, 5, 6, 7]])}, {"attention_mask": torch.tensor([[1, 0, 1, 0]])},
    {"attention_mask": torch.tensor([[0, 1, 1, 1]])}, {"attention_mask": torch.zeros(1, 4)},
    {"attention_mask": {"full_attention": None}}, {"attention_mask": torch.tensor([[1, 2, 1, 1]])},
    {"inputs_embeds": torch.zeros(1, 4, 32)}, {"pixel_values": torch.zeros(1)},
])
def test_fail_closed_for_cache_chunking_vision_and_non_right_padding(kwargs):
    body = configure_prompt_attention(tiny(), True)
    with pytest.raises(ValueError):
        body(input_ids=torch.tensor([[1, 2, 3, 4]]), **kwargs)


def test_fail_closed_for_version_or_undeclared_layer(monkeypatch):
    body = tiny()
    monkeypatch.setattr("transformers.__version__", "5.8.0")
    with pytest.raises(RuntimeError, match="5.17.0"):
        configure_prompt_attention(body, True)
    monkeypatch.setattr("transformers.__version__", "5.17.0")
    body.config.layer_types[0] = "full_attention"
    with pytest.raises(ValueError, match="configured"):
        configure_prompt_attention(body, True)


@pytest.mark.parametrize("full", [False, True])
def test_saved_lora_and_full_checkpoints_require_explicit_bidi_and_roundtrip(tmp_path, full):
    original = tiny()
    head = DecisionHead(torch.randn(24, 32))
    fresh = copy.deepcopy(original)
    if full:
        from safetensors.torch import save_file

        configure_prompt_attention(original, True)
        save_file(original.state_dict(), tmp_path / "model.safetensors")
    else:
        from peft import LoraConfig, get_peft_model

        original = get_peft_model(configure_prompt_attention(original, True),
                                  LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"], lora_dropout=0))
        with torch.no_grad():
            for name, p in original.named_parameters():
                if "lora_B" in name:
                    p.fill_(0.02)
        original.save_pretrained(tmp_path)
    metadata = save_attention_metadata(tmp_path, original)
    assert metadata["mode"] == BIDIRECTIONAL and metadata["full_attention_layers"] == [1, 3]
    torch.save(dict(head.state_dict(), attention_config=metadata, args={FLAG: True}), tmp_path / "head.pt")
    with pytest.raises(ValueError, match="refusing causal"):
        load_trained(fresh, head, str(tmp_path))
    loaded = load_trained(fresh, head, str(tmp_path), experimental_prompt_bidirectional=True)
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        torch.testing.assert_close(hidden(loaded.eval(), ids), hidden(original.eval(), ids), rtol=1e-5, atol=1e-6)
    # A missing sidecar is still guarded by head.pt, which travels with every supported checkpoint.
    (tmp_path / "attention_config.json").unlink()
    with pytest.raises(ValueError, match="refusing causal"):
        check_checkpoint_attention(tmp_path)
    with pytest.raises(ValueError, match="vision"):
        load_vision_model("not-a-real-model", str(tmp_path), device="cpu", experimental_prompt_bidirectional=True)
    with pytest.raises(ValueError, match="vision"):
        load_vision_model("not-a-real-model", str(tmp_path), device="cpu")
    with pytest.raises(ValueError, match="refusing causal"):
        load_model("not-a-real-model", str(tmp_path), device="cpu")


def test_metadata_disagreement_resume_mode_and_vision_are_rejected(tmp_path):
    causal = tiny()
    bidi = configure_prompt_attention(copy.deepcopy(causal), True)
    metadata = save_attention_metadata(tmp_path, bidi)
    torch.save({"args": {FLAG: False}}, tmp_path / "head.pt")
    with pytest.raises(ValueError, match="disagree"):
        check_checkpoint_attention(tmp_path, True)
    for saved, enabled in [(metadata, False), (attention_metadata(causal), True), (None, True)]:
        with pytest.raises(ValueError):
            validate_attention_metadata(saved, enabled, strict=True)
    validate_attention_metadata(None, False, strict=True)  # legacy causal resume stays supported
    with pytest.raises(ValueError, match="vision"):
        load_vision_model("not-a-real-model", experimental_prompt_bidirectional=True)
    with pytest.raises(ValueError, match="requires"):
        configure_prompt_attention(bidi, False)
    metadata["mode"] = "unknown_future_mode"
    (tmp_path / "attention_config.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Unsupported"):
        check_checkpoint_attention(tmp_path, True)


def test_entrypoints_share_one_global_flag_and_reject_remote_or_vision():
    from trainjev.eval_vision import main as vision_main
    from trainjev.evaluate import main as evaluate_main
    from trainjev.serve import parse_args as serve_args
    from trainjev.serve import setup

    assert not getattr(serve_args([]), FLAG)
    assert getattr(serve_args([f"--{FLAG}"]), FLAG)
    for conflict in ["--vision", "--proxy"]:
        args = [f"--{FLAG}", conflict] + (["https://invalid.example"] if conflict == "--proxy" else [])
        with pytest.raises(ValueError, match="local text"):
            setup(serve_args(args))
    with pytest.raises(ValueError, match="vision"):
        vision_main(["--di_root", "unused", "--out", "unused", f"--{FLAG}"])
    from trainjev.evaluate import make_backend

    with pytest.raises(ValueError, match="remote"):
        make_backend(type("Args", (), {"systemone_url": "https://invalid.example", "decide_url": "", FLAG: True})())
    with pytest.raises(SystemExit) as exit_info:
        evaluate_main(["--help"])
    assert exit_info.value.code == 0


def test_runtime_backend_drift_or_config_without_wrapper_fails_closed():
    body = tiny()
    setattr(body.config, FLAG, True)
    with pytest.raises(ValueError, match="without configuring"):
        last_hidden(body, [[1, 2, 3]], 0, "cpu")
    configure_prompt_attention(body, True)
    body.set_attn_implementation("eager")
    with pytest.raises(ValueError, match="backend changed"):
        hidden(body, torch.tensor([[1, 2, 3]]))


def test_bidi_api_health_and_image_422_preserve_all_three_text_answers(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    import trainjev.serve as srv

    body = configure_prompt_attention(tiny(), True)
    head = DecisionHead(torch.randn(24, 32))

    class Tokenizer:
        choice_labels = list("ABCDEFGHIJKLMNOP")

        def __call__(self, text, **kwargs):
            return SimpleNamespace(input_ids=[1, 2, 3, 4])

    monkeypatch.setattr(srv, "S", dict(args=srv.parse_args([f"--{FLAG}", "--device", "cpu"]),
                                      tok=Tokenizer(), body=body, head=head, processor=None, vision=None,
                                      temps={"noul": 1, "choice": 1, "score": 1}, lock=asyncio.Lock()))
    payload = {"state": "text", "questions": {
        "boolean": {"type": "noul", "instructions": "Is this text?"},
        "choice": {"type": "choice", "criteria": {"a": "A", "b": "B"}},
        "score": {"type": "score", "criteria": [str(i) for i in range(6)]}}}
    with TestClient(srv.build_app()) as client:
        health = client.get("/health").json()
        assert health[FLAG] is True and health["images"] is False
        response = client.post("/v1/systemone", json=payload)
        assert response.status_code == 200
        answers = response.json()["answers"]
        assert {k: v["type"] for k, v in answers.items()} == {
            "boolean": "noul", "choice": "choice", "score": "score"}
        assert answers["choice"]["confidence"] == answers["choice"]["probabilities"][answers["choice"]["choice"]]
        assert 0 <= answers["boolean"]["noul"] <= 1 and 0 <= answers["score"]["score"] <= 5
        for kind in ("choice", "score"):
            assert sum(answers[kind]["probabilities"].values()) == pytest.approx(1)
        response = client.post("/v1/systemone", json=dict(payload, assets={"image": {"type": "image"}}))
        assert response.status_code == 422 and "answers" not in response.json()


def test_cpu_fsdp_meta_load_checkpointing_and_resume_metadata(tmp_path, monkeypatch):
    import torch.distributed as dist
    from torch.distributed.fsdp import fully_shard

    from trainjev.resume import load_state, save_state
    from trainjev.train_full import load_sharded

    base = AutoModelForCausalLM.from_config(tiny().config, dtype=torch.float32, attn_implementation="sdpa")
    base_path = tmp_path / "base"
    base.save_pretrained(base_path)
    reference = configure_prompt_attention(base.model, True)
    monkeypatch.setattr("trainjev.train_full.VERBALIZER_IDS", list(range(24)))
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    try:
        def shard(body):
            for layer in body.layers:
                fully_shard(layer)
            fully_shard(body)

        body, init_rows = load_sharded(str(base_path), 0, "cpu", shard, experimental_prompt_bidirectional=True)
        body.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        head = DecisionHead(init_rows)
        ids = [[1, 2, 3, 4], [5, 6]]
        body.train()
        out = last_hidden(body, ids, 0, "cpu")
        torch.testing.assert_close(out, last_hidden(reference, ids, 0, "cpu"), rtol=1e-5, atol=1e-6)
        head.slot_logits(out).square().mean().backward()
        grads = [p.grad.to_local() for p in body.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads) and sum(g.abs().sum() for g in grads) > 0
        optimizer = torch.optim.AdamW([*body.parameters(), *head.parameters()])
        state_path = str(tmp_path / "state")
        save_state(state_path, body, head, optimizer, {"step": 0, "best": [1.0, None], "hist": []})
        saved = json.loads((tmp_path / "state" / "meta.json").read_text())
        assert saved["attention_config"]["mode"] == BIDIRECTIONAL
        assert load_state(state_path, body, head, optimizer)["attention_config"] == saved["attention_config"]
        causal = tiny()
        with pytest.raises(ValueError, match="refusing causal"):
            load_state(state_path, causal, head, torch.optim.AdamW(causal.parameters()))
    finally:
        dist.destroy_process_group()
