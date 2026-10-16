import json

import pytest
import torch

from examples.smoke_systemone import assess
from trainjev.modeling import resolve_checkpoint
from trainjev.resume import require_state


def test_local_checkpoint_is_resolved_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **kw: pytest.fail("Unexpected download"))
    assert resolve_checkpoint(tmp_path) == str(tmp_path)
    assert resolve_checkpoint("") == ""
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(tmp_path / "missing")


def test_hf_snapshot_includes_head_and_temperatures(tmp_path, monkeypatch):
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path)
    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    assert resolve_checkpoint("owner/model", "v0.2.0") == str(tmp_path)
    assert calls[0]["revision"] == "v0.2.0"
    assert {"head.pt", "calibration.json", "adapter_model.safetensors"} <= set(calls[0]["allow_patterns"])


def test_missing_resume_state_never_starts_fresh_training(tmp_path, monkeypatch):
    from trainjev.train import main
    monkeypatch.setattr(torch.distributed, "init_process_group", lambda *a, **kw: pytest.fail("NCCL should not start"))
    for extra in ([], ["--full_ft"]):
        with pytest.raises(FileNotFoundError, match="No completed training state"):
            main(["--out", str(tmp_path), "--resume", *extra])
    state = tmp_path / "state.old"
    state.mkdir()
    (state / "meta.json").write_text(json.dumps({"step": 3}))
    assert require_state(str(tmp_path / "state")) == str(state)


def test_smoke_check_does_not_hide_wrong_predictions():
    result = {"answers": {"damage": {"noul": 0.1},
                           "team": {"probabilities": {"billing": 0.8, "shipping": 0.1, "engineering": 0.1}},
                           "severity": {"probabilities": {str(i): 0.0 if i < 5 else 1.0 for i in range(6)}}}}
    assert assess(result)["correct"] == 3
    result["answers"]["damage"]["noul"] = 0.9
    assert assess(result)["correct"] == 2
    result["answers"]["team"]["probabilities"]["billing"] = float("nan")
    with pytest.raises(ValueError):
        assess(result)
