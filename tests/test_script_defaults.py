"""Check public model defaults and training options without loading models or running training."""
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def shell_runner(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture = tmp_path / "calls.jsonl"
    program = """#!/usr/bin/env python3
import json, os, sys
with open(os.environ['SCRIPT_CALLS'], 'a') as stream:
    stream.write(json.dumps(sys.argv[1:]) + '\\n')
"""
    for name in ("python", "torchrun"):
        path = bin_dir / name
        path.write_text(program)
        path.chmod(0o755)

    def run(script, **settings):
        env = os.environ.copy()
        for key in ("BASE", "CKPT", "GPUS", "DATA", "OUT", "NAME", "DEVICE", "STRATEGY", "INIT_ADAPTER", "MIX",
                    "MIX_FRACTION", "KIND_COUNTS", "LR", "HEAD_LR", "RESUME", "FIT_CALIBRATION", "BASE_ONLY",
                    "ADAPTER_REVISION", "VISION", "ATTENTION", "THOUGHT_DROP", "VARIANT"):
            env.pop(key, None)
        env.update(PATH=str(bin_dir) + os.pathsep + env["PATH"], SCRIPT_CALLS=str(capture),
                   OUT=str(tmp_path / "output"), **settings)
        result = subprocess.run(["bash", str(ROOT / "scripts" / script)], env=env, capture_output=True, text=True)
        calls = [json.loads(line) for line in capture.read_text().splitlines()] if capture.exists() else []
        return result, calls

    return run


def value(arguments, flag):
    return arguments[arguments.index(flag) + 1]


def test_serving_defaults_are_public_hf_models(shell_runner):
    result, calls = shell_runner("run_serve.sh")
    assert result.returncode == 0, result.stderr
    args = calls[0]
    assert value(args, "--base") == "Qwen/Qwen3.5-4B"
    assert value(args, "--adapter") == "shibing624/train-jev-4b-lora"
    assert value(args, "--name") == "train-jev-4b-lora"
    assert "--adapter_revision" not in args


def test_serving_vision_flag(shell_runner):
    result, calls = shell_runner("run_serve.sh", VISION="1")
    assert result.returncode == 0, result.stderr
    assert "--vision" in calls[0]


def test_serving_overrides_and_revision(shell_runner):
    result, calls = shell_runner("run_serve.sh", BASE="Qwen/Qwen3.8-27B", CKPT="shibing624/train-jev-27b-lora",
                                ADAPTER_REVISION="v0.3.0", DEVICE="cpu", STRATEGY="tournament")
    assert result.returncode == 0, result.stderr
    args = calls[0]
    assert value(args, "--name") == "train-jev-27b-lora"
    assert value(args, "--adapter_revision") == "v0.3.0"
    assert value(args, "--device") == "cpu"
    assert value(args, "--strategy") == "tournament"


def test_evaluation_does_not_refit_published_model(shell_runner):
    result, calls = shell_runner("run_eval.sh")
    assert result.returncode == 0, result.stderr
    args = calls[0]
    assert value(args, "--adapter") == "shibing624/train-jev-4b-lora"
    assert value(args, "--base") == "Qwen/Qwen3.5-4B"
    assert "--fit_calibration" not in args


def test_local_calibration_is_explicit(shell_runner):
    result, calls = shell_runner("run_eval.sh", CKPT="runs/custom/best", FIT_CALIBRATION="1")
    assert result.returncode == 0, result.stderr
    assert "--fit_calibration" in calls[0]
    assert value(calls[0], "--adapter") == "runs/custom/best"


@pytest.mark.parametrize("script,base", [("run_train.sh", "Qwen/Qwen3.5-4B"),
                                         ("run_train_27b.sh", "Qwen/Qwen3.8-27B")])
def test_training_launcher_supports_supervision(shell_runner, script, base):
    result, calls = shell_runner(script, INIT_ADAPTER="runs/initial/best", MIX="data/gold/train.jsonl",
                                MIX_FRACTION="0.125", KIND_COUNTS="choice:64,noul:32,score:32",
                                LR="2e-5", HEAD_LR="4e-5")
    assert result.returncode == 0, result.stderr
    args = calls[0]
    assert value(args, "--base") == base
    assert value(args, "--init_adapter") == "runs/initial/best"
    assert value(args, "--mix") == "data/gold/train.jsonl"
    assert value(args, "--mix_fraction") == "0.125"
    assert value(args, "--kind_counts") == "choice:64,noul:32,score:32"
    assert value(args, "--lr") == "2e-5"
    assert value(args, "--head_lr") == "4e-5"


def test_instant_and_autothink_launchers_set_attention(shell_runner):
    instant, instant_calls = shell_runner("run_train_instant.sh")
    assert instant.returncode == 0, instant.stderr
    assert value(instant_calls[-1], "--attention") == "bidirectional"
    auto, auto_calls = shell_runner("run_train_autothink.sh")
    assert auto.returncode == 0, auto.stderr
    assert value(auto_calls[-1], "--attention") == "causal"
    assert value(auto_calls[-1], "--thought_drop") == "0.5"


def test_serve_variant_is_opt_in(shell_runner):
    _, plain_calls = shell_runner("run_serve.sh")
    assert "--variant" not in plain_calls[-1]
    _, auto_calls = shell_runner("run_serve.sh", VARIANT="autothink")
    assert value(auto_calls[-1], "--variant") == "autothink"


@pytest.mark.parametrize("script", ["run_train.sh", "run_train_27b.sh"])
def test_missing_resume_state_fails_before_training(shell_runner, script):
    result, calls = shell_runner(script, RESUME="1")
    assert result.returncode == 2
    assert "No resumable training state" in result.stderr
    assert calls == []
