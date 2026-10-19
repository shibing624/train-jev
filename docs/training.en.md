# Train train-jev from a Qwen backbone

[中文](training.md)

This repository trains probabilistic decision models with **LoRA + a 24-slot decision head**. The full path is a public Qwen backbone → distillation of Jev target distributions → gold-and-rule supervised refinement → calibration and serving.

Use the released [train-jev-4b-lora](https://huggingface.co/shibing624/train-jev-4b-lora) / [train-jev-27b-lora](https://huggingface.co/shibing624/train-jev-27b-lora), or follow this guide to train your own.

## 1. Backbone and hardware

| Size | Backbone | LoRA + head parameters | Reference hardware |
|---|---|---:|---|
| 4B | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | 30.5M + 61k | 3×H20 |
| 27B | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | 108.8M + 123k | 4×H20 |

The backbone is frozen; only LoRA and the decision head are trained. The vision encoder is not updated. The 4B launcher disables gradient checkpointing by default; 27B enables it. Lower `MICRO_TOKENS`, or enable gradient checkpointing in a direct training command, if memory is limited.

```bash
git clone https://github.com/shibing624/train-jev.git && cd train-jev
pip install -r requirements.txt
```

Requires Python 3.12+, a transformers version supporting Qwen3.5 and CUDA. `requirements.txt` includes the `flash-linear-attention` acceleration dependency.

## 2. Training data

| Data | Purpose |
|---|---|
| [SargeDev/jev-distill-corpus-v3](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3) | 655,806 soft-distribution and programmatic targets for initial distillation and replay |
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k), [AQuA-RAT](https://huggingface.co/datasets/deepmind/aqua_rat), [QASC](https://huggingface.co/datasets/allenai/qasc) | Mathematics and science choices |
| [PAWS](https://huggingface.co/datasets/google-research-datasets/paws) | Paraphrase yes/no questions |
| [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) | Correctness 0–4 converted to five-option choices |
| Business rules or your own labels | Routing, evidence, date calculations, refunds and related tasks |

Download the distillation data and backbone:

```bash
bash scripts/prepare_data.sh
# Download the corresponding backbone for 27B
BASE_REPO=Qwen/Qwen3.8-27B BASE_DIR=models/Qwen3.8-27B bash scripts/prepare_data.sh
```

Convert new data to JSONL, one question per row with a target aligned to its options:

```json
{"kind":"choice","state":"A customer was charged twice.","question":"Which team should handle this?","options":["returns","shipping","billing"],"target":[0,0,1],"weight":1.0}
```

- `noul`: fixed options `["false","true"]` and two probabilities.
- `choice`: 2–16 options; one-hot gold labels or teacher soft distributions.
- `score`: six ordered levels `["0",...,"5"]`; use `choice` for other level counts.

A minimal public PAWS conversion, saved to `data/my-gold/train.jsonl`:

```python
import json
from pathlib import Path
from datasets import load_dataset

dataset = load_dataset("google-research-datasets/paws", "labeled_final", split="train")
output = Path("data/my-gold/train.jsonl")
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", encoding="utf-8") as stream:
    for row in dataset:
        if row["label"] not in (0, 1):
            continue
        item = {
            "kind": "noul",
            "state": f"Sentence 1: {row['sentence1']}\nSentence 2: {row['sentence2']}",
            "question": "Do the two sentences have the same meaning?",
            "options": ["false", "true"],
            "target": [1 - row["label"], row["label"]],
            "weight": 1.0,
        }
        stream.write(json.dumps(item, ensure_ascii=False) + "\n")
```

This PAWS example demonstrates the method, not the full published gold-and-rule mixture; do not attach released-model scores to its outputs.

## 3. Initial distillation from the backbone

```bash
GPUS=0,1,2 bash scripts/run_train.sh
# Or train 27B
GPUS=0,1,2,3 bash scripts/run_train_27b.sh
```

Defaults: global batch 128, LoRA r16 / alpha32, LoRA learning rate 1e-4 and head learning rate 2e-4. Initial distillation takes about 5,123 steps. Outputs go to `runs/jev-4b/` or `runs/jev-27b/`; `best/` is the lowest-validation-KL checkpoint.

The method:

- Initialize head rows from the backbone's `lm_head` rows for `false/true`, `0–5` and `A–P`, not a random classifier.
- Use `KL(target || model)` to learn complete distributions; add 0.5×RPS to penalize distant ordinal errors on `score`.
- Shuffle options and targets on 30% of choice rows to reduce position bias.
- AdamW, 3% warm-up then cosine decay, global gradient clipping at 1.0.

## 4. Gold-and-rule refinement

Start from your trained checkpoint, mix in supervised JSONL and continue updating LoRA and the head:

```bash
GPUS=0,1,2 INIT_ADAPTER=runs/jev-4b/best MIX=data/my-gold/train.jsonl \
  MIX_FRACTION=0.125 LR=2e-5 HEAD_LR=4e-5 STEPS=2048 MAX_STEPS=512 SEED=17 \
  KIND_COUNTS=choice:64,noul:32,score:32 OUT=runs/jev-4b-supervised \
  EVAL_EVERY=256 bash scripts/run_train.sh
```

Use the corresponding starting point and launcher for 27B:

```bash
GPUS=0,1,2 INIT_ADAPTER=runs/jev-27b/best MIX=data/my-gold/train.jsonl \
  MIX_FRACTION=0.125 LR=2e-5 HEAD_LR=4e-5 STEPS=2048 MAX_STEPS=512 SEED=17 \
  KIND_COUNTS=choice:64,noul:32,score:32 OUT=runs/jev-27b-supervised \
  EVAL_EVERY=256 bash scripts/run_train_27b.sh
```

Separate multiple files with commas, e.g. `MIX=data/math/train.jsonl,data/public/train.jsonl,data/rules/train.jsonl`. Replay reduces forgetting while new supervision improves decisions; the published recipe uses a 12.5% new-data fraction. `STEPS` defines the learning-rate schedule; `MAX_STEPS` defines where to stop.

The released v0.3.0 models use this two-stage method with 37,147 new supervised rows and the step-512 refinement weights. Public sources let you reuse the recipe, but the complete published preprocessing data and generators are not distributed here; bitwise reproduction is not guaranteed.

## 5. Calibration, resuming and serving

Fit per-type temperatures for your own checkpoint on an independent `calibration.jsonl`:

```bash
CKPT=runs/jev-4b-supervised/step00512 FIT_CALIBRATION=1 bash scripts/run_eval.sh
# 27B
BASE=Qwen/Qwen3.8-27B CKPT=runs/jev-27b-supervised/step00512 \
  FIT_CALIBRATION=1 bash scripts/run_eval.sh
```

These commands use the distillation calibration split by default. For a new business domain, prepare independent calibration data and set `DATA`. Never fit on final evaluation questions. Published HF models include temperatures; `bash scripts/run_eval.sh` evaluates without refitting by default.

Resume interrupted training with the same settings and output directory plus `RESUME=1`:

```bash
GPUS=0,1,2 RESUME=1 OUT=runs/jev-4b bash scripts/run_train.sh
```

`SAVE_STATE_EVERY` controls checkpoint frequency. `best/` is for inference and `state/` restores optimizer and RNG state; keep GPU count and training configuration unchanged when resuming.

Serve a trained checkpoint:

```bash
CKPT=runs/jev-4b-supervised/step00512 bash scripts/run_serve.sh
# Or try the published model directly
bash scripts/run_serve.sh
```

More flags are in the [serving guide](serving.en.md).

## 6. Measured training time and results

| Run | Hardware | Steps | Time |
|---|---|---:|---:|
| 4B initial distillation | 3×H20 | 5,123 | 2.77 h |
| 27B initial distillation | 4×H20 | 5,123 | 33.07 h |
| 4B gold-and-rule experiment | 3×H20 | 2,048 | 1.3 h |
| 27B gold-and-rule experiment | 3×H20 | 1,024 | 7.9 h |

Refinement times describe the completed experiments; released models use step 512, not the final step.

| Released model | [train-jev-pk](../data/train-jev-pk) accuracy ↑ | [transfer-v4](https://github.com/jaredpalmer/kev) accuracy ↑ | transfer-v4 ECE ↓ | [Full Decision Index](https://github.com/apolinario/decision-index) score ↑ |
|---|---:|---:|---:|---:|
| shibing624/train-jev-4b-lora | 74.18% | 0.805 | 0.054 | 39.01 |
| shibing624/train-jev-27b-lora | 78.72% | 0.858 | 0.032 | 55.58 |

These are released-model results under fixed protocols, not expected scores for the PAWS example. Comparisons against [autotrust/JEV-27B-VL](https://huggingface.co/autotrust/JEV-27B-VL), Kev and TypeSafe API, with limitations, are in [the results guide](results.en.md).

## 7. Instant and AutoThink

The default remains causal attention and one forward pass. Released weights do not need a new flag. `scripts/run_train_instant.sh` trains the StartLux Instant mask (bidirectional full-attention layers, linear layers stay causal). `scripts/run_train_autothink.sh` stays causal and drops half of the supervised thoughts (`THOUGHT_DROP`, default 0.5). Average seeds with `scripts/average_lora.py`. `scripts/build_precision_mix.py` writes a local contrast mix; it is not DecisionPivot and not a DI score. Commands, cited public numbers, and risks are in the [Decision Index plan](di_plan.en.md). This repo does not yet have a new score on a shared DI suite.
