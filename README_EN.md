[**🇨🇳中文**](https://github.com/shibing624/train-jev/blob/main/README.md) | [**🌐English**](https://github.com/shibing624/train-jev/blob/main/README_EN.md) | [**📖文档/Docs**](docs/) | [**🤖模型/Models**](https://huggingface.co/shibing624)

-----------------

# train-jev: Training JEV-style Decision Models

[![HF Models](https://img.shields.io/badge/Hugging%20Face-shibing624-green)](https://huggingface.co/shibing624)
[![Github Stars](https://img.shields.io/github/stars/shibing624/train-jev?color=yellow)](https://star-history.com/#shibing624/train-jev&Timeline)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_version](https://img.shields.io/badge/Python-3.12%2B-green.svg)](requirements.txt)
[![GitHub issues](https://img.shields.io/github/issues/shibing624/train-jev.svg)](https://github.com/shibing624/train-jev/issues)

## 📖 Introduction

**train-jev** is an open-source framework for training **System 1 typed-decision models**. It reproduces the training
recipe of JEV ([autotrust/JEV-27B](https://huggingface.co/autotrust/JEV-27B)): a **LoRA + 24-slot decision head** on a
Qwen3.5 backbone, **KL-distilled** from TypeSafe Jev 1.13's full output distributions, returning **calibrated
probabilities in a single forward pass**.

A decision model answers typed questions about a state:

| type | example | output |
|---|---|---|
| `noul` | Is this ticket a platform-wide outage? | P(true) |
| `choice` | Which team should handle it? (2–16 options; up to 256 via a tournament) | a probability per option |
| `score` | How urgent is it? (levels 0–5) | a 6-level distribution and expected score |

train-jev covers the whole pipeline:

- **Training**: multi-GPU DDP via `torchrun`; the full JEV recipe in one file (LoRA r16, fp32 decision head, KL + RPS
  loss, option-permutation augmentation, kind-stratified batches).
- **Calibration**: per-kind temperature fitting, so the probabilities can be used directly as confidence.
- **Evaluation**: four suites covering in-distribution distillation, OOD transfer, a public benchmark
  ([Decision Index](https://github.com/apolinario/decision-index) 0.3), and zero-shot image decisions.
- **Serving**: a TypeSafe [System One API](https://docs.typesafe.ai/api) compatible `POST /v1/systemone`, plus JEV's
  `/v1/decide`. Existing SDKs and harnesses work unchanged.
- **Image decisions**: training is text-only and Qwen3.5's vision encoder is left untouched, so the trained model can
  make decisions about images zero-shot.

> This is an independent research reproduction, not affiliated with TypeSafe AI or autotrust.

## 🔥 News

[2026/10/14] v0.1.0: first release. Reproduces the JEV recipe on Qwen3.5-4B. Training takes 2.8 h on 3× H20.
In-distribution KL is 0.018, essentially level with the official JEV-27B (0.017). Zero-shot image decision skill rises
from 0.632 to 0.697. Qwen3.8-27B (the JEV-27B backbone) is supported, and a 13-dataset classification PK is included.
Full results are in [docs/results.md](docs/results.md).

## 😊 Features

- **One forward pass per decision**: no generation. The model reads a 24-slot head at the last prompt token; the 4B
  model's single-request p50 is about 43 ms on an H20.
- **Starts out identical to the backbone**: the head is initialised from `lm_head`'s rows for the verbalizer tokens
  (`false/true`, `0–5`, `A–P`), so at step 0 it reproduces the base model's restricted next-token distribution
  (|Δp| = 7.7e-6). Distillation sharpens what pretraining already knows.
- **Small trainable footprint**: 30.5M LoRA parameters (0.7% of the backbone) plus a 61k-parameter head; the backbone
  stays frozen.
- **Covers Qwen3.5's hybrid stack**: the LoRA covers the Gated DeltaNet layers (`in_proj_qkv / in_proj_z / out_proj`)
  as well as full attention. Batches are right-padded, so padding never leaks into the linear-attention recurrent state.
- **Answers by option content, not position**: 30% of `choice` rows have their options shuffled during training, with
  the target permuted the same way.
- **Reproducible**: with a fixed seed, every rank draws the same per-step sample. On the same inputs this code is
  bit-identical to the original experiment code: training loss, evaluation logits, server outputs and image
  probabilities all differ by 0.

## 🧠 Method

```mermaid
flowchart LR
    P["[kind] choice<br/>[state] ...<br/>[question] ...<br/>[options]<br/>A) ...<br/>[decision]:"] --> L["Qwen3.5 backbone (frozen bf16)<br/>+ LoRA r16"]
    L -->|hidden at last token| H["24-slot fp32 head<br/>noul | score 0..5 | choice A..P<br/>init = lm_head rows"]
    H --> S["slice by kind → softmax / T_kind"]
    T["Teacher: Jev 1.13 distributions<br/>(jev-distill-corpus-v3)"] -.KL + 0.5·RPS.-> S
```

| item | value |
|---|---|
| backbone | `Qwen/Qwen3.5-4B` (text model for training; the full VL model for image inference) |
| data | [`SargeDev/jev-distill-corpus-v3`](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3) `train`, 655,806 rows |
| LoRA | r=16, α=32, dropout 0.05 on 10 projection types |
| loss | KL(teacher ‖ model) + 0.5 · RPS for `score` |
| augmentation | options shuffled on 30% of `choice` rows |
| batching | 128 rows/step, kind-stratified, length-bucketed |
| optimiser | AdamW β=(0.9, 0.98), LoRA 1e-4 / head 2e-4, 3% warm-up, cosine to 0.02×, clip 1.0, one epoch (5,123 steps) |
| calibration | per-kind temperature on the `calibration` split: noul 1.008 / choice 1.004 / score 0.998 |

Details and rationale: [docs/training.md](docs/training.md).

**Backbones.** Qwen3.5 / 3.6 / 3.8 share the `qwen3_5` architecture and vocabulary, so switching is just `--base`.
Qwen3.5-4B is fully trained and evaluated. **Qwen3.8-27B**, whose config is identical to JEV-27B-VL's, is verified end
to end: load, init gate (|Δp| = 0), training, and corpus and image evaluation. Its LoRA (108.8M) and head (123k) match
the JEV-27B card. Full 9B and 27B runs are next.

## 📊 Results

These are results for **train-jev 4B**. Within each table, every model answers the same rows.

| evaluation | Kev-4B | **train-jev 4B** | JEV-27B-VL | reference |
|---|---:|---:|---:|---|
| jev-distill `test_set_30k`, KL on Jev-labelled rows ↓ | 0.364 | **0.018** | 0.017 | JEV-27B card ≈ 0.017 |
| jev-distill `ood` (unseen task families), KL ↓ | 0.624 | **0.311** | 0.108 | JEV-27B card 0.104 |
| Decision Index 0.3 public index (35/37 benchmarks, 6k-row stratified sample) ↑ | 38.4 | **40.4** | 54.1 | board, same panel: vLLM-SR d3-mini 56.1, ezjev-4B 52.9, Sieve-4B 51.5 |
| Kev `transfer-v4` (656 questions), accuracy ↑ | **0.817** | 0.779 | 0.852 | Jev 1.13: 0.857 |
| zero-shot image decisions, 5 vision sets, mean skill ↑ | n/a | **0.697** | 0.782 | untrained Qwen3.5-4B 0.632 |
| 13-dataset PK (1,300 rows), accuracy ↑ | 0.640 | **0.647** | 0.690 | Jev 1.13 API 0.700 |
| 13-dataset PK, the 8 sources Kev never trained on ↑ | 0.570 | **0.603** | 0.645 | Jev 1.13 API 0.645 |
| 13-dataset PK, ECE ↓ | 0.083 | **0.076** | 0.094 | Jev 1.13 API 0.130 |
| single-request p50 latency (H20) | 17 ms | 43 ms | 96 ms | Jev 1.13 hosted API 370 ms incl. network |

**Takeaways.**

1. **The recipe reproduces**: in-distribution KL, the validation curve and the fitted temperatures (all ≈ 1.0) match
   the official 27B report.
2. **At 4B, model capacity is the bottleneck**: the gap to 27B opens on OOD tasks, knowledge reasoning and many-class
   intent classification. On Decision Index train-jev 4B is in the same tier as Kev-4B; the leading 4B models score
   11–16 points higher.
3. **Image decisions survive text-only training**: all five vision sets improve (+6.5 points mean skill). Kev-4B
   cannot take images.

Per-area Decision Index scores, the PK results, the full training curve and protocol notes are in
[docs/results.md](docs/results.md).

## 💾 Install

```shell
git clone https://github.com/shibing624/train-jev
cd train-jev
pip install -r requirements.txt
```

This is a training repo, not a published Python package. From the repository root, run `python -m trainjev.train`.

Requires Python ≥ 3.12, `transformers>=5.17` (Qwen3.5 / 3.8; transformers 4.x does not know `qwen3_5`), `peft>=0.21`, `torch>=2.6`. On CUDA, install
`flash-linear-attention==0.5.2`; without it the Gated DeltaNet layers fall back to a slow PyTorch path.

| setup | peak memory / GPU | speed | one epoch |
|---|---:|---:|---:|
| 3× H20 96GB, `--no_gc --micro_tokens 4096` (reference) | 48.8 GB | 1.94 s/step | 2.8 h |
| 1× H20 96GB, same flags | 48.8 GB | ~4.5 s/step | ~6.4 h |
| less memory: keep gradient checkpointing on (default) | lower | slower | — |
| **Qwen3.8-27B**: 1× H20 96GB, gradient checkpointing, `--micro_tokens 16384` | 75 GB | ~50 s/step | ~70 GPU-hours |

Inference: the 4B bf16 weights are 9.3 GB, so it fits on a single 24 GB GPU; 27B (text model, ~54 GB) needs an 80 GB-class GPU.

## 🚀 Quick Start

```shell
bash scripts/prepare_data.sh                                     # data/ and models/
python scripts/check_init.py --base models/Qwen3.5-4B            # untrained head == base readout -> PASS
GPUS=0,1,2 bash scripts/run_train.sh                             # -> runs/jev-4b/best
CKPT=runs/jev-4b/best bash scripts/run_eval.sh                   # temperatures + test_set_30k / ood
CKPT=runs/jev-4b/best PORT=8020 bash scripts/run_serve.sh        # System One API
python examples/client.py --url http://127.0.0.1:8020
```

```shell
curl -s localhost:8020/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Ticket #4821: every CSV export fails with a 504 after 60s. Status page: all systems operational.",
  "questions": {
    "is_outage": {"type": "noul", "instructions": "Is this most likely a platform-wide outage?"},
    "route": {"type": "choice", "instructions": "Which team should handle this first?",
              "criteria": {"billing": "payments", "infra": "timeouts and capacity", "product": "feature requests"}}
  }}'
```

```shell
# 13-dataset PK
python scripts/pk_build.py --out data/pk
python scripts/pk_benchmark.py run --url http://127.0.0.1:8020 --suite data/pk --out eval/pk_jev-4b.jsonl
python scripts/pk_benchmark.py report jev-4b=eval/pk_jev-4b.jsonl

# Qwen3.8-27B backbone
python scripts/check_init.py --base models/Qwen3.8-27B
GPUS=0,1,2,3,4,5,6,7 bash scripts/run_train_27b.sh
```

Decision Index, Kev transfer-v4, image and PK evaluation: [docs/evaluation.md](docs/evaluation.md).

## 📁 Project Structure

```
train-jev/
├── trainjev/          # data / modeling / losses / metrics / train / evaluate / eval_vision / serve
├── scripts/           # prepare_data, run_train(_27b), run_eval, run_serve, run_decision_index, check_init, di_*, pk_*
├── examples/          # client.py
├── patches/           # small patch for the decision-index kit
├── docs/              # training.md, evaluation.md, results.md
└── tests/             # unit tests (CPU)
```

## ✅ TODO

- [x] Reproduce the JEV recipe on Qwen3.5-4B and match the official training curve and in-distribution KL
- [x] Temperature calibration, a System One compatible server, the >16-option tournament
- [x] Five evaluation suites: jev-distill (in-distribution/OOD), Decision Index 0.3, Kev transfer-v4, zero-shot images, 13-dataset PK
- [x] Qwen3.8-27B backbone (the JEV-27B base): load, init gate, training and evaluation verified
- [ ] **On top of the current LoRA recipe, run a full fine-tune (or high-rank LoRA) on jev-distill plus broader public
      gold-labelled data.** The goal is to close the two largest gaps to the leading 4B models: Knowledge & Reasoning
      (0.32 vs 0.44) and Language Understanding (0.39 vs 0.58). The top-ranked 4B models (d3-mini, RSI-Jev 4B) are
      fully fine-tuned.
- [ ] Keep calibration when mixing in hard gold labels: they tend to make the model over-confident, so try label
      smoothing or a separate temperature for the gold part
- [ ] Train Qwen3.5-9B and Qwen3.8-27B and run the full evaluation
- [ ] Release train-jev 4B / 9B / 27B weights on Hugging Face
- [ ] Image inputs in the `/v1/systemone` server (currently only in the evaluation script)

## ☎️ Contact

- Issues: [![GitHub issues](https://img.shields.io/github/issues/shibing624/train-jev.svg)](https://github.com/shibing624/train-jev/issues)
- Email: xuming624@qq.com
- WeChat: xuming624 (note: name-company-NLP) to join the NLP group

## ⚠️ License

The code is licensed under [Apache License 2.0](LICENSE). Data and base models follow their own licenses:
[jev-distill-corpus-v3](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3) (Apache-2.0, attribution
requested) and [Qwen3.5 / Qwen3.8](https://huggingface.co/Qwen), and the public datasets used by the PK suite. This project is for research use; the authors accept no
liability for any harm or loss caused by using it.

## 😇 Citation

```latex
@misc{train-jev,
  title={train-jev: Training JEV-style Decision Models},
  author={Ming Xu},
  year={2026},
  howpublished={\url{https://github.com/shibing624/train-jev}},
}
```

## 😍 Contribute

Contributions are welcome. Add unit tests under `tests/` and make sure `python -m pytest` passes before opening a
PR; see [CONTRIBUTING.md](CONTRIBUTING.md).

## 💕 Acknowledgements

- [autotrust/JEV-27B](https://huggingface.co/autotrust/JEV-27B) and [JEV-27B-VL](https://huggingface.co/autotrust/JEV-27B-VL): recipe, prompt template, `serve_decide.py`
- [SargeDev/jev-distill-corpus-v3](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3): training data
- [ZefanCai/Open-Jev](https://huggingface.co/datasets/ZefanCai/Open-Jev): OOD task families
- [jaredpalmer/kev](https://github.com/jaredpalmer/kev): Kev models and the `transfer-v4` suite
- [apolinario/decision-index](https://github.com/apolinario/decision-index): Decision Index reproduction kit
- [Jev's Architecture Unmasked](https://archerhume.com/posts/jevs-architecture-unmasked)
- [Qwen3.5](https://huggingface.co/Qwen), [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)
