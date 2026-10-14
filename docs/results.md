# Results

All numbers are for **train-jev 4B** (Qwen3.5-4B + the JEV recipe, `runs/jev-4b/best` = step 5,123, temperatures fitted
on the `calibration` split). Every comparison in a table uses the same rows for every model. Reference models were run
by us on the same hardware (one NVIDIA H20 each) unless marked *board* or *reported*.

| Model | What it is |
|---|---|
| Qwen3.5-4B (untrained) | base model + head initialised from `lm_head` (the step-0 model; a zero-shot restricted readout) |
| Kev-4B | [jaredpalmer/kev](https://github.com/jaredpalmer/kev) 4B release, served with `kev.serve` |
| **train-jev 4B** | this repository |
| JEV-27B-VL | [autotrust/JEV-27B-VL](https://huggingface.co/autotrust/JEV-27B-VL), served with its own `serve_decide.py` (System 1, thinking off) |

## 1. Training trajectory

Validation KL on the fixed 4,000-row subset (`--val_n 4000`, seed 1), every 500 steps. 3× H20, `--no_gc --micro_tokens 4096`.

| step | rows seen | val KL | noul KL | score KL | choice KL | choice top-1 | elapsed |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 64k | 0.0604 | 0.027 | 0.061 | 0.128 | 0.850 | 0.29 h |
| 1000 | 128k | 0.0414 | 0.015 | 0.049 | 0.088 | 0.864 | 0.56 h |
| 1500 | 192k | 0.0320 | 0.007 | 0.043 | 0.073 | 0.883 | 0.83 h |
| 2000 | 256k | 0.0278 | 0.008 | 0.036 | 0.061 | 0.906 | |
| 2500 | 320k | 0.0244 | 0.004 | 0.032 | 0.059 | 0.906 | |
| 3000 | 384k | 0.0222 | 0.005 | 0.030 | 0.050 | 0.902 | |
| 3500 | 448k | 0.0204 | 0.003 | 0.027 | 0.050 | 0.918 | |
| 4000 | 512k | 0.0181 | 0.004 | 0.024 | 0.043 | 0.919 | |
| 4500 | 576k | 0.0164 | 0.003 | 0.023 | 0.038 | 0.930 | 2.44 h |
| 5000 | 640k | 0.0161 | 0.003 | 0.022 | 0.037 | 0.924 | 2.70 h |
| **5123** | 656k | **0.0160** | 0.003 | 0.022 | 0.037 | 0.927 | 2.77 h |

The JEV-27B card reports val KL 0.058 at step 500 and 0.017 at the released step 5,000; the 4B curve tracks it closely.
Fitted temperatures: noul 1.008, choice 1.004, score 0.998 (JEV-27B: 1.014 / 1.016 / 1.004). Values this close to 1
mean the KL objective already produces calibrated distributions.

## 2. jev-distill-corpus-v3 (in-distribution and OOD)

`test_set_30k`: 6,000-row fixed subset (seed 7). `ood`: 3,000-row fixed subset of Open-Jev task families never seen in
training, with programmatic labels. "Jev-labelled" = rows whose targets are Jev 1.13's own distributions (`yuri_v3`).
Calibrated metrics; `choice` rows with > 16 options are excluded.

| split | model | KL | KL (Jev-labelled) | noul top-1 | noul AUROC | choice top-1 | score top-1 | score MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| test_set_30k | Qwen3.5-4B (untrained) | 0.637 | 0.658 | 0.689 | 0.816 | 0.511 | 0.019 | 1.371 |
| | Kev-4B | 0.387 | 0.364 | 0.781 | 0.861 | 0.575 | 0.334 | 0.779 |
| | **train-jev 4B** | **0.022** | **0.018** | **0.962** | **0.995** | **0.908** | **0.897** | **0.104** |
| | JEV-27B-VL | 0.021 | 0.017 | 0.969 | 0.996 | 0.908 | 0.894 | 0.098 |
| ood | Qwen3.5-4B (untrained) | 0.884 | – | 0.542 | 0.558 | 0.413 | 0.000 | 1.227 |
| | Kev-4B | 0.624 | – | 0.768 | 0.703 | 0.642 | 0.471 | 0.389 |
| | **train-jev 4B** | **0.311** | – | **0.932** | **0.982** | **0.888** | 0.588 | 0.390 |
| | JEV-27B-VL | 0.108 | – | 0.970 | 0.995 | 0.913 | 0.588 | 0.432 |

*Reported* by the JEV-27B card on the full splits: KL 0.019 (all), ≈ 0.017 (Jev-labelled), OOD KL 0.104.

In distribution the 4B model is essentially indistinguishable from the 27B one; the gap opens on OOD task families
(0.311 vs 0.108).

## 3. Decision Index 0.3 (public benchmarks)

[apolinario/decision-index](https://github.com/apolinario/decision-index), edition 0.3. We score a 6,000-row stratified
sample (round-robin over benchmarks, whole groups, the kit's default seed) and build a sample-only suite so the
official scorer computes coverage over the sample (`scripts/di_sample.py`). Every model answers the same 6,000 rows
through the same `/v1/systemone` HTTP engine.

Two of the 37 benchmarks in the 0.3 public index could not be rebuilt in our environment: **HLE** (gated on the HF Hub) and **HoVer**
(a 2 GB Wikipedia download). They are removed from their areas and the remaining benchmarks renormalised
(`scripts/di_index.py`); the board's published per-benchmark skills are re-scored on the same 35-benchmark panel, so
every row below is comparable, but **none of them equals the official Decision Index total**.

| model | index (35) | Knowledge & Reasoning | Language | Retrieval & Classification | Tools & Automation | Arts & Taste |
|---|---:|---:|---:|---:|---:|---:|
| Kev-4B (ours, sample) | 38.43 | 0.314 | 0.355 | 0.457 | 0.565 | 0.166 |
| **train-jev 4B (ours, sample)** | **40.35** | 0.322 | 0.394 | 0.415 | 0.584 | 0.287 |
| JEV-27B-VL (ours, sample) | 54.11 | 0.452 | 0.534 | 0.549 | 0.739 | 0.413 |
| *Kev 4B r10 (board)* | *39.53* | 0.292 | 0.418 | 0.466 | 0.550 | 0.179 |
| *Sieve 4B (board)* | *51.45* | 0.377 | 0.539 | 0.637 | 0.667 | 0.283 |
| *ezjev 4B s2 (board)* | *52.87* | 0.344 | 0.605 | 0.621 | 0.699 | 0.312 |
| *vLLM-SR d3-mini 4B (board)* | *56.06* | 0.443 | 0.578 | 0.637 | 0.720 | 0.377 |
| *JEV-27B (board)* | *55.04* | 0.460 | 0.564 | 0.548 | 0.734 | 0.416 |
| *Kev 27B (board)* | *58.58* | 0.507 | 0.626 | 0.586 | 0.726 | 0.427 |
| *Jev 1.13 (board)* | *59.72* | 0.593 | 0.592 | 0.572 | 0.751 | 0.391 |

Sanity check of the sampled protocol: our Kev-4B is 38.4 against the board's 39.5, and our JEV-27B-VL is 54.1 against
the board's JEV-27B at 55.0. The sample is accurate to about a point at the index level; individual benchmarks
on ~150 rows each are noisy.

**Reading.** train-jev 4B and Kev-4B are in the same tier (≈ 40). The leading 4B models are 11–16 points higher, with
most of the gap in Knowledge & Reasoning and Language Understanding. This is the motivation for the TODO on full
fine-tuning with broader public gold data.

## 4. Kev transfer-v4

`evals/v4/transfer-v4` from [jaredpalmer/kev](https://github.com/jaredpalmer/kev): 656 questions from sources Kev never
trained on, run with `python -m kev.benchmark --remote <server>`.

| model | accuracy | ECE | NLL |
|---|---:|---:|---:|
| Kev-4B | **0.817** | **0.042** | **0.435** |
| train-jev 4B | 0.779 | 0.075 | 0.745 |
| JEV-27B-VL | 0.852 | 0.097 | 0.460 |
| *Jev 1.13 (reported in the Kev repo)* | *0.857* | | |

Kev's recipe (trained on gold labels close to these task types) is stronger here, especially on rule-composition
questions; train-jev's distillation recipe stays closer to Jev's distributions.

## 5. Zero-shot image decisions

Five public Decision Index vision sets, 300 items each (first 300 by `sha256(id)`), choice questions, chance-corrected
skill. The LoRA and head were trained on **text only**; for images we load the full Qwen3.5 VL model, merge the LoRA
into its language model, and leave the vision tower untouched (`trainjev/eval_vision.py`).

| model | CV-Bench | BLINK | RealWorldQA | CharXiv | InfoVQA | mean skill | mean acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B (untrained) | 0.682 | 0.454 | 0.510 | 0.667 | 0.847 | 0.632 | 0.758 |
| **train-jev 4B** | 0.732 | 0.506 | 0.608 | 0.729 | 0.913 | **0.697** | 0.802 |
| JEV-27B-VL | 0.743 | 0.634 | 0.711 | 0.849 | 0.973 | 0.782 | 0.859 |

Text-only distillation improves every vision set (+6.5 points mean skill), i.e. what the head learns about *how to
decide* transfers across modalities. Kev drops the vision tower at load time and cannot take images.

## 6. Supplementary: 13-dataset classification PK

1,300 rows (100 per dataset) from 13 public classification datasets: AG News, DAIR Emotion, BANKING77, MASSIVE (en/zh),
XNLI (en/zh), TNEWS, SST-5, Yelp stars, BoolQ, a prompt-injection set and a toxicity set. Identical prompts for every
system. This harness is from our earlier experiments and is not part of the repository; the numbers are for context.

| | Jev 1.13 (API) | JEV-27B-VL | Kev-4B | **train-jev 4B** |
|---|---:|---:|---:|---:|
| accuracy | 0.700 | 0.690 | 0.640 | **0.647** |
| 95% CI | [0.675, 0.724] | [0.664, 0.715] | [0.614, 0.666] | [0.621, 0.672] |
| ECE | 0.130 | 0.094 | 0.083 | **0.076** |
| p50 latency | 370 ms (incl. network) | 96 ms | 17 ms | 43 ms |
| choice (n=800) | 0.739 | 0.738 | 0.685 | 0.701 |
| score (n=200) | 0.695 | 0.655 | 0.610 | 0.575 |
| boolean (n=300) | 0.600 | 0.587 | 0.540 | 0.550 |

train-jev 4B is the best calibrated of the four. It is weakest on many-class intent sets (BANKING77 0.69 vs 0.81 for
Jev; MASSIVE en 0.73 vs 0.86).

## 7. Notes on the protocol

- **Batched vs single inference.** The server batches the questions of a request by length in bf16. Compared with
  one-at-a-time inference this flips 6 of 816 answers in our check (max |Δp| 0.02). All Decision Index numbers for
  train-jev 4B come from the batched server, run from scratch.
- **Latency** is server wall time on one H20 with HF transformers (no CUDA graphs, no fused kernels beyond fla), one
  request at a time. The Jev 1.13 figure is a third-party measurement of the hosted API and includes the network.
- **Reproducibility.** The code in this repository was checked against the original experiment code on the same
  inputs: identical training loss for the first 11 steps, identical evaluation logits on all 9,000 corpus rows, and
  identical server and vision outputs (|Δ| = 0).
