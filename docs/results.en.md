# Results

[中文](results.md)

Results for the two released models, **train-jev-4b-lora** and **train-jev-27b-lora** (5,123 steps each, temperatures
fitted on the `calibration` split). Every comparison in a table uses the same rows for every model. Reference models
were run by us on the same hardware (one NVIDIA H20 each) unless marked *board* or *reported*.

| Model | What it is |
|---|---|
| Qwen3.5-4B (untrained) | base model + head initialised from `lm_head` (the step-0 model; a zero-shot restricted readout) |
| Kev-4B | [jaredpalmer/kev](https://github.com/jaredpalmer/kev) 4B release, served with `kev.serve` |
| **train-jev-4b-lora** | [shibing624/train-jev-4b-lora](https://huggingface.co/shibing624/train-jev-4b-lora), Qwen3.5-4B + LoRA (this repository) |
| **train-jev-27b-lora** | [shibing624/train-jev-27b-lora](https://huggingface.co/shibing624/train-jev-27b-lora), Qwen3.8-27B + LoRA (this repository) |
| JEV-27B-VL | [autotrust/JEV-27B-VL](https://huggingface.co/autotrust/JEV-27B-VL), served with its own `serve_decide.py` (System 1, thinking off) |
| Jev 1.13 | TypeSafe's hosted API |

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
| | **train-jev-4b-lora** | **0.022** | **0.018** | **0.962** | **0.995** | **0.908** | **0.897** | **0.104** |
| | JEV-27B-VL | 0.021 | 0.017 | 0.969 | 0.996 | 0.908 | 0.894 | 0.098 |
| | train-jev-27b-lora | 0.019 | 0.016 | 0.968 | 0.996 | 0.916 | 0.905 | 0.096 |
| ood | Qwen3.5-4B (untrained) | 0.884 | – | 0.542 | 0.558 | 0.413 | 0.000 | 1.227 |
| | Kev-4B | 0.624 | – | 0.768 | 0.703 | 0.642 | 0.471 | 0.389 |
| | **train-jev-4b-lora** | **0.311** | – | **0.932** | **0.982** | **0.888** | 0.588 | 0.390 |
| | JEV-27B-VL | 0.108 | – | 0.970 | 0.995 | 0.913 | 0.588 | 0.432 |
| | train-jev-27b-lora | 0.110 | – | 0.974 | 0.996 | 0.935 | 0.647 | 0.367 |

*Reported* by the JEV-27B card on the full splits: KL 0.019 (all), ≈ 0.017 (Jev-labelled), OOD KL 0.104.

In distribution the 4B model is essentially indistinguishable from the 27B one; the gap opens on OOD task families
(0.311 vs 0.108).

## 3. Decision Index 0.3 (historical 35-benchmark sample)

[apolinario/decision-index](https://github.com/apolinario/decision-index), edition 0.3. We score a 6,000-row stratified
sample (round-robin over benchmarks, whole groups, the kit's default seed) and build a sample-only suite so the
official scorer computes coverage over the sample (`scripts/di_sample.py`). Every model answers the same 6,000 rows
through the same `/v1/systemone` HTTP engine.

For this historical sample, two of the 37 public benchmarks could not be rebuilt at the time: **HLE** (gated on the HF Hub) and **HoVer**
(a 2 GB Wikipedia download). They are removed from their areas and the remaining benchmarks renormalised
(`scripts/di_index.py`); the board's published per-benchmark skills are re-scored on the same 35-benchmark panel, so
every row below is comparable, but **none of them equals the official Decision Index total**.

| model | index (35) | Knowledge & Reasoning | Language | Retrieval & Classification | Tools & Automation | Arts & Taste |
|---|---:|---:|---:|---:|---:|---:|
| Kev-4B (ours, sample) | 38.43 | 0.314 | 0.355 | 0.457 | 0.565 | 0.166 |
| **train-jev-4b-lora (ours, sample)** | **40.35** | 0.322 | 0.394 | 0.415 | 0.584 | 0.287 |
| JEV-27B-VL (ours, sample) | 54.11 | 0.452 | 0.534 | 0.549 | 0.739 | 0.413 |
| **train-jev-27b-lora (ours, sample)** | **54.57** | 0.438 | 0.552 | 0.567 | 0.737 | 0.415 |
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

The public 37-benchmark text panel including HLE/HoVer was completed on 2026-10-17; see §11. This historical sample and board rescores remain separate.

**Reading.** train-jev-4b-lora and Kev-4B are in the same tier (≈ 40). The leading 4B models are 11–16 points higher, with
most of the gap in Knowledge & Reasoning and Language Understanding. Closing that gap needs broader gold-labelled data (see the roadmap).

## 4. Kev transfer-v4

`evals/v4/transfer-v4` from [jaredpalmer/kev](https://github.com/jaredpalmer/kev): 656 questions from sources Kev never
trained on, run with `python -m kev.benchmark --remote <server>`.

| model | accuracy | ECE | NLL |
|---|---:|---:|---:|
| Kev-4B | 0.817 | **0.042** | 0.435 |
| train-jev-4b-lora | 0.779 | 0.075 | 0.745 |
| JEV-27B-VL | **0.852** | 0.097 | 0.460 |
| train-jev-27b-lora | 0.851 | 0.053 | **0.411** |
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
| **train-jev-4b-lora** | 0.732 | 0.506 | 0.608 | 0.729 | 0.913 | **0.697** | 0.802 |
| JEV-27B-VL | 0.743 | 0.634 | 0.711 | 0.849 | 0.973 | 0.782 | 0.859 |
| **train-jev-27b-lora** | **0.770** | 0.568 | 0.673 | 0.822 | 0.967 | 0.760 | 0.844 |

Text-only distillation improves every vision set (+6.5 points mean skill), i.e. what the head learns about *how to
decide* transfers across modalities. Kev drops the vision tower at load time and cannot take images.

## 6. 13-dataset classification PK

1,300 rows (the first 100 of each scenario after a seed-0 shuffle) from 13 public datasets. Every system gets the same
text (first 4,000 characters), the dataset's label names and one fixed question per primitive; there is no per-task
prompt engineering. `scripts/pk_build.py` rebuilds the suite byte-for-byte from the Hugging Face Hub, and
`scripts/pk_benchmark.py` runs any `/v1/systemone` endpoint and prints the paired report. Re-running train-jev-4b-lora
through the released script reproduces the stored probabilities (max |Δp| 6e-8).

| scenario | primitive | lang | labels | source | in Kev's training sources |
|---|---|---|---:|---|:---:|
| ag_news | choice | en | 4 | `fancyzhx/ag_news` test | ✓ |
| dair_emotion | choice | en | 6 | `dair-ai/emotion` test | |
| banking77 | choice | en | 77 | `legacy-datasets/banking77` test | ✓ |
| massive_en / massive_zh | choice | en / zh | 59 | `mteb/amazon_massive_intent` test | |
| xnli_en / xnli_zh | choice | en / zh | 3 | `facebook/xnli` validation | |
| tnews_zh | choice | zh | 15 | `clue/clue` tnews validation | |
| sst5 | score | en | 5 | `SetFit/sst5` test | ✓ |
| yelp_stars | score | en | 5 | `Yelp/yelp_review_full` test | ✓ |
| boolq | boolean | en | 2 | `google/boolq` validation | ✓ |
| prompt_injection | boolean | en | 2 | `deepset/prompt-injections` test | |
| toxicity | boolean | en | 2 | `google/civil_comments` test (balanced 50/50) | |

| | Jev 1.13 (API) | JEV-27B-VL | Kev-4B | **train-jev-4b-lora** | train-jev-27b-lora |
|---|---:|---:|---:|---:|---:|
| accuracy | 0.700 | 0.690 | 0.640 | 0.647 | **0.678** |
| 95% CI | [0.675, 0.724] | [0.664, 0.715] | [0.614, 0.666] | [0.621, 0.672] | [0.652, 0.703] |
| ECE | 0.130 | 0.094 | 0.083 | **0.076** | 0.103 |
| p50 latency | 370 ms (incl. network) | 96 ms | 17 ms | 43 ms | 82 ms |
| choice (n=800) | 0.739 | 0.738 | 0.685 | 0.701 | **0.725** |
| score (n=200) | 0.695 | 0.655 | 0.610 | 0.575 | 0.630 |
| boolean (n=300) | 0.600 | 0.587 | 0.540 | 0.550 | **0.583** |
| Kev-seen sources (5 scenarios, n=500) | 0.788 | 0.762 | 0.752 | 0.718 | **0.754** |
| **Kev-unseen sources (8 scenarios, n=800)** | 0.645 | 0.645 | 0.570 | 0.603 | **0.630** |

Per scenario:

| scenario | Jev 1.13 | JEV-27B-VL | Kev-4B | train-jev-4b-lora | train-jev-27b-lora |
|---|---:|---:|---:|---:|---:|
| ag_news ✓ | 0.86 | 0.85 | 0.87 | 0.89 | 0.87 |
| banking77 ✓ | 0.81 | 0.75 | 0.81 | 0.69 | 0.73 |
| boolq ✓ | 0.88 | 0.90 | 0.86 | 0.86 | **0.91** |
| sst5 ✓ | 0.70 | 0.62 | 0.49 | 0.51 | 0.58 |
| yelp_stars ✓ | 0.69 | 0.69 | 0.73 | 0.64 | 0.68 |
| dair_emotion | 0.50 | 0.51 | 0.52 | 0.53 | 0.51 |
| massive_en | 0.86 | 0.83 | 0.60 | 0.73 | 0.77 |
| massive_zh | 0.77 | 0.77 | 0.59 | 0.66 | 0.72 |
| prompt_injection | 0.46 | 0.44 | 0.32 | 0.34 | 0.42 |
| tnews_zh | 0.48 | 0.49 | 0.50 | 0.49 | 0.50 |
| toxicity | 0.46 | 0.42 | 0.44 | 0.45 | 0.42 |
| xnli_en | 0.89 | 0.91 | 0.86 | 0.88 | **0.91** |
| xnli_zh | 0.74 | 0.79 | 0.73 | 0.74 | 0.79 |

On the sources Kev was not trained on, train-jev-4b-lora leads Kev-4B by 3.3 points, and it is the best calibrated of the
four overall. Its weak spot is many-class intent sets (BANKING77 0.69 vs 0.81 for Jev; MASSIVE en 0.73 vs 0.86). Each
cell is 100 rows (±~9 points at 95%), so read single scenarios with care.

## 7. Protocol notes

- **Batched vs single inference.** The server batches the questions of a request by length in bf16. Compared with
  one-at-a-time inference this flips 6 of 816 answers in our check (max |Δp| 0.02). All Decision Index numbers for
  train-jev-4b-lora come from the batched server, run from scratch.
- **Latency** is server wall time on one H20 with HF transformers (no CUDA graphs, no fused kernels beyond fla), one
  request at a time. The Jev 1.13 figure is a third-party measurement of the hosted API and includes the network.
- **Reproducibility.** The code in this repository was checked against the original experiment code on the same
  inputs: identical training loss for the first 11 steps, identical evaluation logits on all 9,000 corpus rows, and
  identical server and vision outputs (|Δ| = 0).

## 8. Wide-choice questions (17–256 options)

How the many-class PK scenarios were read, and what is known so far (100 rows each, GPU, from the PK runs above):

| scenario | options | JEV-27B-VL (`single`) | ECE | train-jev-4b-lora (`tournament`) | ECE |
|---|---:|---:|---:|---:|---:|
| banking77 | 77 | 0.75 | 0.101 | 0.69 | 0.153 |
| massive_en | 59 | 0.83 | 0.051 | 0.73 | 0.107 |
| massive_zh | 59 | 0.77 | 0.064 | 0.66 | 0.063 |

The two columns differ in both model size and read-out strategy, so this table does not isolate the strategy. JEV's
model card reports `single` on JEV-27B-VL down to 150 options (CLINC150, all intents: 89.5% names only, 93.8% with a
one-line description; ECE 0.079 / 0.034).

Checks done for train-jev-4b-lora:

- The `tournament` path of the new server reproduces the stored GPU PK run on CPU fp32 (8/8 BANKING77 predictions,
  max |Δp| 0.0027).
- `single` and `tournament` disagree on 2 of the same 8 BANKING77 rows (both 7/8 correct), so the choice matters.
- `single` takes one pass instead of ⌈n/16⌉ + 1 (6 for BANKING77, 11 for CLINC150).

### Same-checkpoint GPU comparison (2026-10-16)

train-jev-4b-lora, the same checkpoint and calibration, one H20, serial requests, fixed seed-0 test samples:
600 each from CLINC150 / BANKING77 / MASSIVE en (1,800 total per strategy). Inputs differ only in strategy;
request traces verify actual option orders/token counts, and source/checkpoint/input SHA256 values did not change.
Nine warm-up requests are excluded. BANKING77/MASSIVE input files match the existing PK 600-row pools.

| strategy | CLINC150 acc | BANKING77 acc | MASSIVE en acc | overall acc | ECE | p50 ms |
|---|---:|---:|---:|---:|---:|---:|
| single | 0.7850 | 0.6300 | 0.7650 | 0.7267 | 0.1622 | 52.3 |
| tournament | 0.8383 | 0.6533 | 0.7050 | 0.7322 | **0.0499** | 108.9 |
| permute | **0.8633** | **0.6750** | **0.7750** | **0.7711** | 0.2180 | 171.2 |

Permute vs single has 130/50 uniquely correct rows (net +80/1,800, +4.44 pp; exact McNemar p=2.13e-9).
Tournament vs single has 131/121 (net +10, p=0.571): this sample does not establish an overall accuracy advantage.
On MASSIVE en, tournament is 6 pp below single (p=4.37e-5), so it is not uniformly better.
Permute's mean confidence is 0.553 against accuracy 0.771: its worse ECE is primarily underconfidence,
not evidence of overconfidence. No temperatures were fitted on these test labels.

These are exploratory same-checkpoint results, not training gains. Keep the default unchanged until independent
development-set strategy/calibration selection and a separate final check; post-average temperature scaling can
be tested on dev without changing argmax, rather than tuning against these 600-row test samples. Latency includes
HTTP/observer overhead with the existing GPU occupier left running, not isolated-hardware throughput.


## 9. 4B full-parameter training vs LoRA

Both runs use Qwen3.5-4B, the same jev-distill-corpus-v3, global batch 128, the same loss, sampling and option
permutation, and 5,123 steps; the checkpoint with the lowest KL on the fixed 4,000-row validation subset is kept
(step 5,123 for both). LoRA: r16, lr 1e-4, 3x H20. Full parameter: every text-backbone weight, lr 2e-6, 4x H20, FSDP2.
Head lr 2e-4, micro-token budget 4,096 and no gradient checkpointing for both; the vision tower is not trained.

| metric | 4B LoRA | 4B full parameter | |
|---|---:|---:|:---:|
| best validation KL (4,000 rows) | 0.0160 | 0.0162 | ↓ |
| test_set_30k KL (6,000 rows, calibrated) | 0.02239 | **0.02174** | ↓ |
| OOD KL (3,000 rows, calibrated) | **0.31073** | 0.46604 | ↓ |
| Decision Index 0.3, 35-benchmark sample (6,000 rows) | **40.35** | 37.52 | ↑ |
| transfer-v4 accuracy (656 rows) | **0.77896** | 0.76220 | ↑ |
| transfer-v4 ECE | 0.07468 | **0.06520** | ↓ |
| transfer-v4 NLL | 0.74455 | **0.66690** | ↓ |
| 13-dataset PK accuracy (1,300 rows) | **0.64692** | 0.61077 | ↑ |
| PK: 5 Kev-seen sources (500 rows) | **0.718** | 0.672 | ↑ |
| PK: 8 Kev-unseen sources (800 rows) | **0.6025** | 0.5725 | ↑ |
| vision mean skill (5 sets, equal weight) | **0.69729** | 0.68757 | ↑ |
| vision mean accuracy (5 sets, equal weight) | **0.80155** | 0.79494 | ↑ |
| training time (incl. periodic validation) | 2.77 h (3 GPUs) | 3.15 h (4 GPUs) | ↓ |

Decision Index area skills:

| method | Knowledge | Language | Retrieval | Tools | Arts |
|---|---:|---:|---:|---:|---:|
| LoRA | 0.322 | 0.394 | 0.415 | 0.584 | 0.287 |
| full parameter | 0.263 | 0.341 | 0.436 | 0.579 | 0.260 |

On PK, full parameter is uniquely right on 42 rows and LoRA on 89 (exact two-sided McNemar p ≈ 4.9e-5). Excluding the
three wide-choice scenarios (BANKING77, MASSIVE en, MASSIVE zh), LoRA still gets 633 of the remaining 1,000 rows right
against 600 (67/34 unique, p = 0.00133), so the gap is not only the wide-choice readout. On transfer-v4 the accuracy
difference (511 vs 500 of 656, 33/22 unique, p = 0.177) is not significant, and full parameter has the better NLL and
ECE there. On OOD choice questions top-1 is close (88.84% vs 88.69%) but KL is 0.31905 vs 0.54443: matching the argmax
does not mean matching the distribution.

**Conclusion: full-parameter training is slightly better on in-distribution KL and on transfer-v4 calibration, but not
on OOD, Decision Index, PK or vision, so the 27B model is trained with LoRA.** This holds for the current
distillation-only data, learning rate and single seed; it does not show that full-parameter training with broader
gold data would also lose. The 2e-6 learning rate came from a short 400-step sweep, not a tuned full-epoch schedule.

## 10. Qwen3.8-27B LoRA (train-jev-27b-lora)

Qwen3.8-27B, jev-distill-corpus-v3, global batch 128, max length 2,048, micro-token budget 16,384, gradient
checkpointing on, r16, LoRA / head lr 1e-4 / 2e-4, 4x H20, seed 0, 5,123 steps. The lowest validation KL is at step
5,123. PK, transfer-v4 and Decision Index use the `tournament` readout for > 16 options; JEV-27B-VL's numbers use
`single` on a different backend, so this table is not a readout ablation.

| metric | train-jev-4b-lora | JEV-27B-VL | **train-jev-27b-lora** | |
|---|---:|---:|---:|:---:|
| best validation KL (4,000 rows) | 0.0160 | – | **0.01423** | ↓ |
| test_set_30k KL (6,000 rows, calibrated) | 0.02239 | 0.02075 | **0.01853** | ↓ |
| test_set_30k KL (Jev-labelled, calibrated) | 0.01795 | 0.017 | **0.01561** | ↓ |
| OOD KL (3,000 rows, calibrated) | 0.31073 | **0.10842** | 0.11006 | ↓ |
| Decision Index 0.3, 35-benchmark sample (6,000 rows) | 40.35 | 54.11 | **54.57** | ↑ |
| transfer-v4 accuracy (656 rows) | 0.77896 | **0.85213** | 0.85061 | ↑ |
| transfer-v4 ECE | 0.07468 | 0.09731 | **0.05308** | ↓ |
| transfer-v4 NLL | 0.74455 | 0.46022 | **0.41130** | ↓ |
| 13-dataset PK accuracy (1,300 rows) | 0.64692 | **0.690** | 0.67769 | ↑ |
| PK ECE | **0.07569** | 0.094 | 0.10294 | ↓ |
| PK: 5 Kev-seen sources (500 rows) | 0.718 | **0.762** | 0.754 | ↑ |
| PK: 8 Kev-unseen sources (800 rows) | 0.6025 | **0.645** | 0.630 | ↑ |
| PK p50 latency (one H20) | **43.5 ms** | 96 ms | 81.9 ms | ↓ |
| vision mean skill (5 sets, 1,444 rows) | 0.69729 | **0.78195** | 0.75999 | ↑ |
| vision mean accuracy (5 sets) | 0.80155 | **0.85923** | 0.84442 | ↑ |
| training time (incl. periodic validation) | 2.77 h (3 GPUs) | – | 33.07 h (4 GPUs) | ↓ |

Validation trajectory (`--val_n 4000`, cumulative wall time):

| step | 500 | 1000 | 1500 | 2000 | 2500 | 3000 | 3500 | 4000 | 4500 | 5000 | **5123** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val KL | 0.0451 | 0.0420 | 0.0295 | 0.0243 | 0.0211 | 0.0193 | 0.0175 | 0.0160 | 0.0146 | 0.0143 | **0.0142** |
| elapsed h | 3.23 | 6.48 | 9.71 | 12.96 | 16.17 | 19.41 | 22.62 | 25.82 | 29.06 | 32.27 | 33.07 |

- **Against train-jev-4b-lora**, the 27B model is better on almost everything: OOD KL −0.20, Decision Index +14.2,
  transfer-v4 accuracy +7.2 points (71/24 unique rows, p = 1.5e-6), PK accuracy +3.1 points (104/64, p = 0.0025),
  vision skill +6.3 points. It is worse on PK ECE (0.103 vs 0.076) and slower (82 vs 43 ms).
- **Against JEV-27B-VL**, accuracy is level: transfer-v4 16/17 unique rows (p = 1), PK 27/43 (p = 0.072), Decision Index
  54.57 vs 54.11 (no confidence interval). Calibration on transfer-v4 is clearly better (ECE 0.053 vs 0.097, NLL 0.411
  vs 0.460); OOD KL is about the same (0.110 vs 0.108).
- **The weak spot is vision**: mean skill 0.760 vs 0.782, lower on BLINK, RealWorldQA, CharXiv and InfoVQA, higher
  only on CV-Bench (0.770 vs 0.743). The vision tower was not trained.

## 11. Decision Index 0.3: full public 37-benchmark text panel (2026-10-17)

Both **v0.2.0 LoRA release checkpoints** answered the same ordered **110,087 requests / 272,229 subquestions**.
This covers every scoreable text request in the exported public 37-benchmark panel, including **501 HLE text requests and 4,000 HoVer requests**.
It is neither full multimodal HLE nor the maintainers' Full leaderboard with private tests. Kev, JEV-27B-VL and the TypeSafe API were not rerun on this input.

Both used HF transformers with a merged bf16 text backbone, fp32 decision head, explicit `single`, and 32,768 context/batch-token budgets.
Original per-kind calibration temperatures were retained: no new training, refitting, truncation or benchmark-dependent strategy.
Frozen source/checkpoint hashes remained unchanged. Independent checks verified every actual payload, every answer key and all 37 catalog counts;
CPU rescoring reproduced the scores.

| Scope | train-jev-4b-lora | train-jev-27b-lora | Coverage / readout |
|---|---:|---:|---|
| Historical 35, 6,000-row sample (§3) | 40.35 | 54.57 | HLE/HoVer removed; wide-choice `tournament` |
| New 37, quick 6,000 requests | 38.14 | 53.29 | 132 HLE / 132 HoVer; `single` |
| **New 37, full 110,087 requests** | **37.40** | **53.88** | All requests successful; `single` |

Full `balanced_raw`: 52.98 / 65.08 (not overall accuracy); `breadth_skill`: 36.35 / 52.94.
Each model has **110,087 unique/latest ok, zero request errors, all 272,229 subquestions answered, coverage=1.0**.
The canonical `complete=false` flag is preserved because its suite check also covers display benchmarks outside this indexed text panel;
it does not mean missing requests in the specified 37-panel export, nor permit claiming the private Full leaderboard.

| Full area skill | train-jev-4b-lora | train-jev-27b-lora |
|---|---:|---:|
| Knowledge & Reasoning | 0.2798 | 0.4127 |
| Language | 0.3881 | 0.5830 |
| Retrieval & Classification | 0.3444 | 0.5395 |
| Tools & Automation | 0.5786 | 0.7230 |
| Arts & Taste | 0.2662 | 0.4122 |

| Full benchmark | Requests | train-jev-4b-lora accuracy | train-jev-27b-lora accuracy |
|---|---:|---:|---:|
| HLE (text subset) | 501 | 9.98% | 7.39% |
| HoVer | 4,000 | 63.25% | 75.08% |

The official HLE chance baseline is 0.1641, so both HLE chance-corrected skills clip to zero; 27B does not win every benchmark.

Timing on one H20 per model, excluding loading/scoring and the pause: 4B's first 2,882 requests span 1,821.5 seconds,
then its resumed 107,205 requests take 10,586.3 seconds (about **3.45 active hours** combined); 27B takes **56,528.0 seconds / 15.70 hours**.
Full request p50 / p95: **45.9 / 459.1 ms** for 4B, **154.6 / 2,470.7 ms** for 27B.
Requests may contain multiple questions and the hardware was shared with occupier/other jobs; these are not exclusive-GPU per-question latency guarantees.

Uncompressed input SHA256: `1a2875db806c4d7d8f5fe0f2ddaa01c7d324a304a10cb15db16d57d5a915136b`.
Machine-readable scores, 37 catalog counts, protocol and fingerprints: [di37_results.json](../reports/v0.2.0/di37_results.json).
27B leads 4B by 16.48 index points on this full panel, a comparison of different backbones/sizes—not a v0.3 training-data gain.
Do not interpret differences across historical 35, new 37 quick and new 37 full as causal training/readout improvements.

## 12. Limitations

- §3 is the historical 35-benchmark sample; §11 is the full public 37-benchmark scoreable text panel. They are not mixed and neither is the
  private Full leaderboard. New HLE coverage is only 501 text requests, not full multimodal HLE. Quick per-benchmark scores remain noisy.
- One training seed per model; seed variance was not measured. The McNemar tests above are not corrected for multiple
  comparisons.
- The in-distribution splits of jev-distill share states (but no complete prompts) with the training split, so
  in-distribution KL is not evidence of generalisation to new states; OOD states do not overlap.
- Vision covers 1,444 rows (CV-Bench, BLINK, RealWorldQA and CharXiv 300 each, InfoVQA 244).
- Conclusions hold for the current distillation-only data; they are not extrapolated to recipes with broader gold data.

## 13. v0.3.0: continued training with gold and rule data

4B and 27B use the same recipe, continuing from their own v0.2.0 LoRA + head with a fresh optimizer: 87.5% jev-distill
replay and 12.5% new supervision per step (two GSM8K distractor views, PAWS, AQuA-RAT, QASC, HelpSteer2, CC0 exact-rule
examples), LoRA / head learning rates 2e-5 / 4e-5, seed 17, a 2,048-step schedule. 4B ran all 2,048 steps (3×H20, 1.3 h);
27B ran 1,024 steps (3×H20, gradient checkpointing, 7.9 h). Checkpoints were chosen on a separate 3,550-question development
set only (highest macro gain with no >16-option regression); both sizes picked step 512.

### 13.1 Development selection (27B, vs 27B v0.2.0)

| step | macro Δpp [95%] | public gold | legacy | wide |
|---:|---:|---:|---:|---:|
| 256 | +3.76 [2.69, 4.77] | +6.59 | +0.69 | +1.67 |
| **512** | **+4.80 [3.74, 5.89]** | +9.35 | +1.53 | +1.67 |
| 768 | +4.70 [3.51, 5.87] | +9.72 | +0.97 | 0.00 |
| 1,024 | +4.21 [3.02, 5.35] | +9.60 | +0.42 | −1.11 |

### 13.2 Sealed same-question PK (2,494 questions, 22 tasks)

Six models, identical inputs, each with its native stack and shipped temperatures. Per-task accuracy and all paired
comparisons: [reports/v0.3.0/external_final2.json](../reports/v0.3.0/external_final2.json); summary table in the
[README](../README_EN.md#v030-continued-training-with-gold-and-rule-data).
27B v0.3.0 vs JEV-27B-VL: paired +5.25pp [4.01, 6.50]; vs 27B v0.2.0 +4.73pp [3.71, 5.80]; vs 4B v0.3.0 +4.54pp [3.15, 5.91].

### 13.3 Historical suites (same protocol as §2–§6)

PK / transfer-v4 / the Decision Index sample go through `/v1/systemone` with `tournament` for wide choices. Distillation KL
uses raw logits: the v0.3.0 temperatures are fitted on a gold calibration set and sharpen score questions against the
jev-distill teacher, so they are not used for KL.

| Metric | 4B v0.2.0 | 4B v0.3.0 | 27B v0.2.0 | 27B v0.3.0 |
|---|---:|---:|---:|---:|
| test_set_30k KL (raw) ↓ | 0.0224 | 0.0230 | 0.0185 | 0.0185 |
| OOD KL (raw) ↓ | 0.312 | 0.483 | 0.110 | 0.123 |
| OOD choice top-1 ↑ | 0.888 | 0.900 | 0.935 | 0.940 |
| 13-dataset PK accuracy ↑ | 0.647 | 0.642 | 0.678 | 0.687 |
| PK ECE ↓ | 0.076 | 0.050 | 0.103 | 0.048 |
| PK: 8 sources Kev never saw ↑ | 0.603 | 0.600 | 0.630 | 0.640 |
| transfer-v4 accuracy ↑ | 0.779 | 0.805 | 0.851 | 0.858 |
| transfer-v4 ECE ↓ | 0.075 | 0.054 | 0.053 | 0.032 |
| transfer-v4 NLL ↓ | 0.745 | 0.596 | 0.411 | 0.377 |
| Decision Index 35-benchmark sample ↑ | 40.35 | 41.49 | 54.57 | ⏳ |
| Decision Index 37-benchmark full ↑ | 37.40 | 39.01 | 53.88 | ⏳ |
| Vision mean skill ↑ | 0.697 | 0.693 | 0.760 | 0.765 |
| PK p50 latency | 43.5 ms | 44.9 ms | 81.9 ms | 84.5 ms |

- PK pairing: 4B v0.3.0 vs v0.2.0 discordant 24 / 30 (p=0.50); 27B 25 / 13 (p=0.073).
- 4B Decision Index 37 full: all 110,087 requests ok; paired question-level macro +0.97pp [0.78, 1.17]; all five areas up
  (tools +3.0, retrieval +2.1, language +1.5, arts +0.9, knowledge +0.7 skill points); 25 benchmarks up / 10 down / 2 equal,
  McNemar-significant 17 up / 3 down; #38 skill 8.0 → 5.8, #44 −2.3 and #37 −1.5 significantly down. The panel only has
  choice / noul questions; re-tempering to the v0.2.0 temperatures changes 0 answers.
- The OOD KL increase is the expected cost of gold supervision moving away from the Jev teacher: accuracy does not drop, and
  refitting temperatures on the jev-distill calibration split gives 0.472 / 0.121 (4B / 27B), close to the raw values.
- Conclusion: the v0.3.0 recipe gives small, consistent out-of-distribution gains (transfer-v4, Decision Index, calibration),
  much smaller than on the sealed PK, which contains in-family rule items and public-gold sources seen in training.
