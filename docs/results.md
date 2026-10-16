# Results

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

**Reading.** train-jev-4b-lora and Kev-4B are in the same tier (≈ 40). The leading 4B models are 11–16 points higher, with
most of the gap in Knowledge & Reasoning and Language Understanding. This is the motivation for the TODO on full
fine-tuning with broader public gold data.

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

## 7. Notes on the protocol

- **Batched vs single inference.** The server batches the questions of a request by length in bf16. Compared with
  one-at-a-time inference this flips 6 of 816 answers in our check (max |Δp| 0.02). All Decision Index numbers for
  train-jev-4b-lora come from the batched server, run from scratch.
- **Latency** is server wall time on one H20 with HF transformers (no CUDA graphs, no fused kernels beyond fla), one
  request at a time. The Jev 1.13 figure is a third-party measurement of the hosted API and includes the network.
- **Reproducibility.** The code in this repository was checked against the original experiment code on the same
  inputs: identical training loss for the first 11 steps, identical evaluation logits on all 9,000 corpus rows, and
  identical server and vision outputs (|Δ| = 0).

## 7. Wide-choice questions (17–256 options)

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

Qwen3.5-4B LoRA A, the same checkpoint/calibration, one H20, serial requests, fixed seed-0 test samples:
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

## 8. 4B 全参训练与 LoRA 对照（2026-10-14）

底座均为 Qwen3.5-4B，使用同一份 jev-distill-corpus-v3、全局 batch 128、相同损失、采样和选项置换，
训练 5,123 步；按固定 4,000 行验证集的最低 KL 选择 checkpoint，两者均选中第 5,123 步。
LoRA 使用 r16、学习率 1e-4，3×H20；全参更新文本底座全部参数，使用学习率 2e-6，4×H20、FSDP2。
两者决策头学习率均为 2e-4，micro-token budget 均为 4,096，均关闭 gradient checkpointing。
视觉塔未参与训练。以下是实际产物的评测结果，不是学习率探针的结果。

| 指标 | 4B LoRA | 4B 全参 | 方向 |
|---|---:|---:|:---:|
| 最佳 validation KL（4,000 行） | 0.0160 | 0.0162 | ↓ |
| test_set_30k KL（6,000 行，校准后） | 0.02239 | **0.02174** | ↓ |
| OOD KL（3,000 行，校准后） | **0.31073** | 0.46604 | ↓ |
| Decision Index 0.3，35 基准重算（6,000 行） | **40.35** | 37.52 | ↑ |
| transfer-v4 accuracy（656 行） | **0.77896** | 0.76220 | ↑ |
| transfer-v4 ECE | 0.07468 | **0.06520** | ↓ |
| transfer-v4 NLL | 0.74455 | **0.66690** | ↓ |
| 13-dataset PK accuracy（1,300 行） | **0.64692** | 0.61077 | ↑ |
| PK：Kev 训过的 5 个来源（500 行） | **0.718** | 0.672 | ↑ |
| PK：Kev 未训过的 8 个来源（800 行） | **0.6025** | 0.5725 | ↑ |
| 视觉决策 mean skill（5 个集合等权平均） | **0.69729** | 0.68757 | ↑ |
| 视觉决策 mean accuracy（5 个集合等权平均） | **0.80155** | 0.79494 | ↑ |
| 训练耗时（含周期验证，不含加载/最终评测） | 2.77 h（3 卡） | 3.15 h（4 卡） | ↓ |

Decision Index 的五个 area skill：

| 方法 | Knowledge | Language | Retrieval | Tools | Arts |
|---|---:|---:|---:|---:|---:|
| LoRA | 0.322 | 0.394 | 0.415 | 0.584 | 0.287 |
| 全参 | 0.263 | 0.341 | 0.436 | 0.579 | 0.260 |

PK 的配对比较中，全参独有正确 42 行、LoRA 独有正确 89 行，双侧精确 McNemar p≈4.9e-5。
该检验只针对这 1,300 行，并不消除训练种子和其他评测集的变异。
历史全参服务的启动命令未显式记录宽选项策略，不能仅凭当前默认值确认其历史读法。
排除 BANKING77 / MASSIVE en / MASSIVE zh 三个宽选项场景后，剩余同题 1,000 行仍为
LoRA 633 行正确、全参 600 行正确；LoRA/全参独有正确 67/34 行，精确 McNemar p=0.00133。
因此 PK 差距不能全部归因于宽选项策略，但完整宽题消融仍待 GPU 空闲后完成。

transfer-v4 的同题准确率差为 511/656 对 500/656，LoRA/全参独有正确 33/22 行，
精确 McNemar p=0.177；这批样本不足以证明其准确率差显著。全参在该集上更好的 NLL/ECE 应同时保留，
不能把结果概括为所有指标都更差。

OOD choice 的 top-1 约为 88.84% / 88.69%，而 KL 为 0.31905 / 0.54443：
top-1 接近并不意味着完整概率分布接近。这里 top-1 只统计 teacher 有唯一最大值的题，
KL 则统计全部 738 行（见 `trainjev/metrics.py`），两者分母口径不同。
这支持“该 OOD 集上的分布匹配退化”，不构成已证明具体原因或所有域外任务准确率下降。
两组验证 KL 仍持续下降，不能仅凭 OOD 差距断言已观察到后期过拟合。

视觉实际覆盖 CV-Bench / BLINK / RealWorldQA / CharXiv 各 300 行、InfoVQA 244 行，共 1,444 行；
两模型的行集合相同。前文“300 items each”是抽样上限，不代表每个集合都实际有 300 行。

**本轮结论：全参在分布内 KL、transfer-v4 的 ECE/NLL 上略好，但 OOD、Decision Index、PK 和视觉平均 skill
没有改善，因此不满足“4B 全参效果更好”的升级条件。按预先约定，Qwen3.8-27B 采用 LoRA，而不是全参。**
这是当前纯蒸馏数据、当前学习率与单个 seed 下的结论，不意味着全参加更广公开金标数据也一定不如 LoRA。
全参学习率 2e-6 来自独立的 400 步 cosine schedule、2,000 行 validation 的短 LR sweep，
不是 5,123 步 schedule 的截断测试，也不足以证明完整 epoch 的超参已充分调优。

### Qwen3.8-27B 后续实验

27B LoRA 的完整训练与评测已完成，见第 10 节。
本轮配置：4×H20、r16、LoRA/head lr 1e-4/2e-4、global batch 128、max length 2,048、
micro-token budget 16,384、开启 gradient checkpointing；实跑 5,123 步。
训练保留可恢复的权重/优化器/RNG 状态，并按相同 corpus、Decision Index、transfer-v4、PK 和视觉行集合评测。

## 9. 完整 Decision Index 0.3 面板准备（2026-10-15）

这部分是数据构建/完整性核验，不是模型评测分数。canonical Decision Index revision 为 `f8552a13`。
当前隔离构建已准备官方文本 public panel 的 **36/37 个基准、106,087 个可评分请求**；
HoVer 的 4,000 请求仍等待 Wikipedia DB 下载完成及官方 SHA256/SQLite 校验。
完整 37 面板的预期请求数为 **110,087**，尚未完成严格 suite import/verify，不报告其完整指数。
旧 suite、6,000 请求采样及历史模型结果均未覆盖。

| 核验项 | 结果 |
|---|---|
| HLE 原始 parquet | 2,500 行，274,276,147 bytes，SHA256 与锁定 revision `5a81a4c7` 的 etag 匹配 |
| HLE 官方文本多选归一化 | 513 行；应用官方排除规则后 501 个可评分请求 |
| release-v1 / release-v2 基础 / GSM8K | 未压缩 SHA256 分别与官方 pin 逐字节匹配 |
| 其余 6 个 added 基准 | 26,419 行，与旧部分文件逐字节一致 |
| added 全量文件 | 缺 HoVer，完整官方 SHA256 尚不匹配，不视为通过 |
| 独立 DI CPU 检查 | 89 项通过，不包含尚未完成的完整面板导入/模型推理验收 |

**范围必须区分：完整 DI 0.3 文本面板不等于完整 HLE 多模态评测。**
HLE 原始数据中有图像的题共 342 道（78 道多选、264 道 exactMatch）；官方文本适配器明确排除这些题
及开放答案题，因此 501 道可评分 HLE 请求均无图像。当前准备的 36 项请求也没有原生图像字段；
少数冻结源文本含 Markdown/HTML 图像引用或字面量 data URI，这些仍按文本处理，不删除来追求覆盖率。
独立视觉决策仍按前述 5 集合协议评测；若扩展 HLE 图像多选或开放答案，必须另立协议、单独报告。

完整面板验证后，当前 27B 与原 JEV-27B-VL 将使用相同冻结请求文件评测。
新 37 面板分数不与旧 35 基准采样分数直接比较；数据 pin 匹配也不等于所有模型请求成功，
还需分别验证实际作答覆盖、错误请求及 scorer 配置。

## 10. Qwen3.8-27B LoRA 训练与评测（2026-10-16）

按第 8 节的结论，27B 采用 LoRA 而非全参。底座 Qwen3.8-27B，数据仍为 jev-distill-corpus-v3，
全局 batch 128、max length 2,048、micro-token budget 16,384、关闭 `--no_gc`（即开启 gradient
checkpointing）、r16、LoRA/head lr 1e-4/2e-4，4×H20，seed 0，实跑 5,123 步。
完整训练退出码 0；按固定 4,000 行验证集最低 KL 选中第 5,123 步，`best/` 与 `step05123/` 的
adapter/head SHA256 一致（`88816e6e…` / `cfb6b9bd…`），可复现该 checkpoint。
以下为实际产物的同题评测；本轮 train-jev-27b-lora 的 PK、transfer-v4、Decision Index 显式使用 `tournament`。
原 JEV-27B-VL 的历史宽题结果使用 `single`，且后端不同；不能把这张跨模型表当作读法消融。

本轮主链路按历史协议对齐，**不等待 HLE/HoVer**：Decision Index 使用原 6,000 请求、35 基准重算面板。
本表 train-jev-27b-lora 结果来自已完成的 HF 合并 bf16 评测，不采用仍未通过数值替换门槛的 vLLM 实验结果。
完整 37 面板、部署后端及下一版数据优化是另行跟进的分支，不是这轮训练/五套旧协议评测的完成前提。

| 指标 | train-jev-4b-lora | JEV-27B-VL | **train-jev-27b-lora** | 方向 |
|---|---:|---:|---:|:---:|
| 最佳 validation KL（4,000 行） | 0.0160 | – | **0.01423** | ↓ |
| test_set_30k KL（6,000 行，校准后） | 0.02239 | 0.02075 | **0.01853** | ↓ |
| test_set_30k KL（Jev-labelled，校准后） | 0.01795 | 0.017 | **0.01561** | ↓ |
| OOD KL（3,000 行，校准后） | 0.31073 | **0.10842** | 0.11006 | ↓ |
| Decision Index 0.3，35 基准重算（6,000 行） | 40.35 | 54.11 | **54.57** | ↑ |
| transfer-v4 accuracy（656 行） | 0.77896 | **0.85213** | 0.85061 | ↑ |
| transfer-v4 ECE | 0.07468 | 0.09731 | **0.05308** | ↓ |
| transfer-v4 NLL | 0.74455 | 0.46022 | **0.41130** | ↓ |
| 13-dataset PK accuracy（1,300 行） | 0.64692 | **0.690** | 0.67769 | ↑ |
| PK ECE | **0.07569** | 0.094 | 0.10294 | ↓ |
| PK：Kev 训过的 5 个来源（500 行） | 0.718 | **0.762** | 0.754 | ↑ |
| PK：Kev 未训过的 8 个来源（800 行） | 0.6025 | **0.645** | 0.630 | ↑ |
| PK p50 延迟（单张 H20） | **43.5 ms** | 96 ms | 81.9 ms | ↓ |
| 视觉决策 mean skill（5 个集合等权，1,444 行） | 0.69729 | **0.78195** | 0.75999 | ↑ |
| 视觉决策 mean accuracy（5 个集合等权） | 0.80155 | **0.85923** | 0.84442 | ↑ |
| 训练耗时（含周期验证，不含加载/最终评测） | 2.77 h（3 卡） | – | 33.07 h（4 卡） | ↓ |

验证轨迹（`--val_n 4000`，含周期验证的累计时间）：

| step | 500 | 1000 | 1500 | 2000 | 2500 | 3000 | 3500 | 4000 | 4500 | 5000 | **5123** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val KL | 0.0451 | 0.0420 | 0.0295 | 0.0243 | 0.0211 | 0.0193 | 0.0175 | 0.0160 | 0.0146 | 0.0143 | **0.0142** |
| elapsed h | 3.23 | 6.48 | 9.71 | 12.96 | 16.17 | 19.41 | 22.62 | 25.82 | 29.06 | 32.27 | 33.07 |

Decision Index 0.3 的五个 area skill（同一 35 基准面板，HLE/HoVer 已剔除并重新归一化）：

| 模型 | index (35) | Knowledge | Language | Retrieval | Tools | Arts |
|---|---:|---:|---:|---:|---:|---:|
| train-jev-4b-lora | 40.35 | 0.322 | 0.394 | 0.415 | 0.584 | 0.287 |
| JEV-27B-VL | 54.11 | 0.452 | 0.534 | 0.549 | 0.739 | 0.413 |
| **train-jev-27b-lora** | **54.57** | 0.438 | **0.552** | **0.567** | 0.737 | **0.415** |

视觉决策五个集合（300 行抽样上限；InfoVQA 实际 244 行，共 1,444 行，三模型行集合一致）：

| 模型 | CV-Bench | BLINK | RealWorldQA | CharXiv | InfoVQA | mean skill |
|---|---:|---:|---:|---:|---:|---:|
| train-jev-4b-lora | 0.732 | 0.506 | 0.608 | 0.729 | 0.913 | 0.697 |
| JEV-27B-VL | 0.743 | **0.634** | **0.711** | **0.849** | **0.973** | **0.782** |
| **train-jev-27b-lora** | **0.770** | 0.568 | 0.673 | 0.822 | 0.967 | 0.760 |

**读法（只依据实测值，不把单项优势等同于整体 SOTA）：**

- 相对 4B LoRA，train-jev-27b-lora 的多数准确率/分布指标更好：OOD KL −0.2007、Decision Index +14.22、
  transfer-v4 准确率 +0.0717 且 NLL −0.3333、PK 准确率 +0.0308、视觉 mean skill +0.0627；
  但 PK ECE 0.10294 劣于 0.07569，PK p50 81.9 ms 慢于 43.5 ms，不能说全部指标更好。
- 相对原 JEV-27B-VL，train-jev-27b-lora 的 Decision Index 54.57 对 54.11（+0.46），
  transfer-v4 NLL 0.41130 对 0.46022、ECE 0.05308 对 0.09731 更好；OOD KL 0.11006 略高于 0.10842。
- train-jev-27b-lora 的弱项是视觉：mean skill 0.760 低于原版 27B 的 0.782（差 2.2 点），
  BLINK / RealWorldQA / CharXiv / InfoVQA 均略低，仅 CV-Bench 更高（0.770 对 0.743）。
  本轮视觉塔未参与训练，这与“文本蒸馏不必然改善视觉”一致。
- PK 上 train-jev-27b-lora 对 4B LoRA 的独有正确为 104/64，精确 McNemar p=0.00252；
  train-jev-27b-lora 对原版为 27/43，p=0.072。后者未达到 0.05 阈值，不代表已证明两者等价。
- transfer-v4 上 train-jev-27b-lora 对 4B LoRA 的独有正确为 71/24，p=1.46e-6；对原版为 16/17，p=1。
  本题集未证实 train-jev-27b-lora 的准确率超过原版，不能从“都比 4B 好”直接推出规模是已验证的唯一原因。
- 这些配对检验未校正多个比较，且不替代跨 seed 或跨任务泛化验证。
- Decision Index 的 +0.46 尚未做配对置信区间/显著性检验，不能断言它在采样噪声内，
  也不足以证明全面超过原版。完整面板、同读法/后端及重复训练仍待完成，不声称 SOTA。

**边界与限制：**

- Decision Index 剔除 HLE / HoVer 后重算为 **35** 基准，仅用于相同样本面板内的比较，**不是官方总分**。
  采样为 6,000 行分层抽样，单基准约 150 行；未估计这些个体差异的置信区间。
- 只跑了一个训练 seed，未测跨种子方差；27B 为全 rank-16 LoRA，未与全参对照。
  训练耗时 33.07 h（4 卡）显著高于 4B 的 2.77 h（3 卡），本表不构成等算力消融。
- 27B corpus 推理 token budget 为 8,192，而 4B 历史结果为 24,576，BF16 批大小不同可能带来小幅数值差异。
- 4B 全参历史服务的宽题读法未在启动日志中记录，不能把三模型宽题策略严格一致视为已证实。
- corpus 的分布内切分与训练共享状态较多，其分数不是全新状态上的泛化证据；OOD 状态未重合。
- 视觉集合实际覆盖 1,444 行（InfoVQA 244 行），前文“300 items each”是抽样上限。
- 结论限于当前纯蒸馏数据、当前学习率与单个 seed，不外推到加入更广公开金标数据的情形。
