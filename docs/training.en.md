# Training

[中文](training.md)

This page explains the JEV System 1 recipe as implemented in `trainjev/train.py` and why each piece is there.

## 0. Quick start

```bash
bash scripts/prepare_data.sh                                    # data/jev-distill-corpus-v3 + models/Qwen3.5-4B
BASE_REPO=Qwen/Qwen3.8-27B BASE_DIR=models/Qwen3.8-27B bash scripts/prepare_data.sh   # the 27B backbone

GPUS=0,1,2 bash scripts/run_train.sh                            # 4B, 3x H20, about 2.8 h
GPUS=0,1,2,3 bash scripts/run_train_27b.sh                      # 27B, 4x H20, about 33 h
GPUS=0 MAX_STEPS=3 OUT=runs/smoke bash scripts/run_train.sh     # smoke test (the LR schedule stays 5,123 steps)
GPUS=0,1,2,3 RESUME=1 OUT=runs/jev-27b bash scripts/run_train_27b.sh   # continue an interrupted run

CKPT=runs/jev-4b/best bash scripts/run_eval.sh                  # fit temperatures, then evaluate
BASE=models/Qwen3.8-27B CKPT=runs/jev-27b/best OUT=eval/jev-27b bash scripts/run_eval.sh
```

The launchers read `GPUS`, `BASE`, `DATA`, `OUT`, `SEED`, `STEPS`, `MAX_STEPS`, `MICRO_TOKENS`, `EVAL_EVERY`,
`SAVE_STATE_EVERY` (default 100) and `RESUME`. `prepare_data.sh` downloads the corpus and backbone at the revisions
the published models were trained on; set `HF_ENDPOINT` for a mirror.

| measured run | peak memory per GPU | wall time (incl. periodic validation) |
|---|---:|---:|
| 4B, 3x H20, gradient checkpointing off, `--micro_tokens 4096` | 48.8 GB | 2.77 h |
| 27B, 4x H20, gradient checkpointing on, `--micro_tokens 16384` | 74.9 GB | 33.07 h (23.2 s/step) |

## 1. Task format

Every training row is one typed question about a state:

```json
{"kind": "choice", "state": "...", "question": "Which team should handle this?",
 "options": ["billing", "infra: timeouts and capacity", "product"], "target": [0.05, 0.90, 0.05],
 "source": "yuri_v3", "weight": 1.0}
```

- `noul`: options `["false", "true"]`, target `[P(false), P(true)]`
- `score`: options `["0", ..., "5"]`, target a distribution over the 6 ordered levels
- `choice`: 2-16 options, target aligned with `options`

`target` is the supplied full probability vector; the corpus includes teacher soft distributions and programmatic
targets. It is not uniformly one-hot or uniformly one API's output. Other jsonl files can use the same schema via
`--mix path.jsonl:N` (repeated N times); `weight` scales loss. Audit new targets and provenance before training.

## 2. Prompt (`bare-v1`)

```
[kind] choice
[state] Ticket #4821 ...
[question] Which team should handle this?
[options]
A) billing
B) infra: timeouts and capacity
C) product
[decision]:
```

`noul` and `score` list the raw option strings (`false`, `true`; `0` .. `5`). The template is byte-for-byte the one in
JEV-27B-VL's `serve_decide.py`, so checkpoints trained here can be compared with JEV on identical inputs. No chat template
and no special tokens are added. An over-long state is cut in the middle (head and tail kept, ` … ` in between); the
question and options are never truncated.

## 3. Model

```
Qwen3.5 text model (frozen bf16) + LoRA  ──► hidden state at the last prompt token ──► 24-slot fp32 head
```

**Decision head.** 24 rows, one per verbalizer token: `false, true | 0..5 | A..P`. A question of kind `k` with `n`
options uses the `n` slots starting at `RANGES[k]`. The head is initialised from the backbone's own `lm_head` rows for
those tokens, so at step 0:

```
softmax(head(h)[slots]) == softmax(lm_head(h)[verbalizer_ids])
```

i.e. the untrained model is exactly the base model's zero-shot next-token readout restricted to the valid answers.
`scripts/check_init.py` verifies this (max |Δp| 7.7e-6 on Qwen3.5-4B, fp32). Training therefore starts from what
pretraining already knows (noul AUROC 0.82, choice top-1 0.51 on `test_set_30k`) instead of from a random classifier.

**LoRA.** r=16, α=32, dropout 0.05 on every projection of both layer types of Qwen3.5's hybrid stack:

| layer type | modules |
|---|---|
| Gated DeltaNet (linear attention) | `in_proj_qkv`, `in_proj_z`, `out_proj` |
| full attention | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| MLP | `gate_proj`, `up_proj`, `down_proj` |

The 4B has 30.5M trainable LoRA parameters plus 61,464 head parameters; the 27B has 108.8M plus 122,904.
Training uses the small head rather than the full vocabulary projection. Original base files are unchanged, but
generation with an enabled/merged adapter can change because the encoder is shared; that behavior was not evaluated.

**Right padding, no mask.** Batches are right-padded and the hidden state is read at each row's last real token. In a
causal model padding after the last real token cannot affect it, and this also holds for the recurrent Gated DeltaNet
layers, whose state at position t depends only on tokens ≤ t. Left padding would feed pad tokens into the
recurrent state.

## 4. Objective

```
loss = Σ_rows weight · [ KL(target ‖ softmax(z)) + 0.5 · RPS(target, softmax(z)) · 1{kind = score} ] / batch
```

- **KL to the supplied distribution** encourages agreement with uncertainty as well as argmax. Temperatures near
  1.0 were observed here, not guaranteed by the loss or guaranteed on every new domain.
- **RPS** (ranked probability score, squared distance between the cumulative distributions) on `score` rows penalises
  mass that is far from the target level more than mass on a neighbouring level, which KL alone ignores.

For K=6, RPS is `sum((cumsum(p)-cumsum(t))[:K-1]**2)/(K-1)`; the final cumulative bin is omitted.
The denominator of the training loss is the global row batch, not the sum of row weights. Targets are normalised
before vectorised computation. The head operates on fp32 hidden/readout values; backbone compute is bf16.

## 5. Data handling

- **Placeholder rows.** The training yuri_v1 source has 137,203 rows, mostly uniform binary targets. The existing
  heuristic also flags any uniform two-way target (138,889 flagged train rows total); it does not estimate a label
  error rate. They are kept at loss weight ×0.05 (`--placeholder_w`) and excluded
  from temperature fitting. `--placeholder_w 0` drops them.
- **Option permutation.** 30% of `choice` rows get their options shuffled, with the target permuted the same way
  (`--perm`). Without this the model learns position priors (the untrained backbone flips its answer under reordering
  41% of the time on JEV-27B's measurement).
- **Kind-stratified batches.** Each 128-row step draws floor(128/6)=21 rows per kind and fills the remaining 65 from
  the global pool. Each sample operation is without replacement within that operation, but the forced and global
  draws can overlap, and steps resample independently. This is not an epoch traversal. The
  sampling RNG is seeded by the step, so every rank sees the same global batch and takes its `rank::world` slice.
- **> 16 options** are dropped from training. At serving time `trainjev/serve.py --strategy` picks how they are read
  (the three strategies of JEV-27B-VL's `serve_decide.py`): `single` (default; one pass, labels A–P from the trained
  head, then Q–Z, AA, AB, … read from the backbone's own `lm_head` rows with zero bias), `tournament` (groups of ≤ 16
  plus a final, only ever reading the trained A–P) or `permute` (`single` over 4 option orders, averaged). See
  [results.en.md §8](results.en.md#8-wide-choice-questions-17256-options).

### Data audit before a new recipe

Use `scripts/audit_training_data.py` before adding a candidate gold pool:

```bash
python scripts/audit_training_data.py \
  --train data/jev-distill-corpus-v3/train.jsonl --gold data/candidate-gold.jsonl \
  --heldout data/jev-distill-corpus-v3/validation.jsonl data/jev-distill-corpus-v3/ood.jsonl \
  --pk-suite data/pk --di-rows data/di-requests.jsonl.gz --out outputs/data-audit
```

Supply every final evaluation input and optionally `--source-jsonl` raw records with upstream provenance. The tool
checks targets/slots, normalized duplicates and target disagreements, state overlap and complete-prompt overlap,
and saves file SHA256 values. It parses top-level `state`, PK `text`, and `payload.body.state` / `body.state` /
`request.state`; missing inputs are reported as incomplete evidence. Prediction-only `rows.json` files are not
substitutes for benchmark request files. Empty contexts are valid but cannot establish state-level independence.

The optional candidate-index output groups by normalized state, not by individual question. It always marks rows
as unapproved: this does not verify label truth, upstream split/licence, source/template independence or isolation
of counterfactual twins. Casefold/whitespace matches need review rather than automatic deletion.

The initial local audit finds extensive shared states between the original corpus train and in-distribution splits,
while OOD states do not overlap. Shared state is not necessarily repeated questions or label leakage: multiple
questions can legitimately use one context. Report both levels and use fresh state/group-disjoint development
data for new-recipe selection. The current reproduction run retains its frozen splits for comparability.

Full-file audit snapshot (2026-10-15; these are not the smaller evaluation samples):

| split | rows | rows whose normalized state is in train | complete prompts also in train |
|---|---:|---:|---:|
| validation | 14,111 | 12,225 | 0 |
| calibration | 13,766 | 11,873 | 0 |
| test | 14,261 | 12,373 | 0 |
| test_set_30k | 29,955 | 28,065 | 0 |
| ood | 13,058 | 0 | 0 |

The training pool has 655,806 rows, one normalized repeated prompt and no observed target disagreement on that
repeat. About 21.2% match the existing placeholder heuristic; this is not an estimated semantic label error rate.
The first candidate gold pool has 39,973 structurally valid rows, with no normalized state or complete-prompt
overlap with the distillation training pool or the supplied held-out inputs. That does not cover the pending full
DI suite or missing transfer request inputs, and does not establish source/template independence. Candidate
indices contain 36,218 train and 3,755 dev rows grouped by state; all remain unapproved for training. The utility
has 20 offline tests, including actual input-format parsing, target disagreements and state-vs-prompt overlap.

## 6. Optimisation

| item | value |
|---|---|
| optimiser | AdamW (fused), β = (0.9, 0.98), weight decay 0 |
| learning rates | LoRA 1e-4, head 2e-4 |
| schedule | linear warm-up over 3% of steps, then one cosine to 0.02× |
| clipping | global grad norm 1.0 |
| steps | one epoch: 655,806 / 128 = 5,123 |
| micro-batching | length-sorted, `rows × max_len ≤ --micro_tokens` padded tokens per GPU |
| DDP | gradients flattened and all-reduced manually after all micro-batches, so ranks may run different numbers of micro-batches |

The loss of each micro-batch is divided by `batch / world` before `backward()`, so after the all-reduce mean the
gradient equals the mean over the global 128-row batch regardless of how rows were split.

## 7. Checkpoints and calibration

Every `--eval_every` steps the model is evaluated on a fixed `--val_n` subset of `validation.jsonl` and saved to
`stepNNNNN/`; the lowest validation KL is also saved to `best/`. Each checkpoint contains:

```
adapter_config.json  adapter_model.safetensors   # PEFT LoRA for the text model
head.pt                                          # {"weight": [24, d], "bias": [24], "verbalizer_ids", "base", "args"}
calibration.json                                 # written by `trainjev.evaluate --fit_calibration`
```

`trainjev.evaluate --fit_calibration` fits one temperature per kind on the `calibration` split (placeholders excluded)
by golden-section search on log T, minimising cross-entropy to the soft targets. The server divides logits by these
temperatures.

### Resuming long runs

`best/` and `stepNNNNN/` are inference checkpoints, **not** complete training states. For long 27B runs, enable
`--save_state_every 100` from the start. All ranks collectively save trainable weights, AdamW state, Python/Torch/CUDA
RNG, the completed step, best-validation record and history into `<out>/state/` using
`torch.distributed.checkpoint`. Full-parameter model and optimiser states remain sharded.

Restart with the same command, output directory and GPU count, adding `--resume`:

```bash
torchrun --nproc_per_node 4 -m trainjev.train \
  --base Qwen/Qwen3.8-27B --data data/jev-distill-corpus-v3 --out runs/jev-27b \
  --micro_tokens 16384 --save_state_every 100 --resume
```

Keep the model, dataset, seed, global batch, learning rates, augmentation and total `--steps` schedule unchanged.
`--max_steps` is only a stopping limit and does not shorten the learning-rate schedule. The checkpoint loader rejects
a different world size because rank-local sampling slices and RNG would change. A state is written to `state.tmp/`
and replaces the previous state only after all ranks finish; `state.old/` is a recovery fallback if the swap was
interrupted. Do not delete these directories to restart a run.

State saving happens on multiples of `--save_state_every`; after a 5,123-step run with interval 100, the last resumable
state is normally step 5,100, while `step05123/` contains the final inference weights. Validate successful training
exit and the final checkpoint separately; do not mistake a three-step smoke checkpoint for a completed run.

### GPU optimisation checks (2026-10-14)

The implementation keeps the handwritten PyTorch/DDP/FSDP2 trainers. It adds rank-0 checkpoint loading after meta
initialisation/sharding for full fine-tuning, batched 24-slot KL/RPS computation, next-step tokenisation prefetch and
complete DCP training states. Tokenisation is prefetched, **not** cached by concatenating independently tokenised
prefixes; prompt token boundaries, sampling and option permutation remain unchanged. Optional shared-prefix and
DDP communication-overlap changes are not enabled.

GPU checks use 4×H20, Qwen3.5-4B, seed 0, global batch 24, micro-token budget 4,096, gradient checkpointing disabled,
a 5,123-step LR schedule and a three-step stopping limit.

| Check | Before | After | Interpretation |
|---|---:|---:|---|
| LoRA step-3 loss | 0.4320 | 0.4333 | Not bit-identical |
| LoRA step-3 validation KL (100 rows) | 0.450388 | 0.450648 | Small, nonzero difference |
| LoRA step-3 s/step | 1.99 | 2.04 | No demonstrated speed gain in this smoke test |
| Full-parameter step-3 loss | 0.4264 | 0.4321 | Not bit-identical |
| Full-parameter step-3 validation KL (100 rows) | 0.449687 | 0.450884 | Not a strict numerical-equivalence pass |
| Full-parameter step-3 s/step | 2.35 | 1.65 | Smoke-test timing only |
| Full-parameter loading peak, summed process RSS | 96.17 GiB | 31.01 GiB | Lower host-memory use |

The LoRA adapter/head maximum parameter differences after three steps are 7.335e-6 / 3.943e-6. The full-parameter
bf16 checkpoint maximum difference is 0.00390625, exceeding the planned 1e-3 threshold; that strict check remains
unresolved and must not be advertised as passing. That maximum occurs at one scalar each in
`layers.{0,21,25,26}.linear_attn.norm.weight`; overall model RMS difference is 1.205e-7, relative L2 difference is
1.021e-5, and 415,674 of 4,205,751,296 elements differ. This is not evidence of widespread large parameter errors.
The RSS figures sum the sampled process tree, include imports/data, and were sampled every second; logged
load-plus-shard time is 14 s / 16 s before/after, so lower host-memory use is not a demonstrated loading speedup.
A separate synthetic head probe finds that TF32 increases the
batch-vs-row maximum logit difference from 7.15e-7 to 6.85e-4, but this does not establish the cause of all training
differences. Three steps do not establish steady-state throughput or convergence equivalence.

**LoRA resume check passed exactly:** three steps, DCP save, fresh-process load and three more steps produce the
same adapter and head as an uninterrupted six-step run (maximum difference 0 for both; identical step-6 validation
KL 0.416699). The 4B resumable state is about 370 MB. This is a resume check, not a completed training run or an
assessment of model quality. The current 27B experiment uses the verified LoRA path and preserves its original
TF32 setting across the saved-state boundary; it does not use the unresolved full-parameter path.

> Dropping placeholder rows changes the training pool and can raise KL on uniform-target validation rows. Define an
> independent development/selection criterion before running that ablation; do not inspect test results and then
> change checkpoint selection. Keep explicit total steps fixed so a smaller pool does not silently shorten training.

## 8. Memory and speed

Reference run: 3× H20 (96 GB), `--no_gc --micro_tokens 4096`: 1.94 s/step, ~8.2k tokens/s overall, 48.8 GB peak per
GPU, 2.77 h for 5,123 steps. On smaller GPUs keep gradient checkpointing on (the default) and/or reduce
`--micro_tokens`; the global batch stays 128 rows either way.

Install `flash-linear-attention==0.5.2` on CUDA. Without it, transformers runs the Gated DeltaNet layers in a slow
reference implementation. `causal-conv1d` is optional.

## 9. Backbones

The tested 4B/27B bases share the dense `qwen3_5` architecture and the same decision verbalizer IDs.
`check_verbalizers` refuses an incompatible tokenizer. Other sizes/architectures require initialization, LoRA target
and inference checks; compatible token IDs alone do not establish trained quality or universal plug-in support.

| backbone | layers (full attention) | hidden | tied embeddings | LoRA params | status |
|---|---:|---:|:---:|---:|---|
| Qwen3.5-4B | 32 (8) | 2560 | yes | 30.5M | trained and evaluated (this release) |
| Qwen3.5-9B | 32 (8) | 4096 | no | | planned |
| **Qwen3.8-27B** | 64 (16) | 5120 | no | 108.8M | verified end to end: load, init gate, training, corpus and image evaluation |

Qwen3.8-27B notes:

- Its architecture dimensions and trainable counts match the JEV-27B recipe. This is not proof of identical
  optimizer history, training data choices or deployment arithmetic; compare under a stated protocol.
- The tokenizer differs from Qwen3.5 only by 7 added audio/TTS special tokens (ids 248070-248076); vocab, merges and
  pre-tokenizer are identical, and 2,000 training prompts tokenise to the same ids.
- `AutoModelForCausalLM` loads only the text model (vision tower and MTP head are skipped); the init gate gives
  |Δp| = 0. `trainjev.eval_vision` loads the full VL model for images.
- Requires `transformers>=5.17` and `peft>=0.21`; transformers 4.x does not know the `qwen3_5` model type.
- Completed four-H20 run: gradient checkpointing,`micro_tokens16384`,74.9 GB/rank, 23.24 s/step,
  33.07 h including periodic validation for 5,123 steps. The launcher now defaults to these measured four GPUs.
  Do not substitute the earlier unmeasured eight-GPU throughput estimate for a measured result.

## 10. Adapting the recipe

| goal | change |
|---|---|
| different backbone size | `--base Qwen/Qwen3.5-9B` or `--base Qwen/Qwen3.8-27B` (same code) |
| add your own labelled data | convert to the row schema; `--mix mydata.jsonl:2` |
| more capacity | `--lora_r 64` (α is set to 2r) |
| hard labels only | one-hot `target`; expect higher confidence, so re-fit temperatures and watch ECE |
