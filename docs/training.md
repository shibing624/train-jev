# Training

This page explains the JEV System 1 recipe as implemented in `trainjev/train.py` and why each piece is there.

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

`target` is a full soft distribution from the teacher, not a hard label. Any jsonl file in this schema can be added to
training with `--mix path.jsonl:N` (repeated N times); `weight` scales its loss.

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

30.5M trainable LoRA parameters on the 4B backbone, plus 61k for the head. `lm_head` is not used on the decision path,
so the base model's generation ability is untouched (the adapter is only merged into the decision path at serving time).

**Right padding, no mask.** Batches are right-padded and the hidden state is read at each row's last real token. In a
causal model padding after the last real token cannot affect it, and this also holds for the recurrent Gated DeltaNet
layers, whose state at position t depends only on tokens ≤ t. Left padding would feed pad tokens into the
recurrent state.

## 4. Objective

```
loss = Σ_rows weight · [ KL(target ‖ softmax(z)) + 0.5 · RPS(target, softmax(z)) · 1{kind = score} ] / batch
```

- **KL to the full teacher distribution** transfers the teacher's uncertainty as well as its argmax. This is why the
  fitted temperatures end up at ≈ 1.0: calibration comes from the objective.
- **RPS** (ranked probability score, squared distance between the cumulative distributions) on `score` rows penalises
  mass that is far from the target level more than mass on a neighbouring level, which KL alone ignores.

## 5. Data handling

- **Placeholder rows.** The `yuri_v1` stream of jev-distill-corpus-v3 (148k memory-relevance rows) carries uniform
  `[0.5, 0.5]` targets. They are kept at loss weight ×0.05 (`--placeholder_w`), following the JEV card, and excluded
  from temperature fitting. `--placeholder_w 0` drops them.
- **Option permutation.** 30% of `choice` rows get their options shuffled, with the target permuted the same way
  (`--perm`). Without this the model learns position priors (the untrained backbone flips its answer under reordering
  41% of the time on JEV-27B's measurement).
- **Kind-stratified batches.** Each 128-row step draws ≥ 1/6 of the batch from each kind, the rest uniformly. The
  sampling RNG is seeded by the step, so every rank sees the same global batch and takes its `rank::world` slice.
- **> 16 options** are dropped from training; at serving time they go through the tournament (`trainjev/serve.py`).

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

> If you train on data without placeholder rows (`--placeholder_w 0`), the validation subset still contains them and
> their KL rises as the model stops predicting 0.5. Select checkpoints on the non-placeholder part of validation, or
> simply use the last step.

## 8. Memory and speed

Reference run: 3× H20 (96 GB), `--no_gc --micro_tokens 4096`: 1.94 s/step, ~8.2k tokens/s overall, 48.8 GB peak per
GPU, 2.77 h for 5,123 steps. On smaller GPUs keep gradient checkpointing on (the default) and/or reduce
`--micro_tokens`; the global batch stays 128 rows either way.

Install `flash-linear-attention==0.5.2` on CUDA. Without it, transformers runs the Gated DeltaNet layers in a slow
reference implementation. `causal-conv1d` is optional.

## 9. Adapting the recipe

| goal | change |
|---|---|
| different backbone size | `--base Qwen/Qwen3.5-9B` (same code; the verbalizer ids are checked at start) |
| add your own labelled data | convert to the row schema; `--mix mydata.jsonl:2` |
| more capacity | `--lora_r 64` (α is set to 2r) |
| hard labels only | one-hot `target`; expect higher confidence, so re-fit temperatures and watch ECE |
