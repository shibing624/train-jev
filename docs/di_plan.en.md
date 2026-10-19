# Closing the Decision Index gap: Instant / AutoThink

[中文](di_plan.md)

This note cites published numbers only. The code in this change does **not** include a new Decision Index run, and it does not subtract StartLux's public subset from train-jev's 37-benchmark panel. The protocols differ, so a difference is not a result.

## Gap (citations, not a new measurement)

Published train-jev numbers (this repo's README / CHANGELOG). The full panel is the public 37-benchmark text set, 110,087 items. The sample is 6,000 items.

| Model | DI 0.3 sample | DI 0.3 full text panel |
|---|---:|---:|
| train-jev-4b-lora | 41.5 | 39.01 |
| train-jev-27b-lora | 56.1 | 55.58 |
| TypeSafe Jev v1.13 (official API, sample only) | 59.7 | – |

StartLux-Decision-v1.1, as reported by StartLux ([repo](https://github.com/StartLuxLabs/StartLux-Decision), [paper](https://arxiv.org/abs/2610.17309), [27B AutoThink card](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-27B-AutoThink)). They select models on the public subset. These are not an independent rerun in this repo.

| Model | DI 0.3 public subset | DecisionPivot pair accuracy (7,800 pairs, gate 0.9) |
|---|---:|---:|
| 27B AutoThink | 73.37 | 86.9 |
| 27B Instant | 72.83 | 75.4 |
| 9B AutoThink / Instant | 70.45 / 69.96 | 59.9 / 47.5 |
| 4B AutoThink / Instant | 68.21 / 67.88 | 59.9 / 41.5 |
| 2B / 0.8B Instant | 59.14 / 53.75 | – |

DI 0.3.2 leaderboard (StartLux README, 2026-10-16): 27B Instant Full 69.12 / 174 (public 72.88, private same-skill 64.82, private new-domain 61.41). The private sets are not in this repo.

In their paper, bidirectional attention versus causal is worth up to about **+2.4** DI points in a controlled comparison, with seed noise about 0.2–0.7. AutoThink over Instant is about **+0.3 to +0.5** DI points. DecisionPivot pair accuracy is where AutoThink matters (about +11.5 to +18.4 points). RL mainly shortens traces (up to about −65% length, up to about +48 points termination).

The bulk of the DI gap is data coverage and full-parameter training, not another gate.

## What each system does

| | train-jev (released) | StartLux Instant | StartLux AutoThink |
|---|---|---|---|
| Base | Qwen3.5-4B / Qwen3.8-27B | Qwen, 0.8B–27B | Qwen, 4B / 9B / 27B |
| Readout | 24-slot head, bare-v1 prompt, last token | `lm_head` letter logits, chat template | same, then a second readout after the thought |
| Attention | causal | bidirectional on full-attention layers only; linear layers stay causal | causal throughout, so it can generate |
| Training | LoRA r16 + head; KL + 0.5 RPS; 30% choice permutation | full-parameter soft CE, then adapters on Lux-Rounds, then a seed average | thought supervision, then RLCD² |
| When unsure | one forward | refuses `think` | write a thought only if the top probability is below the gate; final `(1 - mix) * p0 + mix * p1`, mix 0.75 |

This repo does **not** switch the prompt to their chat template or the readout to letter logits. Released checkpoints stay bare-v1, causal, and one-pass. With no `thought`, the prompt is byte-identical to the previous template.

Bidirectional attention applies only to Qwen full-attention layers (about every fourth layer). On an unpadded row `create_causal_mask` returns `None`, and causality lives on the module `is_causal` flag, so training must flip that flag and keep it flipped while gradient checkpointing recomputes the forward. Serving refuses to flip a causal checkpoint at inference time.

## Priority

1. **Data and full finetune (flag already exists; this PR does not retrain).** `--full_ft` uses FSDP2. StartLux Stage 1 uses on the order of 2.7M examples (0.8B–9B) or 0.8M (27B) and updates every LM parameter, including the output projection. That is the bulk of the DI gap.
2. **Instant: bidirectional training (this PR).** Their ablation ceiling is about 2.4 points, not 30. The mask has to be on during training.
3. **Seed average (this PR, LoRA + head only).** They average 3 seeds at 0.8B and 4 seeds at 2B–27B. Full-finetune safetensors are out of scope for the script.
4. **AutoThink serving, thought dropout, and a local contrast mix (this PR).** `thought_drop` defaults to 0.5 so the one-pass readout keeps training. Local `precision-mix` covers refund, deadline, count, and sum only. It is **not** DecisionPivot and not a DI score. Do not train on the dev split or tune DI on it.
5. **RLCD² objective (library only; no training loop).** Reward is the change in the decision-weighted proper score, minus `lambda` (1e-4) per token past a soft budget (paper B=3072, generation cap 4096). Traces under 8 tokens score 0. Group advantages are centered and clipped to [−5, 5], with no standard-deviation normalisation. Online GRPO needs generation, a reference model, and a GPU. None of that was run here.
6. **Next, not in this PR:** none-of-the-above, their chat template, score questions with 2–10 levels, knockout past 26 options, and the DI 0.3.2 private sets.

## How to train and evaluate

Default training stays causal. Instant and AutoThink are separate launchers.

```bash
# Local contrast pairs. data/ is gitignored; do not commit the jsonl.
python scripts/build_precision_mix.py --pairs 50 --out data/precision-mix

# Stage 2 Instant: continue a causal checkpoint with bidirectional full attention.
INIT_ADAPTER=runs/jev-4b/best MIX=data/precision-mix/train.jsonl MIX_FRACTION=0.25 \
  LR=2e-5 HEAD_LR=4e-5 STEPS=2048 MAX_STEPS=512 OUT=runs/jev-4b-instant \
  bash scripts/run_train_instant.sh

# Seed average (attention mode, verbalizer ids, and LoRA config must match)
python scripts/average_lora.py --out runs/jev-4b-instant-avg \
  runs/seed0/best runs/seed1/best runs/seed2/best runs/seed3/best

# Stage 3 AutoThink: stay causal and drop half of the supervised thoughts.
INIT_ADAPTER=runs/jev-4b/best MIX=data/precision-mix/train.jsonl MIX_FRACTION=0.25 \
  LR=2e-5 HEAD_LR=4e-5 STEPS=2048 MAX_STEPS=512 OUT=runs/jev-4b-autothink \
  bash scripts/run_train_autothink.sh
```

Full finetune adds `--full_ft` to `trainjev.train` (or keep using `trainjev.train_full`). For 27B use `run_train_27b.sh` and set `ATTENTION=bidirectional` or `THOUGHT_DROP` yourself.

Serving:

```bash
# Released models: omit VARIANT. Behaviour matches the previous server.
bash scripts/run_serve.sh

# instant only if head.pt says attention=bidirectional
VARIANT=instant CKPT=runs/jev-4b-instant/best bash scripts/run_serve.sh

# Causal checkpoint. Default gate is 0.6; their published numbers use 0.9.
VARIANT=autothink CKPT=runs/jev-4b-autothink/best bash scripts/run_serve.sh --gate 0.9
```

`POST /v1/decisions` is the same handler as `/v1/systemone`. `think_mode` is `auto`, `never`, or `always` (aliases `hybrid`, `plain`, `think`). Plain and Instant accept only `never`. Image questions and choices wider than 16 options skip the thought. If the thought prompt would exceed `--max_len`, the first readout is kept.

Evaluation stays on `evaluate.py` / `run_eval.sh` and reads `attention` from the checkpoint. Do not fit temperatures on `precision-mix/dev.jsonl` or on the final DI items. Do not publish a new index until a frozen suite is rerun. Report `trainjev.precision.pair_accuracy` only for the local contrast pairs.

## Risks

- Flipping a causal checkpoint to bidirectional at serve time is refused. The reverse, scoring a bidirectional checkpoint as causal, is not meaningful either.
- If the LoRA never saw thought-conditioned prompts, sampled thoughts can be useless. Do not treat the second readout as a gain until that training exists. `thought_drop` keeps thought-free rows.
- Sampled thoughts are not deterministic. `return_thought` is for inspection, not for the score.
- Vision requests and wide choices skip thinking.
- The seed average does not cover full-finetune weights.
- The RLCD² library has unit tests. There is no online run and no new DI or DecisionPivot number.
