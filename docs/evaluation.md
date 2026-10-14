# Evaluation

Five complementary evaluations, from "did the distillation work" to "is it useful on public tasks".

| # | Evaluation | Question it answers | Tool |
|---|---|---|---|
| 1 | jev-distill `test_set_30k` / `ood` | How close is the student to Jev's distributions, in and out of distribution? | `trainjev.evaluate` |
| 2 | Decision Index 0.3 (public) | How good is it across 35+ public benchmarks, against the board? | `decision_index` + `scripts/di_*.py` |
| 3 | Kev `transfer-v4` | Accuracy and calibration on 656 questions from unseen sources | `kev.benchmark --remote` |
| 4 | Decision Index vision sets | Does text-only distillation transfer to images? | `trainjev.eval_vision` |
| 5 | 13-dataset classification PK | Accuracy / calibration on 13 public datasets (en + zh), paired against other systems | `scripts/pk_*.py` |

(2), (3) and (5) talk to a running server, so any System One compatible endpoint can be compared on the
same rows.

## 1. jev-distill-corpus-v3

```shell
# trained checkpoint: fit temperatures on `calibration`, then evaluate (writes <ckpt>/calibration.json)
python -m trainjev.evaluate --base models/Qwen3.5-4B --adapter runs/jev-4b/best --fit_calibration --out eval/jev-4b
# untrained reference
python -m trainjev.evaluate --base models/Qwen3.5-4B --base_only --out eval/base
# any /v1/systemone server (e.g. Kev) or a JEV serve_decide.py /v1/decide endpoint
python -m trainjev.evaluate --systemone_url http://127.0.0.1:8023 --workers 4 --out eval/kev-4b
python -m trainjev.evaluate --decide_url http://127.0.0.1:8011 --out eval/jev-27b-vl

python scripts/compare_corpus.py base=eval/base/summary.json jev-4b=eval/jev-4b/summary.json
```

Defaults: 6,000 rows of `test_set_30k` and 3,000 of `ood` (fixed seed 7; `--splits name:n,...`). `summary.json` has
`raw` and `calibrated` metrics, each also restricted to Jev-labelled rows (`*_jev_labelled`). Metrics:

- **KL(target ‖ model)**: the distillation objective; the headline number.
- **top-1**: argmax agreement on rows whose target has a unique maximum.
- **noul AUROC**, **score MAE** (expected level vs target expected level), **ECE** (10 bins, top-1 confidence).

## 2. Decision Index 0.3

[apolinario/decision-index](https://github.com/apolinario/decision-index) rebuilds the board's frozen public suite
from public sources and scores it with the board's own scorers.

**One-time setup** (~30 GB of downloads):

```shell
git clone https://github.com/apolinario/decision-index third_party/decision-index
cd third_party/decision-index
git apply ../../patches/decision-index-resumable-download.patch   # optional: resume interrupted downloads with curl
pip install -e .
python -m decision_index suite rebuild --work work
python -m decision_index suite import \
  --rows work/artifacts/benchmark-suite/release-v2-rebuilt/selected-rows.jsonl.gz \
  --added-rows work/artifacts/benchmark-suite/release-v2-rebuilt/added-rows.jsonl.gz \
  --gsm8k-rows work/artifacts/benchmark-suite/release-v3-rebuilt/gsm8k-rows.jsonl.gz
cd ../..
python scripts/di_sample.py --di_root third_party/decision-index --n 6000 \
  --rows_out data/di_sample6000.jsonl.gz --suite_out third_party/decision-index/suite-0.3-sample6000
```

**Run and score** a server:

```shell
python -m trainjev.serve --base models/Qwen3.5-4B --adapter runs/jev-4b/best --port 8020 &
URL=http://127.0.0.1:8020 NAME=jev-4b bash scripts/run_decision_index.sh
```

Sampling and coverage. The full public suite is ~140k requests; a stratified 6,000-row sample takes about 50 minutes
per 4B model on one GPU. Scoring against the *sample-only* suite makes the official scorer compute coverage over the
sample; scoring against the full suite would count every unsampled row as unanswered.

Missing benchmarks. If some benchmarks cannot be rebuilt (in our case HLE, gated, and HoVer, a 2 GB download), the
official index treats them as 0 for everyone. `scripts/di_index.py` removes them from their areas instead, and with
`--board di_index.json` (from the [board's Space](https://huggingface.co/spaces/multimodalart/jev-decision-index))
re-scores published models on the same panel:

```shell
python scripts/di_index.py --di_root third_party/decision-index --drop 45,61 \
  jev-4b=eval/di_jev-4b/score_s/index.json --board data/di_index.json
```

## 3. Kev transfer-v4

```shell
git clone https://github.com/jaredpalmer/kev && cd kev   # follow its install instructions
python -m kev.benchmark --remote http://127.0.0.1:8020 --suite evals/v4/transfer-v4 --out ../eval/jev-4b-transfer-v4
```

`report.json` → `clean.acc`, `clean.ece`, `clean.nll`, plus per-task breakdowns.

## 4. Zero-shot image decisions

Rebuild the vision suite once inside the Decision Index checkout:

```shell
(cd third_party/decision-index && python -m decision_index suite rebuild --vision --work work \
   --sets full_cvbench full_blink full_charxiv full_infovqa full_realworldqa)
```

Then:

```shell
python -m trainjev.eval_vision --base models/Qwen3.5-4B --adapter runs/jev-4b/best \
  --di_root third_party/decision-index --per_set 300 --out eval/vision-jev-4b
python -m trainjev.eval_vision --base models/Qwen3.5-4B --base_only \
  --di_root third_party/decision-index --per_set 300 --out eval/vision-base
```

The script loads the full Qwen3.5 VL model, merges the text-only LoRA into its language model, and reads the decision
head at the last token of the `serve_decide.py` image prompt. Output: per-set accuracy, chance and chance-corrected
skill `(acc - chance) / (1 - chance)`.

## Comparing against JEV-27B-VL

JEV-27B-VL ships its own vLLM server (`serve_decide.py`, `/v1/decide`). To put it behind the System One API used by
(2) and (3), run this repository's server in proxy mode:

```shell
python -m trainjev.serve --proxy http://127.0.0.1:8011 --port 8021 --name JEV-27B-VL
```

The proxy uses the same question mapping as the local backend, so both see identical prompts.

## 5. 13-dataset classification PK

```shell
python scripts/pk_build.py --out data/pk          # 13 scenarios x up to 600 rows, from the HF Hub (set HF_ENDPOINT for a mirror)
python scripts/pk_benchmark.py run --url http://127.0.0.1:8020 --suite data/pk --out eval/pk_jev-4b.jsonl
python scripts/pk_benchmark.py run --url http://127.0.0.1:8023 --model kev-latest --suite data/pk --out eval/pk_kev-4b.jsonl
python scripts/pk_benchmark.py report jev-4b=eval/pk_jev-4b.jsonl kev-4b=eval/pk_kev-4b.jsonl
```

`run` sends one request per row (100 rows per scenario by default, `--limit`), is resumable, and records the
prediction, its confidence, the full distribution and the latency. `report` pairs runs on their shared rows and
prints per-scenario accuracy, overall accuracy with a Wilson 95% CI, ECE, an exact McNemar test against the first run,
p50 latency, per-primitive accuracy, and accuracy split by whether the source is in Kev's training data
(`--seen`, scenarios marked `*`).

Fixed questions (no per-task prompt engineering):

| primitive | System One question |
|---|---|
| choice | `choice` over the label names (`_` → space): "Which label best describes this input?" |
| score | `choice` over the ordered level names: "Rate this input on the given scale." |
| boolean | `noul`: "Is the statement or property described true for this input?" |
