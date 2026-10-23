# Evaluation and published-input reproduction

[中文](https://github.com/shibing624/train-jev/docs/evaluation.md)

The completed 4B/27B main result uses five complementary suites. It is not enough to compare training accuracy,
and it is not appropriate to mix a new full panel with an old sample or to change readout/backend silently.

| evaluation | purpose | main size |
|---|---|---:|
| jev-distill test_set_30k / OOD | distribution agreement with supplied targets | 6,000 / 3,000 |
| [Decision Index 0.3 legacy sample](https://github.com/apolinario/decision-index) | multi-area decision skill | 6,000 requests;35 indexed benchmarks |
| [Kev transfer-v4](https://github.com/jaredpalmer/kev) | accuracy and calibration on sources unseen in Kev training | 656 clean decisions |
| [vision](https://github.com/apolinario/decision-index) | retained zero-shot image decisions | 1,444 actual rows over 5 datasets |
| [train-jev-classification-pk](https://github.com/shibing624/train-jev/tree/main/data/train-jev-classification-pk) | paired multilingual accuracy/calibration | 1,300 rows, 13×100 |

Wide-choice diagnostics are a separate sixth comparison: same 4B checkpoint, 600 each CLINC150/BANKING77/MASSIVE en.
All numbers and limitations are in [results.en.md](https://github.com/shibing624/train-jev/docs/results.en.md). No test labels are used to fit temperatures or select a
prompt.

v0.3.0 also uses [train-jev-pk](https://github.com/shibing624/train-jev/tree/main/data/train-jev-pk): 2,494 rows over 22 tasks, historically called the “sealed same-question PK”.
It includes public sources seen in training and in-family rule items, so it is not wholly source-independent generalisation.
These fixed evaluation rows must not be reused for training, checkpoint selection or temperature fitting.

## 1. Corpus and calibration

Download the pinned corpus/base with `scripts/prepare_data.sh`. The evaluation samples are selected by Python
`random.Random(7)`; training validation selection uses seed 1, not the evaluation seed. Targets are supplied vectors;
KL is not human answer accuracy. Top-1/ECE exclude target ties, whereas KL includes all rows, so denominators differ.

```bash
# Training checkpoint: fit on calibration, not test, then save temperatures beside the adapter.
CKPT=runs/jev-4b/best FIT_CALIBRATION=1 OUT=eval/corpus-4b bash scripts/run_eval.sh
BASE=Qwen/Qwen3.8-27B CKPT=runs/jev-27b/best FIT_CALIBRATION=1 OUT=eval/corpus-27b bash scripts/run_eval.sh

# Already-calibrated release: apply its temperatures,do not refit against test.
bash scripts/run_eval.sh
python -m trainjev.evaluate --base Qwen/Qwen3.5-4B --adapter shibing624/train-jev-4b-lora --adapter_revision v0.2.0 \
  --data data/jev-distill-corpus-v3 --out eval/release-4b
python scripts/compare_corpus.py release=eval/release-4b/summary.json
```

`run_eval.sh` defaults to `Qwen/Qwen3.5-4B` and `shibing624/train-jev-4b-lora`, applying the release's saved temperatures without fitting.
Only explicit `FIT_CALIBRATION=1` adds `--fit_calibration`, which excludes placeholder rows and writes a local checkpoint's calibration file.
Remote HF cache refitting is rejected; download to a writable directory first if needed. `ADAPTER_REVISION` or `--adapter_revision` pins an HF release.
Historical 4B corpus batching was 24,576 tokens;27B used 8,192. Match `--batch_tokens` to reproduce the deployment
implementation as well as weights. BF16 merging/batching may cause small numerical changes.

In-distribution source splits share states with training, but the audit found no complete-prompt duplication.
They do not establish new-state generalisation. OOD states were disjoint. New recipes need separate state/group
and preferably source-disjoint development data; do not tune against the final OOD/test score.

## 2. Serving protocol

```bash
BASE=Qwen/Qwen3.5-4B CKPT=shibing624/train-jev-4b-lora PORT=8020 STRATEGY=tournament bash scripts/run_serve.sh
```

Use `POST /v1/systemone`; the local backend is text-only. Readout strategy, batch token budget, max length, precision,
calibration and model versions must be recorded. Historical train-jev-4b-lora / train-jev-27b-lora wide results use tournament; the [autotrust/JEV-27B-VL](https://huggingface.co/autotrust/JEV-27B-VL)
reference uses single. The default server strategy is single, so defaults do not reproduce the historical wide score.
For the native JEV `/v1/decide` backend, use its documented compatibility proxy; do not send scorer metadata as state.

## 3. Decision Index: reproduce the old sample, not a different panel

Use [apolinario/decision-index](https://github.com/apolinario/decision-index) revision `f8552a13b643d39f61185f5951bb5fed16bedf51` for the recorded sampler/scorer:

```bash
git clone https://github.com/apolinario/decision-index third_party/decision-index
git -C third_party/decision-index checkout f8552a13b643d39f61185f5951bb5fed16bedf51
pip install -e third_party/decision-index
```

Rebuild/import the official 0.3 suite using that project's locked source revisions and documentation. Downloads can
be large or require access; incomplete files are not valid suites. Use the upstream downloader or independently
download pinned files and verify their official hashes before import; no local downloader patch is required here.
HLE/HoVer are **not required model evaluations for our main
legacy 35 result**; a complete 37 source suite can also produce exactly the same old sample by explicitly dropping them.

```bash
python scripts/di_sample.py --di_root third_party/decision-index --n 6000 \
  --drop 45,61 --seed 20260919 --rows_out data/di_sample6000.jsonl.gz \
  --suite_out third_party/decision-index/suite-0.3-sample6000
URL=http://127.0.0.1:8020 NAME=jev-4b bash scripts/run_decision_index.sh
```

If the complete suite lives elsewhere, use `--source_suite PATH`. Existing outputs are not overwritten; use a new
versioned path. The sampler takes whole groups round-robin across eligible catalogs, ranked by the hash of
`seed:catalog:group`. The 6,000 input requests span 41 catalogs; the reported public index uses 35 indexed benchmarks
after dropping 45(HLE)/61(HoVer), not all 41 as index components.

`sampling_manifest.json` contains seed, catalog counts, ordered run-ID SHA256 and canonical model-input SHA256.
Rebuilding from the complete 37 source with these settings was actually verified against the original sample:
**all 6,000 ordered run IDs and all ordered state/questions payloads are identical** (fingerprints in the
`sampling_manifest.json` the script writes). Gzip header/container bytes are not the equality test.

Score against the generated **sample-only suite**, not the full suite, otherwise unsampled requests count as missing.
Do not claim this adjusted 35-index as the official complete leaderboard. `scripts/di_index.py` reads the official
area specification and drops 45/61; a new 37 sample/full result needs a different declared protocol and matched
reference rerun. Never fill missing benchmarks with guessed scores or silently ignore failed model requests.

## 4. Kev transfer-v4

Use the same frozen `transfer-v4` suite/source revision for every model; verify its IDs and input hashes. Obtain the
Kev implementation/suite separately as documented by [jaredpalmer/kev](https://github.com/jaredpalmer/kev).

```bash
# In a pinned Kev checkout with the original suite available:
python -m kev.benchmark --remote http://127.0.0.1:8020 --remote-concurrency 1 \
  --suite evals/v4/transfer-v4 --out eval/transfer-jev-4b
```

Report clean 656-row accuracy, NLL, ECE separately from controls/variants (the recorded prediction metadata contains
764 rows). `rows.json` here contains predictions/labels, not complete input text: it cannot prove decontamination.
Label is an index into each row's keys, not necessarily a letter. Paired results require matching ID/question/label
sets. McNemar significance on rows is not a substitute for training-seed or source-level variance.

## 5. Vision

```bash
python -m trainjev.eval_vision --base Qwen/Qwen3.5-4B --adapter runs/jev-4b/best \
  --di_root third_party/decision-index --per_set 300 --out eval/vision-4b
python -m trainjev.eval_vision --base Qwen/Qwen3.8-27B --adapter runs/jev-27b/best \
  --di_root third_party/decision-index --per_set 300 --out eval/vision-27b
```

The [upstream Decision Index kit](https://github.com/apolinario/decision-index) must have the 5 vision assets rebuilt. Actual coverage:CV-Bench/BLINK/RealWorldQA/CharXiv each 300,
InfoVQA 244, total 1,444, not 1,500. Mean skill/accuracy is the equal mean of dataset metrics, not one pooled accuracy.
Match image processing/size, precision and row IDs before comparing implementations. Text LoRA is merged into the
VL language model; the vision tower was not trained. This is not the text HTTP route and not full multimodal HLE.
The official DI 0.3 text adapter intentionally retains 501 HLE text-MC requests, not all 2,500 raw HLE questions.

## 6. train-jev-pk, train-jev-classification-pk and wide diagnostics

Both fixed datasets publish every row's index, not every row's raw text. [train-jev-pk](https://github.com/shibing624/train-jev/tree/main/data/train-jev-pk) has 2,494 rows / 22 tasks:
1,594 inputs are included and 900 rows are index-and-label only. [train-jev-classification-pk](https://github.com/shibing624/train-jev/tree/main/data/train-jev-classification-pk) has 1,300 rows / 13 tasks:
500 inputs are included and 800 rows are index-and-label only. Raw text with uncleared licenses is not redistributed.
Included text retains upstream attribution and licenses; see each dataset README. The code license does not cover upstream text.

Obtain the original source files with the builder under their source terms, then restore the fixed row indices and verify labels and full model-input SHA256 offline:

```bash
python scripts/pk_build.py --out data/pk
python scripts/prepare_eval_data.py --dataset data/train-jev-pk --source-dir data/pk \
  --out eval/complete-inputs.jsonl
python scripts/evaluate_pk.py --dataset data/train-jev-pk --source-dir data/pk \
  --base Qwen/Qwen3.5-4B --adapter shibing624/train-jev-4b-lora --out eval/train-jev-pk-4b

python scripts/prepare_eval_data.py --dataset data/train-jev-classification-pk --source-dir data/pk \
  --out eval/classification-complete-inputs.jsonl
python scripts/evaluate_pk.py --dataset data/train-jev-classification-pk --source-dir data/pk \
  --base Qwen/Qwen3.5-4B --adapter shibing624/train-jev-4b-lora --out eval/train-jev-classification-pk-4b
```

`prepare_eval_data.py` makes no network requests and refuses missing source files, mismatched labels or changed input hashes.
`evaluate_pk.py` also requires every input, uses local HF raw prefill with the published question / score legend intact,
and applies saved calibration without refitting. It does not use the HTTP conversion that would rewrite score-question legends.
train-jev-pk uses equal-task macro averages; historical classification PK ECE is `all_rows.ece10` (pooled 10-bin ECE), not mean task ECE. The bundled-text subset is not a full-suite evaluation.
HF model loading may still download model files. Different GPUs, batching, precision or stacks need not give bit-identical probabilities,
and these commands do not replace historical metrics.

The old `pk_benchmark.py` remains useful for the builder's source files of up to 600 rows each and server comparisons.
The historical 13×100 classification server protocol and wide-choice builder remain available:

```bash
python scripts/pk_benchmark.py run --url http://127.0.0.1:8020 --suite data/pk \
  --strategy tournament --limit 100 --workers 1 --out eval/pk_4b.jsonl
python scripts/pk_benchmark.py report model=eval/pk_4b.jsonl

python scripts/wide_build.py --out data/wide
URL=http://127.0.0.1:8020 NAME=jev-4b SUITE=data/wide LIMIT=600 bash scripts/run_wide.sh
```

Freeze dataset files/labels/SHA before comparing. Historical classification PK uses label names(underscores replaced by spaces),text truncated
to 4,000 characters and the recorded common question strings. Five-level rating tasks use choice over the five
labels, not the trained six-level score protocol. Compare paired scenario/index/gold sets, not unrelated averages.

Wide diagnostics record every actual strategy/option-order hash/model input size. Nine warm-up requests were
excluded; an empty internal batching call is not a model forward. The observed permute gain and underconfidence
are in [results.en.md §8](https://github.com/shibing624/train-jev/docs/results.en.md#8-wide-choice-questions-17256-options). Choose strategy/post-average calibration on independent dev and evaluate separately, not by
fitting temperatures on these 600 test examples. A small smoke is not statistical evidence of SOTA.

## 7. What the reports do not claim

- Fixed recipes and inputs are reproducible; cross-GPU/library/dtype/merge implementations are not bit-exact.
- The main result uses HF merged bf16 inference. An experimental vLLM backend did not pass the probability-agreement
  check, so its results do not replace the reported probabilities.
- A single seed and sample-based index do not prove superiority across deployments or the full/private board.
- Public datasets bundle only inputs with cleared redistribution terms; restore other rows from independently obtained source files under upstream terms, rather than claiming all raw text is packaged. Private logs are not included; models are obtained separately from HF.
