# Changelog

## v0.1.0 (2026-10-14)

First release.

- JEV System 1 recipe on Qwen3.5: LoRA r16 on all Gated DeltaNet / attention / MLP projections, 24-slot fp32 decision
  head initialised from `lm_head`, KL + 0.5·RPS loss, 30% option permutation, kind-stratified 128-row batches,
  multi-GPU DDP (`trainjev.train`).
- Per-kind temperature calibration and corpus evaluation (`trainjev.evaluate`).
- TypeSafe System One compatible server with `/v1/systemone`, `/v1/decide`, the >16-option tournament and a proxy mode
  for JEV-27B-VL (`trainjev.serve`).
- Zero-shot image decision evaluation on the Decision Index vision sets (`trainjev.eval_vision`).
- Decision Index 0.3 helpers: stratified sample + sample-only suite (`scripts/di_sample.py`), panel re-scoring with
  dropped benchmarks (`scripts/di_index.py`).
- 13-dataset classification PK: byte-identical suite builder from the HF Hub (`scripts/pk_build.py`) and a resumable
  runner with a paired report (`scripts/pk_benchmark.py`).
- Qwen3.8-27B backbone verified end to end; `scripts/run_train_27b.sh`.
- Python >= 3.12.
- Results for Qwen3.5-4B: [docs/results.md](docs/results.md).
