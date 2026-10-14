# Contributing to train-jev

Thanks for your interest! Bug reports, evaluation results on new backbones, and improvements to the recipe are all
welcome.

## Development setup

```shell
git clone https://github.com/shibing624/train-jev
cd train-jev
pip install -r requirements.txt
pip install -e ".[dev]"
```

## Before opening a PR

1. Add or update unit tests under `tests/` (they run on CPU).
2. Run the checks:

   ```shell
   ruff check .
   python -m pytest
   # optional, needs a Qwen3.5 tokenizer locally or on the Hub:
   TRAINJEV_TOKENIZER=Qwen/Qwen3.5-4B python -m pytest
   ```

3. If you change anything on the numeric path (prompt, encoding, head, loss, batching, serving), state in the PR how
   you checked it. For example: logits on a fixed `test_set_30k` subset before vs after (`trainjev.evaluate` writes
   them to `<out>/test_set_30k.jsonl`), or the first steps of the training log with the same seed.
4. If you report new results, include the exact command, the hardware, and which rows were evaluated.

## Code style

- Line length 120, enforced by `ruff`.
- Keep the training recipe in `trainjev/train.py` readable end to end; prefer flags over new abstractions.
