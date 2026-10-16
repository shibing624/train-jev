# Serving and inference

[中文](serving.md)

## Start a server

```bash
# a released model from Hugging Face (downloads the base, the LoRA, the decision head and the temperatures)
python -m trainjev.serve --base Qwen/Qwen3.5-4B --adapter shibing624/train-jev-4b-lora --port 8020
python -m trainjev.serve --base Qwen/Qwen3.8-27B --adapter shibing624/train-jev-27b-lora --port 8020

# a checkpoint you trained yourself
CKPT=runs/jev-4b/best PORT=8020 bash scripts/run_serve.sh
```

| flag | default | meaning |
|---|---|---|
| `--base` | `Qwen/Qwen3.5-4B` | backbone: HF id or local directory |
| `--adapter` | – | HF repo id or local checkpoint directory (LoRA + `head.pt` + `calibration.json`) |
| `--adapter_revision` | – | HF tag or commit of the adapter, e.g. `v0.2.0` |
| `--strategy` | `single` | how choices with more than 16 options are read (see below) |
| `--device` | `cuda` | `cpu` runs fp32 on the CPU (slow; for checks only) |
| `--host` / `--port` | `127.0.0.1` / `8020` | bind address |
| `--max_len` | 32768 | prompt token limit; a longer state is cut in the middle |
| `--batch_tokens` | 32768 | padded tokens per forward pass when batching questions |
| `--calibration` | `<adapter>/calibration.json` | temperatures file |
| `--base_only` | off | serve the untrained backbone (head initialised from `lm_head`) as a reference |
| `--proxy URL` | – | put a JEV-27B-VL `serve_decide.py` server behind the same API, for side-by-side comparison |

Memory: the 4B model in bf16 is about 9 GB and fits a 24 GB GPU; the 27B model is about 54 GB and needs an 80 GB GPU.
On CUDA install `flash-linear-attention`; without it the Gated DeltaNet layers of Qwen3.5 run a slow reference path.

Plain PEFT / Transformers loading applies only the LoRA adapter. The decision head (`head.pt`) and the temperatures
(`calibration.json`) are applied by train-jev, which is why the model is served through `trainjev.serve`.

## API

`POST /v1/systemone`, compatible with TypeSafe's [System One API](https://docs.typesafe.ai/api). `GET /health`
returns the model name.

```jsonc
{
  "state": "…",                        // string, or an object / array (serialised to JSON)
  "questions": {
    "<id>": {                          // you choose the id; the model never sees it
      "type": "noul" | "choice" | "score",
      "instructions": "…",
      "criteria": …                    // noul: optional {true, false} descriptions
                                       // choice: {option: description | null}, 2–256 options
                                       // score: [level, …], ordered from lowest to highest
    }
  },
  "strategy": "single"                 // optional, overrides --strategy for this request
}
```

| type | answer |
|---|---|
| `noul` | `noul`: probability of true |
| `choice` | `choice`: most likely option; `probabilities` over all options |
| `score` | `score`: expected level index (from 0); `probabilities` over the levels |

A request may carry any number of questions. They share the state but are scored independently: each question sees
the state and its own question only, never the others' answers. Questions are batched by length.

Six-level `score` questions use the trained `score` slots with the level texts as a legend. Any other number of levels
is asked as a `choice` over the level texts, and `score` is the expected index.

Images in `state` are supported only with `--proxy`. Local image decisions use `trainjev.eval_vision`
(see [evaluation.en.md](evaluation.en.md#5-vision)).

## Examples

With a server running:

```bash
python examples/client.py --url http://127.0.0.1:8020             # one ticket, three question types
python examples/batch_decisions.py --url http://127.0.0.1:8020    # several decisions about one state
python examples/smoke_systemone.py --url http://127.0.0.1:8020    # three fixed questions with expected answers
python examples/wide_choices.py --url http://127.0.0.1:8020 --compare   # 20 options, all three strategies
```

Without a server, one decision straight from Python:

```bash
python examples/local_inference.py --base Qwen/Qwen3.5-4B --adapter shibing624/train-jev-4b-lora \
  --state "My card was charged twice for one order." --question "Which team should handle this request?" \
  --options billing shipping "technical support"
```

## Choices with more than 16 options

The decision head is trained on the labels A–P only. For 17–256 options `--strategy` (or `"strategy"` in the request)
picks how the question is read; all three come from JEV-27B-VL's `serve_decide.py`:

| strategy | how | forward passes (n options) |
|---|---|---:|
| `single` (default) | one pass: A–P from the trained head, then Q–Z, AA, AB, … from the backbone's own `lm_head` rows with zero bias | 1 |
| `tournament` | groups of ≤ 16, then a final of the group winners plus the best runners-up; only the trained A–P are ever read | ⌈n/16⌉ + 1 |
| `permute` | `single` over 4 option orders (the given one + 3 shuffles seeded by the question), averaged | 4 |

Measured on train-jev-4b-lora, 600 questions each from CLINC150 (150 options), BANKING77 (77) and MASSIVE en (59):

| strategy | accuracy (1,800) | ECE ↓ | p50 latency |
|---|---:|---:|---:|
| `single` | 0.727 | 0.162 | 52 ms |
| `tournament` | 0.732 | **0.050** | 109 ms |
| `permute` | **0.771** | 0.218 | 171 ms |

`permute` is the most accurate but under-confident; `tournament` is the best calibrated. Per-dataset numbers are in
[results.en.md](results.en.md#8-wide-choice-questions-17256-options). The published benchmark numbers for both models use
`tournament`; pass `--strategy tournament` to reproduce them.
