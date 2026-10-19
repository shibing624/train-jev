# Serving and inference

[中文](serving.md)

## Start a server

```bash
# script defaults: Qwen/Qwen3.5-4B + shibing624/train-jev-4b-lora
bash scripts/run_serve.sh

# a released model from Hugging Face (downloads the base, the LoRA, the decision head and the temperatures)
python -m trainjev.serve --base Qwen/Qwen3.5-4B --adapter shibing624/train-jev-4b-lora --port 8020
python -m trainjev.serve --base Qwen/Qwen3.8-27B --adapter shibing624/train-jev-27b-lora --port 8020

# a checkpoint you trained yourself
CKPT=runs/jev-4b/best PORT=8020 bash scripts/run_serve.sh

# pin the adapter release and pass a token budget as a trailing CLI argument
ADAPTER_REVISION=v0.3.0 STRATEGY=tournament bash scripts/run_serve.sh --batch_tokens 32768

# text and image decisions on the same endpoint, using the full VL backbone
VISION=1 bash scripts/run_serve.sh
```

`run_serve.sh` supports `BASE`, `CKPT`, `ADAPTER_REVISION`, `DEVICE`, `NAME`, `HOST`, `PORT`, `STRATEGY`, `VISION`, and `VARIANT`,
and forwards trailing arguments to `trainjev.serve`. Script defaults are `BASE=Qwen/Qwen3.5-4B`,
`CKPT=shibing624/train-jev-4b-lora`, `DEVICE=cuda`, `NAME=train-jev-4b-lora`, `HOST=127.0.0.1`, `PORT=8020`
and `STRATEGY=single`; the adapter revision is unpinned by default. The table below lists Python CLI defaults:
unlike the script, the CLI has no default `--adapter`.

| flag | default | meaning |
|---|---|---|
| `--base` | `Qwen/Qwen3.5-4B` | backbone: HF id or local directory |
| `--adapter` | – | HF repo id or local checkpoint directory (LoRA + `head.pt` + `calibration.json`) |
| `--adapter_revision` | – | HF tag or commit of the adapter, e.g. `v0.3.0`; `v0.2.0` for the previous release |
| `--strategy` | `single` | how choices with more than 16 options are read (see below) |
| `--device` | `cuda` | `cpu` runs fp32 on the CPU (slow; for checks only) |
| `--host` / `--port` | `127.0.0.1` / `8020` | bind address |
| `--max_len` | 32768 | prompt token limit; a longer state is cut in the middle |
| `--batch_tokens` | 32768 | padded tokens per forward pass when batching questions |
| `--calibration` | `<adapter>/calibration.json` | temperatures file |
| `--base_only` | off | serve the untrained backbone (head initialised from `lm_head`) as a reference |
| `--vision` | off | load the full Qwen VL backbone for image assets; adds vision-encoder memory |
| `--proxy URL` | – | put an [autotrust/JEV-27B-VL](https://huggingface.co/autotrust/JEV-27B-VL) `serve_decide.py` server behind the same API, for side-by-side comparison |
| `--variant` | `plain` | `plain` is one forward pass at the checkpoint's attention; `instant` requires bidirectional `head.pt`; `autothink` requires a causal checkpoint and the backbone `lm_head` |
| `--gate` / `--mix` | `0.6` / `0.75` | AutoThink writes a thought when the top probability is below the gate; `mix` weights the post-thought readout |
| `--think_budget` | `4096` | maximum thought tokens |
| `--think_temperature` / `--think_top_p` / `--think_top_k` | `0.6` / `0.95` / `20` | thought sampler |

Memory: the 4B model in bf16 is about 9 GB and fits a 24 GB GPU; the 27B model is about 54 GB and needs an 80 GB GPU.
On CUDA install `flash-linear-attention`; without it the Gated DeltaNet layers of Qwen3.5 run a slow reference path.

Plain PEFT / Transformers loading applies only the LoRA adapter. The decision head (`head.pt`) and the temperatures
(`calibration.json`) are applied by train-jev, which is why the model is served through `trainjev.serve`.

## API

`POST /v1/systemone`, compatible with TypeSafe's [System One API](https://docs.typesafe.ai/api). `POST /v1/decisions` is the same handler. `GET /health` returns the model name, `variant`, and `attention`.

Omitting `--variant` keeps a released causal model on one forward pass. `think_mode` is `auto`, `never`, or `always` (aliases `hybrid`, `plain`, `think`). Plain and Instant accept only `never`. AutoThink defaults to `auto`. Image questions and choices wider than 16 options skip the thought. StartLux's published numbers use gate 0.9; the server default stays 0.6. See the [Decision Index plan](di_plan.en.md).

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

Start local image serving with `--vision`. Without it the original text-only loader remains in use and image assets return 422 instead of being silently ignored. Vision and text share the same LoRA-applied language model and FP32 decision head. Image questions run separately; text-only questions retain length-based batching.

## Image input: TypeSafe assets

Using the [TypeSafe Images](https://docs.typesafe.ai/images) assets format, references can appear in string values of `state`, `instructions` and `criteria`. No experimental header is required locally; enable `--vision` at startup. The existing `noul`, `choice` and `score` types work with images; **`box` is not implemented**.

Prefer multipart, which avoids base64's roughly 33% overhead. Put JSON first in a part named `request`, and each file in its own named part:

```bash
curl http://localhost:8020/v1/systemone \
  -F 'request={
    "model": "jev-latest",
    "state": {"door": "a3f1c2e8-door"},
    "assets": {"a3f1c2e8-door": {"type": "image", "part": "door-photo"}},
    "questions": {
      "person_present": {"type": "noul", "instructions": "Is there a person in `door`?"},
      "carrying": {"type": "choice", "instructions": "What is the person in `door` carrying?",
                   "criteria": {"package": "A box or parcel.", "bag": "A shopping bag.", "nothing": "Empty hands."}}
    }
  };type=application/json' \
  -F 'door-photo=@door-camera.jpg;type=image/jpeg'
```

Or use JSON/base64. No fixed client-side image dimensions are required:

```python
import base64
import uuid
from pathlib import Path
import httpx

photo_key = str(uuid.uuid4())
photo = {
    "type": "image",
    "data_base64": base64.b64encode(Path("door-camera.jpg").read_bytes()).decode("ascii"),
}
response = httpx.post(
    "http://localhost:8020/v1/systemone",
    json={
        "model": "jev-latest",
        "state": {"door": photo_key},
        "assets": {photo_key: photo},
        "questions": {"person_present": {"type": "noul", "instructions": "Is there a person in `door`?"}},
    },
    timeout=120,
)
response.raise_for_status()
print(response.json()["answers"])
```

### References and validation

- Replace string values only, not question IDs, Choice option names or other object keys. Whole values match first; longer strings match left to right. Prefer UUIDs to avoid accidental substring matches.
- Nested/multiple images and question-local instruction/criterion examples are supported. A question never sees another question's examples. Each repeated use expands the image again.
- Every asset must be referenced. `type` is optional. Base64 must be standard and padded, without whitespace or a `data:` prefix. `format` is optional and inferred from content; when supplied it must match. Part wins when both forms are given.
- PNG, JPEG, WebP, BMP, TIFF, GIF, ICO and AVIF are supported subject to Pillow codec availability. GIF, multi-page TIFF and APNG use frame 0 as a still image. SVG, PDF, PostScript and video are not supported by this endpoint.
- The 280–1568 per-side constraint is removed. Qwen's processor resizes ordinary images according to its pixel/grid settings; very small images are enlarged proportionally, and extreme aspect ratios are reduced and padded without cropping. Images are EXIF-oriented, converted to RGB and stripped of metadata before vision encoding.
- Resource guards remain: 16 MiB per file, 32 MiB HTTP body, 256 MiB for processed images and expanded content, about 89.5 MP total decoded pixels, 16 distinct assets and 64 references. Pillow decompression-bomb checks remain; resource excesses return 413. These are safety budgets, not TypeSafe's fixed edge dimensions.
- Image prompts exceeding `--max_len` return 422 without silent truncation; legacy text-only truncation is unchanged.
- Bad images/references/JSON return 400; invalid questions or a text-only backend return 422; unsupported media/encoding returns 415. Only uncompressed JSON/multipart is supported here.
- Inference memory exhaustion returns 503 without partial answers; reduce image count or context before retrying. Guaranteed-overlong repeated images are rejected before allocating patch tensors.
- Assets do not accept URLs or server-local paths. No automatic URL fetches or arbitrary file reads. `--proxy` forwards validated images as data URLs to the configured `/v1/decide`; question/criterion images are labeled in that question's upstream state. The legacy proxy inline-image state format still works.

`GET /health` includes `images` for local vision/proxy availability. The request's `model` is a compatibility field; the actual model is selected at startup. The local server does not validate TypeSafe Bearer keys; use reverse-proxy auth, rate limiting and body-size limits for public deployment.

Image input support is not a guarantee of visual accuracy. This remains zero-shot visual transfer after text-only training. Run `trainjev.eval_vision` for task evaluation (see [evaluation.en.md](evaluation.en.md#5-vision)).

## Examples

With a server running:

```bash
python examples/client.py --url http://127.0.0.1:8020             # one ticket, three question types
python examples/batch_decisions.py --url http://127.0.0.1:8020    # several decisions about one state
python examples/smoke_systemone.py --url http://127.0.0.1:8020    # three fixed questions with expected answers
python examples/wide_choices.py --url http://127.0.0.1:8020 --compare   # 20 options, all three strategies
python examples/image_decisions.py --image door-camera.jpg           # TypeSafe multipart image
python examples/image_decisions.py --image door-camera.jpg --transport base64  # JSON/base64
```

Without a server, one decision straight from Python:

```bash
python examples/local_inference.py --base Qwen/Qwen3.5-4B --adapter shibing624/train-jev-4b-lora \
  --state "My card was charged twice for one order." --question "Which team should handle this request?" \
  --options billing shipping "technical support"
```

## Choices with more than 16 options

The decision head is trained on the labels A–P only. For 17–256 options `--strategy` (or `"strategy"` in the request)
picks how the question is read; all three come from [autotrust/JEV-27B-VL](https://huggingface.co/autotrust/JEV-27B-VL)'s `serve_decide.py`:

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
[results.en.md](results.en.md#8-wide-choice-questions-17256-options). Both models' historical classification PK / wide-choice results use
`tournament`; pass `--strategy tournament` to reproduce those runs. The full public Decision Index text panel uses `single`;
do not mix these protocols. For the fixed [train-jev-pk](../data/train-jev-pk) / [train-jev-classification-pk](../data/train-jev-classification-pk)
releases, restore missing inputs and use the local HF evaluator described in the
[evaluation guide](evaluation.en.md#6-train-jev-pk-train-jev-classification-pk-and-wide-diagnostics).
Preserve the original question / score legend instead of substituting HTTP rewriting; bit-identical probabilities are not guaranteed across GPU / batching stacks.
