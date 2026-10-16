"""Local GPU/CPU inference from a train-jev directory or a released HF adapter repo.

python examples/local_inference.py --base models/Qwen3.5-4B --adapter runs/jev-4b/best
CPU fp32 inference needs substantial RAM and is slower; no text is generated.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from trainjev.data import encode  # noqa: E402
from trainjev.modeling import last_hidden, load_model, load_temperatures, resolve_checkpoint  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Local pinned backbone or its HF model id")
    parser.add_argument("--adapter", required=True, help="Local adapter directory or HF release repo id")
    parser.add_argument("--adapter_revision", default=None)
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    parser.add_argument("--state", default="The same order was billed twice. The customer needs help with the payment.")
    parser.add_argument("--question", default="Which team should handle this request?")
    parser.add_argument("--options", nargs="+", default=["billing", "shipping", "technical support"])
    a = parser.parse_args(argv)
    if not 2 <= len(a.options) <= 256:
        parser.error("choice requires 2–256 options")
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu or an HTTP server")
    adapter = resolve_checkpoint(a.adapter, a.adapter_revision)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok, body, head = load_model(a.base, adapter, dtype=dtype, device=device)
    calibration = Path(adapter) / "calibration.json"
    temps = load_temperatures(calibration)
    row = {"kind": "choice", "state": a.state, "question": a.question, "options": a.options}
    ids = encode(tok, row, labels=tok.choice_labels)
    with torch.inference_mode():
        h = last_hidden(body, [ids], tok.pad_token_id or 0, device)[0]
        p = torch.softmax(head.logits(h, "choice", len(a.options)) / temps["choice"], -1).cpu().tolist()
    result = {"device": device, "calibration_loaded": calibration.is_file(), "input_tokens": len(ids),
              "prediction": a.options[max(range(len(p)), key=p.__getitem__)],
              "probabilities": dict(zip(a.options, p)), "strategy": "single", "trained_choice_slots": 16}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
