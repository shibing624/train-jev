"""Evaluate a complete PK release locally with fixed, saved decision-head temperatures."""
import argparse
import json
import math
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_eval_data import load_records

METRIC_NAMES = ("accuracy", "nll", "brier", "ece15")


def prediction(kind, probabilities):
    """Match the noul true-at-0.5 decision rule, not argmax's false-at-tie rule."""
    if kind == "noul":
        return int(probabilities[1] >= 0.5)
    return max(range(len(probabilities)), key=probabilities.__getitem__)


def _check_probabilities(row, probabilities):
    if (not isinstance(probabilities, (list, tuple)) or len(probabilities) != len(row["target"])
            or not all(isinstance(p, (int, float)) and math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
            or not math.isclose(sum(probabilities), 1.0, rel_tol=0, abs_tol=1e-6)):
        raise ValueError(f"{row['id']}: invalid probability vector")


def metrics(rows, probabilities, bin_count=15):
    """Hard-label accuracy, mean NLL, summed-class Brier, and equal-width ECE bins."""
    if not rows or len(rows) != len(probabilities):
        raise ValueError("Metrics require nonempty rows and exactly one probability vector per row")
    correct = nll = brier = 0.0
    bins = [[0, 0.0, 0.0] for _ in range(bin_count)]
    for row, probs in zip(rows, probabilities):
        _check_probabilities(row, probs)
        gold = row["target"].index(1)
        pred = prediction(row["kind"], probs)
        hit = pred == gold
        correct += hit
        nll -= math.log(max(probs[gold], 1e-12))
        brier += sum((p - t) ** 2 for p, t in zip(probs, row["target"]))
        confidence = probs[pred]
        # Same upper-inclusive boundaries as trainjev.metrics.ece.
        bucket = bins[next(i for i in range(bin_count) if confidence <= (i + 1) / bin_count)]
        bucket[0] += 1
        bucket[1] += confidence
        bucket[2] += hit
    n = len(rows)
    ece = sum(abs(conf_sum - hit_sum) / n for count, conf_sum, hit_sum in bins if count)
    return {"n": n, "correct": int(correct), "accuracy": correct / n, "nll": nll / n,
            "brier": brier / n, f"ece{bin_count}": ece}


def summarise(rows, probabilities):
    """Equal-task macro is primary; all-row micro metrics are explicitly separate."""
    all_rows = metrics(rows, probabilities)
    # The historical 13-scenario classification PK reports pooled 10-bin ECE.
    all_rows["ece10"] = metrics(rows, probabilities, bin_count=10)["ece10"]
    grouped = {}
    for row, probs in zip(rows, probabilities):
        task_rows, task_probs = grouped.setdefault(row["task"], ([], []))
        task_rows.append(row)
        task_probs.append(probs)
    per_task = {task: metrics(task_rows, task_probs) for task, (task_rows, task_probs) in grouped.items()}
    macro = {key: sum(values[key] for values in per_task.values()) / len(per_task) for key in METRIC_NAMES}
    return {"rows": len(rows), "tasks": len(per_task), "macro": macro, "per_task": per_task, "all_rows": all_rows}


def predict_probabilities(rows, *, base, adapter, adapter_revision=None, device="cuda", batch_tokens=8192,
                          max_len=8192):
    import torch

    from trainjev.data import micro_batches, prompt
    from trainjev.modeling import last_hidden, load_model, load_temperatures, resolve_checkpoint

    if batch_tokens <= 0 or max_len <= 0:
        raise ValueError("--batch_tokens and --max_len must be positive")
    if device != "cpu" and not device.startswith("cuda"):
        raise ValueError("--device must be cpu or cuda[:index]; CPU evaluation must be explicitly requested")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu explicitly for CPU evaluation")
    checkpoint = resolve_checkpoint(adapter, adapter_revision)
    temperatures = load_temperatures(str(Path(checkpoint) / "calibration.json") if checkpoint else "")
    for kind in {row["kind"] for row in rows}:
        temperature = temperatures.get(kind)
        if (not isinstance(temperature, (int, float)) or isinstance(temperature, bool)
                or not math.isfinite(temperature) or temperature <= 0):
            raise ValueError(f"Saved temperature for {kind} must be finite and positive")
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    tok, body, head = load_model(base, checkpoint, dtype=dtype, device=device)
    encoded = []
    for row in rows:
        text = prompt(row["kind"], row["state"], row["question"], row["options"], labels=tok.choice_labels)
        ids = tok(text, add_special_tokens=False).input_ids
        if not ids or len(ids) > max_len:
            raise ValueError(f"{row['id']}: input has {len(ids)} tokens, allowed 1..{max_len}; "
                             "refusing truncation or partial scores")
        encoded.append(ids)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    probabilities = [None] * len(rows)
    with torch.inference_mode():
        for batch in micro_batches([(len(ids), i) for i, ids in enumerate(encoded)], batch_tokens):
            hidden = last_hidden(body, [encoded[i] for i in batch], pad_id, device)
            for index, h in zip(batch, hidden):
                row = rows[index]
                logits = head.logits(h, row["kind"], len(row["options"]))
                probs = torch.softmax(logits.float() / temperatures[row["kind"]], dim=-1).cpu().tolist()
                _check_probabilities(row, probs)
                probabilities[index] = probs
    return probabilities, temperatures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--adapter", default="shibing624/train-jev-4b-lora")
    parser.add_argument("--adapter_revision", default=None)
    parser.add_argument("--dataset", default="data/train-jev-pk")
    parser.add_argument("--source-dir", help="Independently downloaded legacy PK text/label JSONL directory")
    parser.add_argument("--out", required=True, help="Evaluation output directory")
    parser.add_argument("--device", default="cuda", help="cuda[:index], or explicitly cpu")
    parser.add_argument("--batch_tokens", type=int, default=8192)
    parser.add_argument("--max_len", type=int, default=8192)
    args = parser.parse_args(argv)
    if args.batch_tokens <= 0 or args.max_len <= 0:
        raise ValueError("--batch_tokens and --max_len must be positive")
    out = Path(args.out).expanduser()
    for name in ("summary.json", "predictions.jsonl"):
        if (out / name).exists() or (out / name).is_symlink():
            raise FileExistsError(f"Refusing to overwrite existing evaluation output: {out / name}")
    for root in (args.dataset, args.source_dir):
        if root is not None and out.resolve().is_relative_to(Path(root).expanduser().resolve()):
            raise ValueError("--out must be outside the dataset and source directories")
    rows = load_records(args.dataset, args.source_dir)
    probabilities, temperatures = predict_probabilities(
        rows, base=args.base, adapter=args.adapter, adapter_revision=args.adapter_revision,
        device=args.device, batch_tokens=args.batch_tokens, max_len=args.max_len)
    summary = summarise(rows, probabilities)
    summary["temperatures"] = temperatures
    # Do not disclose resolved adapter/cache paths or any raw state in published artifacts.
    summary["protocol"] = {"aggregation": "equal-task macro", "ece_bins": 15,
                           "nll_probability_floor": 1e-12, "brier": "sum over classes",
                           "recalibrated": False, "truncated": False,
                           "classification_pk_ece": "all_rows.ece10 (pooled, not mean task ECE)"}
    out.mkdir(parents=True, exist_ok=True)
    with (out / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        for row, probs in zip(rows, probabilities):
            pred = prediction(row["kind"], probs)
            gold = row["target"].index(1)
            result = {key: row[key] for key in ("id", "task", "eval_group", "group_id", "kind", "target")}
            result.update(probs=probs, prediction=pred, gold=gold, correct=pred == gold, confidence=probs[pred])
            stream.write(json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    with (out / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    macro = summary["macro"]
    print(f"{summary['rows']} rows / {summary['tasks']} tasks: equal-task macro "
          f"accuracy={macro['accuracy']:.4f} NLL={macro['nll']:.4f} "
          f"Brier={macro['brier']:.4f} ECE15={macro['ece15']:.4f}; "
          f"all-row accuracy={summary['all_rows']['accuracy']:.4f} ECE10={summary['all_rows']['ece10']:.4f}")
    return summary


if __name__ == "__main__":
    main()
