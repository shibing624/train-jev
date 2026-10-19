"""Validate a released PK index and materialize licensed, independently supplied text."""
import argparse
import hashlib
import json
import re
from pathlib import Path


def canonical_input(row):
    """Tokenizer-independent input bytes; nested state key order is significant."""
    payload = {key: row[key] for key in ("kind", "state", "question", "options")}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _read_jsonl(path):
    records = []
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{path.name}: row_index {index}: invalid JSONL (including blank lines)") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path.name}: row_index {index}: expected a JSON object")
            records.append(row)
    return records


def _integer(value, minimum):
    return type(value) is int and value >= minimum


def _basename(value):
    if (not isinstance(value, str) or not value or value in (".", "..")
            or "/" in value or "\\" in value or ":" in value or "\x00" in value):
        raise ValueError("source.file must be a safe basename, not a path")
    return value


def _validate_row(row):
    for key in ("id", "task", "eval_group", "group_id"):
        if not isinstance(row.get(key), str) or not row[key]:
            raise ValueError(f"{key} must be a nonempty string")
    if not isinstance(row.get("question"), str):
        raise ValueError("question must be a string")
    kind, options, target = row.get("kind"), row.get("options"), row.get("target")
    if not isinstance(options, list) or not all(isinstance(option, str) for option in options):
        raise ValueError("options must be a list of strings")
    n = len(options)
    if not ((kind == "choice" and 2 <= n <= 256) or (kind == "score" and n == 6)
            or (kind == "noul" and n == 2)):
        raise ValueError("invalid primitive range: choice needs 2..256, score 6, noul 2 options")
    if (not isinstance(target, list) or len(target) != n
            or not all(type(value) in (int, float) and value in (0, 1) for value in target)
            or sum(target) != 1):
        raise ValueError("target must be a onehot vector matching options")
    digest = row.get("input_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("input_sha256 must be a lowercase SHA256 digest")
    source = row.get("source")
    if not isinstance(source, dict):
        raise ValueError("source must be an object")
    _basename(source.get("file"))
    if not _integer(source.get("row_index"), 0):
        raise ValueError("source.row_index must be a nonnegative integer")
    if not isinstance(source.get("native_label"), str):
        raise ValueError("source.native_label must be a string")
    if "text_limit" not in source or not (source["text_limit"] is None or _integer(source["text_limit"], 1)):
        raise ValueError("source.text_limit must be null or a positive integer")
    for key in ("repo", "license"):
        if key in source and not isinstance(source[key], str):
            raise ValueError(f"source.{key} must be a string")
    if "state" in row and not isinstance(row["state"], (str, dict)):
        raise ValueError("state must be a string or object")


def load_records(dataset_path, source_dir=None):
    """Return every validated row, or fail without permitting partial evaluation."""
    dataset = Path(dataset_path).expanduser()
    if not dataset.is_dir():
        raise ValueError(f"Dataset must be a directory containing manifest.json and rows.jsonl: {dataset}")
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not _integer(manifest.get("rows"), 1):
        raise ValueError("manifest.rows must be a positive integer")
    files = manifest.get("files")
    info = files.get("rows.jsonl") if isinstance(files, dict) else None
    if not isinstance(info, dict) or not isinstance(info.get("sha256"), str):
        raise ValueError("manifest must include files['rows.jsonl'].sha256")
    path = dataset / "rows.jsonl"
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != info["sha256"]:
        raise ValueError("rows.jsonl SHA256 mismatch against dataset manifest")
    if "bytes" in info and (not _integer(info["bytes"], 0) or len(raw) != info["bytes"]):
        raise ValueError("rows.jsonl byte count mismatch against dataset manifest")
    # Parse the bytes that were hashed, not a second read of a potentially changed index.
    rows = []
    lines = raw.split(b"\n")
    if lines[-1] == b"":
        lines.pop()
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"rows.jsonl: row_index {index}: invalid JSONL") from exc
        if not isinstance(row, dict):
            raise ValueError(f"rows.jsonl: row_index {index}: expected a JSON object")
        rows.append(row)
    if len(rows) != manifest["rows"]:
        raise ValueError("rows.jsonl row count mismatch against dataset manifest")
    seen = set()
    for index, row in enumerate(rows):
        try:
            _validate_row(row)
        except ValueError as exc:
            raise ValueError(f"rows.jsonl: row_index {index}: {exc}") from exc
        if row["id"] in seen:
            raise ValueError(f"Duplicate row id: {row['id']}")
        seen.add(row["id"])
    missing = [row for row in rows if "state" not in row]
    if missing and source_dir is None:
        raise ValueError(f"{len(missing)} rows omit restricted raw text. Supply --source-dir with independently "
                         "downloaded legacy PK text/label JSONL files; partial scores are not allowed. "
                         "No data is downloaded automatically.")
    source_root = Path(source_dir).expanduser().resolve() if missing else None
    cache = {}
    for row in rows:
        if "state" not in row:
            source = row["source"]
            name = _basename(source["file"])
            source_path = (source_root / name).resolve()
            if source_path.parent != source_root:
                raise ValueError(f"Unsafe source path (including symlinks): {name}")
            if name not in cache:
                if not source_path.is_file():
                    raise ValueError(f"Missing source file {name} in --source-dir; partial scores are not allowed")
                cache[name] = _read_jsonl(source_path)
            index = source["row_index"]
            if index >= len(cache[name]):
                raise ValueError(f"{row['id']}: source.row_index {index} is out of range in {name}")
            original = cache[name][index]
            if original.get("label") != source["native_label"]:
                raise ValueError(f"{row['id']}: native_label mismatch in {name} at row_index {index}")
            text = original.get("text")
            if not isinstance(text, str):
                raise ValueError(f"{row['id']}: source text must be a string")
            row["state"] = text if source["text_limit"] is None else text[:source["text_limit"]]
        try:
            digest = hashlib.sha256(canonical_input(row)).hexdigest()
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{row['id']}: input is not finite JSON") from exc
        if digest != row["input_sha256"]:
            raise ValueError(f"{row['id']}: input_sha256 mismatch; source text, order, or prompt fields changed")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/train-jev-pk")
    parser.add_argument("--source-dir", help="Independently downloaded legacy PK text/label JSONL directory")
    parser.add_argument("--out", required=True, help="New fully materialized JSONL file (never overwritten)")
    args = parser.parse_args(argv)
    out = Path(args.out).expanduser()
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {out}")
    destination = out.resolve()
    for root in (args.dataset, args.source_dir):
        if root is not None and destination.is_relative_to(Path(root).expanduser().resolve()):
            raise ValueError("--out must be outside the dataset and source directories")
    rows = load_records(args.dataset, args.source_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    print(f"Materialized {len(rows)} validated rows to {out}")


if __name__ == "__main__":
    main()
