# -*- coding: utf-8 -*-
"""Audit JEV-schema training data and supplied held-out inputs without generating labels.

Reports normalized state/prompt overlap, target/schema issues and available provenance.
Candidate split indices are not an approval for training; omitted or unparseable held-out
inputs and missing upstream provenance remain explicit limitations.
"""
import argparse
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def canonical(value):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    return " ".join(text.casefold().split())


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def iter_records(path):
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        if not isinstance(data, list):
            raise ValueError(f"{path}: JSON root must be a record list")
        yield from enumerate(data, 1)
        return
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield line_no, json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: malformed JSON") from exc


def input_state(row):
    if not isinstance(row, dict):
        raise ValueError("record must be an object")
    if "state" in row:
        return row["state"]
    if "text" in row:
        return row["text"]
    payload = row.get("payload")
    body = payload.get("body") if isinstance(payload, dict) else None
    for candidate in (body, row.get("body"), row.get("request")):
        if isinstance(candidate, dict) and "state" in candidate:
            return candidate["state"]
    raise ValueError("no supported input state/text; prediction metadata is not an input")


def prompt_fingerprint(row):
    return fingerprint([row["kind"], canonical(row["state"]), canonical(row["question"]), row["options"]])


def schema_issues(row):
    if not isinstance(row, dict) or not {"kind", "state", "question", "options", "target"} <= row.keys():
        return ["missing required fields"]
    kind, opts, target = row["kind"], row["options"], row["target"]
    if not isinstance(opts, list) or not isinstance(target, list):
        return ["options/target must be lists"]
    errors = []
    if kind not in {"noul", "score", "choice"}:
        errors.append("unknown kind")
    if len(opts) != len(target):
        errors.append("target/options length mismatch")
    if kind == "choice" and not 2 <= len(opts) <= 16:
        errors.append("choice outside current training slots")
    if kind == "noul" and (opts != ["false", "true"] or len(target) != 2):
        errors.append("invalid noul slots")
    if kind == "score" and (opts != [str(i) for i in range(6)] or len(target) != 6):
        errors.append("invalid score slots")
    if not target or not all(isinstance(x, (int, float)) and math.isfinite(x) and x >= 0 for x in target):
        errors.append("invalid target values")
    elif abs(sum(target) - 1) > 1e-3:
        errors.append("target not normalized")
    return errors


def audit_training(path):
    source, domain, family, options, placeholders, errors = [Counter() for _ in range(6)]
    prompts, states, targets = set(), set(), {}
    repeated, changed_target, changed_argmax, count = 0, 0, 0, 0
    examples = []
    for line_no, row in iter_records(path):
        count += 1
        invalid = schema_issues(row)
        if invalid:
            errors.update(invalid)
            if len(examples) < 10:
                examples.append({"line": line_no, "issues": invalid})
            continue
        source[str(row.get("source", "<missing>"))] += 1
        domain[str(row.get("domain", "<missing>"))] += 1
        family[str(row.get("family", "<missing>"))] += 1
        options[f"{row['kind']}:{len(row['options'])}"] += 1
        t = row["target"]
        if row.get("source") == "yuri_v1" or (len(t) == 2 and len({round(x, 6) for x in t}) == 1):
            placeholders[row["kind"]] += 1
        key = prompt_fingerprint(row)
        target = tuple(round(x, 8) for x in t)
        repeated += key in prompts
        if key in targets and targets[key] != target:
            changed_target += 1
            old = targets[key]
            changed_argmax += max(range(len(old)), key=old.__getitem__) != max(range(len(t)), key=t.__getitem__)
        prompts.add(key)
        states.add(fingerprint(row["state"]))
        targets.setdefault(key, target)
    report = {"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size, "rows": count,
              "source": dict(source), "domain": dict(domain), "family": dict(family), "kind_options": dict(options),
              "placeholder_heuristic": dict(placeholders), "schema_issues": dict(errors), "issue_examples": examples,
              "normalization": "Unicode casefold and whitespace collapse; matches require review, "
                               "not automatic deletion",
              "normalized_repeated_prompts": repeated, "normalized_prompt_target_differences": changed_target,
              "normalized_prompt_argmax_differences": changed_argmax, "unique_states": len(states),
              "unique_prompts": len(prompts)}
    return report, prompts, states


def audit_heldout(path, train_prompts, train_states, gold_prompts, gold_states):
    states, prompts, unsupported, count = set(), set(), 0, 0
    state_hits_train, state_hits_gold, prompt_rows, empty_states = 0, 0, 0, 0
    examples = []
    for line_no, row in iter_records(path):
        count += 1
        try:
            value = input_state(row)
            if value is None:
                raise ValueError("null state")
            if value == "" or value == [] or value == {}:
                empty_states += 1
            else:
                key = fingerprint(value)
                states.add(key)
                state_hits_train += key in train_states
                state_hits_gold += key in gold_states
            if {"kind", "state", "question", "options"} <= row.keys():
                prompts.add(prompt_fingerprint(row))
                prompt_rows += 1
        except ValueError as exc:
            unsupported += 1
            if len(examples) < 3:
                examples.append({"line": line_no, "reason": str(exc)})
    report = {"path": str(path), "sha256": file_sha256(path), "rows": count, "unique_states": len(states),
              "unsupported_rows": unsupported, "unsupported_examples": examples, "empty_state_rows": empty_states,
              "input_coverage": "complete" if unsupported == 0 else "incomplete",
              "train_shared_unique_states": len(states & train_states),
              "gold_shared_unique_states": len(states & gold_states),
              "train_state_matched_rows": state_hits_train, "gold_state_matched_rows": state_hits_gold,
              "complete_prompt_rows": prompt_rows, "train_shared_complete_prompts": len(prompts & train_prompts),
              "gold_shared_complete_prompts": len(prompts & gold_prompts)}
    return report, states


def source_provenance(path):
    fields, sources, origins = Counter(), Counter(), Counter()
    count = 0
    for _, row in iter_records(path):
        count += 1
        meta = row.get("_meta", {})
        if isinstance(meta, dict):
            fields.update(meta.keys())
            origin = {k: meta[k] for k in ("source", "repo", "revision", "split", "provenance") if k in meta}
            origins[json.dumps(origin, ensure_ascii=False, sort_keys=True)] += 1
        for q in row.get("questions", {}).values():
            sources[json.dumps(q.get("src"), ensure_ascii=False, sort_keys=True)] += 1
    return {"path": str(path), "sha256": file_sha256(path), "records": count, "metadata_fields": dict(fields),
            "question_sources": dict(sources), "declared_origins": dict(origins), "independently_verified": False}


def candidate_partition(state, seed):
    return "dev_candidate" if int(fingerprint([seed, canonical(state)])[:8], 16) % 10 == 0 else "train_candidate"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True,
                        help="Candidate labels; this tool cannot verify their truth")
    parser.add_argument("--heldout", type=Path, nargs="*", default=[])
    parser.add_argument("--pk-suite", type=Path, help="Also audit this directory's *.jsonl text/label records")
    parser.add_argument("--di-rows", type=Path,
                        help="A JSONL or gzip input suite; prediction files are not substitutes")
    parser.add_argument("--source-jsonl", type=Path, nargs="*", default=[], help="Raw gold records with provenance")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    a = parser.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    train, train_prompts, train_states = audit_training(a.train)
    gold, gold_prompts, gold_states = audit_training(a.gold)
    inputs = list(a.heldout)
    if a.pk_suite:
        found = sorted(a.pk_suite.glob("*.jsonl"))
        if not found:
            raise ValueError(f"{a.pk_suite}: no PK input files")
        inputs += found
    if a.di_rows:
        inputs.append(a.di_rows)
    inputs = list(dict.fromkeys(inputs))
    heldout, blocked_states = [], set()
    for path in inputs:
        info, states = audit_heldout(path, train_prompts, train_states, gold_prompts, gold_states)
        heldout.append(info)
        blocked_states |= states
    counts = Counter()
    with (a.out / "candidate_gold_indices.jsonl").open("w") as f:
        for line_no, row in iter_records(a.gold):
            if schema_issues(row):
                part = "quarantine_invalid_schema"
                state = None
            else:
                state = fingerprint(row["state"])
                part = ("quarantine_heldout_overlap" if state in blocked_states
                        else candidate_partition(row["state"], a.seed))
            counts[part] += 1
            f.write(json.dumps({"source_line": line_no, "state_sha256": state, "partition": part,
                                "approved": False}, ensure_ascii=False) + "\n")
    report = {"format_version": 1, "status": "audit_only_not_training_approval", "semantic_labels_verified": False,
              "train": train, "candidate_gold": gold, "heldout_inputs": heldout,
              "gold_vs_train": {"shared_states": len(gold_states & train_states),
                                "shared_prompts": len(gold_prompts & train_prompts)},
              "declared_provenance": [source_provenance(p) for p in a.source_jsonl],
              "candidate_partitions": dict(counts), "seed": a.seed,
              "limitations": ["Only supplied inputs are checked; missing final evaluation sets remain unaudited.",
                              "Unsupported records are missing evidence, never evidence of zero overlap.",
                              "State sharing is not necessarily question/label leakage; prompt sharing is separate.",
                              "Normalization can merge case-sensitive meanings; matches require review.",
                              "Shared-state partitions do not prove source/template/counterfactual-group isolation.",
                              "Empty states have no state fingerprint; question-level overlap needs full prompts.",
                              "Source declarations, licences and semantic correctness require separate verification."]}
    (a.out / "audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"train_rows": train["rows"], "gold_rows": gold["rows"], "candidate_partitions": dict(counts),
                      "unsupported_inputs": sum(x["unsupported_rows"] for x in heldout),
                      "report": str(a.out / "audit_report.json"), "approved_for_training": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
