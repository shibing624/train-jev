"""CPU-only integrity, materialization, aggregation, and fake inference tests."""
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import evaluate_pk as evaluate
from scripts import prepare_eval_data as prepare


@pytest.mark.parametrize("name,count,available,tasks", [
    ("train-jev-pk", 2494, 1594, 22),
    ("train-jev-classification-pk", 1300, 500, 13),
])
def test_bundled_evaluation_index_integrity(name, count, available, tasks):
    directory = Path(__file__).resolve().parents[1] / "data" / name
    manifest = json.loads((directory / "manifest.json").read_text())
    raw = (directory / "rows.jsonl").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == manifest["files"]["rows.jsonl"]["sha256"]
    rows = [json.loads(line) for line in raw.split(b"\n") if line]
    assert len(rows) == manifest["rows"] == count
    assert len({row["id"] for row in rows}) == count
    assert len({row["task"] for row in rows}) == tasks
    assert sum("state" in row for row in rows) == manifest["available_inputs"] == available
    for row in rows:
        prepare._validate_row(row)
        if "state" in row:
            assert hashlib.sha256(prepare.canonical_input(row)).hexdigest() == row["input_sha256"]
    assert b"apdcephfs" not in raw
    with pytest.raises(ValueError, match="partial scores are not allowed"):
        prepare.load_records(directory)


def make_row(identifier="r0", state="text", kind="choice", n=2, task="public/example"):
    if kind == "noul":
        options = ["false", "true"]
    elif kind == "score":
        options = [str(i) for i in range(6)]
    else:
        options = [f"option {i}" for i in range(n)]
    row = {"id": identifier, "task": task, "eval_group": "sealed", "group_id": f"g/{identifier}",
           "kind": kind, "state": state, "question": "Decide without extra text.", "options": options,
           "target": [0, 1] + [0] * (len(options) - 2),
           "source": {"file": "example.jsonl", "row_index": 0, "native_label": "native", "text_limit": None}}
    row["input_sha256"] = hashlib.sha256(prepare.canonical_input(row)).hexdigest()
    return row


def make_dataset(tmp_path, rows):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
    (dataset / "rows.jsonl").write_bytes(raw)
    manifest = {"name": "fixture", "rows": len(rows), "tasks": sorted({row["task"] for row in rows}),
                "files": {"rows.jsonl": {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}}}
    (dataset / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return dataset


def make_source(tmp_path, records):
    source = tmp_path / "source"
    source.mkdir()
    (source / "example.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    return source


def test_canonical_input_exact_bytes_and_order():
    row = make_row(state={"z": "中文", "a": {"b": 1, "a": 2}})
    expected = ('{"kind":"choice","state":{"z":"中文","a":{"b":1,"a":2}},'
                '"question":"Decide without extra text.","options":["option 0","option 1"]}').encode("utf-8")
    assert prepare.canonical_input(row) == expected
    row["target"] = [1, 0]
    row["id"] = "other"
    assert prepare.canonical_input(row) == expected
    row["state"] = {"a": {"b": 1, "a": 2}, "z": "中文"}
    assert prepare.canonical_input(row) != expected


def test_load_records_preserves_row_and_state_order(tmp_path):
    rows = [make_row("second", state={"z": 1, "a": 2}), make_row("first", kind="score")]
    dataset = make_dataset(tmp_path, rows)
    assert prepare.load_records(dataset) == rows
    assert list(prepare.load_records(dataset)[0]["state"]) == ["z", "a"]


def test_unicode_line_separators_are_not_jsonl_record_boundaries(tmp_path):
    rows = [make_row(state="中文\u2028line\u2029end\x85next")]
    assert prepare.load_records(make_dataset(tmp_path, rows)) == rows


def test_manifest_hash_mismatch(tmp_path):
    dataset = make_dataset(tmp_path, [make_row()])
    with (dataset / "rows.jsonl").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        prepare.load_records(dataset)


@pytest.mark.parametrize("field,value,message", [("rows", 2, "row count"), ("bytes", 1, "byte count")])
def test_manifest_counts(tmp_path, field, value, message):
    dataset = make_dataset(tmp_path, [make_row()])
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if field == "rows":
        manifest[field] = value
    else:
        manifest["files"]["rows.jsonl"][field] = value
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=message):
        prepare.load_records(dataset)


@pytest.mark.parametrize("mutation", ["state", "question", "options", "state_order"])
def test_input_hash_mismatch(tmp_path, mutation):
    row = make_row(state={"z": 1, "a": 2})
    if mutation == "state_order":
        row["state"] = {"a": 2, "z": 1}
    elif mutation == "options":
        row["options"].reverse()
    else:
        row[mutation] = "changed"
    dataset = make_dataset(tmp_path, [row])
    with pytest.raises(ValueError, match="input_sha256 mismatch"):
        prepare.load_records(dataset)


@pytest.mark.parametrize("target", [[0.5, 0.5], [1, 1], [0, 0], [1], [True, False], [float("nan"), 0]])
def test_reject_non_onehot_targets(tmp_path, target):
    row = make_row()
    row["target"] = target
    with pytest.raises(ValueError, match="onehot"):
        prepare.load_records(make_dataset(tmp_path, [row]))


@pytest.mark.parametrize("kind,n", [("unknown", 2), ("choice", 1), ("choice", 257), ("score", 5), ("noul", 3)])
def test_reject_invalid_primitive_ranges(tmp_path, kind, n):
    row = make_row(n=max(2, n))
    row["kind"] = kind
    row["options"] = [str(i) for i in range(n)]
    with pytest.raises(ValueError, match="primitive range"):
        prepare.load_records(make_dataset(tmp_path, [row]))


@pytest.mark.parametrize("kind,n", [("choice", 2), ("choice", 17), ("choice", 256), ("score", 6), ("noul", 2)])
def test_valid_primitive_ranges(tmp_path, kind, n):
    rows = [make_row(kind=kind, n=n)]
    assert prepare.load_records(make_dataset(tmp_path, rows)) == rows


def test_duplicate_ids(tmp_path):
    dataset = make_dataset(tmp_path, [make_row(), make_row()])
    with pytest.raises(ValueError, match="Duplicate row id"):
        prepare.load_records(dataset)


def test_missing_state_requires_source_and_no_partial_scores(tmp_path):
    row = make_row("restricted")
    del row["state"]
    dataset = make_dataset(tmp_path, [make_row("available"), row])
    with pytest.raises(ValueError, match="--source-dir.*partial scores are not allowed"):
        prepare.load_records(dataset)


@pytest.mark.parametrize("limit", [None, 4])
def test_restore_full_source_index_and_slice(tmp_path, limit):
    text = "中文text with suffix"
    row = make_row(state=text if limit is None else text[:limit])
    row["source"].update(row_index=2, text_limit=limit)
    expected = dict(row)
    del row["state"]
    dataset = make_dataset(tmp_path, [row])
    source = make_source(tmp_path, [{"text": "unselected", "label": "different"},
                                    {"text": "also unselected", "label": "native"},
                                    {"text": text, "label": "native"}])
    before = {path: path.read_bytes() for root in (dataset, source) for path in root.iterdir()}
    assert prepare.load_records(dataset, source) == [expected]
    assert all(path.read_bytes() == raw for path, raw in before.items())


def test_source_native_label_mismatch(tmp_path):
    row = make_row()
    del row["state"]
    dataset = make_dataset(tmp_path, [row])
    source = make_source(tmp_path, [{"text": "text", "label": "wrong"}])
    with pytest.raises(ValueError, match="native_label mismatch"):
        prepare.load_records(dataset, source)


def test_source_text_mismatch(tmp_path):
    row = make_row()
    del row["state"]
    dataset = make_dataset(tmp_path, [row])
    source = make_source(tmp_path, [{"text": "wrong", "label": "native"}])
    with pytest.raises(ValueError, match="input_sha256 mismatch"):
        prepare.load_records(dataset, source)


@pytest.mark.parametrize("index", [-1, True, 1.0, 99])
def test_invalid_source_indices(tmp_path, index):
    row = make_row()
    row["source"]["row_index"] = index
    del row["state"]
    dataset = make_dataset(tmp_path, [row])
    source = make_source(tmp_path, [{"text": "text", "label": "native"}])
    with pytest.raises(ValueError, match="row_index"):
        prepare.load_records(dataset, source)


@pytest.mark.parametrize("name", ["../secret.jsonl", "/tmp/secret.jsonl", "sub/file.jsonl", "..\\secret.jsonl",
                                  ".", "..", "C:secret.jsonl"])
def test_source_path_traversal(tmp_path, name):
    row = make_row()
    row["source"]["file"] = name
    del row["state"]
    with pytest.raises(ValueError, match="safe basename"):
        prepare.load_records(make_dataset(tmp_path, [row]), tmp_path)


def test_source_symlink_escape(tmp_path):
    row = make_row()
    del row["state"]
    dataset = make_dataset(tmp_path, [row])
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "outside.jsonl"
    outside.write_text('{"text":"text","label":"native"}\n')
    (source / "example.jsonl").symlink_to(outside)
    with pytest.raises(ValueError, match="Unsafe source path"):
        prepare.load_records(dataset, source)


def test_missing_source_file(tmp_path):
    row = make_row()
    del row["state"]
    dataset = make_dataset(tmp_path, [row])
    with pytest.raises(ValueError, match="Missing source file.*partial scores"):
        prepare.load_records(dataset, tmp_path)


def test_materialize_and_refuse_overwrite(tmp_path):
    rows = [make_row("b", state={"z": "中文", "a": 1}), make_row("a")]
    dataset = make_dataset(tmp_path, rows)
    out = tmp_path / "new" / "full.jsonl"
    argv = ["--dataset", str(dataset), "--out", str(out)]
    prepare.main(argv)
    assert [json.loads(line) for line in out.read_text().splitlines()] == rows
    raw = out.read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        prepare.main(argv)
    assert out.read_bytes() == raw


@pytest.mark.parametrize("script", [prepare, evaluate])
def test_invalid_dataset_creates_no_outputs(tmp_path, monkeypatch, script):
    row = make_row()
    del row["state"]
    dataset = make_dataset(tmp_path, [row])
    out = tmp_path / "new-parent" / "output"
    monkeypatch.setattr(evaluate, "predict_probabilities", lambda *args, **kwargs: pytest.fail("inference called"))
    with pytest.raises(ValueError, match="--source-dir"):
        script.main(["--dataset", str(dataset), "--out", str(out)])
    assert not out.parent.exists()


@pytest.mark.parametrize("script", [prepare, evaluate])
def test_outputs_cannot_modify_dataset_or_source(tmp_path, script):
    dataset = make_dataset(tmp_path, [make_row()])
    with pytest.raises(ValueError, match="outside the dataset"):
        script.main(["--dataset", str(dataset), "--out", str(dataset / "new")])
    assert not (dataset / "new").exists()


def test_macro_not_micro_and_all_rows_totals():
    rows = [make_row("small", task="rules/small")] + [make_row(str(i), task="public/large") for i in range(9)]
    probs = [[0.1, 0.9]] + [[0.9, 0.1]] * 9
    result = evaluate.summarise(rows, probs)
    assert result["rows"] == 10 and result["tasks"] == 2
    assert result["macro"]["accuracy"] == pytest.approx(0.5)
    assert result["all_rows"]["accuracy"] == pytest.approx(0.1)
    assert result["all_rows"]["correct"] == 1 and result["all_rows"]["n"] == 10
    assert result["macro"]["nll"] == pytest.approx((-math.log(0.9) - math.log(0.1)) / 2)
    assert result["macro"]["brier"] == pytest.approx((0.02 + 1.62) / 2)
    assert result["macro"]["ece15"] == pytest.approx(0.5)


def test_noul_true_at_tie_and_other_kinds_argmax():
    assert evaluate.prediction("noul", [0.5, 0.5]) == 1
    assert evaluate.prediction("choice", [0.5, 0.5]) == 0
    assert evaluate.prediction("score", [0.5, 0.5, 0, 0, 0, 0]) == 0
    result = evaluate.metrics([make_row(kind="noul")], [[0.5, 0.5]])
    assert result["accuracy"] == 1 and result["nll"] == pytest.approx(math.log(2))
    assert result["brier"] == 0.5 and result["ece15"] == 0.5


def test_metrics_perfect_and_confidence_bin_boundaries():
    rows = [make_row(str(i)) for i in range(3)]
    assert evaluate.metrics(rows, [[0, 1]] * 3) == {
        "n": 3, "correct": 3, "accuracy": 1, "nll": 0, "brier": 0, "ece15": 0}
    result = evaluate.metrics(rows, [[0.9, 0.1], [0.1, 0.9], [0.1, 0.9]])
    assert result["ece15"] == pytest.approx(abs(0.9 - 2 / 3))


def test_ece_matches_published_upper_inclusive_boundaries():
    from trainjev.metrics import ece

    rows = [make_row("low", n=5), make_row("high", n=5)]
    probabilities = [[0.2] * 5, [0.19, 0.24, 0.19, 0.19, 0.19]]
    expected = ece([0.2, 0.24], [0, 1], bins=15)
    assert evaluate.metrics(rows, probabilities)["ece15"] == pytest.approx(expected)


def test_classification_ece_is_pooled_not_macro():
    rows = [make_row("correct", task="classification/a"), make_row("wrong", task="classification/b")]
    result = evaluate.summarise(rows, [[0.1, 0.9], [0.9, 0.1]])
    assert result["all_rows"]["ece10"] == pytest.approx(0.4)
    assert result["macro"]["ece15"] == pytest.approx(0.5)


@pytest.mark.parametrize("probs", [[], [None], [[0.5, 0.5]] * 2, [[float("nan"), 1]], [[-1, 2]], [[0.1, 0.1]]])
def test_metrics_reject_incomplete_or_invalid_probabilities(probs):
    with pytest.raises(ValueError):
        evaluate.summarise([make_row()], probs)


def fake_backend(monkeypatch):
    import torch

    from trainjev import modeling

    calls = SimpleNamespace(prompts=[], batches=[], heads=[], loads=[], revisions=[])

    class Tokenizer:
        choice_labels = list("ABCDEFGHIJKLMNOPQ")
        pad_token_id = 7

        def __call__(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            calls.prompts.append(text)
            return SimpleNamespace(input_ids=[1] * (2 if "[state] short\n" in text else 4))

    class Head:
        attention_mode = "causal"

        def logits(self, hidden, kind, n):
            calls.heads.append((kind, n, float(hidden[0])))
            return torch.zeros(n) if kind == "noul" else torch.arange(n, dtype=torch.float32)

    def load_model(base, adapter, **kwargs):
        calls.loads.append((base, adapter, kwargs))
        return Tokenizer(), object(), Head()

    def last_hidden(body, ids, pad_id, device, attention="causal"):
        assert pad_id == 7 and device == "cpu" and attention == "causal"
        calls.batches.append([len(tokens) for tokens in ids])
        return torch.tensor([[len(tokens)] for tokens in ids], dtype=torch.float32)

    def resolve_checkpoint(adapter, revision=None):
        calls.revisions.append((adapter, revision))
        return "/private/internal/adapter"

    monkeypatch.setattr(modeling, "load_model", load_model)
    monkeypatch.setattr(modeling, "last_hidden", last_hidden)
    monkeypatch.setattr(modeling, "resolve_checkpoint", resolve_checkpoint)
    monkeypatch.setattr(modeling, "load_temperatures", lambda path: {"noul": 1.0, "choice": 2.0, "score": 4.0})
    return calls


def test_fake_inference_wide_choice_score_saved_temperatures_and_order(monkeypatch):
    calls = fake_backend(monkeypatch)
    rows = [make_row("long", state="long", kind="score"), make_row("short", state="short", kind="noul"),
            make_row("wide", state="wide", n=17)]
    probs, temps = evaluate.predict_probabilities(
        rows, base="fake", adapter="fake/repo", adapter_revision="sealed-tag", device="cpu", batch_tokens=8)
    assert [len(p) for p in probs] == [6, 2, 17]
    assert probs[1] == [0.5, 0.5]
    assert probs[0][1] / probs[0][0] == pytest.approx(math.exp(1 / temps["score"]))
    assert probs[2][1] / probs[2][0] == pytest.approx(math.exp(1 / temps["choice"]))
    assert calls.batches == [[2, 4], [4]]
    assert calls.revisions == [("fake/repo", "sealed-tag")]
    assert calls.loads[0][2]["device"] == "cpu"
    assert "Q) option 16" in calls.prompts[2]
    assert "[options]\n0\n1\n2\n3\n4\n5\n[decision]:" in calls.prompts[0]
    assert all("[target]" not in text and "[gold]" not in text for text in calls.prompts)


def test_overlength_refused_before_any_fake_inference(monkeypatch):
    calls = fake_backend(monkeypatch)
    with pytest.raises(ValueError, match="refusing truncation or partial scores"):
        evaluate.predict_probabilities([make_row()], base="fake", adapter="fake", device="cpu", max_len=3)
    assert calls.batches == []


def test_cuda_is_not_silently_replaced_by_cpu(monkeypatch):
    import torch

    calls = fake_backend(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="--device cpu explicitly"):
        evaluate.predict_probabilities([make_row()], base="fake", adapter="fake")
    assert calls.loads == []


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), True, None])
def test_invalid_saved_temperatures(monkeypatch, temperature):
    from trainjev import modeling

    calls = fake_backend(monkeypatch)
    monkeypatch.setattr(modeling, "load_temperatures", lambda path: {"choice": temperature})
    with pytest.raises(ValueError, match="temperature"):
        evaluate.predict_probabilities([make_row()], base="fake", adapter="fake", device="cpu")
    assert calls.loads == []


def test_evaluation_outputs_and_refuse_completed_overwrite(tmp_path, monkeypatch, capsys):
    rows = [make_row("r1", kind="noul"), make_row("r2", kind="score", task="rules/score")]
    dataset = make_dataset(tmp_path, rows)
    calls = fake_backend(monkeypatch)
    out = tmp_path / "evaluation" / "result"
    argv = ["--dataset", str(dataset), "--out", str(out), "--device", "cpu", "--adapter", "fake/repo"]
    summary = evaluate.main(argv)
    assert "equal-task macro" in capsys.readouterr().out
    assert json.loads((out / "summary.json").read_text()) == summary
    predictions = [json.loads(line) for line in (out / "predictions.jsonl").read_text().splitlines()]
    assert [row["id"] for row in predictions] == ["r1", "r2"]
    assert predictions[0]["correct"] is True
    assert predictions[0]["target"] == [0, 1]
    assert not any(key in row for row in predictions for key in ("state", "question", "options", "source"))
    assert "private/internal" not in "".join(path.read_text() for path in out.iterdir())
    before = len(calls.loads)
    with pytest.raises(FileExistsError, match="overwrite"):
        evaluate.main(argv)
    assert len(calls.loads) == before
