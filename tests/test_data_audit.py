import gzip
import json

import pytest

from scripts.audit_training_data import (
    audit_heldout,
    audit_training,
    candidate_partition,
    fingerprint,
    input_state,
    iter_records,
    main,
    schema_issues,
    source_provenance,
)


def row(**changes):
    r = {"kind": "choice", "state": "Some state", "question": "Pick one", "options": ["x", "y"],
         "target": [1.0, 0.0], "source": "fixture"}
    r.update(changes)
    return r


def write_rows(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


@pytest.mark.parametrize("changes", [
    {"target": [float("nan"), 0]}, {"target": [-1, 2]}, {"target": [0.1, 0.1]},
    {"target": [1]}, {"kind": "unknown"}, {"options": list(range(17)), "target": [1 / 17] * 17},
    {"kind": "noul", "options": ["true", "false"]}, {"kind": "score"},
])
def test_schema_rejects_bad_targets_and_slots(changes):
    assert not schema_issues(row())
    assert schema_issues(row(**changes))
    assert schema_issues({"state": "missing other fields"})


def test_normalized_duplicates_and_target_disagreement(tmp_path):
    path = write_rows(tmp_path / "train.jsonl", [row(), row(), row(target=[0, 1]), row(question="Another")])
    report, prompts, states = audit_training(path)
    assert report["rows"] == 4
    assert report["normalized_repeated_prompts"] == 2
    assert report["normalized_prompt_target_differences"] == 1
    assert report["normalized_prompt_argmax_differences"] == 1
    assert len(prompts) == 2 and len(states) == 1
    assert fingerprint(" A  STATE ") == fingerprint("a state")


@pytest.mark.parametrize("value", [
    {"state": "s"}, {"text": "s"}, {"payload": {"body": {"state": "s"}}},
    {"body": {"state": "s"}}, {"request": {"state": "s"}},
])
def test_input_formats_are_actually_parsed(value):
    assert input_state(value) == "s"


def test_prediction_metadata_is_missing_evidence(tmp_path):
    with pytest.raises(ValueError, match="no supported input"):
        input_state({"id": "answer", "p": [0.1, 0.9]})
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps([{"id": "answer", "p": [0.1, 0.9]}]))
    report, states = audit_heldout(path, set(), set(), set(), set())
    assert report["unsupported_rows"] == 1
    assert report["input_coverage"] == "incomplete"
    assert not states


def test_empty_context_is_valid_but_not_a_shared_state(tmp_path):
    path = write_rows(tmp_path / "input.jsonl", [row(state="")])
    report, states = audit_heldout(path, set(), {fingerprint("")}, set(), set())
    assert report["input_coverage"] == "complete"
    assert report["empty_state_rows"] == 1
    assert report["train_shared_unique_states"] == 0
    assert report["complete_prompt_rows"] == 1
    assert not states


def test_state_overlap_is_not_automatically_prompt_overlap(tmp_path):
    train = write_rows(tmp_path / "train.jsonl", [row()])
    _, prompts, states = audit_training(train)
    heldout = write_rows(tmp_path / "heldout.jsonl", [row(question="Different question")])
    report, _ = audit_heldout(heldout, prompts, states, set(), set())
    assert report["train_shared_unique_states"] == 1
    assert report["train_state_matched_rows"] == 1
    assert report["train_shared_complete_prompts"] == 0
    heldout = write_rows(heldout, [row()])
    report, _ = audit_heldout(heldout, prompts, states, set(), set())
    assert report["train_shared_complete_prompts"] == 1


def test_gzip_and_malformed_records(tmp_path):
    path = tmp_path / "input.jsonl.gz"
    with gzip.open(path, "wt") as f:
        f.write(json.dumps({"text": "s"}) + "\n")
    assert list(iter_records(path)) == [(1, {"text": "s"})]
    path = tmp_path / "bad.jsonl"
    path.write_text("{malformed}\n")
    with pytest.raises(ValueError, match="malformed JSON"):
        list(iter_records(path))
    path = tmp_path / "wrong-root.json"
    path.write_text('{}')
    with pytest.raises(ValueError, match="record list"):
        list(iter_records(path))


def test_declared_origin_is_not_an_independent_verification(tmp_path):
    raw = {"state": "s", "questions": {"q": {"src": "example"}},
           "_meta": {"repo": "example/data", "revision": "abc", "split": "test"}}
    path = write_rows(tmp_path / "train.jsonl", [raw])
    report = source_provenance(path)
    assert report["metadata_fields"]["split"] == 1
    assert report["independently_verified"] is False
    origin = json.loads(next(iter(report["declared_origins"])))
    assert origin["split"] == "test"


def test_cli_indices_are_grouped_and_never_approved(tmp_path):
    train = write_rows(tmp_path / "train.jsonl", [row(state="training state")])
    gold = write_rows(tmp_path / "gold.jsonl", [row(state="shared"), row(state="shared", question="Second"),
                                               row(state="heldout state"), {"state": "invalid"}])
    heldout = write_rows(tmp_path / "test.jsonl", [{"text": "heldout state"}])
    out = tmp_path / "audit"
    before = (train.read_bytes(), gold.read_bytes(), heldout.read_bytes())
    main(["--train", str(train), "--gold", str(gold), "--heldout", str(heldout), "--out", str(out)])
    report = json.loads((out / "audit_report.json").read_text())
    indices = [json.loads(x) for x in (out / "candidate_gold_indices.jsonl").read_text().splitlines()]
    assert report["status"] == "audit_only_not_training_approval"
    assert report["semantic_labels_verified"] is False
    assert indices[0]["partition"] == indices[1]["partition"]
    assert indices[2]["partition"] == "quarantine_heldout_overlap"
    assert indices[3]["partition"] == "quarantine_invalid_schema"
    assert all(i["approved"] is False for i in indices)
    assert candidate_partition(" SHARED ", 0) == candidate_partition("shared", 0)
    assert before == (train.read_bytes(), gold.read_bytes(), heldout.read_bytes())
