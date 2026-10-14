import importlib.util
import json
import math
import os

import pytest

_spec = importlib.util.spec_from_file_location(
    "pk_benchmark", os.path.join(os.path.dirname(__file__), "..", "scripts", "pk_benchmark.py"))
pk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pk)


def test_wilson_contains_point_estimate():
    lo, hi = pk.wilson(70, 100)
    assert lo < 0.7 < hi and 0.59 < lo < 0.62 and 0.78 < hi < 0.80


def test_mcnemar_exact():
    a = [True] * 10 + [False] * 2
    b = [False] * 10 + [True] * 2
    w, lo, p = pk.mcnemar(a, b)
    assert (w, lo) == (10, 2)
    assert p == pytest.approx(2 * sum(math.comb(12, i) for i in range(3)) / 2 ** 12)
    assert pk.mcnemar([True], [True]) == (0, 0, 1.0)


def test_ece_perfect_and_overconfident():
    assert pk.ece([1.0, 1.0], [1, 1]) == pytest.approx(0.0)
    assert pk.ece([0.95] * 4, [1, 0, 0, 0]) == pytest.approx(0.7)


def test_report_pairs_runs(tmp_path, capsys):
    def write(name, preds):
        p = tmp_path / f"{name}.jsonl"
        with open(p, "w") as f:
            for i, pred in enumerate(preds):
                f.write(json.dumps({"scenario": "ag_news" if i < 2 else "xnli_en", "index": i % 2,
                                    "primitive": "choice", "gold": "a", "pred": pred, "conf": 0.9,
                                    "latency_s": 0.01}) + "\n")
        return str(p)

    a = write("a", ["a", "a", "a", "b"])
    b = write("b", ["a", "b", "b", "b"])
    pk.main(["report", f"A={a}", f"B={b}", "--seen", "ag_news"])
    out = capsys.readouterr().out
    assert "paired rows 4" in out and "ag_news*" in out
    assert "0.750" in out and "0.250" in out


def test_one_builds_systemone_requests(monkeypatch):
    sent = []

    def fake_post(url, body):
        sent.append(body)
        q = body["questions"]["q"]
        if q["type"] == "noul":
            return {"answers": {"q": {"noul": 0.8}}}, 0.01
        opts = list(q["criteria"])
        return {"answers": {"q": {"probabilities": {o: (0.7 if i == 1 else 0.3 / (len(opts) - 1))
                                                    for i, o in enumerate(opts)}}}}, 0.01

    monkeypatch.setattr(pk, "post", fake_post)
    meta = {"name": "s", "primitive": "score", "lang": "en", "labels": ["very_bad", "ok", "good"]}
    r = pk.one("u", "m", meta, 0, {"text": "x" * 5000, "label": "ok"})
    q = sent[-1]["questions"]["q"]
    assert q["instructions"] == pk.SCORE_Q and list(q["criteria"]) == ["very bad", "ok", "good"]
    assert len(sent[-1]["state"]) == 4000 and r["pred"] == "ok"
    meta = {"name": "b", "primitive": "boolean", "lang": "en", "labels": ["false", "true"]}
    r = pk.one("u", "m", meta, 0, {"text": "t", "label": "true"})
    assert sent[-1]["questions"]["q"] == {"type": "noul", "instructions": pk.NOUL_Q}
    assert r["pred"] == "true" and r["conf"] == pytest.approx(0.8)
