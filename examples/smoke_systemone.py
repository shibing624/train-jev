"""Qualitative three-case protocol/model smoke check, not an accuracy benchmark.

python examples/smoke_systemone.py --url http://127.0.0.1:8020
Expected labels are declared before inference; failures are printed, never hidden.
"""
import argparse
import json
import math
import urllib.request

REQUEST = {
    "state": "The order arrived undamaged. Billing charged the same order twice. "
             "All users are unable to sign in and the service is completely unavailable. "
             "Severity rubric: 0=no issue, 1=cosmetic, 2=minor single-user issue, 3=major subset affected, "
             "4=most users blocked, 5=all users blocked.",
    "questions": {
        "damage": {"type": "noul", "instructions": "Does the record explicitly say the delivered item is damaged?"},
        "team": {"type": "choice", "instructions": "Which team should handle the duplicate charge?",
                 "criteria": {"billing": "payments and invoices", "shipping": "delivery", "engineering": "software"}},
        "severity": {"type": "score", "instructions": "Rate the sign-in incident using the given severity rubric.",
                     "criteria": ["no issue", "cosmetic", "minor single user", "major subset",
                                  "most users", "all users"]},
    },
}
EXPECTED = {"damage": "false", "team": "billing", "severity": "5"}


def assess(response):
    answers = response["answers"]
    true_probability = answers["damage"]["noul"]
    if not math.isfinite(true_probability) or not 0 <= true_probability <= 1:
        raise ValueError("Invalid noul probability")
    predictions = {"damage": "true" if true_probability >= 0.5 else "false"}
    for name, n in (("team", 3), ("severity", 6)):
        p = answers[name]["probabilities"]
        if (len(p) != n or not all(math.isfinite(v) and 0 <= v <= 1 for v in p.values())
                or abs(sum(p.values()) - 1) > 1e-5):
            raise ValueError(f"Invalid {name} distribution")
        predictions[name] = max(p, key=p.__getitem__)
    return {"protocol_valid": True, "expected": EXPECTED, "predictions": predictions,
            "correct": sum(predictions[k] == v for k, v in EXPECTED.items()), "total": 3,
            "not_a_general_accuracy_estimate": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8020")
    a = parser.parse_args(argv)
    request = urllib.request.Request(a.url.rstrip("/") + "/v1/systemone", json.dumps(REQUEST).encode(),
                                     {"content-type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.load(response)
    print(json.dumps({"check": assess(result), "response": result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
