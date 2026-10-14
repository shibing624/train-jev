"""Minimal client for a train-jev server (or any TypeSafe System One endpoint).

    python -m trainjev.serve --base Qwen/Qwen3.5-4B --adapter runs/jev-4b/best --port 8020
    python examples/client.py --url http://127.0.0.1:8020
"""
import argparse
import json
import time
import urllib.request

REQUEST = {
    "model": "jev-4b",
    "state": "Ticket #4821 from a Pro-plan customer: 'Since this morning every CSV export fails with a 504 after "
             "about 60 seconds. We have a board meeting in two hours and need the Q3 numbers.' "
             "Status page: all systems operational. Export volume today is 3x the weekly average.",
    "questions": {
        "is_outage": {"type": "noul", "instructions": "Is this most likely a platform-wide outage?"},
        "route": {"type": "choice", "instructions": "Which team should handle this ticket first?",
                  "criteria": {"billing": "payments, invoices, plan changes",
                               "infra": "timeouts, capacity, performance of backend jobs",
                               "product": "feature requests and UX feedback",
                               "security": "account compromise, data exposure"}},
        "urgency": {"type": "score", "instructions": "How urgent is this ticket?",
                    "criteria": ["not urgent", "low", "moderate", "high", "very high", "critical"]},
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8020")
    a = ap.parse_args()
    req = urllib.request.Request(a.url + "/v1/systemone", json.dumps(REQUEST).encode(),
                                 {"content-type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.loads(r.read())
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"round trip {1000 * (time.perf_counter() - t):.1f} ms")


if __name__ == "__main__":
    main()
