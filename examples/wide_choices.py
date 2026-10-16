"""Compare wide-choice readouts on the same request, without changing the checkpoint.

python examples/wide_choices.py --url http://127.0.0.1:8020 --compare
Only A–P are trained head rows; extra single/permute rows use the base LM readout.
"""
import argparse
import json
import time
import urllib.request

LABELS = ["duplicate_payment", "refund_request", "card_activation", "lost_card", "delivery_tracking",
          "change_address", "cancel_order", "account_login", "reset_password", "change_email",
          "subscription_upgrade", "subscription_cancel", "invoice_request", "bank_transfer",
          "cash_withdrawal", "currency_exchange", "product_question", "technical_bug",
          "privacy_request", "other"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8020")
    parser.add_argument("--strategy", choices=["single", "tournament", "permute"], default="single")
    parser.add_argument("--compare", action="store_true")
    a = parser.parse_args(argv)
    results = {}
    for strategy in (["single", "tournament", "permute"] if a.compare else [a.strategy]):
        body = {"state": "I was charged twice for the same subscription payment and need the duplicate charge fixed.",
                "strategy": strategy, "questions": {"intent": {"type": "choice",
                "instructions": "Which intent best describes this message?", "criteria": {x: x for x in LABELS}}}}
        request = urllib.request.Request(a.url.rstrip("/") + "/v1/systemone", json.dumps(body).encode(),
                                         {"content-type": "application/json"})
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.load(response)
        p = result["answers"]["intent"]["probabilities"]
        results[strategy] = {"prediction": max(p, key=p.__getitem__),
                             "top5": sorted(p.items(), key=lambda x: -x[1])[:5],
                             "client_ms": 1000 * (time.perf_counter() - start), "server_ms": result.get("latency_ms")}
    print(json.dumps({"options": len(LABELS), "trained_choice_slots": 16, "results": results}, indent=2))


if __name__ == "__main__":
    main()
