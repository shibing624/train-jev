"""Three independent typed decisions about one state through POST /v1/systemone.

python examples/batch_decisions.py --url http://127.0.0.1:8020
This estimates decisions; it does not issue a refund or call an external tool.
"""
import argparse
import json
import time
import urllib.request

REQUEST = {
    "model": "train-jev-4b",
    "state": "Refund policy: a defective product is eligible if requested within 30 days. "
             "This order cost 49 dollars, the request was made after 10 days, and the product is defective. "
             "The same payment was charged twice. The billing team handles payments and refunds.",
    "questions": {
        "eligible": {"type": "noul", "instructions": "Is the order eligible for a refund under the stated policy?"},
        "route": {"type": "choice", "instructions": "Which team should handle the duplicate payment?",
                  "criteria": {"billing": "payments and refunds", "shipping": "delivery status",
                               "engineering": "software bugs"}},
        "urgency": {"type": "score", "instructions": "Rate the urgency of resolving this customer's payment issue.",
                    "criteria": ["none", "very low", "low", "moderate", "high", "critical"]},
    },
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8020")
    a = parser.parse_args(argv)
    request = urllib.request.Request(a.url.rstrip("/") + "/v1/systemone", json.dumps(REQUEST).encode(),
                                     {"content-type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.load(response)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"client round trip: {1000 * (time.perf_counter() - start):.1f} ms")


if __name__ == "__main__":
    main()
