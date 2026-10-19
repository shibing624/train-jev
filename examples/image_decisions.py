"""Send a door-camera image using TypeSafe assets: multipart (default) or JSON/base64.

Start the server with --vision, then:
    python examples/image_decisions.py --image door-camera.jpg
    python examples/image_decisions.py --image door-camera.jpg --transport base64
"""
import argparse
import base64
import json
import uuid
from pathlib import Path

import httpx


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8020")
    parser.add_argument("--transport", choices=["multipart", "base64"], default="multipart")
    args = parser.parse_args(argv)
    extension = args.image.suffix.lower()
    if extension not in {".png", ".jpg", ".jpeg"}:
        parser.error("--image must be a PNG or JPEG file")
    reference = str(uuid.uuid4())
    request = {
        "model": "train-jev-4b-lora",
        "state": {"door": reference},
        "assets": {reference: {"type": "image"}},
        "questions": {
            "person_present": {"type": "noul", "instructions": "Is there a person in `door`?"},
            "carrying": {"type": "choice", "instructions": "What is the person in `door` carrying?",
                         "criteria": {"package": "A box or parcel.", "bag": "Groceries or a shopping bag.",
                                      "nothing": "Empty hands."}},
        },
    }
    headers = {"TypeSafe-Danger-Zone-Images": "true"}
    url = args.url.rstrip("/") + "/v1/systemone"
    if args.transport == "base64":
        request["assets"][reference].update(
            format="png" if extension == ".png" else "jpg",
            data_base64=base64.b64encode(args.image.read_bytes()).decode("ascii"),
        )
        response = httpx.post(url, headers=headers, json=request, timeout=120)
    else:
        request["assets"][reference]["part"] = "door-photo"
        content_type = "image/png" if extension == ".png" else "image/jpeg"
        with args.image.open("rb") as image:
            response = httpx.post(url, headers=headers, timeout=120, files=[
                ("request", (None, json.dumps(request), "application/json")),
                ("door-photo", (args.image.name, image, content_type)),
            ])
    response.raise_for_status()
    result = response.json()["answers"]
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
