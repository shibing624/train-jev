"""CPU-only image asset and real multipart/HTTP parsing tests."""

import asyncio
import base64
import io
import json
import struct
import zlib

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from PIL import Image
from starlette.datastructures import Headers, UploadFile

import trainjev.assets as assets_module
from trainjev.assets import (
    MARKER_PREFIX,
    VISION_TOKEN,
    AssetBundle,
    AssetError,
    AssetImage,
    prepare_assets,
    read_request,
)


def image_bytes(fmt="PNG", size=(280, 280), mode="RGB", **kwargs):
    image = Image.new(mode, size, "red" if mode != "L" else 127)
    out = io.BytesIO()
    image.save(out, format=fmt, **kwargs)
    return out.getvalue()


@pytest.fixture(scope="module")
def png():
    return image_bytes()


@pytest.fixture(scope="module")
def jpeg():
    return image_bytes("JPEG")


def image_asset(raw, fmt="png", **kwargs):
    return {"format": fmt, "data_base64": base64.b64encode(raw).decode("ascii"), **kwargs}


def request_body(raw, state="<photo>", **kwargs):
    return {
        "state": state,
        "assets": {"<photo>": image_asset(raw)},
        "questions": {"first": {"type": "noul", "instructions": "Is this red?"}},
        **kwargs,
    }


def prepare(body, uploads=None, enabled=True):
    return prepare_assets(body, uploads or {}, enabled=enabled)


def expect_error(body, uploads=None, status=400, enabled=True):
    with pytest.raises(AssetError) as error:
        prepare(body, uploads, enabled)
    assert error.value.status_code == status
    assert error.value.message == str(error.value)
    return str(error.value)


@pytest.mark.parametrize("overflow", [False, True])
def test_prepared_images_are_closed_if_asset_validation_fails(png, monkeypatch, overflow):
    prepared = []
    original = assets_module._prepare_image

    def record(*args, **kwargs):
        asset = original(*args, **kwargs)
        prepared.append(asset.image)
        return asset

    monkeypatch.setattr(assets_module, "_prepare_image", record)
    body = request_body(png, state=["<photo>", "<bad>"])
    body["assets"]["<bad>"] = {"format": "png", "data_base64": "invalid"}
    if overflow:
        monkeypatch.setattr(assets_module, "MAX_PROCESSED_BYTES", 1)
    expect_error(body, status=413 if overflow else 400)
    assert len(prepared) == 1
    with pytest.raises(ValueError, match="closed"):
        prepared[0].getpixel((0, 0))


@pytest.mark.parametrize("fmt", ["png", "jpg", "jpeg"])
def test_rgb_lossless_png_and_marker_bundle(png, jpeg, fmt):
    raw = png if fmt == "png" else jpeg
    body = request_body(raw)
    body["assets"]["<photo>"]["format"] = fmt
    normalized, bundle = prepare(body)
    marker = f"{MARKER_PREFIX}0\ue001"
    assert normalized["state"] == marker
    assert body["state"] == "<photo>"
    assert normalized["assets"] is body["assets"]
    assert list(bundle.markers) == [marker]
    asset = bundle.markers[marker]
    assert asset.image.mode == "RGB" and asset.image.size == (280, 280)
    encoded = base64.b64decode(asset.data_url.removeprefix("data:image/png;base64,"))
    assert len(encoded) == asset.processed_bytes
    with Image.open(io.BytesIO(encoded)) as decoded:
        assert decoded.format == "PNG"
        assert decoded.tobytes() == asset.image.tobytes()
    assert bundle.parts(f"before {marker}{marker} after") == ["before ", asset, asset, " after"]
    assert bundle.render(f"{marker} text {marker}") == (f"{VISION_TOKEN} text {VISION_TOKEN}",
                                                       [asset.image, asset.image])
    assert bundle.has_images(marker)
    assert not bundle.has_images("<photo>")
    assert bundle.parts("") == []
    assert bundle.parts("plain") == ["plain"]
    assert bundle.parts(marker) == [asset]
    assert bundle.render("") == ("", [])


def test_empty_assets_preserve_identity_and_no_header():
    body = {"state": ["text", {"nested": None}], "questions": {}}
    normalized, bundle = prepare(body, enabled=False)
    assert normalized is body
    assert bundle.markers == {}
    assert bundle.parts("text") == ["text"]
    assert bundle.render("text") == ("text", [])
    assert not bundle.has_images("text")
    expect_error(body, {"dangling": (b"bytes", "image/png")})


def test_header_required_and_optional_type(png):
    body = request_body(png)
    expect_error(body, enabled=False)
    prepare(body)
    body["assets"]["<photo>"]["type"] = "image"
    prepare(body)
    body["assets"]["<photo>"]["type"] = None
    expect_error(body)


@pytest.mark.parametrize("fmt,raw_fmt", [("png", "JPEG"), ("jpg", "PNG"), ("jpeg", "PNG")])
def test_format_mismatch(fmt, raw_fmt):
    body = request_body(image_bytes(raw_fmt))
    body["assets"]["<photo>"]["format"] = fmt
    assert "match" in expect_error(body)


@pytest.mark.parametrize("fmt", ["GIF", "WEBP", "BMP", "TIFF"])
def test_unsupported_actual_format(fmt):
    expect_error(request_body(image_bytes(fmt)))


@pytest.mark.parametrize("value", ["", "PNG", "gif", None, 3, [], {}])
def test_unsupported_declared_format(png, value):
    body = request_body(png)
    body["assets"]["<photo>"]["format"] = value
    expect_error(body)


@pytest.mark.parametrize("mode", ["RGBA", "L", "P"])
def test_supported_modes_convert_rgb(mode):
    raw = image_bytes(mode=mode)
    _, bundle = prepare(request_body(raw))
    assert next(iter(bundle.markers.values())).image.mode == "RGB"


def test_exif_transpose_and_metadata_stripping():
    image = Image.new("RGB", (280, 300), "blue")
    exif = Image.Exif()
    exif[274] = 6
    out = io.BytesIO()
    image.save(out, format="JPEG", exif=exif)
    body = request_body(out.getvalue())
    body["assets"]["<photo>"]["format"] = "jpeg"
    _, bundle = prepare(body)
    asset = next(iter(bundle.markers.values()))
    assert asset.image.size == (300, 280)
    encoded = base64.b64decode(asset.data_url.split(",", 1)[1])
    with Image.open(io.BytesIO(encoded)) as decoded:
        assert not decoded.info
        assert not decoded.getexif()


@pytest.mark.parametrize("single_frame", [False, True])
def test_apng_rejected(single_frame):
    first = Image.new("RGB", (280, 280), "red")
    second = Image.new("RGB", (280, 280), "blue")
    out = io.BytesIO()
    first.save(out, format="PNG", save_all=True, append_images=[] if single_frame else [second], loop=0)
    raw = out.getvalue()
    if single_frame:
        # Pillow saves single-frame save_all as an ordinary PNG; add valid APNG control chunks.
        def chunk(kind, data):
            return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

        animation = chunk(b"acTL", struct.pack(">II", 1, 0))
        animation += chunk(b"fcTL", struct.pack(">IIIIIHHBB", 0, 280, 280, 0, 0, 1, 10, 0, 0))
        raw = raw[:33] + animation + raw[33:]
    with Image.open(io.BytesIO(raw)) as decoded:
        assert decoded.n_frames == (1 if single_frame else 2)
    assert "Animated" in expect_error(request_body(raw))


@pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
def test_truncated_images_rejected(fmt):
    raw = image_bytes(fmt)
    for cut in (2, 20, len(raw) // 2):
        body = request_body(raw[:-cut])
        body["assets"]["<photo>"]["format"] = "png" if fmt == "PNG" else "jpg"
        expect_error(body)


@pytest.mark.parametrize("damage", ["cut_one", "cut_four", "extra", "bomb", "bad_iend"])
def test_png_integrity_beyond_pillow_verify(png, damage):
    chunks, position = [], 8
    while position < len(png):
        size = struct.unpack_from(">I", png, position)[0]
        kind = png[position + 4:position + 8]
        data = png[position + 8:position + 8 + size]
        if kind == b"IDAT":
            if damage == "cut_one":
                data = data[:-1]
            elif damage == "cut_four":
                data = data[:-4]
            elif damage == "extra":
                data += b"extra"
            elif damage == "bomb":
                data = zlib.compress(b"x" * (1024 * 1024))
        checksum = zlib.crc32(kind + data)
        if kind == b"IEND" and damage == "bad_iend":
            checksum ^= 1
        chunks.append(struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum))
        position += size + 12
    expect_error(request_body(png[:8] + b"".join(chunks)))


@pytest.mark.parametrize("value", ["!!!!", "Zg", "Zg=", "Zg===", "Zh==", "Zm8===", "Zm9v=", "Z g==",
                                    "Zg==\n", "Zg-_", "data:image/png;base64,Zg==", "é===", 7, None, []])
def test_invalid_base64_never_exposes_data(png, value):
    body = request_body(png)
    body["assets"]["<photo>"]["data_base64"] = value
    message = expect_error(body)
    assert "data:image" not in message
    if isinstance(value, str) and len(value) > 6:
        assert value not in message


@pytest.mark.parametrize("size,valid", [((279, 280), False), ((280, 279), False), ((1569, 280), False),
                                       ((280, 1569), False), ((280, 1568), True), ((1568, 1568), True)])
def test_dimensions(size, valid):
    body = request_body(image_bytes(size=size))
    if valid:
        _, bundle = prepare(body)
        assert next(iter(bundle.markers.values())).image.size == size
    else:
        expect_error(body)


def test_dimensions_checked_before_pixel_load(monkeypatch):
    raw = image_bytes(size=(279, 280))

    def forbidden(*args, **kwargs):
        pytest.fail("invalid dimensions must be rejected before loading pixels")

    monkeypatch.setattr(Image.Image, "load", forbidden)
    expect_error(request_body(raw))


@pytest.mark.parametrize("ref", ["", " ", "\t\n", "x" * 1025, "界" * 342])
def test_invalid_reference(png, ref):
    expect_error(request_body(png, state=ref, assets={ref: image_asset(png)}))


def test_utf8_reference_byte_limits_and_total(monkeypatch, png):
    ref = "界" * 341
    prepare(request_body(png, state=ref, assets={ref: image_asset(png)}))
    monkeypatch.setattr(assets_module, "MAX_TOTAL_REF_BYTES", 1022)
    expect_error(request_body(png, state=ref, assets={ref: image_asset(png)}))


@pytest.mark.parametrize("entry", [{}, {"format": "png"}, {"data_base64": ""}, {"url": "https://example.com/a.png"},
                                    {"path": "/tmp/a.png"}, {"type": "file", "part": "photo"}, None, []])
def test_invalid_or_remote_asset_entries(entry):
    expect_error({"state": "ref", "assets": {"ref": entry}})


@pytest.mark.parametrize("value", [None, [], "", 0])
def test_assets_must_be_mapping(value):
    expect_error({"assets": value})


def test_leftmost_longest_whole_values_keys_order_and_regex_literal(png):
    refs = ["cat", "category", "[pic]", "a", "ab"]
    body = {
        "state": {"category": ["category", "categories cat", "categorycat", "[pic]", "ababa"]},
        "assets": {ref: image_asset(png) for ref in refs},
        "questions": {},
    }
    normalized, bundle = prepare(body)
    markers = list(bundle.markers)
    assert list(normalized["state"]) == ["category"]
    assert normalized["state"]["category"] == [
        markers[1], markers[0] + "egories " + markers[0], markers[1] + markers[0], markers[2],
        markers[4] + markers[4] + markers[3],
    ]
    assert body["state"]["category"][0] == "category"


def test_instructions_criteria_independence_and_other_fields_unchanged(png):
    refs = ["[state]", "[one]", "[two]"]
    body = {
        "state": {"[one]": ["[state]", {"inner": "context [state]"}]},
        "assets": {ref: image_asset(png) for ref in refs},
        "metadata": "[one]",
        "questions": {
            "[one]": {"type": "choice", "instructions": ["[one]", {"example": "[one]"}],
                      "criteria": {"[two]": {"desc": "compare [one]", "pictures": ["[state]", "[one]"]}},
                      "other": "[two]"},
            "second": {"type": "score", "instructions": "[two]", "criteria": ["[two]", "none"]},
            "plain": {"type": "noul", "instructions": "plain"},
        },
    }
    normalized, bundle = prepare(body)
    a, b, c = bundle.markers
    assert list(normalized) == list(body)
    assert list(normalized["questions"]) == list(body["questions"])
    assert normalized["metadata"] == "[one]"
    assert normalized["questions"]["[one]"]["other"] == "[two]"
    assert list(normalized["questions"]["[one]"]["criteria"]) == ["[two]"]
    first_text = json.dumps(normalized["questions"]["[one]"], ensure_ascii=False)
    assert bundle.has_images(first_text) and b in first_text and c not in first_text
    assert bundle.render(first_text)[0].count(VISION_TOKEN) == 5
    assert normalized["questions"]["second"]["criteria"] == [c, "none"]
    assert not bundle.has_images(json.dumps(normalized["questions"]["plain"], ensure_ascii=False))
    assert normalized["state"]["[one]"] == [a, {"inner": "context " + a}]


@pytest.mark.parametrize("body", [
    {"state": {"ref": "plain"}},
    {"state": "plain", "questions": {"ref": {"type": "noul", "instructions": "plain"}}},
    {"state": "plain", "questions": {"q": {"type": "choice", "criteria": {"ref": "plain"}}}},
    {"state": "plain", "metadata": "ref"},
    {"state": "plain", "questions": {"q": {"type": "ref", "other": "ref"}}},
])
def test_only_eligible_values_count_as_references(png, body):
    body = {**body, "assets": {"ref": image_asset(png)}}
    assert "must occur" in expect_error(body)


@pytest.mark.parametrize("token", [MARKER_PREFIX, VISION_TOKEN, "<|vision_start|>", "<|vision_end|>", "<|image_pad|>"])
@pytest.mark.parametrize("location", ["state", "key", "metadata", "instructions", "criteria"])
def test_internal_token_injection_rejected(png, token, location):
    body = request_body(png)
    if location == "key":
        body[token] = "plain"
    elif location in {"instructions", "criteria"}:
        body["questions"]["first"][location] = token
    else:
        body[location] = token
    assert "internal" in expect_error(body)


def test_part_precedence_and_optional_format(png):
    body = {"state": "ref", "assets": {"ref": {"part": "photo", "data_base64": "NOT VALID BASE64"}}}
    normalized, bundle = prepare(body, {"photo": (png, "image/png")})
    assert bundle.has_images(normalized["state"])
    body["assets"]["ref"]["format"] = "jpeg"
    expect_error(body, {"photo": (png, "image/png")})


@pytest.mark.parametrize("content_type", [None, "", "image/png", "image/jpeg", "image/jpg", "application/octet-stream"])
def test_allowed_upload_content_types(png, content_type):
    prepare({"state": "ref", "assets": {"ref": {"part": "photo"}}}, {"photo": (png, content_type)})


def test_missing_shared_dangling_reserved_and_invalid_uploads(png):
    body = {"state": ["one", "two"], "assets": {"one": {"part": "a"}, "two": {"part": "b"}}}
    expect_error(body, {"a": (png, "image/png")})
    body["assets"]["two"]["part"] = "a"
    expect_error(body, {"a": (png, "image/png")})
    del body["assets"]["two"]
    expect_error(body, {"a": (png, "image/png"), "b": (png, None)})
    expect_error(body, {"a": (png, "image/gif")})
    expect_error(body, {"a": png})
    expect_error(body, {"a": ("not bytes", None)})
    body["assets"]["one"]["part"] = "request"
    expect_error(body, {"request": (png, "image/png")})


@pytest.mark.parametrize("count", [16, 17])
def test_unique_image_count(png, count):
    refs = [f"[{index}]" for index in range(count)]
    body = {"state": refs, "assets": {ref: image_asset(png) for ref in refs}}
    if count == 16:
        _, bundle = prepare(body)
        assert len(bundle.markers) == count
    else:
        expect_error(body, status=413)


@pytest.mark.parametrize("count", [64, 65])
def test_repeated_occurrences_count(png, count):
    body = request_body(png, state="<photo>" * count)
    if count == 64:
        normalized, bundle = prepare(body)
        assert len(bundle.render(normalized["state"])[1]) == 64
    else:
        expect_error(body, status=413)


def test_occurrence_limit_is_shared_across_slots(png):
    body = request_body(png, state=["<photo>"] * 32)
    body["questions"]["first"]["criteria"] = ["<photo>"] * 32
    prepare(body)
    body["questions"]["first"]["instructions"] = "<photo>"
    expect_error(body, status=413)


def test_decoded_file_byte_limit(png, monkeypatch):
    monkeypatch.setattr(assets_module, "MAX_IMAGE_BYTES", len(png))
    prepare(request_body(png))
    too_big = png + b"!"
    expect_error(request_body(too_big), status=413)
    expect_error({"state": "ref", "assets": {"ref": {"part": "p"}}}, {"p": (too_big, None)}, status=413)


def test_real_16_mib_file_limit(png):
    raw = png + b"\0" * (assets_module.MAX_IMAGE_BYTES - len(png))
    prepare({"state": "ref", "assets": {"ref": {"part": "p"}}}, {"p": (raw, None)})
    expect_error({"state": "ref", "assets": {"ref": {"part": "p"}}}, {"p": (raw + b"x", None)}, status=413)


def test_processed_aggregate_limit(png, monkeypatch):
    _, bundle = prepare(request_body(png))
    size = next(iter(bundle.markers.values())).processed_bytes
    monkeypatch.setattr(assets_module, "MAX_PROCESSED_BYTES", size * 2)
    body = {"state": ["one", "two"], "assets": {"one": image_asset(png), "two": image_asset(png)}}
    prepare(body)
    monkeypatch.setattr(assets_module, "MAX_PROCESSED_BYTES", size * 2 - 1)
    expect_error(body, status=413)


def test_expanded_text_and_every_image_use_byte_limit(png, monkeypatch):
    body = request_body(png, state="<photo><photo>")
    normalized, bundle = prepare(body)
    text = json.dumps({key: normalized[key] for key in ("state", "questions")}, ensure_ascii=False)
    rendered, _ = bundle.render(text)
    image = next(iter(bundle.markers.values()))
    exact = len(rendered.encode("utf-8")) + image.processed_bytes * 2
    monkeypatch.setattr(assets_module, "MAX_EXPANDED_BYTES", exact)
    prepare(body)
    monkeypatch.setattr(assets_module, "MAX_EXPANDED_BYTES", exact - 1)
    expect_error(body, status=413)
    monkeypatch.setattr(assets_module, "MAX_EXPANDED_BYTES", exact)
    body["questions"]["first"]["instructions"] += "界"
    expect_error(body, status=413)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), b"not JSON", {1: "key"}, (1, 2)])
def test_direct_json_numeric_and_type_validation(value):
    expect_error({"state": value})


def test_direct_json_depth_and_surrogates():
    value = "text"
    for _ in range(63):
        value = [value]
    prepare({"state": value})
    expect_error({"state": [value]})
    expect_error({"state": "\ud800"})
    expect_error({"\ud800": "text"})


@pytest.fixture
def client():
    app = FastAPI()

    @app.post("/")
    async def endpoint(req: Request):
        try:
            body, uploads = await read_request(req)
            normalized, bundle = prepare_assets(body, uploads,
                                                enabled=req.headers.get("TypeSafe-Danger-Zone-Images") == "true")
            text = json.dumps({key: normalized[key] for key in ("state", "questions") if key in normalized},
                              ensure_ascii=False)
            rendered, images = bundle.render(text)
            return {"body": normalized, "rendered": rendered, "images": len(images),
                    "upload_names": list(uploads), "image_sizes": [list(image.size) for image in images]}
        except AssetError as error:
            return JSONResponse({"error": str(error)}, status_code=error.status_code)

    with TestClient(app) as test_client:
        yield test_client


def test_real_json_endpoint(client, png):
    response = client.post("/", json=request_body(png), headers={"TypeSafe-Danger-Zone-Images": "true"})
    assert response.status_code == 200
    assert response.json()["images"] == 1
    assert response.json()["image_sizes"] == [[280, 280]]
    assert VISION_TOKEN in response.json()["rendered"]
    assert client.post("/", json=request_body(png)).status_code == 400
    assert client.post("/", json={"state": "plain"}).status_code == 200


@pytest.mark.parametrize("raw", [b"", b"{", b"[]", b"null", b'{"x":1,"x":2}', b'{"x":{"a":1,"a":2}}',
                                  b'{"x":NaN}', b'{"x":Infinity}', b'{"x":-Infinity}', b'{"x":1e999}',
                                  b'{"x":"\xff"}', b'\xef\xbb\xbf{}', b'{"x":"\\ud800"}', b'{} trailing',
                                  b'{"x":' + b'[' * 65 + b'0' + b']' * 65 + b'}'])
def test_malformed_json_endpoint(client, raw):
    response = client.post("/", content=raw, headers={"content-type": "application/json"})
    assert response.status_code == 400
    assert "error" in response.json()


@pytest.mark.parametrize("media", ["text/plain", "application/octet-stream", "application/x-www-form-urlencoded", ""])
def test_unsupported_media(client, media):
    response = client.post("/", content=b"{}", headers={"content-type": media})
    assert response.status_code == 415


@pytest.mark.parametrize("encoding", ["gzip", "deflate", "br", "identity, gzip", ""])
def test_unsupported_compression(client, encoding):
    response = client.post("/", content=b"{}", headers={"content-type": "application/json",
                                                       "content-encoding": encoding})
    assert response.status_code == 415


@pytest.mark.parametrize("json_as_file", [False, True])
def test_real_multipart_endpoint(client, png, jpeg, json_as_file):
    body = {"state": ["one", "two", "one"],
            "assets": {"one": {"part": "p1"}, "two": {"type": "image", "part": "p2", "format": "jpeg"}},
            "questions": {"q": {"type": "noul", "instructions": "compare one and two"}}}
    files = [("request", ("request.json" if json_as_file else None, json.dumps(body), "application/json")),
             ("p1", ("one.png", png, "image/png")), ("p2", ("two.jpg", jpeg, "image/jpeg"))]
    response = client.post("/", files=files, headers={"TypeSafe-Danger-Zone-Images": "true"})
    assert response.status_code == 200, response.text
    assert response.json()["images"] == 5
    assert response.json()["upload_names"] == ["p1", "p2"]


@pytest.mark.parametrize("parts", [
    [("photo", ("p.png", b"bytes", "image/png")), ("request", (None, "{}", "application/json"))],
    [("request", (None, "{}", "application/json")), ("request", (None, "{}", "application/json"))],
    [("request", (None, "{}", "application/json")), ("", ("p.png", b"bytes", "image/png"))],
    [("request", (None, "{}", "application/json")), ("p", (None, "text", "image/png"))],
    [("request", (None, "{}", "application/json")), ("p", ("p.gif", b"bytes", "image/gif"))],
    [("request", (None, "{}", "application/json")), ("p", ("p.png", b"bytes", "image/png")),
     ("p", ("p.png", b"bytes", "image/png"))],
    [("request", ("request.json", b'{"x":NaN}', "application/json"))],
    [("request", (None, b'{"x":"\xff"}', "application/json"))],
])
def test_bad_multipart_parts(client, parts):
    response = client.post("/", files=parts, headers={"TypeSafe-Danger-Zone-Images": "true"})
    assert response.status_code == 400


@pytest.mark.parametrize("raw,content_type", [
    (b"", "multipart/form-data"),
    (b"not multipart", "multipart/form-data; boundary=abc"),
    (b'--abc\r\nContent-Disposition: form-data; name="request"\r\n\r\n{}\r\n',
     "multipart/form-data; boundary=abc"),
    (b'--abc\r\nContent-Disposition: form-data\r\n\r\n{}\r\n--abc--\r\n',
     "multipart/form-data; boundary=abc"),
])
def test_malformed_multipart_framing(client, raw, content_type):
    response = client.post("/", content=raw, headers={"content-type": content_type})
    assert response.status_code == 400


def test_multipart_part_count_bound(client, png):
    parts = [("request", (None, "{}", "application/json"))]
    parts.extend((f"p{i}", ("p.png", png, "image/png")) for i in range(17))
    assert client.post("/", files=parts).status_code == 400


@pytest.mark.parametrize("failure", ["none", "json", "type", "duplicate", "incomplete"])
def test_all_upload_files_closed(client, png, monkeypatch, failure):
    closed = []
    original = UploadFile.close

    async def track(self):
        await original(self)
        closed.append(self)

    monkeypatch.setattr(UploadFile, "close", track)
    body = {"state": "ref", "assets": {"ref": {"part": "p"}}}
    request_json = "{bad" if failure == "json" else json.dumps(body)
    parts = [("request", ("request.json", request_json, "application/json")),
             ("p", ("p.png", png, "image/gif" if failure == "type" else "image/png"))]
    if failure == "duplicate":
        parts.append(("p", ("duplicate.png", png, "image/png")))
    if failure == "incomplete":
        raw = (b'--abc\r\nContent-Disposition: form-data; name="request"; filename="r.json"\r\n\r\n'
               + json.dumps(body).encode() + b'\r\n--abc\r\nContent-Disposition: form-data; '
               b'name="p"; filename="p.png"\r\n\r\n' + png)
        response = client.post("/", content=raw, headers={"content-type": "multipart/form-data; boundary=abc"})
    else:
        response = client.post("/", files=parts, headers={"TypeSafe-Danger-Zone-Images": "true"})
    assert response.status_code == (200 if failure == "none" else 400)
    assert len(closed) == (3 if failure == "duplicate" else 2)
    assert all(upload.file.closed for upload in closed)


class StreamingRequest:
    def __init__(self, chunks, headers=None):
        self.headers = Headers(headers or {"content-type": "application/json"})
        self.chunks = chunks
        self.read_chunks = 0

    async def stream(self):
        for chunk in self.chunks:
            self.read_chunks += 1
            yield chunk

    async def body(self):
        pytest.fail("read_request must never perform an unbounded body read")


def test_bounded_stream_ignores_underreported_content_length(monkeypatch):
    monkeypatch.setattr(assets_module, "MAX_BODY_BYTES", 8)
    req = StreamingRequest([b'{"a":', b'1234}', b"never read"],
                           {"content-type": "application/json", "content-length": "2"})
    with pytest.raises(AssetError) as error:
        asyncio.run(read_request(req))
    assert error.value.status_code == 413
    assert req.read_chunks == 2
    req = StreamingRequest([b"{}      "])
    assert asyncio.run(read_request(req)) == ({}, {})


def test_real_32_mib_stream_limit():
    chunk = b" " * (1024 * 1024)
    req = StreamingRequest([b"{}" + chunk[2:]] + [chunk] * 31)
    assert asyncio.run(read_request(req)) == ({}, {})
    req = StreamingRequest([chunk] * 33)
    with pytest.raises(AssetError) as error:
        asyncio.run(read_request(req))
    assert error.value.status_code == 413
    assert req.read_chunks == 33


def test_declared_body_limit_before_stream(client, monkeypatch):
    monkeypatch.setattr(assets_module, "MAX_BODY_BYTES", 4)
    response = client.post("/", content=b"{}", headers={"content-type": "application/json", "content-length": "5"})
    assert response.status_code == 413


@pytest.mark.parametrize("length", ["bad", "-1"])
def test_bad_content_length(length):
    req = StreamingRequest([b"{}"], {"content-type": "application/json", "content-length": length})
    with pytest.raises(AssetError) as error:
        asyncio.run(read_request(req))
    assert error.value.status_code == 400
    assert req.read_chunks == 0


def test_bundle_can_be_constructed_without_request(png):
    with Image.open(io.BytesIO(png)) as image:
        asset = AssetImage(image.convert("RGB"), "data:image/png;base64,", 0)
    bundle = AssetBundle({f"{MARKER_PREFIX}0\ue001": asset})
    assert bundle.render("plain")[0] == "plain"
