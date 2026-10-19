"""Bounded TypeSafe image request parsing and reference expansion, without inference."""

import base64
import binascii
import io
import json
import math
import re
import struct
import warnings
import zlib
from dataclasses import dataclass, field
from typing import Any

from PIL import Image, ImageOps

VISION_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"
MARKER_PREFIX = "\ue000trainjev_image_"
MAX_BODY_BYTES = 32 * 1024 * 1024
MAX_IMAGE_BYTES = 16 * 1024 * 1024
MAX_PROCESSED_BYTES = 256 * 1024 * 1024
MAX_EXPANDED_BYTES = 256 * 1024 * 1024
MAX_DECODED_PIXELS = 89_478_485
MAX_IMAGES = 16
MAX_OCCURRENCES = 64
MAX_REF_BYTES = 1024
MAX_TOTAL_REF_BYTES = 65536
MAX_DEPTH = 64
_FORMATS = {"png": "PNG", "jpg": "JPEG", "jpeg": "JPEG", "webp": "WEBP", "gif": "GIF", "bmp": "BMP",
            "tif": "TIFF", "tiff": "TIFF", "ico": "ICO", "avif": "AVIF", "mpo": "JPEG"}


class AssetError(ValueError):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass
class AssetImage:
    image: Image.Image
    data_url: str
    processed_bytes: int


@dataclass
class AssetBundle:
    markers: dict[str, AssetImage] = field(default_factory=dict)

    def parts(self, text: str) -> list[str | AssetImage]:
        if not self.markers:
            return [text] if text else []
        pattern = re.compile("|".join(re.escape(marker) for marker in self.markers))
        result: list[str | AssetImage] = []
        start = 0
        for match in pattern.finditer(text):
            if match.start() > start:
                result.append(text[start:match.start()])
            result.append(self.markers[match.group()])
            start = match.end()
        if start < len(text):
            result.append(text[start:])
        return result

    def render(self, text: str) -> tuple[str, list[Image.Image]]:
        chunks, images = [], []
        for part in self.parts(text):
            if isinstance(part, str):
                chunks.append(part)
            else:
                chunks.append(VISION_TOKEN)
                images.append(part.image)
        return "".join(chunks), images

    def has_images(self, text: str) -> bool:
        return any(marker in text for marker in self.markers)

    def close(self) -> None:
        for asset in self.markers.values():
            asset.image.close()


def _validate_json(value: Any, depth: int = 0, *, reject_markers: bool = False) -> None:
    if isinstance(value, (dict, list)):
        if depth >= MAX_DEPTH:
            raise AssetError("JSON nesting exceeds 64 levels")
        if isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(key, str):
                    raise AssetError("JSON object keys must be strings")
                _validate_json(key, depth + 1, reject_markers=reject_markers)
                _validate_json(item, depth + 1, reject_markers=reject_markers)
        else:
            for item in value:
                _validate_json(item, depth + 1, reject_markers=reject_markers)
    elif isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeError:
            raise AssetError("JSON strings must be valid UTF-8") from None
        if reject_markers and any(token in value for token in
                                  (MARKER_PREFIX, "<|vision_start|>", "<|vision_end|>", "<|image_pad|>")):
            raise AssetError("Image requests cannot contain internal image tokens")
    elif value is None or isinstance(value, (bool, int)):
        pass
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise AssetError("JSON numbers must be finite")
    else:
        raise AssetError("Request must contain only JSON values")


def _json_body(raw: bytes) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise AssetError("Duplicate JSON object key")
            result[key] = value
        return result

    def nonfinite(_):
        raise AssetError("JSON numbers must be finite")

    try:
        text = raw.decode("utf-8")
        if text.startswith("\ufeff"):
            raise AssetError("JSON must not contain a UTF-8 BOM")
        body = json.loads(text, object_pairs_hook=pairs, parse_constant=nonfinite)
        if not isinstance(body, dict):
            raise AssetError("Request JSON must be an object")
        _validate_json(body)
        return body
    except AssetError:
        raise
    except (ValueError, UnicodeError, RecursionError, OverflowError):
        raise AssetError("Malformed UTF-8 JSON request") from None


async def _bounded_body(req) -> bytes:
    length = req.headers.get("content-length")
    if length is not None:
        try:
            size = int(length)
        except ValueError:
            raise AssetError("Invalid Content-Length") from None
        if size < 0:
            raise AssetError("Invalid Content-Length")
        if size > MAX_BODY_BYTES:
            raise AssetError("Request body exceeds 32 MiB", 413)
    raw = bytearray()
    async for chunk in req.stream():
        if len(raw) + len(chunk) > MAX_BODY_BYTES:
            raise AssetError("Request body exceeds 32 MiB", 413)
        raw.extend(chunk)
    return bytes(raw)


def _image_content_type(content_type: str | None) -> str:
    if content_type is not None and not isinstance(content_type, str):
        raise AssetError("Invalid image part content type")
    media = (content_type or "").split(";", 1)[0].strip().lower()
    if media not in {"application/octet-stream", ""} and not media.startswith("image/"):
        raise AssetError("Unsupported image part content type")
    return media


async def read_request(req) -> tuple[dict, dict[str, tuple[bytes, str | None]]]:
    """Read JSON or multipart from a bounded stream; close even partially parsed files."""
    encoding = req.headers.get("content-encoding", "identity").strip().lower()
    if encoding != "identity":
        raise AssetError("Unsupported Content-Encoding", 415)
    media = req.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if media not in {"application/json", "multipart/form-data"}:
        raise AssetError("Use application/json or multipart/form-data", 415)
    raw = await _bounded_body(req)
    if media == "application/json":
        return _json_body(raw), {}

    from starlette.datastructures import UploadFile
    from starlette.formparsers import MultiPartException, MultiPartParser

    class StrictParser(MultiPartParser):
        def __init__(self):
            super().__init__(req.headers, stream(), max_files=MAX_IMAGES + 1,
                             max_fields=MAX_IMAGES + 1, max_part_size=MAX_BODY_BYTES)
            self.names = set()
            self.uploads = []
            self.complete = False

        def on_headers_finished(self):
            super().on_headers_finished()
            part = self._current_part
            if part.file is not None:
                self.uploads.append(part.file)
            name = part.field_name
            if not name or not name.strip() or name in self.names:
                raise MultiPartException("Part names must be unique and nonempty")
            if not self.names and name != "request":
                raise MultiPartException("The first part must be named request")
            self.names.add(name)
            if len(self.names) > MAX_IMAGES + 1:
                raise MultiPartException("Too many multipart parts")
            if name != "request" and part.file is None:
                raise MultiPartException("Image parts must be file uploads")

        def on_part_end(self):
            part = self._current_part
            if part.file is None:
                try:
                    text = bytes(part.data).decode("utf-8")
                except UnicodeError:
                    raise MultiPartException("Request JSON must be UTF-8") from None
                self.items.append((part.field_name, text))
            else:
                super().on_part_end()

        def on_end(self):
            self.complete = True
            super().on_end()

    async def stream():
        for offset in range(0, len(raw), 65536):
            yield raw[offset:offset + 65536]

    parser = StrictParser()
    try:
        form = await parser.parse()
        if not parser.complete or not form or form.multi_items()[0][0] != "request":
            raise AssetError("Malformed multipart request")
        request_part = form.multi_items()[0][1]
        if isinstance(request_part, UploadFile):
            request_raw = await request_part.read()
        else:
            request_raw = request_part.encode("utf-8")
        body = _json_body(request_raw)
        uploads = {}
        for name, part in form.multi_items()[1:]:
            if not isinstance(part, UploadFile):
                raise AssetError("Image parts must be file uploads")
            _image_content_type(part.content_type)
            uploads[name] = (await part.read(), part.content_type)
        return body, uploads
    except AssetError:
        raise
    except Exception:
        raise AssetError("Malformed multipart request") from None
    finally:
        for upload in parser.uploads:
            await upload.close()
        # Some parser versions allocate a file before failing to finish its headers.
        for file in parser._files_to_close_on_error:
            if not file.closed:
                file.close()


def _decode_base64(value: Any) -> bytes:
    if not isinstance(value, str):
        raise AssetError("data_base64 must be a string")
    if len(value) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
        raise AssetError("Image file exceeds 16 MiB", 413)
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error):
        raise AssetError("Invalid standard base64 image data") from None
    # validate=True alone accepts redundant padding and nonzero padding bits.
    if base64.b64encode(raw).decode("ascii") != value:
        raise AssetError("Image base64 must use canonical alphabet and padding")
    if len(raw) > MAX_IMAGE_BYTES:
        raise AssetError("Image file exceeds 16 MiB", 413)
    return raw


def _check_png(raw: bytes, dimensions: tuple[int, int]) -> None:
    decoder = zlib.decompressobj()
    width, height = dimensions
    # Even 16-bit RGBA Adam7 data fits this bound, including row filters/padding.
    remaining = width * height * 8 + height * 14 + 1024
    position = 8
    while position + 12 <= len(raw):
        size = struct.unpack_from(">I", raw, position)[0]
        kind = raw[position + 4:position + 8]
        end = position + size + 12
        if end > len(raw):
            break
        if kind == b"IDAT":
            decoded = decoder.decompress(raw[position + 8:end - 4], remaining + 1)
            remaining -= len(decoded)
            if remaining < 0 or decoder.unused_data:
                raise AssetError("Malformed PNG image data")
        if kind == b"IEND":
            if size != 0 or raw[position + 8:end] != b"\xaeB`\x82" or not decoder.eof:
                break
            return
        position = end
    raise AssetError("Truncated or malformed PNG image")


def _prepare_image(raw: bytes, declared_format: str | None) -> AssetImage:
    if len(raw) > MAX_IMAGE_BYTES:
        raise AssetError("Image file exceeds 16 MiB", 413)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as source:
                actual = "JPEG" if source.format == "MPO" else source.format
                if actual not in set(_FORMATS.values()):
                    raise AssetError("Unsupported raster image format; SVG, PDF and PostScript are not supported")
                if declared_format is not None and _FORMATS[declared_format] != actual:
                    raise AssetError("Declared image format does not match image data")
                if source.width * source.height > MAX_DECODED_PIXELS:
                    raise AssetError("Decoded image exceeds the pixel memory budget", 413)
                if actual == "PNG":
                    _check_png(raw, source.size)
                elif actual == "JPEG" and b"\xff\xd9" not in raw:
                    raise AssetError("Truncated or malformed JPEG image")
                source.verify()
            with Image.open(io.BytesIO(raw)) as source:
                source.seek(0)
                source.load()
                image = ImageOps.exif_transpose(source).convert("RGB")
            encoded = io.BytesIO()
            # A fresh RGB image strips untrusted metadata, including EXIF/ICC/text chunks.
            clean = Image.new("RGB", image.size)
            clean.paste(image)
            image.close()
            clean.save(encoded, format="PNG")
            processed = encoded.getvalue()
        return AssetImage(clean, "data:image/png;base64," + base64.b64encode(processed).decode("ascii"), len(processed))
    except AssetError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise AssetError("Decoded image exceeds the pixel memory budget", 413) from None
    except Exception:
        raise AssetError("Invalid, truncated, or unsupported image data") from None


def prepare_assets(body: dict, uploads: dict[str, tuple[bytes, str | None]]) -> tuple[dict, AssetBundle]:
    """Replace references in state/question values only, preserving caller data and keys."""
    if not isinstance(body, dict) or not isinstance(uploads, dict):
        raise AssetError("Request body and uploads must be objects")
    _validate_json(body)
    assets = body.get("assets", {})
    if not isinstance(assets, dict):
        raise AssetError("assets must be an object")
    if not assets:
        if uploads:
            raise AssetError("Every image upload must be referenced by one asset")
        return body, AssetBundle()
    if len(assets) > MAX_IMAGES:
        raise AssetError("At most 16 image assets are supported", 413)
    _validate_json(body, reject_markers=True)
    total_ref_bytes, used_parts = 0, set()
    references = {}
    for index, (ref, asset) in enumerate(assets.items()):
        size = len(ref.encode("utf-8"))
        if not ref.strip() or not 1 <= size <= MAX_REF_BYTES:
            raise AssetError("Asset references must be nonblank and at most 1024 UTF-8 bytes")
        total_ref_bytes += size
        if total_ref_bytes > MAX_TOTAL_REF_BYTES:
            raise AssetError("Total asset references exceed 65536 UTF-8 bytes")
        if not isinstance(asset, dict) or set(asset) - {"type", "format", "part", "data_base64"}:
            raise AssetError("Invalid image asset fields; URLs and paths are not supported")
        if "type" in asset and asset["type"] != "image":
            raise AssetError("Asset type must be image")
        fmt = asset.get("format")
        if "format" in asset and (not isinstance(fmt, str) or fmt.lower() not in _FORMATS):
            raise AssetError("Image format must name a supported raster image format")
        if "part" in asset:
            part = asset["part"]
            if not isinstance(part, str) or not part.strip() or part == "request":
                raise AssetError("Invalid image part name")
            if part not in uploads:
                raise AssetError("Image asset names a missing upload part")
            if part in used_parts:
                raise AssetError("Each upload part must belong to exactly one asset")
            used_parts.add(part)
        elif "data_base64" not in asset:
            raise AssetError("Image assets require part or data_base64")
        references[ref] = f"{MARKER_PREFIX}{index}\ue001"
    if set(uploads) != used_parts:
        raise AssetError("Every image upload must be referenced by one asset")

    pattern = re.compile("|".join(re.escape(ref) for ref in sorted(references, key=len, reverse=True)))
    used_refs, occurrences = set(), 0

    def replace(ref):
        nonlocal occurrences
        occurrences += 1
        if occurrences > MAX_OCCURRENCES:
            raise AssetError("At most 64 expanded image occurrences are supported", 413)
        used_refs.add(ref)
        return references[ref]

    def normalize(value):
        if isinstance(value, str):
            if value in references:
                return replace(value)
            return pattern.sub(lambda match: replace(match.group()), value)
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        return value

    normalized = dict(body)
    if "state" in body:
        normalized["state"] = normalize(body["state"])
    if isinstance(body.get("questions"), dict):
        questions = {}
        for qid, question in body["questions"].items():
            if isinstance(question, dict):
                question = dict(question)
                for name in ("instructions", "criteria"):
                    if name in question:
                        question[name] = normalize(question[name])
            questions[qid] = question
        normalized["questions"] = questions
    if used_refs != set(assets):
        raise AssetError("Every image asset must occur in state, instructions, or criteria values")

    bundle = AssetBundle()
    try:
        processed_bytes, decoded_pixels = 0, 0
        for ref, asset in assets.items():
            if "part" in asset:
                upload = uploads[asset["part"]]
                if not isinstance(upload, tuple) or len(upload) != 2 or not isinstance(upload[0], bytes):
                    raise AssetError("Invalid image upload")
                raw, content_type = upload
                _image_content_type(content_type)
            else:
                raw = _decode_base64(asset["data_base64"])
            fmt = asset.get("format")
            image = _prepare_image(raw, fmt.lower() if isinstance(fmt, str) else None)
            bundle.markers[references[ref]] = image
            processed_bytes += image.processed_bytes
            decoded_pixels += image.image.width * image.image.height
            if decoded_pixels > MAX_DECODED_PIXELS:
                raise AssetError("Image assets exceed the decoded pixel memory budget", 413)
            if processed_bytes > MAX_PROCESSED_BYTES:
                raise AssetError("Processed image assets exceed 256 MiB", 413)
        rendered_fields = {key: normalized[key] for key in ("state", "questions") if key in normalized}
        try:
            text = json.dumps(rendered_fields, ensure_ascii=False, allow_nan=False)
        except (ValueError, RecursionError, OverflowError):
            raise AssetError("Invalid JSON request values") from None
        rendered, _ = bundle.render(text)
        expanded_bytes = len(rendered.encode("utf-8"))
        expanded_bytes += sum(part.processed_bytes for part in bundle.parts(text) if isinstance(part, AssetImage))
        if expanded_bytes > MAX_EXPANDED_BYTES:
            raise AssetError("Expanded image request exceeds 256 MiB", 413)
        return normalized, bundle
    except BaseException:
        bundle.close()
        raise
