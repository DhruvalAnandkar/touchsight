"""Name the picture.

A vision model is used when VLM_PROVIDER and a key are set. VLM_MODEL may be
one model or a comma-separated preference list. HTTP 429, a quota error,
HTTP 500, 502, 503, 504, and timeouts try the next model immediately. The reply has to be this JSON, and a bad
reply is tried once on that same model:

    {"subject", "summary" (two sentences), "parts": [{name, description, importance, bbox}]}

`name` is at most three words and there are at most eight parts. A good reply
is cached by image hash. With no key, or after every model fails, the seven
built-in samples use samples/labels.json. Anything else is left unnamed so
later stages can call the regions "Region N". The fallback is not cached.
"""

from __future__ import annotations

import base64
import hashlib
import inspect
import json
import logging
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import httpx
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from pipeline.http_client import CONNECT_TIMEOUT_S, is_quota_response, request
from pipeline.types import BBox, Part

logger = logging.getLogger("touchsight.pipeline.understand")

_LABELS_PATH = Path(__file__).resolve().parents[1] / "samples" / "labels.json"
_PROMPT = (
    "You are labeling a textbook diagram so a blind student can learn from it by touch. "
    "Reply with JSON only, no markdown. "
    "Identify what the diagram represents (use the context hint, any visible title or text, and the overall structure). "
    'Name each part with the correct subject term a teacher would use (for example "Left ventricle", "Nucleus", "Mantle"), max 3 words. '
    'NEVER name a part by its color, shape, or position (no "blue", "polygon", "section", "upper left", "shape", "oval", "rectangle") '
    "unless the diagram has no meaning beyond shapes. "
    "Each description explains what the part IS or DOES in one sentence for a student, not what it looks like. "
    "The summary explains what the diagram teaches, in 2 sentences. "
    'Schema: {"subject": string, "summary": exactly two sentences, "parts": at most 8 objects. '
    'Each part is {"name": string, "description": one sentence, "importance": integer 1 to 5, "bbox": [x0, y0, x1, y1]}}. '
    "bbox is normalized 0-1, origin at the top-left. "
    "Name the important regions a finger would need, not every speck."
)
_VISUAL_WORDS = frozenset(
    {
        "red",
        "blue",
        "green",
        "yellow",
        "orange",
        "purple",
        "pink",
        "brown",
        "black",
        "white",
        "gray",
        "grey",
        "navy",
        "maroon",
        "burgundy",
        "cyan",
        "magenta",
        "teal",
        "gold",
        "golden",
        "olive",
        "beige",
        "tan",
        "violet",
        "indigo",
        "silver",
        "rust",
        "aqua",
        "lime",
        "crimson",
        "scarlet",
        "turquoise",
        "section",
        "polygon",
        "shape",
        "oval",
        "rectangle",
        "circle",
        "panel",
        "region",
        "area",
    }
)


class UnderstoodPart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=40)
    description: str = Field(min_length=1, max_length=240)
    importance: int = Field(ge=1, le=5)
    bbox: BBox

    @field_validator("name")
    @classmethod
    def at_most_three_words(cls, value: str) -> str:
        words = value.split()
        if not 1 <= len(words) <= 3:
            raise ValueError("name must be at most 3 words")
        return value

    @field_validator("description")
    @classmethod
    def one_sentence(cls, value: str) -> str:
        pieces = [piece for piece in value.split(".") if piece.strip()]
        if len(pieces) != 1:
            raise ValueError("description must be one sentence")
        return value


class Understanding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str = Field(min_length=1, max_length=80)
    summary: str = Field(min_length=1, max_length=500)
    parts: list[UnderstoodPart] = Field(default_factory=list, max_length=8)

    @field_validator("summary")
    @classmethod
    def two_sentences(cls, value: str) -> str:
        pieces = [piece for piece in value.split(".") if piece.strip()]
        if len(pieces) != 2:
            raise ValueError("summary must be two sentences")
        return value

    def to_parts(self) -> list[Part]:
        return [
            Part(
                name=part.name,
                description=part.description,
                importance=part.importance,
                approx_bbox=part.bbox,
            )
            for part in self.parts
        ]


@dataclass
class UnderstandingResult:
    subject: str
    summary: str
    parts: list[Part]
    source: str  # "vlm", "sample", or "generic"
    model: str = ""
    from_cache: bool = False


@dataclass
class VlmConfig:
    provider: str = ""
    model: str = ""
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    cache_dir: Path | None = None
    fetcher: Callable[..., str] | None = None


def cache_key(image_hash: str, hint: str = "") -> str:
    """Same picture with a different hint must not reuse the old labels."""

    text = " ".join(hint.split())
    if not image_hash or not text:
        return image_hash
    return hashlib.sha256(f"{image_hash}\n{text}".encode("utf-8")).hexdigest()


def sample_hint(image_hash: str) -> str:
    entry = _label_entry(image_hash)
    if entry is None:
        return ""
    return " ".join(str(entry.get("hint") or "").split())


def parse_understanding(payload: str) -> Understanding:
    data = json.loads(_strip_fences(payload))
    if isinstance(data, dict):
        _convert_bboxes(data)
    return Understanding.model_validate(data)


def _strip_fences(payload: str) -> str:
    """Gemini often wraps the JSON in a ```json fence."""

    text = payload.strip()
    if not text.startswith("```"):
        return text
    rest = text[3:]
    newline = rest.find("\n")
    if newline == -1:
        rest = rest.removeprefix("json").removeprefix("JSON")
    else:
        rest = rest[newline + 1 :]
    if rest.rstrip().endswith("```"):
        rest = rest.rstrip()[:-3]
    return rest.strip()


def understand(
    image_bgr: np.ndarray,
    image_hash: str,
    vlm: VlmConfig | None = None,
    hint: str = "",
) -> UnderstandingResult:
    vlm = vlm or VlmConfig()
    cached = _read_cache(vlm.cache_dir, cache_key(image_hash, hint))
    if cached is not None:
        return cached
    if _can_call(vlm):
        try:
            parsed, used_model = _call_models(image_bgr, vlm, hint)
        except Exception as exc:
            logger.warning("vision model failed (%s); using fallback labels", exc)
        else:
            key = cache_key(image_hash, hint)
            _write_cache(vlm.cache_dir, key, parsed, used_model)
            return UnderstandingResult(
                parsed.subject,
                parsed.summary,
                parsed.to_parts(),
                "vlm",
                model=used_model,
            )
    sample = _sample_labels(image_hash)
    if sample is not None:
        return sample
    return UnderstandingResult(
        subject="Diagram",
        summary="This picture was split into regions along its outlines. The regions are numbered in reading order.",
        parts=[],
        source="generic",
    )


def _can_call(vlm: VlmConfig) -> bool:
    if vlm.fetcher is not None:
        return True
    return bool(vlm.provider.strip() and vlm.api_key.strip())


def _user_prompt(hint: str, extra: str = "") -> str:
    text = " ".join(hint.split())
    if text:
        base = f"Context hint: {text}"
    else:
        base = "No context hint was given. Read any title or text in the picture."
    if extra:
        return f"{base} {extra}"
    return base


def _visual_names(parsed: Understanding) -> list[str]:
    bad: list[str] = []
    for part in parsed.parts:
        words = set(re.findall(r"[a-z]+", part.name.lower()))
        if words & _VISUAL_WORDS:
            bad.append(part.name)
    return bad


def _model_list(vlm: VlmConfig) -> list[str]:
    """Ordered models from a comma-separated VLM_MODEL string. A single name stays one entry."""

    raw = vlm.model.strip() or os.environ.get("VLM_MODEL", "").strip()
    names = [part.strip() for part in raw.split(",") if part.strip()]
    if names:
        return names
    if vlm.provider.strip().lower() == "anthropic":
        return ["claude-3-5-sonnet-latest"]
    return ["gpt-4o-mini"]


def _accepts_model(fetcher: Callable[..., str]) -> bool:
    try:
        signature = inspect.signature(fetcher)
    except (TypeError, ValueError):
        return False
    positional = 0
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            return True
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            positional += 1
    return positional >= 3


def _complete(vlm: VlmConfig, prompt: str, png: bytes, model: str) -> str:
    if vlm.fetcher is not None:
        if _accepts_model(vlm.fetcher):
            return vlm.fetcher(prompt, png, model)
        return vlm.fetcher(prompt, png)
    return _http_complete(vlm, prompt, png, model)


def _call_models(image_bgr: np.ndarray, vlm: VlmConfig, hint: str) -> tuple[Understanding, str]:
    """Try each model. Quota errors, 500, 502, 503, 504, and timeouts move on immediately."""

    models = _model_list(vlm)
    last_error: Exception | None = None
    for index, model in enumerate(models):
        try:
            return _call_model(image_bgr, vlm, hint, model), model
        except Exception as exc:
            last_error = exc
            if index + 1 < len(models):
                logger.warning("vision model %s failed (%s); trying the next model", model, exc)
    assert last_error is not None
    raise last_error


def _call_model(image_bgr: np.ndarray, vlm: VlmConfig, hint: str, model: str) -> Understanding:
    png = _png_bytes(image_bgr)
    prompt = _user_prompt(hint)
    parsed: Understanding | None = None
    last_error: Exception | None = None
    for _attempt in range(2):
        raw = _complete(vlm, prompt, png, model)
        try:
            parsed = parse_understanding(raw)
            break
        except (ValidationError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            prompt = _user_prompt(hint, "The previous reply was not valid JSON for that schema. Reply with JSON only.")
    if parsed is None:
        assert last_error is not None
        raise last_error
    bad = _visual_names(parsed)
    if not bad:
        return parsed
    retry = _user_prompt(
        hint,
        "These names are visual, not subject terms: "
        + ", ".join(bad)
        + ". Rename every part with the term a teacher would use. Reply with JSON only.",
    )
    try:
        raw = _complete(vlm, retry, png, model)
        retried = parse_understanding(raw)
    except Exception as exc:
        logger.warning("vision label retry failed (%s); keeping the first reply", exc)
        return parsed
    still = _visual_names(retried)
    if still:
        logger.warning("vision labels still use visual names: %s", ", ".join(still))
    return retried


def _http_complete(vlm: VlmConfig, prompt: str, png: bytes, model: str) -> str:
    provider = vlm.provider.strip().lower()
    encoded = base64.b64encode(png).decode("ascii")
    if provider == "anthropic":
        chosen = model.strip() or "claude-3-5-sonnet-latest"
        response = request(
            "POST",
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": vlm.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": chosen,
                "max_tokens": 1200,
                "system": _PROMPT,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {"type": "base64", "media_type": "image/png", "data": encoded},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            },
        )
        body = response.json()
        return str(body["content"][0]["text"])
    if provider in {"openai", "openai-compatible"}:
        return _openai_complete(vlm, prompt, encoded, model)
    raise ValueError(f"unknown vision provider {vlm.provider}")


def _openai_settings(vlm: VlmConfig) -> tuple[str, str, str]:
    base = vlm.base_url.strip() or os.environ.get("VLM_BASE_URL", "").strip() or "https://api.openai.com/v1"
    key = vlm.api_key.strip() or os.environ.get("VLM_API_KEY", "").strip()
    model = _model_list(vlm)[0]
    return base, key, model


def _openai_complete(vlm: VlmConfig, prompt: str, encoded: str, model: str) -> str:
    """OpenAI-compatible chat call.

    HTTP 429, quota errors, HTTP 500, 502, 503, 504, and timeouts are raised
    immediately so the next model can run. There is no wait and no retry on
    the same model for those failures.
    """

    base, key, fallback = _openai_settings(vlm)
    chosen = model.strip() or fallback
    payload = {
        "model": chosen,
        "messages": [
            {"role": "system", "content": _PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
                ],
            },
        ],
    }
    timeout = httpx.Timeout(90.0, connect=CONNECT_TIMEOUT_S)
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            with httpx.Client(
                base_url=base.rstrip("/") + "/",
                timeout=timeout,
                headers={"authorization": f"Bearer {key}"},
            ) as client:
                response = client.post("chat/completions", json=payload)
        except httpx.TimeoutException:
            raise
        except httpx.TransportError as exc:
            last_error = exc
            continue
        if is_quota_response(response.status_code, response.text) or response.status_code in {500, 502, 503, 504}:
            response.raise_for_status()
        if response.status_code >= 500 and attempt == 0:
            last_error = httpx.HTTPStatusError(
                f"HTTP {response.status_code}",
                request=response.request,
                response=response,
            )
            continue
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise exc
        body = response.json()
        content = body["choices"][0]["message"].get("content")
        if isinstance(content, list):
            return "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
        if not content:
            raise ValueError("empty vision reply")
        return str(content)
    assert last_error is not None
    raise last_error


def _png_bytes(image_bgr: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", image_bgr)
    if not ok:
        raise ValueError("could not encode the image")
    return buffer.tobytes()


def _label_entry(image_hash: str) -> dict[str, object] | None:
    if not image_hash or not _LABELS_PATH.is_file():
        return None
    catalog = json.loads(_LABELS_PATH.read_text(encoding="utf-8"))
    entry = catalog.get(image_hash)
    if not isinstance(entry, dict):
        return None
    return entry


def _sample_labels(image_hash: str) -> UnderstandingResult | None:
    entry = _label_entry(image_hash)
    if entry is None:
        return None
    payload = {key: value for key, value in entry.items() if key != "hint"}
    parsed = Understanding.model_validate(payload)
    return UnderstandingResult(parsed.subject, parsed.summary, parsed.to_parts(), "sample")


def _convert_bboxes(data: dict[str, object]) -> None:
    parts = data.get("parts")
    if not isinstance(parts, list):
        return
    for part in parts:
        if isinstance(part, dict) and "bbox" in part:
            part["bbox"] = _xywh(part["bbox"])


def _xywh(bbox: object) -> dict[str, float]:
    if isinstance(bbox, list):
        if len(bbox) != 4:
            raise ValueError("bbox must have four numbers")
        x0, y0, x1, y1 = (_unit(value) for value in bbox)
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        width = x1 - x0
        height = y1 - y0
        if width <= 0 or height <= 0:
            raise ValueError("bbox has no area")
        return {"x": round(x0, 4), "y": round(y0, 4), "w": round(width, 4), "h": round(height, 4)}
    if isinstance(bbox, dict) and {"x", "y", "w", "h"} <= set(bbox):
        return {
            "x": _unit(bbox["x"]),
            "y": _unit(bbox["y"]),
            "w": _unit(bbox["w"]),
            "h": _unit(bbox["h"]),
        }
    raise ValueError("bbox must be [x0, y0, x1, y1] or {x, y, w, h}")


def _unit(value: object) -> float:
    number = float(value)  # type: ignore[arg-type]
    if number < -0.02 or number > 1.02:
        raise ValueError("bbox values must be between 0 and 1")
    return min(max(number, 0.0), 1.0)


def _read_cache(cache_dir: Path | None, image_hash: str) -> UnderstandingResult | None:
    if cache_dir is None or not image_hash:
        return None
    path = cache_dir / f"{image_hash}.json"
    if not path.is_file():
        return None
    parsed = Understanding.model_validate_json(path.read_text(encoding="utf-8"))
    model_path = path.with_suffix(".model")
    model = model_path.read_text(encoding="utf-8").strip() if model_path.is_file() else ""
    return UnderstandingResult(parsed.subject, parsed.summary, parsed.to_parts(), "vlm", model=model, from_cache=True)


def _write_cache(cache_dir: Path | None, image_hash: str, parsed: Understanding, model: str = "") -> None:
    if cache_dir is None or not image_hash:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{image_hash}.json").write_text(parsed.model_dump_json(indent=2), encoding="utf-8")
    if model:
        (cache_dir / f"{image_hash}.model").write_text(model, encoding="utf-8")
