"""Comma-separated vision models fail over on HTTP 429 without waiting."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import numpy as np

from pipeline.understand import VlmConfig, understand

_MODELS = "gemini-3.7-flash,gemini-3.6-flash,gemini-3.5-flash,gemini-3.8-flash"
_VALID = {
    "subject": "Heart",
    "summary": "This is a heart. It has four chambers.",
    "parts": [
        {
            "name": "Left atrium",
            "description": "The upper chamber on the right of the drawing.",
            "importance": 4,
            "bbox": {"x": 0.5, "y": 0.1, "w": 0.3, "h": 0.3},
        }
    ],
}


def _install_client(monkeypatch: object, respond: Callable[[dict[str, object]], httpx.Response]) -> list[float]:
    slept: list[float] = []
    monkeypatch.setattr("pipeline.understand.time.sleep", lambda seconds: slept.append(seconds))  # type: ignore[attr-defined]

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *args: object) -> bool:
            return False

        def post(self, url: str, **kwargs: object) -> httpx.Response:
            payload = kwargs["json"]
            assert isinstance(payload, dict)
            return respond(payload)

    monkeypatch.setattr("pipeline.understand.httpx.Client", FakeClient)  # type: ignore[attr-defined]
    return slept


def _response(status_code: int, payload: dict[str, object]) -> httpx.Response:
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    return httpx.Response(status_code, json=payload, request=request)


def test_429_switches_to_the_next_model(monkeypatch: object) -> None:
    attempted: list[str] = []
    content = json.dumps(_VALID)

    def respond(payload: dict[str, object]) -> httpx.Response:
        model_name = str(payload["model"])
        attempted.append(model_name)
        if model_name == "gemini-3.7-flash":
            return _response(
                429,
                {"error": {"message": "quota exceeded", "status": "RESOURCE_EXHAUSTED"}},
            )
        return _response(200, {"choices": [{"message": {"content": content}}]})

    slept = _install_client(monkeypatch, respond)
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    result = understand(
        image,
        "hash-model-list",
        VlmConfig(
            provider="openai",
            model=_MODELS,
            api_key="test",
            base_url="https://example.test/v1",
        ),
    )

    assert slept == []
    assert attempted == ["gemini-3.7-flash", "gemini-3.6-flash"]
    assert result.source == "vlm"
    assert result.model == "gemini-3.6-flash"
    assert result.parts[0].name == "Left atrium"
