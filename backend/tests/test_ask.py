"""Diagram questions stay on the picture, and tests never call a real model."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from pipeline.ask import REFUSAL, system_prompt


def _client(tmp_path: Path, **overrides: str) -> TestClient:
    fields: dict[str, object] = {
        "data_dir": tmp_path / "data",
        "demo_cache_dir": tmp_path / "demo",
        "dpi": 72,
        "openai_api_key": "",
        "vlm_api_key": "",
        "vlm_provider": "",
    }
    fields.update(overrides)
    settings = Settings(**fields)  # type: ignore[arg-type]
    return TestClient(create_app(settings))


def test_system_prompt_refuses_anything_outside_the_diagram() -> None:
    prompt = system_prompt()
    assert REFUSAL in prompt
    assert "only" in prompt
    assert "why is there blood in the heart" in prompt


def test_missing_key_does_not_call_the_model(tmp_path: Path, monkeypatch) -> None:
    def explode(*_args, **_kwargs):
        raise AssertionError("the model was called")

    monkeypatch.setattr("pipeline.ask.httpx.Client", explode)
    client = _client(tmp_path)
    response = client.post(
        "/api/ask",
        json={"question": "why is there blood in the heart", "subject": "Human heart", "parts": []},
    )
    assert response.status_code == 200
    assert response.json() == {"answer": "", "source": "unavailable"}


def test_model_answer_is_limited_to_the_diagram(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "choices": [
                    {"message": {"content": "Blood is in the heart because the heart is what pumps it around the body."}}
                ]
            }

    class FakeClient:
        def __init__(self, *_args, **kwargs) -> None:
            captured["headers"] = kwargs.get("headers")
            captured["base_url"] = kwargs.get("base_url")

        def __enter__(self):
            return self

        def __exit__(self, *_exc) -> bool:
            return False

        def post(self, url: str, json: dict[str, object] | None = None) -> FakeResponse:
            captured["url"] = url
            captured["json"] = json
            return FakeResponse()

    monkeypatch.setattr("pipeline.ask.httpx.Client", FakeClient)
    client = _client(
        tmp_path,
        openai_api_key="sk-test",
        openai_base_url="https://api.openai.com/v1",
        openai_answer_model="gpt-4o-mini",
        vlm_api_key="vision-key-must-not-be-used",
    )
    response = client.post(
        "/api/ask",
        json={
            "question": "why is there blood in the heart",
            "subject": "Human heart anatomy",
            "summary": "The heart moves blood.",
            "parts": [
                {
                    "name": "Right ventricle",
                    "description": "Pumps blood toward the lungs.",
                }
            ],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "model"
    assert "pumps it" in body["answer"]
    assert captured["headers"] == {"authorization": "Bearer sk-test"}
    sent = captured["json"]
    assert isinstance(sent, dict)
    messages = sent["messages"]
    assert isinstance(messages, list)
    assert REFUSAL in messages[0]["content"]
    assert "Right ventricle" in messages[1]["content"]
    assert "why is there blood in the heart" in messages[1]["content"]
