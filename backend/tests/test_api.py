"""API tests: upload, poll, cache, and a bad file."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from tests.drawings import bar_chart, encode_png


def _client(tmp_path: Path) -> TestClient:
    settings = Settings(
        data_dir=tmp_path / "data",
        demo_cache_dir=tmp_path / "demo",
        dpi=72,
        max_regions=8,
        sam_api_token="",
        vlm_provider="",
        anthropic_api_key="",
        openai_api_key="",
    )
    return TestClient(create_app(settings))


def test_convert_returns_an_outline_svg(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.post(
        "/api/convert",
        files={"file": ("bars.png", encode_png(bar_chart()), "image/png")},
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    job = client.get(f"/api/jobs/{job_id}")
    assert job.status_code == 200
    body = job.json()
    assert body["status"] == "complete"
    assert body["cached"] is False
    assert body["region_count"] == 4
    assert body["source"] == "opencv"
    names = [stage["name"] for stage in body["stages"]]
    assert names == ["rectify", "understand", "segment", "simplify", "texture", "braille", "relief", "export"]
    assert body["stages"][0]["status"] == "skipped"
    assert body["stages"][1]["status"] == "complete"
    svg = client.get(body["artifacts"]["outline_svg"])
    assert svg.status_code == 200
    assert b"<svg" in svg.content
    assert "image/svg+xml" in svg.headers["content-type"]


def test_second_upload_is_served_from_cache(tmp_path: Path) -> None:
    client = _client(tmp_path)
    payload = {"file": ("bars.png", encode_png(bar_chart()), "image/png")}
    first = client.post("/api/convert", files=payload).json()["job_id"]
    second = client.post("/api/convert", files=payload).json()["job_id"]
    assert first != second
    job = client.get(f"/api/jobs/{second}").json()
    assert job["cached"] is True
    assert job["status"] == "complete"
    assert job["region_count"] == 4


def test_rejects_a_non_image(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.post("/api/convert", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 400


def test_health(tmp_path: Path) -> None:
    assert _client(tmp_path).get("/api/health").json() == {"status": "ok"}
