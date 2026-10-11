"""The shipped hero images resolve through the hash cache without recomputing."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.images import image_hash
from app.main import create_app

ROOT = Path(__file__).resolve().parents[1] / "demo_cache"


def test_shipped_samples_match_their_hashes_and_sheets() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    samples = manifest["samples"]
    assert [sample["id"] for sample in samples] == [
        "heart",
        "animal-cell",
        "bar-chart",
        "world-map",
        "plant-cell",
        "neighborhood-map",
        "park",
    ]
    for sample in samples:
        image = (ROOT / sample["file"]).read_bytes()
        assert image_hash(image) == sample["hash"]
        sheet = ROOT / "by_hash" / sample["hash"] / "outline.svg"
        assert b"<polygon" in sheet.read_bytes()


def test_uploading_a_hero_image_is_a_cache_hit(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        demo_cache_dir=ROOT,
        dpi=72,
        sam_api_token="",
        vlm_provider="",
        anthropic_api_key="",
        openai_api_key="",
    )
    client = TestClient(create_app(settings))
    listed = client.get("/api/samples").json()
    assert len(listed) == 7
    assert [sample["id"] for sample in listed[:4]] == [
        "heart",
        "animal-cell",
        "bar-chart",
        "world-map",
    ]
    sample = listed[0]
    image = (ROOT / "images" / f"{sample['id']}.png").read_bytes()
    created = client.post(
        "/api/convert",
        files={"file": (f"{sample['id']}.png", image, "image/png")},
    )
    job = client.get(f"/api/jobs/{created.json()['job_id']}").json()
    assert job["cached"] is True
    assert job["status"] == "complete"
    assert job["region_count"] >= 4
    svg = client.get(job["artifacts"]["outline_svg"])
    assert svg.status_code == 200
    assert b"<polygon" in svg.content
