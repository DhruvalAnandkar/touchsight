"""Build the hero images and their precomputed outline cache."""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.images import image_hash  # noqa: E402
from app.schemas import CacheMeta, StageProgress  # noqa: E402
from pipeline.page import TactileConfig  # noqa: E402
from pipeline.run import run_outline  # noqa: E402
from pipeline.understand import VlmConfig, sample_hint  # noqa: E402
from pipeline.types import STAGE_ORDER, StageUpdate  # noqa: E402
from samples.diagrams import animal_cell, heart, world_map, write_source_svgs  # noqa: E402

OUT = ROOT / "demo_cache"


def _plant_cell() -> np.ndarray:
    image = np.full((900, 900, 3), 255, np.uint8)
    ink = (25, 25, 25)
    cv2.circle(image, (450, 450), 340, ink, 8)
    cv2.circle(image, (450, 440), 120, ink, 8)
    cv2.ellipse(image, (280, 300), (90, 48), 25, 0, 360, ink, 8)
    cv2.ellipse(image, (640, 560), (100, 52), -15, 0, 360, ink, 8)
    cv2.circle(image, (600, 280), 48, ink, 8)
    return image


def _bar_chart() -> np.ndarray:
    image = np.full((720, 960, 3), 255, np.uint8)
    colors = [(46, 96, 196), (52, 150, 96), (40, 170, 210), (190, 96, 48)]
    heights = [420, 260, 500, 180]
    for index, (color, height) in enumerate(zip(colors, heights)):
        x = 90 + index * 220
        cv2.rectangle(image, (x, 620 - height), (x + 120, 620), color, thickness=-1)
    return image


def _neighborhood_map() -> np.ndarray:
    image = np.full((780, 960, 3), 248, np.uint8)
    polys = [
        np.array([[80, 80], [460, 60], [500, 360], [120, 420]]),
        np.array([[460, 60], [880, 90], [860, 380], [500, 360]]),
        np.array([[120, 420], [500, 360], [540, 700], [90, 680]]),
        np.array([[500, 360], [860, 380], [820, 700], [540, 700]]),
    ]
    fills = [(86, 122, 176), (78, 158, 104), (186, 140, 72), (164, 92, 138)]
    for poly, color in zip(polys, fills):
        cv2.fillPoly(image, [poly], color)
        cv2.polylines(image, [poly], True, (30, 30, 30), 6)
    return image


def _park_scene() -> np.ndarray:
    image = np.full((780, 960, 3), 255, np.uint8)
    cv2.rectangle(image, (70, 70), (890, 280), (214, 168, 96), thickness=-1)
    cv2.circle(image, (180, 160), 58, (40, 170, 230), thickness=-1)
    cv2.ellipse(image, (700, 150), (120, 46), 0, 0, 360, (250, 250, 252), thickness=-1)
    points = np.array([[250, 620], [470, 300], [700, 620]])
    cv2.fillPoly(image, [points], (70, 130, 70))
    cv2.ellipse(image, (520, 640), (200, 55), 0, 0, 360, (160, 110, 40), thickness=-1)
    return image


SAMPLES = [
    ("heart", "Heart", "Four chambers that share walls.", heart),
    ("animal-cell", "Animal cell", "Cytoplasm, nucleus, nucleolus, and organelles.", animal_cell),
    ("bar-chart", "Bar chart", "Four separated bars, the kind of chart a worksheet uses.", _bar_chart),
    ("world-map", "World map", "Continents on a white ocean. Neighbors share a border.", world_map),
    ("plant-cell", "Plant cell", "A textbook cell: wall, nucleus, and organelles.", _plant_cell),
    ("neighborhood-map", "Neighborhood map", "Four districts that share borders.", _neighborhood_map),
    ("park", "Park", "Sky, sun, cloud, hill, and pond as separate shapes.", _park_scene),
]


def _cached_names(directory: Path) -> str:
    region_map = json.loads((directory / "region_map.json").read_text(encoding="utf-8"))
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    provider = next((stage.get("provider") for stage in meta["stages"] if stage.get("name") == "understand"), "")
    names = ", ".join(region["name"] for region in region_map["regions"])
    return f"provider={provider} names={names}"


def main() -> None:
    images = OUT / "images"
    by_hash = OUT / "by_hash"
    images.mkdir(parents=True, exist_ok=True)
    by_hash.mkdir(parents=True, exist_ok=True)
    write_source_svgs(ROOT / "samples")
    manifest = []
    kept: set[str] = set()
    config = TactileConfig()
    print(f"{'sample':<18} {'regions':>7} {'inline':>6} {'key':>5} {'source':<8} {'render_s':>8}")
    for sample_id, title, description, _draw in SAMPLES:
        payload = (images / f"{sample_id}.png").read_bytes()
        digest = image_hash(payload)
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"could not read {sample_id}")
        stages = [StageProgress(name=name, status="pending") for name in STAGE_ORDER]

        def on_progress(update: StageUpdate, store: list[StageProgress] = stages) -> None:
            for index, stage in enumerate(store):
                if stage.name == update.name:
                    store[index] = StageProgress(
                        name=update.name,
                        status=update.status,
                        elapsed_ms=update.elapsed_ms,
                        message=update.message,
                        provider=update.provider,
                    )

        started = time.perf_counter()
        vlm = VlmConfig(cache_dir=ROOT / "data" / "vlm") if sample_id == "heart" else VlmConfig()
        hint = sample_hint(digest) if sample_id == "heart" else ""
        result = run_outline(
            image,
            config,
            image_hash=digest,
            vlm=vlm,
            hint=hint,
            on_progress=on_progress,
        )
        elapsed = time.perf_counter() - started
        provider = next((stage.provider for stage in stages if stage.name == "understand"), "")
        understand_message = next((stage.message for stage in stages if stage.name == "understand"), "")
        names = ", ".join(f"{region.name}={region.pattern}" for region in result.regions)
        if provider not in {"curated", "vision-llm"}:
            print(f"{sample_id} provider={provider} time={elapsed:.2f}s names={names}", flush=True)
            if (by_hash / digest).is_dir():
                kept.add(digest)
                manifest.append(
                    {
                        "id": sample_id,
                        "title": title,
                        "description": description,
                        "file": f"images/{sample_id}.png",
                        "hash": digest,
                    }
                )
            continue
        kept.add(digest)
        directory = by_hash / digest
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "outline.svg").write_text(result.svg, encoding="utf-8")
        (directory / "tactile.svg").write_text(result.svg, encoding="utf-8")
        if result.pdf:
            (directory / "tactile.pdf").write_bytes(result.pdf)
        if result.stl:
            (directory / "relief.stl").write_bytes(result.stl)
        if result.rectify_before:
            (directory / "rectify-before.jpg").write_bytes(result.rectify_before)
        if result.rectify_after:
            (directory / "rectify-after.jpg").write_bytes(result.rectify_after)
        (directory / "region_map.json").write_text(
            json.dumps(result.region_map(), indent=2),
            encoding="utf-8",
        )
        meta = CacheMeta(
            image_hash=digest,
            region_count=len(result.regions),
            source=result.source,
            stages=stages,
        )
        (directory / "meta.json").write_text(meta.model_dump_json(indent=2), encoding="utf-8")
        manifest.append(
            {
                "id": sample_id,
                "title": title,
                "description": description,
                "file": f"images/{sample_id}.png",
                "hash": digest,
            }
        )
        print(f"{sample_id} provider={provider} note={understand_message} time={elapsed:.2f}s {names}", flush=True)
    earth = ROOT / "samples" / "test" / "earth-layers.png"
    if earth.is_file():
        _cache_earth(earth, by_hash, kept, config)
    else:
        print("earth-layers missing", flush=True)
    for child in by_hash.iterdir():
        if child.is_dir() and child.name not in kept:
            shutil.rmtree(child)
    (OUT / "manifest.json").write_text(json.dumps({"samples": manifest}, indent=2), encoding="utf-8")


def _cache_earth(path: Path, by_hash: Path, kept: set[str], config: TactileConfig) -> None:
    payload = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        print("earth-layers unreadable", flush=True)
        return
    digest = image_hash(payload)
    stages = [StageProgress(name=name, status="pending") for name in STAGE_ORDER]

    def on_progress(update: StageUpdate) -> None:
        for index, stage in enumerate(stages):
            if stage.name == update.name:
                stages[index] = StageProgress(
                    name=update.name,
                    status=update.status,
                    elapsed_ms=update.elapsed_ms,
                    message=update.message,
                    provider=update.provider,
                )

    started = time.perf_counter()
    result = run_outline(image, config, image_hash=digest, vlm=VlmConfig(), on_progress=on_progress)
    elapsed = time.perf_counter() - started
    provider = next((stage.provider for stage in stages if stage.name == "understand"), "")
    names = ", ".join(region.name for region in result.regions)
    print(f"earth-layers provider={provider} time={elapsed:.2f}s names={names}", flush=True)
    if provider != "curated":
        return
    kept.add(digest)
    directory = by_hash / digest
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "outline.svg").write_text(result.svg, encoding="utf-8")
    (directory / "tactile.svg").write_text(result.svg, encoding="utf-8")
    if result.pdf:
        (directory / "tactile.pdf").write_bytes(result.pdf)
    if result.stl:
        (directory / "relief.stl").write_bytes(result.stl)
    (directory / "region_map.json").write_text(json.dumps(result.region_map(), indent=2), encoding="utf-8")
    meta = CacheMeta(
        image_hash=digest,
        region_count=len(result.regions),
        source=result.source,
        stages=stages,
    )
    (directory / "meta.json").write_text(meta.model_dump_json(indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
