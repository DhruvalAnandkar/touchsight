"""Orchestrate the tactile sheet.

fit → understand → segment → simplify → texture → braille → export.
Relief stays skipped. SAM 2 runs only when a client is passed in and parts exist.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import cv2
import numpy as np

from pipeline.braille import place_labels
from pipeline.page import TactileConfig, fit_to_page, px_to_mm
from pipeline.pdf import try_svg_to_pdf
from pipeline.providers.sam2 import Sam2Client
from pipeline.rectify import correct_exif, rectify
from pipeline.relief import relief_package
from pipeline.segment import segment
from pipeline.sheet import render_sheet
from pipeline.simplify import simplify
from pipeline.texture import assign_from_label_masks, assign_patterns
from pipeline.types import STAGE_ORDER, BBox, Part, SimplifyResult, StageUpdate
from pipeline.understand import UnderstandingResult, VlmConfig, understand

ProgressFn = Callable[[StageUpdate], None]

_SKIPPED: dict[str, str] = {}

_LABEL_PROVIDER = {
    "vlm": "vision-llm",
    "sample": "curated",
    "generic": "fallback",
    "caller": "caller",
}


def run_outline(
    image_bgr: np.ndarray,
    config: TactileConfig,
    *,
    max_regions: int = 8,
    sam: Sam2Client | None = None,
    parts: Sequence[Part] = (),
    image_hash: str = "",
    vlm: VlmConfig | None = None,
    hint: str = "",
    image_bytes: bytes = b"",
    on_progress: ProgressFn | None = None,
) -> SimplifyResult:
    _emit(on_progress, StageUpdate("rectify", "running"))
    started = time.perf_counter()
    oriented = _oriented(image_bgr, image_bytes)
    straightened = rectify(oriented)
    image_bgr = straightened.image
    _emit(
        on_progress,
        StageUpdate(
            "rectify",
            straightened.status if straightened.status == "skipped" else "complete",
            elapsed_ms=_elapsed(started),
            message="page straightened" if straightened.status == "complete" else "no page contour",
            provider="opencv" if straightened.status == "complete" else "skipped",
        ),
    )

    _emit(on_progress, StageUpdate("understand", "running"))
    started = time.perf_counter()
    if parts:
        understood = UnderstandingResult("Diagram", "The caller named these regions. They are matched to the outlines by overlap.", list(parts), "caller")
    else:
        understood = understand(image_bgr, image_hash, vlm, hint)
    if understood.source == "sample" and straightened.transform is not None:
        understood.parts = _remap_parts(understood.parts, straightened.transform, oriented.shape, image_bgr.shape)
    _emit(
        on_progress,
        StageUpdate(
            "understand",
            "complete",
            elapsed_ms=_elapsed(started),
            message=_understand_message(understood),
            provider=_LABEL_PROVIDER.get(understood.source, understood.source),
        ),
    )

    _emit(on_progress, StageUpdate("segment", "running"))
    started = time.perf_counter()
    fitted = fit_to_page(image_bgr, config)
    segmented = segment(
        fitted.image,
        understood.parts,
        max_regions=max_regions,
        sam=sam,
    )
    _emit(
        on_progress,
        StageUpdate(
            "segment",
            "complete",
            elapsed_ms=_elapsed(started),
            message=f"{len(segmented.regions)} regions via {segmented.source}",
            provider="sam2" if segmented.source == "sam2" else "opencv",
        ),
    )

    _emit(on_progress, StageUpdate("simplify", "running"))
    started = time.perf_counter()
    simplified = simplify(segmented, fitted, config)
    _emit(
        on_progress,
        StageUpdate(
            "simplify",
            "complete",
            elapsed_ms=_elapsed(started),
            message=f"{len(simplified.regions)} outlines, {simplified.dropped} dropped",
            provider="local",
        ),
    )

    _emit(on_progress, StageUpdate("texture", "running"))
    started = time.perf_counter()
    patterns = _patterns(simplified.regions, segmented.regions)
    for region in simplified.regions:
        region.pattern = "empty" if region.outline_only else patterns.get(region.id, "empty")
    _emit(
        on_progress,
        StageUpdate(
            "texture",
            "complete",
            elapsed_ms=_elapsed(started),
            message="patterns assigned so neighbors differ",
            provider="local",
        ),
    )

    _emit(on_progress, StageUpdate("braille", "running"))
    started = time.perf_counter()
    labels = place_labels([(region.id, region.name, region.polygon_mm) for region in simplified.regions])
    where = {label.region_id: label.where for label in labels}
    for region in simplified.regions:
        region.label_where = where.get(region.id, "key")
    inline = sum(1 for label in labels if label.where == "inline")
    key = sum(1 for label in labels if label.where == "key")
    _emit(
        on_progress,
        StageUpdate(
            "braille",
            "complete",
            elapsed_ms=_elapsed(started),
            message=f"{inline} inline, {key} in the key",
            provider="local",
        ),
    )

    _emit(on_progress, StageUpdate("relief", "running"))
    started = time.perf_counter()
    try:
        masked = [region for region in segmented.regions if region.mask is not None]
        package = relief_package([region.mask for region in masked], [region.name for region in masked])
        simplified.stl = package.stl
        simplified.heightmap_png = package.heightmap_png
        simplified.region_id_png = package.region_id_png
    except Exception:
        simplified.stl = b""
    if simplified.stl:
        _emit(
            on_progress,
            StageUpdate(
                "relief",
                "complete",
                elapsed_ms=_elapsed(started),
                message="relief.stl",
                provider="local",
            ),
        )
    else:
        _emit(
            on_progress,
            StageUpdate(
                "relief",
                "skipped",
                elapsed_ms=_elapsed(started),
                message="Relief mesh was not built",
                provider="fallback",
            ),
        )

    _emit(on_progress, StageUpdate("export", "running"))
    started = time.perf_counter()
    height, width = fitted.image.shape[:2]
    simplified.image_origin_mm = (
        px_to_mm(fitted.offset_px[0], fitted.dpi),
        px_to_mm(fitted.offset_px[1], fitted.dpi),
    )
    simplified.image_size_mm = (px_to_mm(width, fitted.dpi), px_to_mm(height, fitted.dpi))
    simplified.svg = render_sheet(
        simplified.page_mm,
        simplified.regions,
        simplified.segments,
        labels,
        subject=understood.subject,
        config=config,
    )
    simplified.subject = understood.subject
    simplified.summary = understood.summary
    simplified.label_source = understood.source
    simplified.inline_labels = inline
    simplified.key_labels = key
    simplified.pdf = try_svg_to_pdf(simplified.svg)
    simplified.rectify_before = _thumb(straightened.before)
    simplified.rectify_after = _thumb(straightened.after)
    _emit(
        on_progress,
        StageUpdate(
            "export",
            "complete",
            elapsed_ms=_elapsed(started),
            message="tactile.svg" if simplified.pdf else "tactile.svg (no pdf)",
            provider="local",
        ),
    )
    _assert_stages_covered()
    return simplified


def _assert_stages_covered() -> None:
    expected = set(STAGE_ORDER)
    reported = set(_SKIPPED) | {"rectify", "understand", "segment", "simplify", "texture", "braille", "relief", "export"}
    if expected != reported:
        missing = expected - reported
        raise RuntimeError(f"pipeline stages not reported: {sorted(missing)}")


def _remap_parts(
    parts: Sequence[Part],
    transform: np.ndarray,
    source_shape: tuple[int, ...],
    dest_shape: tuple[int, ...],
) -> list[Part]:
    """Move sample boxes from the original photo into the straightened page."""

    source_h, source_w = source_shape[:2]
    dest_h, dest_w = dest_shape[:2]
    moved: list[Part] = []
    for part in parts:
        box = part.approx_bbox
        corners = np.array(
            [
                [box.x * source_w, box.y * source_h],
                [(box.x + box.w) * source_w, box.y * source_h],
                [(box.x + box.w) * source_w, (box.y + box.h) * source_h],
                [box.x * source_w, (box.y + box.h) * source_h],
            ],
            dtype=np.float32,
        )
        mapped = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), transform).reshape(-1, 2)
        x0 = float(np.clip(mapped[:, 0].min() / dest_w, 0.0, 0.98))
        y0 = float(np.clip(mapped[:, 1].min() / dest_h, 0.0, 0.98))
        x1 = float(np.clip(mapped[:, 0].max() / dest_w, x0 + 0.02, 1.0))
        y1 = float(np.clip(mapped[:, 1].max() / dest_h, y0 + 0.02, 1.0))
        moved.append(part.model_copy(update={"approx_bbox": BBox(x=x0, y=y0, w=x1 - x0, h=y1 - y0)}))
    return moved


def _oriented(image_bgr: np.ndarray, image_bytes: bytes) -> np.ndarray:
    if not image_bytes:
        return image_bgr
    try:
        photo = correct_exif(image_bytes)
    except Exception:
        return image_bgr
    return _limit_side(photo, 2400)


def _limit_side(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    return cv2.resize(
        image,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _thumb(image: np.ndarray) -> bytes:
    small = _limit_side(image, 160)
    ok, buffer = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    if not ok:
        return b""
    return buffer.tobytes()


def _patterns(simplified_regions: Sequence[object], segmented_regions: Sequence[object]) -> dict[str, str]:
    masks = {str(getattr(region, "name", "")): getattr(region, "mask") for region in segmented_regions}
    items: list[tuple[str, np.ndarray, int]] = []
    missing = False
    polygons: list[tuple[str, list[tuple[float, float]]]] = []
    for region in simplified_regions:
        if getattr(region, "outline_only", False):
            continue
        region_id = str(getattr(region, "id", ""))
        polygons.append((region_id, list(getattr(region, "polygon_mm", []))))
        mask = masks.get(str(getattr(region, "name", "")))
        if mask is None:
            missing = True
            continue
        items.append((region_id, mask, int(getattr(region, "importance", 3))))
    if items and not missing:
        return assign_from_label_masks(items)
    return assign_patterns(polygons)


def _understand_message(understood: UnderstandingResult) -> str:
    if understood.source != "vlm":
        return f"{understood.source}, {len(understood.parts)} parts"
    where = "cached" if understood.from_cache else "live"
    name = _pretty_model(understood.model) if understood.model else "vision-llm"
    return f"{name} · {where}"


def _pretty_model(name: str) -> str:
    words: list[str] = []
    for part in name.replace("_", "-").split("-"):
        if not part:
            continue
        if part.replace(".", "").isdigit():
            words.append(part)
        else:
            words.append(part[:1].upper() + part[1:])
    return " ".join(words) or name


def _emit(on_progress: ProgressFn | None, update: StageUpdate) -> None:
    if on_progress is not None:
        on_progress(update)


def _elapsed(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)
