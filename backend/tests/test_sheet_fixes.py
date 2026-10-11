"""Heart-mask patterns, Braille margins, and the 2 mm texture inset."""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon

from pipeline.braille import block_size, translate, wrap_lines
from pipeline.page import PAGE_MARGIN_MM, TactileConfig
from pipeline.segment import segment
from pipeline.sheet import fit_title_lines, render_sheet
from pipeline.texture import assign_from_label_masks, inset_ring
from pipeline.types import SimplifiedRegion

HEART = Path(__file__).resolve().parents[1] / "demo_cache" / "images" / "heart.png"
RADIUS_MM = 0.75


def test_heart_chambers_get_four_different_patterns() -> None:
    encoded = np.fromfile(HEART, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    assert image is not None
    result = segment(image)
    assert len(result.regions) == 4
    patterns = assign_from_label_masks(
        [(region.id, region.mask, region.importance) for region in result.regions]
    )
    assert len(patterns) == 4
    assert set(patterns) == {region.id for region in result.regions}
    assert len(set(patterns.values())) == 4, patterns


def test_long_title_drops_to_the_subject_before_wrapping() -> None:
    subject = "Four chambers of the human heart"
    names = "Right atrium, Left atrium, Right ventricle, Left ventricle"
    max_width = 215.9 - 2 * PAGE_MARGIN_MM
    caption = f"{subject}: {names}"
    assert wrap_lines(caption, max_width, max_lines=2) is None
    assert block_size(len(translate(subject)))[0] > max_width
    lines = fit_title_lines(subject, names, max_width)
    assert lines == wrap_lines(subject, max_width, max_lines=2)
    assert len(lines) == 2
    assert "atrium" not in " ".join(lines).lower()


def test_braille_dots_stay_inside_page_margins() -> None:
    config = TactileConfig()
    page_w, page_h = config.page_mm
    margin = config.margin_mm
    subject = "Four chambers of the human heart"
    legend = "Pulmonary trunk and the superior vena cava beside the atrium"
    square = [(40.0, 60.0), (150.0, 60.0), (150.0, 170.0), (40.0, 170.0)]
    region = _region("r1", legend, square, pattern="dots")
    svg = render_sheet(
        (page_w, page_h),
        [region],
        [],
        [],
        subject=subject,
        config=config,
    )
    body = svg.split("</defs>", 1)[-1]
    circles = _circles(body)
    assert circles
    text_budget = page_w - 2 * margin - 8.0 - 2.0
    assert block_size(len(translate(legend)))[0] > text_budget
    for cx, cy, radius in circles:
        assert cx - radius >= margin - 0.02
        assert cx + radius <= page_w - margin + 0.02
        assert cy - radius >= margin - 0.02
        assert cy + radius <= page_h - margin + 0.02


def test_pattern_fill_is_the_two_millimetre_inset() -> None:
    square = [(30.0, 40.0), (120.0, 40.0), (120.0, 140.0), (30.0, 140.0)]
    config = TactileConfig()
    svg = render_sheet(
        config.page_mm,
        [_region("r1", "Chamber", square, pattern="dots")],
        [],
        [],
        subject="Heart",
        config=config,
    )
    filled = _filled_polygons(svg)
    assert len(filled) == 1
    points = filled[0]
    expected = inset_ring(square)
    assert len(points) == len(expected)
    for got, want in zip(points, expected, strict=True):
        assert abs(got[0] - want[0]) < 0.02
        assert abs(got[1] - want[1]) < 0.02
    original = Polygon(square)
    inset = Polygon(points)
    assert original.buffer(0.05).covers(inset)
    gap = original.exterior.distance(inset.exterior)
    assert abs(gap - 2.0) < 0.15


def test_empty_pattern_has_no_texture_fill() -> None:
    square = [(30.0, 40.0), (120.0, 40.0), (120.0, 140.0), (30.0, 140.0)]
    config = TactileConfig()
    svg = render_sheet(
        config.page_mm,
        [_region("r1", "Outline", square, pattern="empty")],
        [],
        [],
        subject="Heart",
        config=config,
    )
    assert _filled_polygons(svg) == []
    assert 'fill="url(' not in svg


def _region(
    region_id: str,
    name: str,
    polygon: list[tuple[float, float]],
    *,
    pattern: str,
) -> SimplifiedRegion:
    return SimplifiedRegion(
        id=region_id,
        name=name,
        description="",
        importance=3,
        polygon_mm=polygon,
        area_mm2=1000.0,
        pattern=pattern,
    )


def _circles(svg: str) -> list[tuple[float, float, float]]:
    found: list[tuple[float, float, float]] = []
    for match in re.finditer(r"<circle\b([^>]*)/?>", svg):
        attrs = match.group(1)
        cx = _attr(attrs, "cx")
        cy = _attr(attrs, "cy")
        radius = _attr(attrs, "r")
        if cx is None or cy is None or radius is None:
            continue
        found.append((cx, cy, radius))
    return found


def _filled_polygons(svg: str) -> list[list[tuple[float, float]]]:
    polygons: list[list[tuple[float, float]]] = []
    for match in re.finditer(r"<polygon\b([^>]*)/?>", svg):
        attrs = match.group(1)
        fill = _raw(attrs, "fill") or ""
        if not fill.startswith("url(#"):
            continue
        raw = _raw(attrs, "points")
        assert raw is not None
        polygons.append(_points(raw))
    return polygons


def _points(value: str) -> list[tuple[float, float]]:
    numbers = [float(part) for part in value.replace(",", " ").split()]
    return list(zip(numbers[0::2], numbers[1::2], strict=True))


def _attr(attrs: str, name: str) -> float | None:
    raw = _raw(attrs, name)
    return None if raw is None else float(raw)


def _raw(attrs: str, name: str) -> str | None:
    match = re.search(rf'{name}="([^"]*)"', attrs)
    return None if match is None else match.group(1)
