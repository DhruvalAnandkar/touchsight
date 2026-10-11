"""Tests for contour smoothing, clearance, and the outline SVG."""

from __future__ import annotations

import numpy as np

from pipeline.page import FittedImage, TactileConfig
from pipeline.simplify import douglas_peucker, enforce_clearance, simplify
from pipeline.types import SegmentResult, SegmentedRegion


def _square_region(x: int, y: int, size: int, shape: tuple[int, int], name: str) -> SegmentedRegion:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[y : y + size, x : x + size] = 255
    return SegmentedRegion(
        id="tmp",
        name=name,
        description="",
        importance=3,
        mask=mask,
        bbox=(x, y, size, size),
    )


def _fitted(shape: tuple[int, int], dpi: float = 25.4) -> FittedImage:
    height, width = shape
    return FittedImage(image=np.zeros((height, width, 3), np.uint8), offset_px=(0, 0), page_px=(width, height), dpi=dpi)


def test_douglas_peucker_drops_collinear_points() -> None:
    points = np.array(
        [
            [0, 0],
            [5, 0],
            [10, 0],
            [10, 5],
            [10, 10],
            [5, 10],
            [0, 10],
            [0, 5],
        ],
        dtype=np.float64,
    )
    approx = douglas_peucker(points, epsilon=0.5)
    assert len(approx) == 4


def test_clearance_separates_touching_regions() -> None:
    shape = (120, 260)
    left = np.zeros(shape, np.uint8)
    right = np.zeros(shape, np.uint8)
    left[20:100, 20:120] = 255
    right[20:100, 120:220] = 255
    radius = 4
    cleared = enforce_clearance([left, right], radius)
    assert len(cleared) == 2
    outside = np.where(cleared[1] > 0, 0, 255).astype(np.uint8)
    import cv2

    distance = cv2.distanceTransform(outside, cv2.DIST_L2, 5)
    gap = float(distance[cleared[0] > 0].min())
    assert gap >= (2 * radius) - 1


def test_simplify_drops_regions_below_minimum_area() -> None:
    shape = (200, 200)
    big = _square_region(20, 20, 80, shape, "Big")
    tiny = _square_region(140, 140, 6, shape, "Tiny")
    config = TactileConfig(dpi=25, min_stroke_mm=1.0, min_gap_mm=1.0, min_area_mm2=80, epsilon_mm=0.4)
    result = simplify(
        SegmentResult(width=200, height=200, source="opencv", regions=[big, tiny]),
        _fitted(shape, dpi=25.4),
        config,
    )
    assert [region.name for region in result.regions] == ["Big"]
    assert result.dropped >= 1


def test_outline_svg_uses_minimum_stroke_and_page_size() -> None:
    shape = (180, 180)
    region = _square_region(30, 30, 90, shape, "Square")
    config = TactileConfig(page="a4", dpi=25, min_stroke_mm=1.5, min_gap_mm=2.0, min_area_mm2=20, epsilon_mm=0.4)
    result = simplify(
        SegmentResult(width=180, height=180, source="opencv", regions=[region]),
        _fitted(shape, dpi=25.4),
        config,
    )
    assert result.page_mm == (210.0, 297.0)
    assert 'width="210mm"' in result.svg
    assert 'height="297mm"' in result.svg
    assert 'stroke-width="1.5"' in result.svg
    assert 'id="r1"' in result.svg
    assert len(result.regions[0].polygon_mm) >= 4


def test_adjacent_squares_share_one_border() -> None:
    shape = (90, 160)
    left = _square_region(10, 10, 60, shape, "Left")
    right = _square_region(70, 10, 60, shape, "Right")
    result = simplify(
        SegmentResult(width=160, height=90, source="opencv", regions=[left, right]),
        _fitted(shape, dpi=25.4),
        TactileConfig(dpi=25, min_stroke_mm=1.5, min_gap_mm=2.5, min_area_mm2=10, epsilon_mm=0.4),
    )
    assert len(result.segments) == 7
    assert result.svg.count("<line ") == 7


def test_wide_sheet_is_landscape_and_tall_sheet_is_portrait() -> None:
    from pipeline.page import fit_to_page

    wide = fit_to_page(np.full((120, 480, 3), 255, np.uint8), TactileConfig())
    tall = fit_to_page(np.full((480, 120, 3), 255, np.uint8), TactileConfig())
    assert wide.page_mm is not None and wide.page_mm[0] > wide.page_mm[1]
    assert tall.page_mm is not None and tall.page_mm[1] > tall.page_mm[0]


def test_empty_segmentation_still_returns_a_sheet() -> None:
    shape = (40, 40)
    result = simplify(
        SegmentResult(width=40, height=40, source="opencv", regions=[]),
        _fitted(shape),
        TactileConfig(),
    )
    assert result.regions == []
    assert result.svg.startswith("<?xml") or "<svg" in result.svg
