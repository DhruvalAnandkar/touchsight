"""Tests for the OpenCV fallback and the SAM hand-off."""

from __future__ import annotations

import numpy as np

from pipeline.providers.sam2 import SamError, match_masks_to_boxes
from pipeline.segment import segment
from pipeline.types import BBox, Part
from tests.drawings import bar_chart, line_cell, separated_circles


class _FailingSam:
    available = True

    def segment_boxes(self, image_bgr: np.ndarray, boxes_xyxy: list[tuple[int, int, int, int]]) -> list[np.ndarray]:
        raise SamError("down")


class _BoxSam:
    available = True

    def segment_boxes(self, image_bgr: np.ndarray, boxes_xyxy: list[tuple[int, int, int, int]]) -> list[np.ndarray]:
        masks: list[np.ndarray] = []
        for x1, y1, x2, y2 in boxes_xyxy:
            mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255
            masks.append(mask)
        return masks


def test_bar_chart_splits_into_four_regions() -> None:
    result = segment(bar_chart())
    assert result.source == "opencv"
    assert len(result.regions) == 4


def test_line_drawing_keeps_enclosed_compartments() -> None:
    result = segment(line_cell())
    assert 4 <= len(result.regions) <= 8
    areas = sorted(int((region.mask > 0).sum()) for region in result.regions)
    assert areas[-1] > areas[0] * 3


def test_region_cap_drops_the_smallest() -> None:
    result = segment(separated_circles(10), max_regions=3)
    assert len(result.regions) == 3
    assert [region.id for region in result.regions] == ["r1", "r2", "r3"]


def test_part_bbox_names_the_matching_region() -> None:
    image = bar_chart()
    # The rightmost bar sits near x=420..490 of 560, y=240..340 of 420.
    part = Part(
        name="Imports",
        description="The shortest bar",
        importance=5,
        approx_bbox=BBox(x=0.72, y=0.50, w=0.18, h=0.40),
    )
    result = segment(image, [part])
    named = [region for region in result.regions if region.name == "Imports"]
    assert len(named) == 1
    assert named[0].description == "The shortest bar"
    assert named[0].importance == 5


def test_sam_failure_falls_back_to_opencv() -> None:
    part = Part(name="Anything", approx_bbox=BBox(x=0.1, y=0.1, w=0.3, h=0.3))
    result = segment(bar_chart(), [part], sam=_FailingSam())
    assert result.source == "opencv"
    assert len(result.regions) == 4


def test_sam_masks_keep_part_names() -> None:
    part = Part(name="Nucleus", description="Control center", importance=4, approx_bbox=BBox(x=0.2, y=0.2, w=0.3, h=0.3))
    result = segment(line_cell(), [part], sam=_BoxSam())
    assert result.source == "sam2"
    assert result.regions[0].name == "Nucleus"
    assert result.regions[0].id == "r1"


def test_mask_box_matching_is_one_to_one() -> None:
    left = np.zeros((40, 80), np.uint8)
    right = np.zeros((40, 80), np.uint8)
    left[:, :30] = 255
    right[:, 50:] = 255
    matched = match_masks_to_boxes([right, left], [(0, 0, 30, 40), (50, 0, 80, 40)])
    assert int((matched[0][:, :30] > 0).sum()) > 0
    assert int((matched[1][:, 50:] > 0).sum()) > 0
