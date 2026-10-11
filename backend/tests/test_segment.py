"""Tests for the OpenCV fallback and the SAM hand-off."""

from __future__ import annotations

import cv2
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


def test_sam_failure_falls_back_to_grabcut() -> None:
    part = Part(name="Anything", approx_bbox=BBox(x=0.1, y=0.1, w=0.3, h=0.3))
    result = segment(bar_chart(), [part], sam=_FailingSam())
    assert result.source == "grabcut"
    assert len(result.regions) >= 1
    assert result.regions[0].name == "Anything"


def test_sam_masks_keep_part_names() -> None:
    part = Part(name="Nucleus", description="Control center", importance=4, approx_bbox=BBox(x=0.2, y=0.2, w=0.3, h=0.3))
    result = segment(line_cell(), [part], sam=_BoxSam())
    assert result.source == "sam2"
    assert result.regions[0].name == "Nucleus"
    assert result.regions[0].id == "r1"


def _rounded_rect_mask(
    shape: tuple[int, int], x1: int, y1: int, x2: int, y2: int, radius: int
) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    cv2.rectangle(mask, (x1 + radius, y1), (x2 - radius, y2), 255, thickness=-1)
    cv2.rectangle(mask, (x1, y1 + radius), (x2, y2 - radius), 255, thickness=-1)
    for center in (
        (x1 + radius, y1 + radius),
        (x2 - radius, y1 + radius),
        (x1 + radius, y2 - radius),
        (x2 - radius, y2 - radius),
    ):
        cv2.circle(mask, center, radius, 255, thickness=-1)
    return mask


def _hatched_illustration() -> tuple[np.ndarray, list[Part]]:
    """Two filled shapes on white, each crossed by many thin dark hatch strokes."""

    height, width = 480, 480
    image = np.full((height, width, 3), 255, np.uint8)
    disk = np.zeros((height, width), np.uint8)
    cv2.circle(disk, (130, 240), 90, 255, thickness=-1)
    panel = _rounded_rect_mask((height, width), 280, 130, 440, 360, 28)
    image[disk > 0] = (50, 120, 190)
    image[panel > 0] = (70, 160, 90)
    strokes = np.zeros((height, width), np.uint8)
    for offset in range(-height, width, 4):
        cv2.line(strokes, (offset, 0), (offset + height, height), 255, 1)
    for row in range(0, height, 5):
        cv2.line(strokes, (0, row), (width, row), 255, 1)
    image[np.logical_and(disk > 0, strokes > 0)] = (25, 30, 35)
    image[np.logical_and(panel > 0, strokes > 0)] = (20, 35, 25)
    parts = [
        Part(
            name="Disk",
            description="Hatched disk",
            importance=3,
            approx_bbox=BBox(x=30 / width, y=140 / height, w=200 / width, h=200 / height),
        ),
        Part(
            name="Panel",
            description="Hatched panel",
            importance=4,
            approx_bbox=BBox(x=260 / width, y=110 / height, w=200 / width, h=270 / height),
        ),
    ]
    return image, parts


def test_concentric_circles_stay_three_regions() -> None:
    size = 300
    image = np.full((size, size, 3), 255, np.uint8)
    cv2.circle(image, (150, 150), 120, (40, 90, 40), thickness=-1)
    cv2.circle(image, (150, 150), 80, (30, 90, 170), thickness=-1)
    cv2.circle(image, (150, 150), 40, (40, 210, 230), thickness=-1)

    def box(radius: int) -> BBox:
        return BBox(
            x=(150 - radius) / size,
            y=(150 - radius) / size,
            w=(2 * radius) / size,
            h=(2 * radius) / size,
        )

    parts = [
        Part(name="Outer", description="The outer ring.", importance=5, approx_bbox=box(120)),
        Part(name="Middle", description="The middle ring.", importance=5, approx_bbox=box(80)),
        Part(name="Inner", description="The inner circle.", importance=5, approx_bbox=box(40)),
    ]
    result = segment(image, parts)
    assert result.source == "grabcut"
    assert len(result.regions) == 3
    assert {region.name for region in result.regions} == {"Outer", "Middle", "Inner"}


def test_grabcut_segments_hatched_illustration_into_two_regions() -> None:
    image, parts = _hatched_illustration()
    result = segment(image, parts)
    assert result.source == "grabcut"
    assert len(result.regions) == 2
    assert {region.name for region in result.regions} == {"Disk", "Panel"}
    assert all(not region.outline_only for region in result.regions)


def test_mask_box_matching_is_one_to_one() -> None:
    left = np.zeros((40, 80), np.uint8)
    right = np.zeros((40, 80), np.uint8)
    left[:, :30] = 255
    right[:, 50:] = 255
    matched = match_masks_to_boxes([right, left], [(0, 0, 30, 40), (50, 0, 80, 40)])
    assert int((matched[0][:, :30] > 0).sum()) > 0
    assert int((matched[1][:, 50:] > 0).sum()) > 0
