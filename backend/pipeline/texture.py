"""Five tactile fills. Neighbors never share one.

Patterns, in the order a region tries them:

- dots: 1.5 mm dots on a 4 mm pitch
- diagonal: 0.8 mm line on a 4 mm pitch
- crosshatch: both diagonals, 0.8 mm on a 4 mm pitch
- horizontal: 0.8 mm line on a 5 mm pitch
- empty: outline only

The fill stops 2 mm inside the outline so a finger meets the wall before the texture.
"""

from __future__ import annotations

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.validation import make_valid

PATTERNS: tuple[str, ...] = ("dots", "diagonal", "crosshatch", "horizontal", "empty")
FILL_MARGIN_MM = 2.0
# Dilate each label mask by this many pixels before the overlap test.
ADJACENCY_RADIUS_PX = 3


def assign_patterns(regions: list[tuple[str, list[tuple[float, float]]]]) -> dict[str, str]:
    """Greedy coloring. `regions` is (id, polygon_mm)."""

    polygons = {region_id: _polygon(points) for region_id, points in regions}
    ids = [region_id for region_id, _points in regions]
    neighbors: dict[str, set[str]] = {region_id: set() for region_id in ids}
    for index, left_id in enumerate(ids):
        left = polygons[left_id]
        if left is None:
            continue
        for right_id in ids[index + 1 :]:
            right = polygons[right_id]
            if right is None:
                continue
            if left.buffer(0.4).intersects(right):
                neighbors[left_id].add(right_id)
                neighbors[right_id].add(left_id)
    order = sorted(ids, key=lambda region_id: (-len(neighbors[region_id]), region_id))
    chosen: dict[str, str] = {}
    for region_id in order:
        used = {chosen[other] for other in neighbors[region_id] if other in chosen}
        free = [pattern for pattern in PATTERNS if pattern not in used]
        chosen[region_id] = free[0] if free else "empty"
    return chosen


def assign_from_label_masks(items: list[tuple[str, np.ndarray, int]]) -> dict[str, str]:
    """Greedy-color regions from their label masks.

    `items` is `(region_id, uint8 mask of 0 or 255, importance 1-5)`.
    Two regions are adjacent when their masks, each dilated by 3 px, overlap.
    The masks are used as segmented, not as simplified or shrunk polygons.
    Higher importance is colored first. Equal importance breaks ties by id.
    """

    dilated = {region_id: _dilate_mask(mask) for region_id, mask, _importance in items}
    ids = [region_id for region_id, _mask, _importance in items]
    importance = {region_id: int(rank) for region_id, _mask, rank in items}
    neighbors: dict[str, set[str]] = {region_id: set() for region_id in ids}
    for index, left_id in enumerate(ids):
        left = dilated[left_id]
        for right_id in ids[index + 1 :]:
            right = dilated[right_id]
            if left.shape != right.shape:
                continue
            if cv2.countNonZero(cv2.bitwise_and(left, right)) > 0:
                neighbors[left_id].add(right_id)
                neighbors[right_id].add(left_id)
    order = sorted(ids, key=lambda region_id: (-importance[region_id], region_id))
    chosen: dict[str, str] = {}
    for region_id in order:
        used = {chosen[other] for other in neighbors[region_id] if other in chosen}
        free = [pattern for pattern in PATTERNS if pattern not in used]
        chosen[region_id] = free[0] if free else "empty"
    return chosen


def _dilate_mask(mask: np.ndarray) -> np.ndarray:
    foreground = (np.asarray(mask) > 0).astype(np.uint8) * 255
    diameter = ADJACENCY_RADIUS_PX * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    return cv2.dilate(foreground, kernel)


def inset_ring(points: list[tuple[float, float]], margin_mm: float = FILL_MARGIN_MM) -> list[tuple[float, float]]:
    polygon = _polygon(points)
    if polygon is None:
        return []
    inset = polygon.buffer(-margin_mm)
    if inset.is_empty:
        return []
    if inset.geom_type == "MultiPolygon":
        inset = max(inset.geoms, key=lambda item: item.area)
    if inset.geom_type != "Polygon":
        return []
    return [(round(x, 2), round(y, 2)) for x, y in inset.exterior.coords]


def _polygon(points: list[tuple[float, float]]) -> Polygon | None:
    if len(points) < 3:
        return None
    polygon = Polygon(points)
    if not polygon.is_valid:
        polygon = make_valid(polygon)
    if polygon.geom_type == "MultiPolygon":
        polygon = max(polygon.geoms, key=lambda item: item.area)
    if polygon.is_empty or polygon.geom_type != "Polygon":
        return None
    return polygon
