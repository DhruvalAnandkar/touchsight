"""Turn region masks into a tactile outline SVG.

Contours are smoothed with Douglas–Peucker. Shared walls are drawn once, from
a boundary graph of the label image, and anything smaller than the minimum
touch area is dropped. Coordinates are millimetres on a Letter or A4 sheet.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import cv2
import numpy as np
import svgwrite
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.validation import make_valid

from pipeline.borders import (
    SNAP_MM,
    boundary_paths,
    close_cracks,
    label_map,
    snap_points,
)
from pipeline.page import FittedImage, TactileConfig, mm_to_px, px_to_mm
from pipeline.timing import timed
from pipeline.types import BoundarySegment, SegmentResult, SimplifiedRegion, SimplifyResult

logger = logging.getLogger("touchsight.pipeline.simplify")


def douglas_peucker(points: np.ndarray, epsilon: float, *, closed: bool = True) -> np.ndarray:
    """Douglas–Peucker. `points` is an (N, 2) array. Closed rings stay closed."""

    minimum = 3 if closed else 2
    if len(points) < minimum:
        return np.asarray(points, dtype=np.float64)
    contour = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(contour, float(epsilon), closed=closed)
    return approx.reshape(-1, 2).astype(np.float64)


def enforce_clearance(masks: Sequence[np.ndarray], radius_px: int) -> list[np.ndarray]:
    """Erode every mask so adjacent outlines separate by about `2 * radius_px`."""

    if radius_px <= 0:
        return [mask.copy() for mask in masks if int((mask > 0).sum()) > 0]
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * radius_px + 1, 2 * radius_px + 1),
    )
    cleared: list[np.ndarray] = []
    for mask in masks:
        eroded = cv2.erode(mask, kernel)
        if int((eroded > 0).sum()) > 0:
            cleared.append(eroded)
    return cleared


def clearance_radius_px(config: TactileConfig, dpi: float) -> int:
    """Inset each region by half the stroke plus half the gap.

    A stroke is centered on the contour, so the clear paper between two ink
    bands is `2 * radius - stroke`. Setting radius to `(gap + stroke) / 2`
    leaves exactly `gap` millimetres of blank paper between strokes.
    """

    stroke_px = mm_to_px(config.min_stroke_mm, dpi)
    gap_px = mm_to_px(config.min_gap_mm, dpi)
    return int(np.ceil((gap_px + stroke_px) / 2.0))


@timed("simplify")
def simplify(
    segmented: SegmentResult,
    fitted: FittedImage,
    config: TactileConfig | None = None,
) -> SimplifyResult:
    config = config or TactileConfig()
    dpi = fitted.dpi
    sheet_mm = fitted.page_mm or config.page_mm
    epsilon_px = max(1.0, mm_to_px(config.epsilon_mm, dpi))
    min_area_px = config.min_area_mm2 * (mm_to_px(1.0, dpi) ** 2)
    kept_masks: list[np.ndarray] = []
    kept_sources = []
    dropped = 0
    for region in segmented.regions:
        if not region.outline_only and int((region.mask > 0).sum()) < min_area_px:
            dropped += 1
            continue
        kept_masks.append(region.mask)
        kept_sources.append(region)
    if not kept_masks:
        svg = render_outline_svg(sheet_mm, [], [], config.min_stroke_mm, segmented.source)
        return SimplifyResult(
            svg=svg,
            page_mm=sheet_mm,
            regions=[],
            source=segmented.source,
            dropped=dropped,
            segments=[],
        )
    labels = close_cracks(label_map(kept_masks), radius_px=_crack_radius_px(dpi))
    simplified = _regions_from_labels(
        labels,
        kept_sources,
        segmented,
        fitted,
        epsilon_px,
        min_area_px,
        config.min_area_mm2,
    )
    dropped += len(kept_sources) - len(simplified)
    for index, region in enumerate(simplified, start=1):
        region.id = f"r{index}"
    segments = _segments_mm(labels, fitted, epsilon_px)
    svg = render_outline_svg(sheet_mm, simplified, segments, config.min_stroke_mm, segmented.source)
    logger.info(
        "simplify kept=%s dropped=%s segments=%s stroke_mm=%.2f",
        len(simplified),
        dropped,
        len(segments),
        config.min_stroke_mm,
    )
    return SimplifyResult(
        svg=svg,
        page_mm=sheet_mm,
        regions=simplified,
        source=segmented.source,
        dropped=max(dropped, 0),
        segments=segments,
    )


def render_outline_svg(
    page_mm: tuple[float, float],
    regions: Sequence[SimplifiedRegion],
    segments: Sequence[BoundarySegment],
    stroke_mm: float,
    source: str,
) -> str:
    width_mm, height_mm = page_mm
    drawing = svgwrite.Drawing(
        size=(f"{_fmt_mm(width_mm)}mm", f"{_fmt_mm(height_mm)}mm"),
        viewBox=f"0 0 {_fmt_mm(width_mm)} {_fmt_mm(height_mm)}",
        profile="full",
        debug=False,
    )
    drawing.attribs["role"] = "img"
    names = ", ".join(region.name for region in regions) or "empty sheet"
    drawing.attribs["aria-label"] = f"Tactile outline: {names}"
    drawing.set_desc(
        title="TouchSight tactile outline",
        desc=f"Raised-line drawing ({source}). Regions: {names}.",
    )
    drawing.add(drawing.rect(insert=(0, 0), size=(width_mm, height_mm), fill="#ffffff"))
    inset = stroke_mm
    drawing.add(
        drawing.rect(
            insert=(inset, inset),
            size=(max(width_mm - 2 * inset, stroke_mm), max(height_mm - 2 * inset, stroke_mm)),
            fill="none",
            stroke="#141414",
            stroke_width=stroke_mm,
            stroke_linejoin="round",
        )
    )
    # Polygons stay in the sheet for later hit-testing. They are not stroked;
    # the ink is the shared-edge graph below.
    for region in regions:
        group = drawing.g(id=region.id)
        group.set_desc(title=region.name, desc=region.description or None)
        group.add(
            drawing.polygon(
                points=region.polygon_mm,
                fill="none",
                stroke="none",
            )
        )
        drawing.add(group)
    ink = drawing.g(
        id="boundaries",
        fill="none",
        stroke="#141414",
        stroke_width=stroke_mm,
        stroke_linejoin="round",
        stroke_linecap="round",
    )
    for segment in segments:
        x1, y1 = segment.start_mm
        x2, y2 = segment.end_mm
        ink.add(drawing.line(start=(x1, y1), end=(x2, y2)))
    drawing.add(ink)
    return drawing.tostring()


def _polygon_mm(
    mask: np.ndarray,
    offset_px: tuple[float, float],
    dpi: float,
    epsilon_px: float,
    min_area_px: float,
) -> tuple[list[tuple[float, float]], float] | None:
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area_px * 0.25:
        return None
    approx = cv2.approxPolyDP(contour, epsilon_px, closed=True)
    if len(approx) < 3:
        approx = cv2.approxPolyDP(contour, max(0.5, epsilon_px / 4), closed=True)
    if len(approx) < 3:
        return None
    offset_x, offset_y = offset_px
    points: list[tuple[float, float]] = []
    for point in approx.reshape(-1, 2):
        points.append(
            (
                round(px_to_mm(float(point[0]) + offset_x, dpi), 2),
                round(px_to_mm(float(point[1]) + offset_y, dpi), 2),
            )
        )
    points = _unique_ring(points)
    if len(points) < 3:
        return None
    polygon = _as_polygon(points)
    if polygon is None:
        return None
    return points, float(polygon.area)


def _as_polygon(points: Sequence[tuple[float, float]]) -> Polygon | None:
    polygon: BaseGeometry = Polygon(points)
    if not polygon.is_valid:
        polygon = make_valid(polygon)
    if isinstance(polygon, MultiPolygon):
        parts = [part for part in polygon.geoms if isinstance(part, Polygon) and not part.is_empty]
        if not parts:
            return None
        polygon = max(parts, key=lambda part: part.area)
    if not isinstance(polygon, Polygon) or polygon.is_empty or polygon.area <= 0:
        return None
    return polygon


def _unique_ring(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    unique: list[tuple[float, float]] = []
    for point in points:
        if not unique or point != unique[-1]:
            unique.append(point)
    if len(unique) > 1 and unique[0] == unique[-1]:
        unique.pop()
    return unique


def _source_region(mask: np.ndarray, segmented: SegmentResult):
    best = None
    best_overlap = 0
    foreground = mask > 0
    for region in segmented.regions:
        overlap = int(np.logical_and(foreground, region.mask > 0).sum())
        if overlap > best_overlap:
            best_overlap = overlap
            best = region
    return best


def _fmt_mm(value: float) -> str:
    rounded = round(float(value), 2)
    if rounded == int(rounded):
        return str(int(rounded))
    return f"{rounded:.2f}".rstrip("0").rstrip(".")


def _crack_radius_px(dpi: float) -> int:
    return max(1, int(round(mm_to_px(SNAP_MM, dpi))))


def _regions_from_labels(
    labels: np.ndarray,
    kept_sources: Sequence[object],
    segmented: SegmentResult,
    fitted: FittedImage,
    epsilon_px: float,
    min_area_px: float,
    min_area_mm2: float,
) -> list[SimplifiedRegion]:
    regions: list[SimplifiedRegion] = []
    for label in sorted(int(value) for value in np.unique(labels) if int(value) > 0):
        mask = np.where(labels == label, 255, 0).astype(np.uint8)
        source = kept_sources[label - 1] if label - 1 < len(kept_sources) else None
        if source is None:
            source = _source_region(mask, segmented)
        outline_only = bool(getattr(source, "outline_only", False))
        if not outline_only and int((mask > 0).sum()) < min_area_px:
            continue
        polygon = _polygon_mm(
            mask,
            fitted.offset_px,
            fitted.dpi,
            epsilon_px,
            0.0 if outline_only else min_area_px,
        )
        if polygon is None:
            continue
        points, area_mm2 = polygon
        if not outline_only and area_mm2 < min_area_mm2:
            continue
        regions.append(
            SimplifiedRegion(
                id="",
                name=getattr(source, "name", "") or f"Region {len(regions) + 1}",
                description=getattr(source, "description", ""),
                importance=int(getattr(source, "importance", 3)),
                polygon_mm=points,
                area_mm2=area_mm2,
                outline_only=outline_only,
            )
        )
    return regions


def _segments_mm(labels: np.ndarray, fitted: FittedImage, epsilon_px: float) -> list[BoundarySegment]:
    raw: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for path in boundary_paths(labels):
        closed = len(path) >= 4 and path[0] == path[-1]
        body = path[:-1] if closed else path
        approx = douglas_peucker(np.asarray(body, dtype=np.float64), epsilon_px, closed=closed)
        points = [(float(point[0]), float(point[1])) for point in approx]
        if closed and len(points) >= 3:
            pairs = ((points[index], points[(index + 1) % len(points)]) for index in range(len(points)))
        else:
            pairs = ((points[index], points[index + 1]) for index in range(len(points) - 1))
        for start, end in pairs:
            if start != end:
                raw.append((start, end))
    flat: list[tuple[float, float]] = []
    for start, end in raw:
        flat.append(_px_point_mm(start, fitted))
        flat.append(_px_point_mm(end, fitted))
    snapped = snap_points(flat, SNAP_MM)
    deduped: list[tuple[tuple[float, float], tuple[float, float]]] = []
    seen: set[tuple[tuple[float, float], tuple[float, float]]] = set()
    for index in range(0, len(snapped), 2):
        start, end = snapped[index], snapped[index + 1]
        if start == end:
            continue
        key = tuple(sorted((start, end)))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((start, end))
    merged = _merge_colinear(deduped)
    return [BoundarySegment(start_mm=start, end_mm=end) for start, end in merged]


def _px_point_mm(point: tuple[float, float], fitted: FittedImage) -> tuple[float, float]:
    offset_x, offset_y = fitted.offset_px
    return (
        round(px_to_mm(point[0] + offset_x, fitted.dpi), 2),
        round(px_to_mm(point[1] + offset_y, fitted.dpi), 2),
    )


def _merge_colinear(
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Join straight runs that meet at a degree-2 vertex. Junctions stay put."""

    neighbors: dict[tuple[float, float], list[tuple[float, float]]] = {}
    for start, end in segments:
        neighbors.setdefault(start, []).append(end)
        neighbors.setdefault(end, []).append(start)
    used: set[frozenset[tuple[tuple[float, float], tuple[float, float]]]] = set()
    merged: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for start, end in segments:
        key = frozenset((start, end))
        if key in used:
            continue
        used.add(key)
        path = [start, end]
        path = _extend_straight(path, neighbors, used, forward=True)
        path = _extend_straight(path, neighbors, used, forward=False)
        merged.append((path[0], path[-1]))
    return merged


def _extend_straight(
    path: list[tuple[float, float]],
    neighbors: dict[tuple[float, float], list[tuple[float, float]]],
    used: set[frozenset[tuple[tuple[float, float], tuple[float, float]]]],
    *,
    forward: bool,
) -> list[tuple[float, float]]:
    while True:
        previous, current = (path[-2], path[-1]) if forward else (path[1], path[0])
        adjacent = neighbors.get(current, [])
        if len(adjacent) != 2:
            break
        nxt = adjacent[0] if adjacent[1] == previous else adjacent[1]
        edge = frozenset((current, nxt))
        if edge in used or not _colinear(previous, current, nxt):
            break
        used.add(edge)
        if forward:
            path.append(nxt)
        else:
            path.insert(0, nxt)
    return path


def _colinear(
    start: tuple[float, float],
    middle: tuple[float, float],
    end: tuple[float, float],
) -> bool:
    v1 = (middle[0] - start[0], middle[1] - start[1])
    v2 = (end[0] - middle[0], end[1] - middle[1])
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    scale = max(abs(v1[0]) + abs(v1[1]), abs(v2[0]) + abs(v2[1]), 1.0)
    return abs(cross) <= 0.02 * scale and dot > 0
