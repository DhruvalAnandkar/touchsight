"""Segment an image into at most eight regions.

Primary path: a hosted SAM 2 client, prompted by part boxes. When part boxes
are present but SAM is unavailable, GrabCut separates each part from the white
background. With no part boxes, OpenCV Canny edges become watershed barriers
and each enclosed area is a region.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

import cv2
import numpy as np

from pipeline.providers.sam2 import Sam2Client, SamError
from pipeline.timing import timed
from pipeline.types import Part, SegmentResult, SegmentedRegion

logger = logging.getLogger("touchsight.pipeline.segment")

MAX_REGIONS = 8


@timed("segment")
def segment(
    image_bgr: np.ndarray,
    parts: Sequence[Part] = (),
    *,
    max_regions: int = MAX_REGIONS,
    sam: Sam2Client | None = None,
) -> SegmentResult:
    height, width = image_bgr.shape[:2]
    limit = min(MAX_REGIONS, max(1, max_regions))
    if sam is not None and sam.available and parts:
        try:
            boxes = [part.approx_bbox.to_xyxy(width, height) for part in parts]
            masks = sam.segment_boxes(image_bgr, boxes)
            regions = _regions_from_part_masks(masks, list(parts))
            if regions:
                logger.info("segment source=sam2 regions=%s", len(regions))
                return SegmentResult(width=width, height=height, source="sam2", regions=regions[:limit])
        except (SamError, OSError, ValueError) as exc:
            logger.warning("SAM 2 unavailable (%s); using GrabCut fallback", exc)

    if parts:
        regions = _grabcut_regions(image_bgr, list(parts), limit)
        logger.info("segment source=grabcut regions=%s", len(regions))
        return SegmentResult(width=width, height=height, source="grabcut", regions=regions)

    masks, thin = _opencv_masks(image_bgr)
    masks = _limit_regions(masks, limit)
    regions = _label_regions(masks, image_bgr.shape[:2], parts)
    _keep_named_thin_regions(thin, regions, parts, image_bgr.shape[:2], limit)
    logger.info("segment source=opencv regions=%s", len(regions))
    return SegmentResult(width=width, height=height, source="opencv", regions=regions)


def _grabcut_regions(image_bgr: np.ndarray, parts: list[Part], limit: int) -> list[SegmentedRegion]:
    """One GrabCut mask per part box, then importance decides contested pixels."""

    height, width = image_bgr.shape[:2]
    smooth = cv2.bilateralFilter(image_bgr, 9, 75, 75)
    raw = [_grabcut_part_mask(image_bgr, smooth, part) for part in parts]
    min_area = 0.003 * height * width
    cleaned: list[np.ndarray] = []
    for mask in raw:
        filled = _fill_holes(mask)
        speckless = _remove_specks(filled, min_area)
        cleaned.append(_smooth_mask_edges(speckless))
    resolved = _assign_by_importance(cleaned, parts)
    return _regions_from_part_masks(resolved, parts)[:limit]


def _grabcut_part_mask(image_bgr: np.ndarray, smooth: np.ndarray, part: Part) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = part.approx_bbox.to_xyxy(width, height)
    x1, y1, x2, y2 = _clip_xyxy(int(x1), int(y1), int(x2), int(y2), width, height)
    empty = np.zeros((height, width), np.uint8)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return empty

    gc_mask = np.full((height, width), cv2.GC_BGD, np.uint8)
    gc_mask[y1:y2, x1:x2] = cv2.GC_PR_FGD
    ix1, iy1, ix2, iy2 = _shrunk_core(x1, y1, x2, y2)
    white = image_bgr.min(axis=2) >= 245
    outside_core = np.ones((height, width), dtype=bool)
    outside_core[iy1:iy2, ix1:ix2] = False
    # Clearly white paper outside the inner core is definite background.
    gc_mask[np.logical_and(white, outside_core)] = cv2.GC_BGD
    core = np.zeros((height, width), dtype=bool)
    core[iy1:iy2, ix1:ix2] = True
    gc_mask[np.logical_and(core, ~white)] = cv2.GC_FGD

    if not np.any((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)):
        return empty

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(
            smooth,
            gc_mask,
            (x1, y1, max(1, x2 - x1), max(1, y2 - y1)),
            bgd_model,
            fgd_model,
            5,
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error:
        logger.warning("grabCut failed for part %s", part.name)
        return empty
    foreground = (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)
    return np.where(foreground, 255, 0).astype(np.uint8)


def _clip_xyxy(
    x1: int, y1: int, x2: int, y2: int, width: int, height: int
) -> tuple[int, int, int, int]:
    x1 = min(max(x1, 0), width)
    y1 = min(max(y1, 0), height)
    x2 = min(max(x2, 0), width)
    y2 = min(max(y2, 0), height)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _shrunk_core(x1: int, y1: int, x2: int, y2: int) -> tuple[int, int, int, int]:
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    dx = max(1, int(round(box_w * 0.18)))
    dy = max(1, int(round(box_h * 0.18)))
    ix1 = min(x1 + dx, x2 - 1)
    iy1 = min(y1 + dy, y2 - 1)
    ix2 = max(ix1 + 1, x2 - dx)
    iy2 = max(iy1 + 1, y2 - dy)
    return ix1, iy1, ix2, iy2


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    padded = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    flood_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(padded, holes)
    return filled[1:-1, 1:-1]


def _remove_specks(mask: np.ndarray, min_area: float) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return np.where(binary > 0, 255, 0).astype(np.uint8)
    kept = np.zeros(binary.shape, np.uint8)
    for label in range(1, count):
        if float(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            kept[labels == label] = 255
    return kept


def _smooth_mask_edges(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)


def _assign_by_importance(masks: list[np.ndarray], parts: Sequence[Part]) -> list[np.ndarray]:
    """Higher importance overwrites. Equal importance keeps the earlier part."""

    height, width = masks[0].shape[:2]
    owner = np.zeros((height, width), dtype=np.int32)
    order = sorted(range(len(parts)), key=lambda index: (int(parts[index].importance), -index))
    for index in order:
        owner[masks[index] > 0] = index + 1
    return [np.where(owner == index + 1, 255, 0).astype(np.uint8) for index in range(len(parts))]


def _opencv_masks(image_bgr: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    labels = _canny_watershed(image_bgr)
    masks, thin = _masks_from_labels(labels, image_bgr)
    if masks or thin:
        return masks, thin
    logger.info("watershed found no regions; using enclosed-line fallback")
    return _enclosed_line_masks(image_bgr), []


def _color_canny(image_bgr: np.ndarray) -> np.ndarray:
    """Edges on each channel, so regions that share a luminance still split."""

    edges = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    for channel in cv2.split(image_bgr):
        blur = cv2.GaussianBlur(channel, (5, 5), 0)
        gradient_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(gradient_x, gradient_y)
        magnitude_u8 = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        otsu, _binary = cv2.threshold(magnitude_u8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        lower = max(8, int(otsu * 0.5))
        upper = max(lower + 8, int(otsu))
        channel_edges = cv2.Canny(blur, lower, upper)
        edges = cv2.bitwise_or(edges, channel_edges)
    return cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)


def _canny_watershed(image_bgr: np.ndarray) -> np.ndarray:
    """Label map. Positive ids are regions, 0 is unknown, -1 is a boundary."""

    height, width = image_bgr.shape[:2]
    edges = _color_canny(image_bgr)
    free = cv2.bitwise_not(edges)
    # Seeds must sit inside a region, not at the global distance peak. A fixed
    # fraction of the max distance keeps only the page background on a mostly
    # white diagram and swallows every bar, cell, and country.
    radius = int(np.clip(round(min(height, width) * 0.015), 4, 9))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    seeds = cv2.erode(free, kernel)
    _count, markers = cv2.connectedComponents(seeds)
    markers = markers.astype(np.int32)
    markers[edges > 0] = 0
    if int(markers.max()) < 1:
        return np.zeros((height, width), dtype=np.int32)
    color = image_bgr.copy()
    cv2.watershed(color, markers)
    return markers


def _masks_from_labels(
    labels: np.ndarray, image_bgr: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    height, width = labels.shape[:2]
    min_area = max(80, int(0.0015 * height * width))
    background = _paper_background_label(labels, image_bgr)
    masks: list[np.ndarray] = []
    thin: list[np.ndarray] = []
    for label in np.unique(labels):
        if int(label) <= 0 or int(label) == background:
            continue
        mask = np.where(labels == label, 255, 0).astype(np.uint8)
        mask = _largest_component(mask)
        if _keep_mask(mask, min_area):
            masks.append(mask)
        elif int((mask > 0).sum()) >= 30:
            thin.append(mask)
    masks.sort(key=lambda item: int((item > 0).sum()), reverse=True)
    return masks, thin


def _paper_background_label(labels: np.ndarray, image_bgr: np.ndarray) -> int | None:
    """The border label is background when it is paper-colored or almost the whole page.

    Interior white (a cell's cytoplasm) does not touch the border, so it stays.
    A colored region that happens to touch the border (sky, a full-bleed photo)
    stays too.
    """

    inset = 2
    if labels.shape[0] <= inset * 2 + 1 or labels.shape[1] <= inset * 2 + 1:
        return None
    frame = np.concatenate(
        [
            labels[inset, inset:-inset],
            labels[-inset - 1, inset:-inset],
            labels[inset:-inset, inset],
            labels[inset:-inset, -inset - 1],
        ]
    )
    frame = frame[frame > 0]
    if frame.size == 0:
        return None
    label = int(np.bincount(frame).argmax())
    mask = labels == label
    area_fraction = float(mask.mean())
    if area_fraction > 0.72 or _looks_like_paper(image_bgr, mask):
        return label
    return None


def _looks_like_paper(image_bgr: np.ndarray, mask: np.ndarray) -> bool:
    pixels = image_bgr[mask]
    if pixels.shape[0] < 10:
        return False
    high = pixels.max(axis=1).astype(np.int16)
    low = pixels.min(axis=1).astype(np.int16)
    saturation = float((high - low).mean())
    luminance = float(pixels.mean())
    return luminance >= 210 and saturation < 28


def _largest_component(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def _keep_mask(mask: np.ndarray, min_area: int) -> bool:
    ys, xs = np.where(mask > 0)
    area = int(xs.size)
    if area < min_area or xs.size == 0:
        return False
    box_w = int(xs.max() - xs.min()) + 1
    box_h = int(ys.max() - ys.min()) + 1
    short_side = min(box_w, box_h)
    if short_side < 8:
        return False
    if area / float(box_w * box_h) < 0.18:
        return False
    return True


def _enclosed_line_masks(image_bgr: np.ndarray) -> list[np.ndarray]:
    """Flood-fill fallback for closed line drawings when watershed returns nothing."""

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _threshold, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    space = cv2.bitwise_not(ink)
    height, width = space.shape
    flood_mask = np.zeros((height + 2, width + 2), np.uint8)
    for point in ((0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)):
        if space[point[1], point[0]] > 0:
            cv2.floodFill(space, flood_mask, point, 0)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats((space > 0).astype(np.uint8), 8)
    min_area = max(80, int(0.0015 * height * width))
    masks: list[np.ndarray] = []
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) < min_area:
            continue
        mask = np.where(labels == label, 255, 0).astype(np.uint8)
        if _keep_mask(mask, min_area):
            masks.append(mask)
    return masks


def _limit_regions(masks: list[np.ndarray], max_regions: int) -> list[np.ndarray]:
    """Keep at most `max_regions` masks. Extras merge into a touching neighbor, or drop."""

    ordered = sorted(masks, key=lambda item: int((item > 0).sum()), reverse=True)
    if len(ordered) <= max_regions:
        return ordered
    kept = ordered[:max_regions]
    for extra in ordered[max_regions:]:
        host = _touching_index(extra, kept)
        if host is None:
            continue
        kept[host] = cv2.bitwise_or(kept[host], extra)
    return kept


def _touching_index(mask: np.ndarray, others: list[np.ndarray]) -> int | None:
    dilated = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    best_index: int | None = None
    best_overlap = 0
    for index, other in enumerate(others):
        overlap = int(np.logical_and(dilated > 0, other > 0).sum())
        if overlap > best_overlap:
            best_overlap = overlap
            best_index = index
    return best_index


def _regions_from_part_masks(masks: Sequence[np.ndarray], parts: list[Part]) -> list[SegmentedRegion]:
    paired = list(zip(masks, parts, strict=True))
    regions: list[SegmentedRegion] = []
    for mask, part in paired:
        cleaned = _largest_component((mask > 0).astype(np.uint8) * 255)
        if int((cleaned > 0).sum()) < 30:
            continue
        regions.append(_region_from_mask(cleaned, part.name, part.description, part.importance, ""))
    ordered = _sort_reading_order(regions)
    for position, region in enumerate(ordered, start=1):
        region.id = f"r{position}"
    return ordered


def _label_regions(
    masks: list[np.ndarray],
    shape: tuple[int, int],
    parts: Sequence[Part],
) -> list[SegmentedRegion]:
    assignment = _match_parts(masks, parts, shape)
    areas = [int((mask > 0).sum()) for mask in masks]
    rank = {index: order for order, index in enumerate(sorted(range(len(masks)), key=lambda i: areas[i], reverse=True))}
    regions: list[SegmentedRegion] = []
    for index, mask in enumerate(masks):
        part = assignment.get(index)
        if part is not None and _is_title_text(part):
            continue
        if part is not None:
            name, description, importance = part.name, part.description, part.importance
        else:
            name, description = "", ""
            importance = max(1, 5 - rank[index])
        regions.append(_region_from_mask(mask, name, description, importance, ""))
    ordered = _sort_reading_order(regions)
    for position, region in enumerate(ordered, start=1):
        region.id = f"r{position}"
        if not region.name:
            region.name = f"Region {position}"
    return ordered


def _is_title_text(part: Part) -> bool:
    return re.search(r"\btitle\b", f"{part.name} {part.description}", re.IGNORECASE) is not None


def _keep_named_thin_regions(
    thin: list[np.ndarray],
    regions: list[SegmentedRegion],
    parts: Sequence[Part],
    shape: tuple[int, int],
    limit: int,
) -> None:
    """A thin band the model named, such as a crust ring, stays as an outline with no texture."""

    if thin and parts:
        used = {region.name.strip().lower() for region in regions}
        free = [part for part in parts if not _is_title_text(part) and part.name.strip().lower() not in used]
        height, width = shape
        pairs: list[tuple[float, int, int]] = []
        for mask_index, mask in enumerate(thin):
            for part_index, part in enumerate(free):
                score = _thin_score(mask, part, width, height)
                if score >= 0.08:
                    pairs.append((score + part.importance * 0.01, mask_index, part_index))
        pairs.sort(reverse=True)
        used_masks: set[int] = set()
        used_parts: set[int] = set()
        for _score, mask_index, part_index in pairs:
            if mask_index in used_masks or part_index in used_parts:
                continue
            used_masks.add(mask_index)
            used_parts.add(part_index)
            part = free[part_index]
            region = _region_from_mask(thin[mask_index], part.name, part.description, part.importance, "")
            region.outline_only = True
            regions.append(region)
        ordered = _sort_reading_order(regions)
        regions[:] = ordered
    while len(regions) > limit:
        index = next((i for i, region in enumerate(regions) if region.name.startswith("Region ")), None)
        if index is None:
            break
        regions.pop(index)
    for position, region in enumerate(regions, start=1):
        region.id = f"r{position}"
        if not region.name:
            region.name = f"Region {position}"


def _thin_score(mask: np.ndarray, part: Part, width: int, height: int) -> float:
    iou = _iou(_normalized_bbox(mask, width, height), part.approx_bbox)
    x1, y1, x2, y2 = part.approx_bbox.to_xyxy(width, height)
    window = mask[y1:y2, x1:x2]
    overlap = float((window > 0).mean()) if window.size else 0.0
    return max(iou, overlap)


def _match_parts(
    masks: list[np.ndarray],
    parts: Sequence[Part],
    shape: tuple[int, int],
) -> dict[int, Part]:
    height, width = shape
    pairs: list[tuple[float, int, int]] = []
    for mask_index, mask in enumerate(masks):
        mask_box = _normalized_bbox(mask, width, height)
        for part_index, part in enumerate(parts):
            score = _iou(mask_box, part.approx_bbox)
            if score >= 0.08:
                pairs.append((score + part.importance * 0.01, mask_index, part_index))
    pairs.sort(reverse=True)
    used_masks: set[int] = set()
    used_parts: set[int] = set()
    assignment: dict[int, Part] = {}
    part_list = list(parts)
    for _score, mask_index, part_index in pairs:
        if mask_index in used_masks or part_index in used_parts:
            continue
        used_masks.add(mask_index)
        used_parts.add(part_index)
        assignment[mask_index] = part_list[part_index]
    return assignment


def _normalized_bbox(mask: np.ndarray, width: int, height: int) -> tuple[float, float, float, float]:
    ys, xs = np.where(mask > 0)
    x0 = float(xs.min()) / width
    y0 = float(ys.min()) / height
    box_w = float(xs.max() - xs.min() + 1) / width
    box_h = float(ys.max() - ys.min() + 1) / height
    return x0, y0, box_w, box_h


def _iou(
    mask_box: tuple[float, float, float, float],
    part_box: object,
) -> float:
    ax, ay, aw, ah = mask_box
    bx, by, bw, bh = part_box.x, part_box.y, part_box.w, part_box.h  # type: ignore[attr-defined]
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _region_from_mask(
    mask: np.ndarray,
    name: str,
    description: str,
    importance: int,
    region_id: str,
) -> SegmentedRegion:
    ys, xs = np.where(mask > 0)
    x0, y0 = int(xs.min()), int(ys.min())
    bbox = (x0, y0, int(xs.max()) - x0 + 1, int(ys.max()) - y0 + 1)
    return SegmentedRegion(
        id=region_id,
        name=name,
        description=description,
        importance=int(importance),
        mask=mask,
        bbox=bbox,
    )


def _sort_reading_order(regions: list[SegmentedRegion]) -> list[SegmentedRegion]:
    def key(region: SegmentedRegion) -> tuple[float, float]:
        ys, xs = np.where(region.mask > 0)
        return float(ys.mean()), float(xs.mean())

    return sorted(regions, key=key)
