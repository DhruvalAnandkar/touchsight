"""Shared-border graph for a tactile outline.

Each region is painted into one label image. An edge is emitted wherever two
pixels disagree, including the step from a region to the background. Colinear
pixel steps collapse into one segment, so a wall that two regions share is
drawn once.
"""

from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np

Point = tuple[int, int]
SNAP_MM = 1.5


def label_map(masks: list[np.ndarray]) -> np.ndarray:
    labels = np.zeros(masks[0].shape[:2], dtype=np.int32)
    for index, mask in enumerate(masks, start=1):
        labels[mask > 0] = index
    return labels


def close_cracks(labels: np.ndarray, radius_px: int) -> np.ndarray:
    """Let regions meet across a hairline gap, and leave the page background alone.

    Watershed often leaves a one-pixel crack between neighbors. That crack is
    not connected to the border of the sheet. Pixels in it take the nearest
    region, so the shared wall is a single edge. The white margin stays white.
    """

    if radius_px <= 0 or int(labels.max()) < 1:
        return labels
    height, width = labels.shape
    unlabeled = np.where(labels == 0, 255, 0).astype(np.uint8)
    flood = unlabeled.copy()
    flood_mask = np.zeros((height + 2, width + 2), np.uint8)
    for point in ((0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)):
        if flood[point[1], point[0]] > 0:
            cv2.floodFill(flood, flood_mask, point, 128)
    cracks = flood == 255
    if not np.any(cracks):
        return labels
    best_dist = np.full(labels.shape, np.inf, dtype=np.float32)
    best_label = np.zeros(labels.shape, dtype=np.int32)
    for label in range(1, int(labels.max()) + 1):
        region = np.where(labels == label, 0, 255).astype(np.uint8)
        distance = cv2.distanceTransform(region, cv2.DIST_L2, 5)
        closer = cracks & (distance < best_dist)
        best_dist[closer] = distance[closer]
        best_label[closer] = label
    filled = labels.copy()
    near = cracks & (best_dist <= float(radius_px))
    filled[near] = best_label[near]
    return filled


def boundary_paths(labels: np.ndarray) -> list[list[Point]]:
    """Pixel-corner paths. Loops are closed (first point repeated)."""

    neighbors = _edge_graph(labels)
    if not neighbors:
        return []
    junctions = {point for point, adjacent in neighbors.items() if len(adjacent) != 2}
    used: set[frozenset[Point]] = set()
    paths: list[list[Point]] = []
    for start in junctions:
        for nxt in list(neighbors[start]):
            key = frozenset((start, nxt))
            if key in used:
                continue
            path = [start, nxt]
            used.add(key)
            while path[-1] not in junctions:
                current = path[-1]
                previous = path[-2]
                options = [
                    candidate
                    for candidate in neighbors[current]
                    if candidate != previous and frozenset((current, candidate)) not in used
                ]
                if len(options) != 1:
                    break
                used.add(frozenset((current, options[0])))
                path.append(options[0])
            if len(path) >= 2:
                paths.append(path)
    paths.extend(_loops(neighbors, used))
    return paths


def snap_points(
    points: list[tuple[float, float]],
    tolerance_mm: float = SNAP_MM,
) -> list[tuple[float, float]]:
    """Cluster vertices closer than `tolerance_mm`. Not used on dense pixel chains."""

    if not points:
        return []
    parent = list(range(len(points)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    cell = max(tolerance_mm, 1e-6)
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, (x_mm, y_mm) in enumerate(points):
        buckets[(int(np.floor(x_mm / cell)), int(np.floor(y_mm / cell)))].append(index)
    for index, (x_mm, y_mm) in enumerate(points):
        cx, cy = int(np.floor(x_mm / cell)), int(np.floor(y_mm / cell))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other in buckets.get((cx + dx, cy + dy), []):
                    if other <= index:
                        continue
                    ox, oy = points[other]
                    if (ox - x_mm) ** 2 + (oy - y_mm) ** 2 <= tolerance_mm ** 2:
                        parent[find(other)] = find(index)
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(points)):
        groups[find(index)].append(index)
    snapped = [(0.0, 0.0)] * len(points)
    for members in groups.values():
        mean_x = sum(points[index][0] for index in members) / len(members)
        mean_y = sum(points[index][1] for index in members) / len(members)
        point = (round(mean_x, 2), round(mean_y, 2))
        for index in members:
            snapped[index] = point
    return snapped


def _edge_graph(labels: np.ndarray) -> dict[Point, set[Point]]:
    height, width = labels.shape
    neighbors: dict[Point, set[Point]] = defaultdict(set)

    def add(start: Point, end: Point) -> None:
        if start == end:
            return
        neighbors[start].add(end)
        neighbors[end].add(start)

    for y in range(height):
        row = labels[y]
        for x in range(width - 1):
            if row[x] != row[x + 1]:
                add((x + 1, y), (x + 1, y + 1))
        if row[0] != 0:
            add((0, y), (0, y + 1))
        if row[width - 1] != 0:
            add((width, y), (width, y + 1))
    for y in range(height - 1):
        for x in range(width):
            if labels[y, x] != labels[y + 1, x]:
                add((x, y + 1), (x + 1, y + 1))
    for x in range(width):
        if labels[0, x] != 0:
            add((x, 0), (x + 1, 0))
        if labels[height - 1, x] != 0:
            add((x, height), (x + 1, height))
    return neighbors


def _loops(neighbors: dict[Point, set[Point]], used: set[frozenset[Point]]) -> list[list[Point]]:
    loops: list[list[Point]] = []
    for start, adjacent in neighbors.items():
        for nxt in adjacent:
            key = frozenset((start, nxt))
            if key in used:
                continue
            path = [start, nxt]
            used.add(key)
            previous, current = start, nxt
            while current != start:
                options = [
                    candidate
                    for candidate in neighbors[current]
                    if candidate != previous and frozenset((current, candidate)) not in used
                ]
                if not options:
                    break
                previous, current = current, options[0]
                used.add(frozenset((previous, current)))
                path.append(current)
            if len(path) >= 4 and path[0] == path[-1]:
                loops.append(path)
    return loops


def drop_repeats(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    kept: list[tuple[float, float]] = []
    for point in points:
        if not kept or point != kept[-1]:
            kept.append(point)
    return kept
