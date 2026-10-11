"""Raised relief of a tactile sheet, as a watertight STL.

The plate is 2 mm. Pixels inside a region gain another 1 mm, and a boundary
gains another 3 mm. ``cv2.distanceTransform`` spreads that ridge about 2 px
so it is wide enough to print. X and Y scale so the longer side of the mask
is at most ``max_mm``; Z stays in those millimetre heights.
"""

from __future__ import annotations

import io
import math

import cv2
import numpy as np
from stl import Mode
from stl import mesh as stl_mesh

BASE_MM = 2.0
REGION_MM = 1.0
BOUNDARY_MM = 3.0
RIDGE_PX = 2.0
MAX_CELLS = 120
MAX_STL_BYTES = 5_000_000
_HEADER = b"TouchSight relief mesh".ljust(80, b" ")[:80]


def relief_stl(masks: list[np.ndarray], *, max_mm: float = 150.0) -> bytes:
    """Binary STL of a solid relief. The longer side is at most ``max_mm``."""

    if not masks:
        raise ValueError("relief_stl needs at least one mask")
    heights = _heightmap(masks)
    image_h, image_w = heights.shape
    if image_h < 2 or image_w < 2:
        raise ValueError("relief_stl needs a mask at least 2 pixels on each side")

    stride = max(1, math.ceil(max(image_h, image_w) / MAX_CELLS))
    limit = max(image_h, image_w)
    while True:
        payload = _mesh_bytes(heights, stride, max_mm)
        if len(payload) <= MAX_STL_BYTES or stride >= limit:
            return payload
        stride += max(1, stride // 2)


def _heightmap(masks: list[np.ndarray]) -> np.ndarray:
    labels = np.zeros(masks[0].shape[:2], dtype=np.int32)
    for index, mask in enumerate(masks, start=1):
        labels[np.asarray(mask) > 0] = index

    heights = np.full(labels.shape, BASE_MM, dtype=np.float64)
    heights[labels > 0] += REGION_MM

    boundary = _boundary(labels)
    if np.any(boundary):
        source = np.full(labels.shape, 255, dtype=np.uint8)
        source[boundary] = 0
        distance = cv2.distanceTransform(source, cv2.DIST_L2, 5)
        falloff = np.clip((RIDGE_PX - distance) / RIDGE_PX, 0.0, 1.0)
        heights += BOUNDARY_MM * falloff
    return heights


def _boundary(labels: np.ndarray) -> np.ndarray:
    """Pixels whose 4-neighbor carries a different label, including background."""

    changed = np.zeros(labels.shape, dtype=bool)
    horizontal = labels[:, 1:] != labels[:, :-1]
    changed[:, 1:] |= horizontal
    changed[:, :-1] |= horizontal
    vertical = labels[1:, :] != labels[:-1, :]
    changed[1:, :] |= vertical
    changed[:-1, :] |= vertical
    return changed


def _samples(length: int, stride: int) -> np.ndarray:
    step = max(1, stride)
    index = np.arange(0, length, step, dtype=np.int32)
    if int(index[-1]) != length - 1:
        index = np.append(index, np.int32(length - 1))
    return index


def _triangle_count(n_rows: int, n_cols: int) -> int:
    cells_x = n_cols - 1
    cells_y = n_rows - 1
    if cells_x < 1 or cells_y < 1:
        return 0
    return 4 * cells_x * cells_y + 4 * (cells_x + cells_y)


def _mesh_bytes(heights: np.ndarray, stride: int, max_mm: float) -> bytes:
    image_h, image_w = heights.shape
    row_index = _samples(image_h, stride)
    col_index = _samples(image_w, stride)
    sampled = heights[np.ix_(row_index, col_index)]

    long_px = float(max(image_h, image_w))
    plate_w = image_w / long_px * float(max_mm)
    plate_h = image_h / long_px * float(max_mm)
    xs = _axis_mm(col_index, image_w, plate_w)
    ys = _axis_mm(row_index, image_h, plate_h)
    triangles = _solid(xs, ys, sampled.astype(np.float32))

    data = np.zeros(triangles.shape[0], dtype=stl_mesh.Mesh.dtype)
    data["vectors"] = triangles
    solid = stl_mesh.Mesh(data, calculate_normals=False, remove_empty_areas=False)
    buffer = io.BytesIO()
    solid.save(
        "touchsight-relief.stl",
        fh=buffer,
        mode=Mode.BINARY,
        update_normals=True,
    )
    raw = bytearray(buffer.getvalue())
    raw[:80] = _HEADER
    return bytes(raw)


def _axis_mm(index: np.ndarray, length: int, plate_mm: float) -> np.ndarray:
    if length <= 1:
        return np.zeros(len(index), dtype=np.float32)
    position = index.astype(np.float64) / float(length - 1) * plate_mm
    return position.astype(np.float32)


def _solid(xs: np.ndarray, ys: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Top grid, vertical walls, and a bottom cap. Each edge meets two faces."""

    rows, cols = z.shape
    n_top = rows * cols
    top = np.empty((rows, cols, 3), dtype=np.float32)
    top[:, :, 0] = xs[np.newaxis, :]
    top[:, :, 1] = ys[:, np.newaxis]
    top[:, :, 2] = z
    bottom = top.copy()
    bottom[:, :, 2] = np.float32(0.0)
    verts = np.concatenate((top.reshape(-1, 3), bottom.reshape(-1, 3)), axis=0)

    count = _triangle_count(rows, cols)
    faces = np.empty((count, 3), dtype=np.int32)
    cursor = 0

    def vid(r: int, c: int, on_top: bool) -> int:
        return r * cols + c + (0 if on_top else n_top)

    def push(a: int, b: int, c: int) -> None:
        nonlocal cursor
        faces[cursor, 0] = a
        faces[cursor, 1] = b
        faces[cursor, 2] = c
        cursor += 1

    for r in range(rows - 1):
        for c in range(cols - 1):
            v00 = vid(r, c, True)
            v10 = vid(r, c + 1, True)
            v01 = vid(r + 1, c, True)
            v11 = vid(r + 1, c + 1, True)
            push(v00, v10, v11)
            push(v00, v11, v01)

    for r in range(rows - 1):
        for c in range(cols - 1):
            v00 = vid(r, c, False)
            v10 = vid(r, c + 1, False)
            v01 = vid(r + 1, c, False)
            v11 = vid(r + 1, c + 1, False)
            push(v00, v11, v10)
            push(v00, v01, v11)

    for c in range(cols - 1):
        t0, t1 = vid(0, c, True), vid(0, c + 1, True)
        b0, b1 = vid(0, c, False), vid(0, c + 1, False)
        push(t0, b0, b1)
        push(t0, b1, t1)

    last_r = rows - 1
    for c in range(cols - 1):
        t0, t1 = vid(last_r, c, True), vid(last_r, c + 1, True)
        b0, b1 = vid(last_r, c, False), vid(last_r, c + 1, False)
        push(t0, t1, b1)
        push(t0, b1, b0)

    for r in range(rows - 1):
        t0, t1 = vid(r, 0, True), vid(r + 1, 0, True)
        b0, b1 = vid(r, 0, False), vid(r + 1, 0, False)
        push(t0, t1, b1)
        push(t0, b1, b0)

    last_c = cols - 1
    for r in range(rows - 1):
        t0, t1 = vid(r, last_c, True), vid(r + 1, last_c, True)
        b0, b1 = vid(r, last_c, False), vid(r + 1, last_c, False)
        push(t0, b0, b1)
        push(t0, b1, t1)

    if cursor != count:
        raise RuntimeError(f"relief mesh built {cursor} triangles, expected {count}")
    return verts[faces]
