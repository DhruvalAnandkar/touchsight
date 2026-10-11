"""Raised relief of a tactile sheet, as a watertight STL and preview images.

The plate is 2 mm. Each region is a pillow, about 0.3 mm at the edge and
about 1.6 mm on the medial axis. A boundary ridge adds another 1.5 mm across
about 2 px. Vessels stand an extra 0.8 mm proud. X and Y scale so the longer
side is at most ``max_mm``; Z stays in those millimetre heights.
"""

from __future__ import annotations

import io
import math
import struct
import zlib
from dataclasses import dataclass

import cv2
import numpy as np
from stl import Mode
from stl import mesh as stl_mesh

BASE_MM = 2.0
PILLOW_EDGE_MM = 0.3
PILLOW_PEAK_MM = 1.6
BOUNDARY_MM = 1.5
RIDGE_PX = 2.0
VESSEL_BOOST_MM = 0.8
ASPECT_THIN = 4.0
# A circle is 1. Thin ribbons fall well below a filled organ.
COMPACTNESS_THIN = 0.30
MAX_CELLS = 120
MAX_STL_BYTES = 5_000_000
_HEADER = b"TouchSight relief mesh".ljust(80, b" ")[:80]
_VESSEL_WORDS = ("artery", "vein", "vena", "aorta")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class ReliefPackage:
    stl: bytes
    heightmap_png: bytes
    region_id_png: bytes
    height_max_mm: float


def relief_stl(masks: list[np.ndarray], *, max_mm: float = 150.0) -> bytes:
    """Binary STL of a solid relief. The longer side is at most ``max_mm``."""

    heights = _build_heights(masks, names=None)
    return _stl_bytes(heights, max_mm)


def relief_package(
    masks: list[np.ndarray],
    names: list[str] | None = None,
    *,
    max_mm: float = 150.0,
) -> ReliefPackage:
    """STL, 16-bit heightmap, and 8-bit region ids for one tactile relief.

    ``heightmap_png`` stores millimetres as 16-bit gray: 0 is 0 mm and 65535
    is ``height_max_mm``. ``region_id_png`` is 0 on the background and N on
    mask N (1-based, later masks win overlaps).
    """

    heights = _build_heights(masks, names=names)
    labels = _labels(masks)
    peak = float(heights.max())
    return ReliefPackage(
        stl=_stl_bytes(heights, max_mm),
        heightmap_png=_png_gray16(_quantize_height(heights, peak)),
        region_id_png=_png_gray8(np.clip(labels, 0, 255).astype(np.uint8)),
        height_max_mm=peak,
    )


def _build_heights(masks: list[np.ndarray], names: list[str] | None) -> np.ndarray:
    if not masks:
        raise ValueError("relief_stl needs at least one mask")
    labels = _labels(masks)
    image_h, image_w = labels.shape
    if image_h < 2 or image_w < 2:
        raise ValueError("relief_stl needs a mask at least 2 pixels on each side")

    heights = np.full(labels.shape, BASE_MM, dtype=np.float64)
    for index, mask in enumerate(masks, start=1):
        region = labels == index
        if not np.any(region):
            continue
        binary = np.zeros(labels.shape, dtype=np.uint8)
        binary[region] = 255
        distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        peak = float(distance.max())
        if peak <= 0.0:
            curve = np.zeros(distance.shape, dtype=np.float64)
        else:
            t = np.clip(distance / peak, 0.0, 1.0)
            curve = 0.5 * (1.0 - np.cos(np.pi * t))
        pillow = PILLOW_EDGE_MM + (PILLOW_PEAK_MM - PILLOW_EDGE_MM) * curve
        name = None if names is None or index - 1 >= len(names) else names[index - 1]
        original = np.zeros(labels.shape, dtype=np.uint8)
        original[np.asarray(mask) > 0] = 255
        if _name_is_vessel(name) or _shape_is_vessel(original):
            pillow = pillow + VESSEL_BOOST_MM
        heights[region] = BASE_MM + pillow[region]

    boundary = _boundary(labels)
    if np.any(boundary):
        source = np.full(labels.shape, 255, dtype=np.uint8)
        source[boundary] = 0
        distance = cv2.distanceTransform(source, cv2.DIST_L2, 5)
        falloff = np.clip((RIDGE_PX - distance) / RIDGE_PX, 0.0, 1.0)
        heights += BOUNDARY_MM * falloff
    return heights


def _labels(masks: list[np.ndarray]) -> np.ndarray:
    labels = np.zeros(np.asarray(masks[0]).shape[:2], dtype=np.int32)
    for index, mask in enumerate(masks, start=1):
        labels[np.asarray(mask) > 0] = index
    return labels


def _name_is_vessel(name: str | None) -> bool:
    if not name:
        return False
    folded = str(name).casefold()
    return any(word in folded for word in _VESSEL_WORDS)


def _shape_is_vessel(binary: np.ndarray) -> bool:
    """Long, thin regions: a stretched box, or little area for the perimeter."""

    ys, xs = np.nonzero(binary)
    count = int(ys.size)
    if count == 0:
        return False
    box_h = int(ys.max() - ys.min()) + 1
    box_w = int(xs.max() - xs.min()) + 1
    short_side = max(1, min(box_h, box_w))
    if max(box_h, box_w) / short_side >= ASPECT_THIN:
        return True
    contours, _hierarchy = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = 0.0
    for contour in contours:
        perimeter += float(cv2.arcLength(contour, True))
    if perimeter <= 1.0:
        return False
    compactness = 4.0 * math.pi * float(count) / (perimeter * perimeter)
    return compactness < COMPACTNESS_THIN


def _quantize_height(heights: np.ndarray, peak: float) -> np.ndarray:
    if peak <= 0.0:
        return np.zeros(heights.shape, dtype=np.uint16)
    scaled = np.clip(np.rint(heights * (65535.0 / peak)), 0, 65535).astype(np.uint16)
    if int(scaled.max()) < 65535:
        scaled.reshape(-1)[int(np.argmax(heights))] = np.uint16(65535)
    return scaled


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)


def _png_gray(samples: np.ndarray, bit_depth: int) -> bytes:
    height, width = samples.shape
    raw = bytearray()
    if bit_depth == 16:
        packed = np.ascontiguousarray(samples, dtype=">u2")
        stride = width * 2
        blob = packed.tobytes()
        for y in range(height):
            raw.append(0)
            raw.extend(blob[y * stride : (y + 1) * stride])
    else:
        packed = np.ascontiguousarray(samples, dtype=np.uint8)
        for y in range(height):
            raw.append(0)
            raw.extend(packed[y].tobytes())
    ihdr = struct.pack(">IIBBBBB", width, height, bit_depth, 0, 0, 0, 0)
    compressed = zlib.compress(bytes(raw), 9)
    return (
        _PNG_SIGNATURE
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", compressed)
        + _png_chunk(b"IEND", b"")
    )


def _png_gray16(samples: np.ndarray) -> bytes:
    return _png_gray(samples, 16)


def _png_gray8(samples: np.ndarray) -> bytes:
    return _png_gray(samples, 8)


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


def _stl_bytes(heights: np.ndarray, max_mm: float) -> bytes:
    image_h, image_w = heights.shape
    stride = max(1, math.ceil(max(image_h, image_w) / MAX_CELLS))
    limit = max(image_h, image_w)
    while True:
        payload = _mesh_bytes(heights, stride, max_mm)
        if len(payload) <= MAX_STL_BYTES or stride >= limit:
            return payload
        stride += max(1, stride // 2)


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
