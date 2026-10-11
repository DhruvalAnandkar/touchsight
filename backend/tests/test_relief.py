"""Watertight relief mesh, pillowed heights, and region ids."""

from __future__ import annotations

import io

import cv2
import numpy as np
from stl import mesh

from pipeline.relief import ReliefPackage, relief_package, relief_stl


def test_adjacent_squares_make_a_watertight_relief() -> None:
    canvas = (36, 48)
    left = np.zeros(canvas, dtype=np.uint8)
    right = np.zeros(canvas, dtype=np.uint8)
    left[6:30, 4:24] = 255
    right[6:30, 24:44] = 255

    payload = relief_stl([left, right])
    assert payload == relief_stl([left, right])
    assert payload == relief_package([left, right]).stl
    assert len(payload) < 5_000_000

    solid = mesh.Mesh.from_file("touchsight-relief.stl", fh=io.BytesIO(payload))
    points = solid.vectors.reshape(-1, 3)
    span_x = float(points[:, 0].max() - points[:, 0].min())
    span_y = float(points[:, 1].max() - points[:, 1].min())
    assert span_x <= 150.0
    assert span_y <= 150.0
    # Base plate is 2 mm. Pillows and boundary ridges rise above it.
    assert float(points[:, 2].max()) > 2.0
    _assert_watertight(solid)


def test_disk_center_pillows_above_the_rim() -> None:
    mask = _disk((180, 180), (90, 90), 70)
    package = relief_package([mask], ["Atrium"])
    image = _decode(package.heightmap_png)
    assert image.dtype == np.uint16
    assert image.ndim == 2
    mm = _height_mm(package, image)
    # Far background is the 2 mm plate. 65535 is the peak millimetre.
    assert abs(float(mm.min()) - 2.0) < 0.02
    assert abs(float(mm.max()) - package.height_max_mm) < 0.02
    center = float(mm[90, 90])
    near_edge = float(mm[90, 90 + 58])
    assert center > near_edge
    assert abs(center - 3.6) < 0.12


def test_boundary_ridge_is_about_1_5mm_above_the_region_edge() -> None:
    mask = _disk((180, 180), (90, 90), 70)
    mm = _height_mm(relief_package([mask], ["Atrium"]))
    row = 90
    inside = np.flatnonzero(mask[row] > 0)
    edge_x = int(inside.max())
    # Just past the 2 px ridge, still on the rim of the pillow.
    inner_x = edge_x - 6
    delta = float(mm[row, edge_x] - mm[row, inner_x])
    assert abs(delta - 1.5) < 0.35


def test_named_artery_stands_above_a_similar_atrium() -> None:
    mask = _disk((96, 96), (48, 48), 30)
    artery = _height_mm(relief_package([mask], ["Coronary artery"]))
    atrium = _height_mm(relief_package([mask], ["Atrium"]))
    assert float(artery[48, 48]) > float(atrium[48, 48])
    assert abs(float(artery[48, 48] - atrium[48, 48]) - 0.8) < 0.05


def test_long_thin_mask_stands_proud_without_a_name() -> None:
    thin = np.zeros((48, 160), dtype=np.uint8)
    thin[14:34, 8:152] = 255
    disk = _disk((80, 80), (40, 40), 24)
    thin_mm = _height_mm(relief_package([thin]))
    disk_mm = _height_mm(relief_package([disk]))
    assert float(thin_mm[24, 80]) > float(disk_mm[40, 40])
    assert abs(float(thin_mm[24, 80] - disk_mm[40, 40]) - 0.8) < 0.15


def test_region_ids_follow_mask_order() -> None:
    canvas = (36, 48)
    left = np.zeros(canvas, dtype=np.uint8)
    right = np.zeros(canvas, dtype=np.uint8)
    left[6:30, 4:24] = 255
    right[6:30, 24:44] = 255
    package = relief_package([left, right], ["Atrium", "Ventricle"])
    ids = _decode(package.region_id_png)
    assert ids.dtype == np.uint8
    assert ids.ndim == 2
    assert int(ids[0, 0]) == 0
    assert int(ids[18, 10]) == 1
    assert int(ids[18, 30]) == 2


def _disk(shape: tuple[int, int], center: tuple[int, int], radius: int) -> np.ndarray:
    canvas = np.zeros(shape, dtype=np.uint8)
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    canvas[(yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= radius * radius] = 255
    return canvas


def _decode(payload: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert image is not None
    return image


def _height_mm(package: ReliefPackage, image: np.ndarray | None = None) -> np.ndarray:
    decoded = _decode(package.heightmap_png) if image is None else image
    assert decoded.dtype == np.uint16
    return decoded.astype(np.float64) / 65535.0 * package.height_max_mm


def _assert_watertight(solid: mesh.Mesh) -> None:
    """Each undirected edge is used by exactly two triangles."""

    counts: dict[tuple[tuple[float, float, float], tuple[float, float, float]], int] = {}
    for triangle in solid.vectors:
        for index in range(3):
            key = _edge_key(triangle[index], triangle[(index + 1) % 3])
            counts[key] = counts.get(key, 0) + 1
    assert counts, "mesh has no edges"
    broken = [count for count in counts.values() if count != 2]
    assert not broken, f"{len(broken)} edges are not shared by exactly two triangles"


def _edge_key(
    a: np.ndarray, b: np.ndarray
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    start = (float(a[0]), float(a[1]), float(a[2]))
    end = (float(b[0]), float(b[1]), float(b[2]))
    return (start, end) if start <= end else (end, start)
