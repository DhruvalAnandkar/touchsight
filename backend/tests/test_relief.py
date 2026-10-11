"""Watertight relief mesh from two adjacent squares."""

from __future__ import annotations

import io

import numpy as np
from stl import mesh

from pipeline.relief import relief_stl


def test_adjacent_squares_make_a_watertight_relief() -> None:
    canvas = (36, 48)
    left = np.zeros(canvas, dtype=np.uint8)
    right = np.zeros(canvas, dtype=np.uint8)
    left[6:30, 4:24] = 255
    right[6:30, 24:44] = 255

    payload = relief_stl([left, right])
    assert payload == relief_stl([left, right])
    assert len(payload) < 5_000_000

    solid = mesh.Mesh.from_file("touchsight-relief.stl", fh=io.BytesIO(payload))
    points = solid.vectors.reshape(-1, 3)
    span_x = float(points[:, 0].max() - points[:, 0].min())
    span_y = float(points[:, 1].max() - points[:, 1].min())
    assert span_x <= 150.0
    assert span_y <= 150.0
    # Base plate is 2 mm. Boundary ridges are the high points.
    assert float(points[:, 2].max()) > 2.0
    _assert_watertight(solid)


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
