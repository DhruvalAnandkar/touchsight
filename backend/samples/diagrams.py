"""Original tactile-diagram drawings.

These are drawn for TouchSight and dedicated to the public domain (CC0 1.0).
See samples/CREDITS.md. The pipeline consumes the raster; the SVG files are
the same geometry, written so the source is a drawing rather than a photo.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

INK = (28, 28, 28)
WHITE = (255, 255, 255)


def heart() -> np.ndarray:
    """Four-chamber heart. Chambers share walls."""

    image = np.full((1040, 860, 3), WHITE, np.uint8)
    cleft = (430, 190)
    av_l = (175, 470)
    av_c = (430, 500)
    av_r = (700, 470)
    apex = (430, 880)
    chambers = [
        [cleft, (300, 145), (175, 220), (145, 350), av_l, av_c],
        [cleft, av_c, av_r, (720, 350), (690, 220), (560, 145)],
        [av_l, av_c, apex, (275, 700), (195, 560)],
        [av_c, av_r, (675, 560), (595, 720), apex],
    ]
    colors = [
        (200, 130, 55),
        (70, 70, 200),
        (150, 80, 30),
        (35, 35, 145),
    ]
    for poly, color in zip(chambers, colors):
        cv2.fillPoly(image, [np.array(poly, dtype=np.int32)], color)
    # A one-pixel wall is enough for Canny and stays inside the crack closer,
    # so the shared border is one stroke instead of a ring that watershed keeps.
    septum = np.array([cleft, av_c, apex], dtype=np.int32)
    groove = np.array([av_l, av_c, av_r], dtype=np.int32)
    cv2.polylines(image, [septum, groove], False, INK, 2)
    return image


def animal_cell() -> np.ndarray:
    """Animal cell: cytoplasm, nucleus, nucleolus, and three organelles."""

    image = np.full((1000, 1000, 3), WHITE, np.uint8)
    # Fills sit well inside one another so a thin stroke stays an edge, not a region.
    cv2.ellipse(image, (500, 500), (380, 300), 0, 0, 360, (180, 210, 235), thickness=-1)
    cv2.circle(image, (390, 470), 108, (170, 80, 150), thickness=-1)
    cv2.circle(image, (420, 450), 32, (90, 30, 95), thickness=-1)
    cv2.ellipse(image, (730, 580), (78, 52), 16, 0, 360, (210, 140, 40), thickness=-1)
    cv2.ellipse(image, (310, 610), (64, 28), -20, 0, 360, (40, 110, 190), thickness=-1)
    cv2.ellipse(image, (640, 330), (74, 30), 10, 0, 360, (50, 150, 70), thickness=-1)
    cv2.ellipse(image, (500, 500), (380, 300), 0, 0, 360, INK, thickness=4)
    cv2.circle(image, (390, 470), 108, INK, thickness=4)
    cv2.circle(image, (420, 450), 32, INK, thickness=4)
    cv2.ellipse(image, (730, 580), (78, 52), 16, 0, 360, INK, thickness=4)
    cv2.ellipse(image, (310, 610), (64, 28), -20, 0, 360, INK, thickness=4)
    cv2.ellipse(image, (640, 330), (74, 30), 10, 0, 360, INK, thickness=4)
    return image


def world_map() -> np.ndarray:
    """Six landmasses. The Americas share an isthmus; Europe touches Africa and Asia."""

    image = np.full((720, 1280, 3), WHITE, np.uint8)
    lands = [
        # North America, sharing the isthmus edge with South America.
        [(90, 150), (190, 70), (320, 55), (400, 130), (360, 210), (300, 275), (235, 265), (190, 210), (120, 190)],
        [(235, 265), (300, 275), (345, 360), (330, 470), (270, 580), (200, 500), (165, 390), (195, 310)],
        # Europe, sharing one edge with Africa and one with Asia.
        [(520, 95), (650, 75), (735, 125), (710, 185), (640, 205), (575, 200), (500, 155)],
        [(575, 200), (640, 205), (760, 300), (690, 490), (540, 510), (485, 360), (530, 250)],
        [(735, 125), (980, 70), (1120, 130), (1100, 250), (940, 300), (790, 230), (710, 185)],
        # Australia, separated by ocean.
        [(980, 390), (1120, 370), (1170, 450), (1090, 520), (960, 490), (940, 430)],
    ]
    colors = [
        (150, 130, 50),
        (90, 150, 60),
        (80, 140, 190),
        (50, 100, 190),
        (40, 160, 170),
        (70, 80, 170),
    ]
    _fill_polygons(image, lands, colors, stroke=4)
    return image


def write_source_svgs(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "heart.svg").write_text(_heart_svg(), encoding="utf-8")
    (directory / "animal-cell.svg").write_text(_cell_svg(), encoding="utf-8")
    (directory / "world-map.svg").write_text(_map_svg(), encoding="utf-8")


def _fill_polygons(
    image: np.ndarray,
    polygons: list[list[tuple[int, int]]],
    colors: list[tuple[int, int, int]],
    stroke: int,
) -> None:
    arrays = [np.array(poly, dtype=np.int32) for poly in polygons]
    for poly, color in zip(arrays, colors):
        cv2.fillPoly(image, [poly], color)
    for poly in arrays:
        cv2.polylines(image, [poly], True, INK, stroke)


def _pts(points: list[tuple[int, int]]) -> str:
    return " ".join(f"{x},{y}" for x, y in points)


def _heart_svg() -> str:
    cleft, av_l, av_c, av_r, apex = (430, 190), (175, 470), (430, 500), (700, 470), (430, 880)
    chambers = [
        ("#3782c8", [cleft, (300, 145), (175, 220), (145, 350), av_l, av_c]),
        ("#c84646", [cleft, av_c, av_r, (720, 350), (690, 220), (560, 145)]),
        ("#1e5096", [av_l, av_c, apex, (275, 700), (195, 560)]),
        ("#912323", [av_c, av_r, (675, 560), (595, 720), apex]),
    ]
    walls = "\n".join(
        f'  <polygon points="{_pts(points)}" fill="{color}" stroke="none"/>' for color, points in chambers
    )
    walls += (
        f'\n  <polyline points="{_pts([cleft, av_c, apex])}" fill="none" stroke="#1c1c1c" stroke-width="2"/>'
        f'\n  <polyline points="{_pts([av_l, av_c, av_r])}" fill="none" stroke="#1c1c1c" stroke-width="2"/>'
    )
    return _svg(860, 1040, "Heart, four chambers", walls)


def _cell_svg() -> str:
    body = """
  <ellipse cx="500" cy="500" rx="380" ry="300" fill="#ebd2b4" stroke="#1c1c1c" stroke-width="4"/>
  <circle cx="390" cy="470" r="108" fill="#9650aa" stroke="#1c1c1c" stroke-width="4"/>
  <circle cx="420" cy="450" r="32" fill="#5f1e5a" stroke="#1c1c1c" stroke-width="4"/>
  <ellipse cx="730" cy="580" rx="78" ry="52" transform="rotate(16 730 580)" fill="#288cd2" stroke="#1c1c1c" stroke-width="4"/>
  <ellipse cx="310" cy="610" rx="64" ry="28" transform="rotate(-20 310 610)" fill="#be6e28" stroke="#1c1c1c" stroke-width="4"/>
  <ellipse cx="640" cy="330" rx="74" ry="30" transform="rotate(10 640 330)" fill="#469632" stroke="#1c1c1c" stroke-width="4"/>
""".strip("\n")
    return _svg(1000, 1000, "Animal cell", body)


def _map_svg() -> str:
    lands = [
        ("#328296", [(90, 150), (190, 70), (320, 55), (400, 130), (360, 210), (300, 275), (235, 265), (190, 210), (120, 190)]),
        ("#3c965a", [(235, 265), (300, 275), (345, 360), (330, 470), (270, 580), (200, 500), (165, 390), (195, 310)]),
        ("#be8c50", [(520, 95), (650, 75), (735, 125), (710, 185), (640, 205), (575, 200), (500, 155)]),
        ("#be6432", [(575, 200), (640, 205), (760, 300), (690, 490), (540, 510), (485, 360), (530, 250)]),
        ("#aaa028", [(735, 125), (980, 70), (1120, 130), (1100, 250), (940, 300), (790, 230), (710, 185)]),
        ("#aa5046", [(980, 390), (1120, 370), (1170, 450), (1090, 520), (960, 490), (940, 430)]),
    ]
    body = "\n".join(
        f'  <polygon points="{_pts(points)}" fill="{color}" stroke="#1c1c1c" stroke-width="4" stroke-linejoin="round"/>'
        for color, points in lands
    )
    return _svg(1280, 720, "World map", body)


def _svg(width: int, height: int, title: str, body: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">\n'
        f"  <title>{title}</title>\n"
        f'  <rect width="100%" height="100%" fill="#ffffff"/>\n'
        f"{body}\n"
        "</svg>\n"
    )
