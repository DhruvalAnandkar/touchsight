"""Drawings used by the segmentation tests."""

from __future__ import annotations

import cv2
import numpy as np


def bar_chart() -> np.ndarray:
    image = np.full((420, 560, 3), 255, np.uint8)
    colors = [(40, 90, 200), (40, 160, 90), (30, 180, 220), (180, 80, 40)]
    heights = [220, 140, 280, 100]
    for index, (color, height) in enumerate(zip(colors, heights)):
        x = 60 + index * 120
        cv2.rectangle(image, (x, 340 - height), (x + 70, 340), color, thickness=-1)
    return image


def line_cell() -> np.ndarray:
    image = np.full((480, 480, 3), 255, np.uint8)
    ink = (20, 20, 20)
    cv2.circle(image, (240, 240), 180, ink, 3)
    cv2.circle(image, (240, 230), 70, ink, 3)
    cv2.ellipse(image, (150, 160), (40, 22), 30, 0, 360, ink, 3)
    cv2.ellipse(image, (330, 300), (46, 24), -20, 0, 360, ink, 3)
    cv2.circle(image, (300, 150), 22, ink, 3)
    return image


def separated_circles(count: int = 10) -> np.ndarray:
    image = np.full((640, 640, 3), 255, np.uint8)
    for index in range(count):
        row, col = divmod(index, 5)
        center = (70 + col * 120, 80 + row * 280)
        cv2.circle(image, center, 36, (30, 80, 160), thickness=-1)
    return image


def encode_png(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("png encode failed")
    return buffer.tobytes()
