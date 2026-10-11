"""Decode uploads and hash them for the result cache."""

from __future__ import annotations

import hashlib

import cv2
import numpy as np


def image_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def decode_image(data: bytes, max_side: int = 2400) -> np.ndarray:
    """Return a BGR image. Transparent pixels are composited onto white."""

    if not data:
        raise ValueError("That file is empty.")
    array = np.frombuffer(data, dtype=np.uint8)
    decoded = cv2.imdecode(array, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise ValueError("That file is not an image we can read.")
    if decoded.ndim == 2:
        image = cv2.cvtColor(decoded, cv2.COLOR_GRAY2BGR)
    elif decoded.shape[2] == 4:
        image = _composite_on_white(decoded)
    else:
        image = decoded[:, :, :3]
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest > max_side:
        scale = max_side / longest
        image = cv2.resize(
            image,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return np.ascontiguousarray(image)


def _composite_on_white(bgra: np.ndarray) -> np.ndarray:
    color = bgra[:, :, :3].astype(np.float32)
    alpha = bgra[:, :, 3:4].astype(np.float32) / 255.0
    white = np.full_like(color, 255.0)
    return (color * alpha + white * (1.0 - alpha)).astype(np.uint8)
