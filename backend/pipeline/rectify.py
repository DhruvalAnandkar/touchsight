"""Straighten a photographed page and honor JPEG EXIF orientation.

`correct_exif` decodes image bytes to BGR and applies orientation tags 1-8.
`rectify` perspective-warps a page when a large quadrilateral covers more than
30% of the frame. Later stages should use `RectifyResult.image`.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

_PAGE_AREA_RATIO = 0.30
_ORIENTATION_TAG = 0x0112


@dataclass
class RectifyResult:
    image: np.ndarray  # BGR, the image later stages should use
    status: str  # "complete" if warped, "skipped" if unchanged
    before: np.ndarray  # BGR copy of the input after EXIF correction, before the warp
    after: np.ndarray  # same as image
    transform: np.ndarray | None = None  # 3x3 map from the input pixels to `image`


def rectify(image_bgr: np.ndarray) -> RectifyResult:
    """If a large 4-sided page contour covers more than 30% of the image, four-point perspective-warp it to a rectangle and crop the page (the warp itself removes the background margins). Otherwise return the image unchanged with status skipped."""

    before = image_bgr.copy()
    quad = _find_page_quad(before)
    if quad is None:
        return RectifyResult(image=image_bgr, status="skipped", before=before, after=image_bgr)
    warped, transform = _warp_quad(before, quad)
    return RectifyResult(image=warped, status="complete", before=before, after=warped, transform=transform)


def correct_exif(data: bytes) -> np.ndarray:
    """Decode image bytes to BGR and apply EXIF orientation (tags 1-8). If there is no EXIF, behave like a normal decode (white composite for alpha is NOT required). Do not add a dependency: parse the JPEG APP1 orientation yourself or use Pillow only if `import PIL` already works in backend/.venv. If PIL is missing, implement the orientation tag read."""

    image = _decode_bgr(data)
    return _apply_orientation(image, _jpeg_orientation(data))


def _decode_bgr(data: bytes) -> np.ndarray:
    if not data:
        raise ValueError("empty image")
    buffer = np.frombuffer(data, dtype=np.uint8)
    # OpenCV applies EXIF itself unless this flag is set. Decode the stored
    # pixels, then apply tags 1-8 once below.
    flags = cv2.IMREAD_COLOR
    ignore = getattr(cv2, "IMREAD_IGNORE_ORIENTATION", 0)
    image = cv2.imdecode(buffer, flags | ignore)
    if image is None:
        raise ValueError("could not decode image")
    return image


def _apply_orientation(image: np.ndarray, orientation: int | None) -> np.ndarray:
    if orientation == 2:
        return cv2.flip(image, 1)
    if orientation == 3:
        return cv2.rotate(image, cv2.ROTATE_180)
    if orientation == 4:
        return cv2.flip(image, 0)
    if orientation == 5:
        return cv2.transpose(image)
    if orientation == 6:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if orientation == 7:
        return cv2.flip(cv2.transpose(image), -1)
    if orientation == 8:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def _jpeg_orientation(data: bytes) -> int | None:
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return None
    index = 2
    length = len(data)
    while index + 1 < length:
        if data[index] != 0xFF:
            return None
        while index < length and data[index] == 0xFF:
            index += 1
        if index >= length:
            return None
        marker = data[index]
        index += 1
        if marker in (0xD9, 0xDA):
            return None
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:
            continue
        if index + 2 > length:
            return None
        segment_length = int.from_bytes(data[index : index + 2], "big")
        if segment_length < 2 or index + segment_length > length:
            return None
        if marker == 0xE1:
            payload = data[index + 2 : index + segment_length]
            if payload.startswith(b"Exif\x00\x00"):
                orientation = _tiff_orientation(payload[6:])
                if orientation in range(1, 9):
                    return orientation
        index += segment_length
    return None


def _tiff_orientation(tiff: bytes) -> int | None:
    if len(tiff) < 8:
        return None
    if tiff[:2] == b"II":
        order = "little"
    elif tiff[:2] == b"MM":
        order = "big"
    else:
        return None

    def u16(offset: int) -> int:
        return int.from_bytes(tiff[offset : offset + 2], order)

    def u32(offset: int) -> int:
        return int.from_bytes(tiff[offset : offset + 4], order)

    if u16(2) != 42:
        return None
    ifd = u32(4)
    if ifd < 0 or ifd + 2 > len(tiff):
        return None
    count = u16(ifd)
    for entry in range(count):
        offset = ifd + 2 + entry * 12
        if offset + 12 > len(tiff):
            return None
        if u16(offset) != _ORIENTATION_TAG:
            continue
        value_type = u16(offset + 2)
        value_count = u32(offset + 4)
        if value_count != 1 or value_type not in (3, 4):
            continue
        if value_type == 3:
            return u16(offset + 8)
        return u32(offset + 8)
    return None


def _find_page_quad(image: np.ndarray) -> np.ndarray | None:
    height, width = image.shape[:2]
    if height < 4 or width < 4:
        return None
    for mask in _candidate_masks(image):
        quad = _largest_quad(mask, width, height)
        if quad is not None:
            return quad
    return None


def _candidate_masks(image: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    masks: list[np.ndarray] = []

    border = np.concatenate((gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]))
    difference = np.abs(gray.astype(np.float32) - float(np.median(border)))
    page = np.where(difference > 15.0, 255, 0).astype(np.uint8)
    page = cv2.morphologyEx(page, cv2.MORPH_CLOSE, kernel, iterations=2)
    masks.append(_filled(page))

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    closed = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
    masks.append(_filled(closed))
    masks.append(_filled(cv2.morphologyEx(cv2.bitwise_not(otsu), cv2.MORPH_CLOSE, kernel)))

    median = float(np.median(blurred))
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, max(lower + 1, 1.33 * median)))
    edges = cv2.Canny(blurred, lower, upper)
    edges = cv2.morphologyEx(cv2.dilate(edges, kernel), cv2.MORPH_CLOSE, kernel, iterations=2)
    masks.append(edges)
    return masks


def _filled(mask: np.ndarray) -> np.ndarray:
    contours = _contours(mask, cv2.RETR_EXTERNAL)
    filled = np.zeros(mask.shape[:2], dtype=np.uint8)
    if contours:
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def _largest_quad(mask: np.ndarray, width: int, height: int) -> np.ndarray | None:
    image_area = float(width * height)
    best_area = _PAGE_AREA_RATIO * image_area
    best: np.ndarray | None = None
    for contour in _contours(mask, cv2.RETR_LIST):
        if float(cv2.contourArea(contour)) <= best_area:
            continue
        quad = _as_quad(contour)
        if quad is None or _is_image_border(quad, width, height) or not _needs_perspective(quad):
            continue
        area = float(cv2.contourArea(quad.reshape(-1, 1, 2)))
        if area <= best_area:
            continue
        best_area = area
        best = quad
    return best


def _as_quad(contour: np.ndarray) -> np.ndarray | None:
    if len(contour) < 4:
        return None
    hull = cv2.convexHull(contour)
    for source in (hull, contour):
        peri = cv2.arcLength(source, True)
        if peri <= 0:
            continue
        for factor in (0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.08):
            approx = cv2.approxPolyDP(source, factor * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                return approx.reshape(4, 2).astype(np.float32)
    return _four_sharp_corners(hull)


def _four_sharp_corners(hull: np.ndarray) -> np.ndarray | None:
    points = hull.reshape(-1, 2).astype(np.float32)
    count = len(points)
    if count < 4:
        return None
    if count == 4:
        return points
    angles = np.empty(count, dtype=np.float32)
    for index in range(count):
        previous = points[(index - 1) % count] - points[index]
        nxt = points[(index + 1) % count] - points[index]
        denom = float(np.linalg.norm(previous) * np.linalg.norm(nxt))
        if denom < 1e-6:
            angles[index] = 180.0
            continue
        cosine = float(np.clip(np.dot(previous, nxt) / denom, -1.0, 1.0))
        angles[index] = np.degrees(np.arccos(cosine))
    chosen = sorted(np.argsort(angles)[:4].tolist())
    quad = points[chosen]
    contour = quad.reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour.astype(np.float32)):
        return None
    return quad


def _needs_perspective(quad: np.ndarray) -> bool:
    """A flat diagram can have a four-sided outline. Warp only a tilted or keystoned page."""

    ordered = _order_corners(quad)
    edges = [ordered[(index + 1) % 4] - ordered[index] for index in range(4)]
    deviations: list[float] = []
    for edge in edges:
        angle = abs(float(np.degrees(np.arctan2(float(edge[1]), float(edge[0]))))) % 180.0
        deviations.append(min(angle, abs(angle - 90.0), abs(angle - 180.0)))
    lengths = [float(np.linalg.norm(edge)) for edge in edges]

    def ratio(first: float, second: float) -> float:
        short = min(first, second)
        if short < 1.0:
            return 1.0
        return max(first, second) / short

    keystone = max(ratio(lengths[0], lengths[2]), ratio(lengths[1], lengths[3]))
    return max(deviations) >= 6.0 or keystone >= 1.15


def _is_image_border(quad: np.ndarray, width: int, height: int) -> bool:
    ordered = _order_corners(quad)
    expected = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    if bool(np.all(np.linalg.norm(ordered - expected, axis=1) <= 2.0)):
        return True
    area_ratio = float(cv2.contourArea(ordered.reshape(-1, 1, 2))) / float(width * height)
    touches = (
        float(ordered[:, 0].min()) <= 2.0
        and float(ordered[:, 1].min()) <= 2.0
        and float(ordered[:, 0].max()) >= width - 3.0
        and float(ordered[:, 1].max()) >= height - 3.0
    )
    return area_ratio >= 0.98 and touches


def _order_corners(points: np.ndarray) -> np.ndarray:
    quad = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = quad.sum(axis=1)
    ordered[0] = quad[int(np.argmin(sums))]
    ordered[2] = quad[int(np.argmax(sums))]
    diffs = quad[:, 1] - quad[:, 0]
    ordered[1] = quad[int(np.argmin(diffs))]
    ordered[3] = quad[int(np.argmax(diffs))]
    return ordered


def _warp_quad(image: np.ndarray, quad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    top_left, top_right, bottom_right, bottom_left = _order_corners(quad)
    width = max(np.linalg.norm(bottom_right - bottom_left), np.linalg.norm(top_right - top_left))
    height = max(np.linalg.norm(top_right - bottom_right), np.linalg.norm(top_left - bottom_left))
    destination_width = max(1, int(round(float(width))))
    destination_height = max(1, int(round(float(height))))
    destination = np.array(
        [
            [0, 0],
            [destination_width - 1, 0],
            [destination_width - 1, destination_height - 1],
            [0, destination_height - 1],
        ],
        dtype=np.float32,
    )
    source = np.stack((top_left, top_right, bottom_right, bottom_left)).astype(np.float32)
    transform = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(image, transform, (destination_width, destination_height)), transform


def _contours(mask: np.ndarray, mode: int) -> list[np.ndarray]:
    found = cv2.findContours(mask, mode, cv2.CHAIN_APPROX_SIMPLE)
    contours = found[0] if len(found) == 2 else found[1]
    return list(contours)
