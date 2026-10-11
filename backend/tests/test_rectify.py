"""Page rectification and JPEG EXIF orientation."""

from __future__ import annotations

import cv2
import numpy as np

from pipeline.rectify import correct_exif, rectify


def _page_with_marker() -> np.ndarray:
    page = np.full((480, 360, 3), 245, np.uint8)
    page[36:120, 28:150] = (20, 20, 20)
    page[150:230, 180:320] = (30, 40, 220)
    page[280:340, 40:200] = (210, 70, 40)
    page[380:440, 200:330] = (40, 180, 60)
    page[40:70, 220:250] = (0, 220, 220)
    return page


def _tilted_capture(page: np.ndarray) -> tuple[np.ndarray, float]:
    page_height, page_width = page.shape[:2]
    canvas_height, canvas_width = 1000, 800
    source = np.array(
        [[0, 0], [page_width - 1, 0], [page_width - 1, page_height - 1], [0, page_height - 1]],
        dtype=np.float32,
    )
    destination = np.array(
        [[90, 80], [690, 120], [650, 900], [70, 820]],
        dtype=np.float32,
    )
    area_ratio = float(cv2.contourArea(destination.reshape(-1, 1, 2))) / float(canvas_width * canvas_height)
    transform = cv2.getPerspectiveTransform(source, destination)
    canvas = np.full((canvas_height, canvas_width, 3), 138, np.uint8)
    warped = cv2.warpPerspective(page, transform, (canvas_width, canvas_height))
    mask = cv2.warpPerspective(np.full(page.shape[:2], 255, np.uint8), transform, (canvas_width, canvas_height))
    canvas[mask > 0] = warped[mask > 0]
    return canvas, area_ratio


def _normalized_correlation(actual: np.ndarray, expected: np.ndarray) -> float:
    resized = cv2.resize(actual, (expected.shape[1], expected.shape[0]), interpolation=cv2.INTER_AREA)
    margin = max(4, int(0.05 * min(expected.shape[:2])))
    left = resized[margin:-margin, margin:-margin].astype(np.float32)
    right = expected[margin:-margin, margin:-margin].astype(np.float32)
    left_flat = left.ravel() - left.mean()
    right_flat = right.ravel() - right.mean()
    denom = float(np.linalg.norm(left_flat) * np.linalg.norm(right_flat))
    return float(left_flat @ right_flat) / denom


def _mean_abs(actual: np.ndarray, expected: np.ndarray) -> float:
    resized = cv2.resize(actual, (expected.shape[1], expected.shape[0]), interpolation=cv2.INTER_AREA)
    margin = max(4, int(0.05 * min(expected.shape[:2])))
    left = resized[margin:-margin, margin:-margin].astype(np.float32)
    right = expected[margin:-margin, margin:-margin].astype(np.float32)
    return float(np.mean(np.abs(left - right)))


def test_rectify_straightens_tilted_page() -> None:
    page = _page_with_marker()
    capture, area_ratio = _tilted_capture(page)
    assert area_ratio > 0.30

    result = rectify(capture)

    assert result.status == "complete"
    assert result.before.shape == capture.shape
    assert result.before is not capture
    np.testing.assert_array_equal(result.before, capture)
    assert result.after is result.image
    assert result.image.shape[0] < capture.shape[0]
    assert result.image.shape[1] < capture.shape[1]
    assert _normalized_correlation(result.image, page) > 0.90
    assert _mean_abs(result.image, page) < 28.0


def test_rectify_skips_image_without_page() -> None:
    image = np.full((180, 220, 3), 90, np.uint8)
    cv2.circle(image, (60, 50), 18, (210, 210, 210), thickness=-1)

    result = rectify(image)

    assert result.status == "skipped"
    assert result.image is image
    assert result.after is result.image
    assert result.before is not image
    np.testing.assert_array_equal(result.before, image)
    np.testing.assert_array_equal(result.image, image)


def _jpeg_with_orientation(image: np.ndarray, orientation: int, byte_order: str = "little") -> bytes:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    assert ok
    payload = encoded.tobytes()
    assert payload[:2] == b"\xff\xd8"
    return payload[:2] + _exif_app1(orientation, byte_order) + payload[2:]


def _exif_app1(orientation: int, byte_order: str) -> bytes:
    order = "little" if byte_order == "little" else "big"
    endian = b"II" if order == "little" else b"MM"
    entry = (
        (0x0112).to_bytes(2, order)
        + (3).to_bytes(2, order)
        + (1).to_bytes(4, order)
        + int(orientation).to_bytes(2, order)
        + b"\x00\x00"
    )
    ifd = (1).to_bytes(2, order) + entry + (0).to_bytes(4, order)
    tiff = endian + (42).to_bytes(2, order) + (8).to_bytes(4, order) + ifd
    body = b"Exif\x00\x00" + tiff
    return b"\xff\xe1" + (len(body) + 2).to_bytes(2, "big") + body


def _expected_orientation(image: np.ndarray, orientation: int) -> np.ndarray:
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


def test_correct_exif_applies_orientation_tags() -> None:
    image = np.full((48, 96, 3), 20, np.uint8)
    image[:16, :16] = (0, 0, 255)
    image[:16, -16:] = (0, 255, 0)
    image[-16:, :16] = (255, 0, 0)
    image[-16:, -16:] = (255, 255, 255)

    plain = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tobytes()
    decoded = correct_exif(plain)
    assert decoded.shape == image.shape
    assert abs(float(decoded[:16, :16, 2].mean()) - 255) < 25

    for orientation in range(1, 9):
        for byte_order in ("little", "big"):
            corrected = correct_exif(_jpeg_with_orientation(image, orientation, byte_order))
            expected = _expected_orientation(image, orientation)
            assert corrected.shape == expected.shape
            assert _mean_abs(corrected, expected) < 18.0

    turned = correct_exif(_jpeg_with_orientation(image, 6))
    assert turned.shape == (96, 48, 3)
    assert turned[:16, -16:, 2].mean() > 200
    assert turned[:16, :16, 2].mean() < 40
