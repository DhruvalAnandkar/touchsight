"""The Replicate client parses the hosted SAM 2 response without calling it."""

from __future__ import annotations

import cv2
import httpx
import numpy as np

from pipeline.providers.sam2 import ReplicateSam2, decode_mask_png


def _png(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return buffer.tobytes()


def test_decode_mask_png_thresholds_and_resizes() -> None:
    mask = np.zeros((8, 8), np.uint8)
    mask[2:6, 2:6] = 255
    decoded = decode_mask_png(_png(mask), (8, 8))
    assert decoded.shape == (8, 8)
    assert int((decoded > 0).sum()) == 16


def test_replicate_client_matches_downloaded_masks_to_boxes() -> None:
    mask = np.zeros((40, 40), np.uint8)
    mask[5:30, 5:30] = 255
    mask_png = _png(mask)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(
                201,
                json={
                    "id": "pred-1",
                    "status": "succeeded",
                    "output": {
                        "combined_mask": "https://masks.test/combined.png",
                        "individual_masks": ["https://masks.test/one.png"],
                    },
                },
            )
        if request.url.host == "masks.test":
            return httpx.Response(200, content=mask_png)
        return httpx.Response(404)

    client = ReplicateSam2("token", client=httpx.Client(transport=httpx.MockTransport(handler)))
    image = np.full((40, 40, 3), 255, np.uint8)
    masks = client.segment_boxes(image, [(5, 5, 30, 30)])
    assert len(masks) == 1
    assert int((masks[0] > 0).sum()) == int((mask > 0).sum())
