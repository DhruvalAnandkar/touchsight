"""Hosted SAM 2, behind a small interface.

Replicate's `meta/sam-2` model generates masks automatically and returns mask
image URLs. It does not take box prompts. When the caller has part boxes, we
keep the generated mask with the best overlap for each box — the boxes are the
prompts that decide which masks survive. Any failure raises `SamError` so
segmentation can fall back to OpenCV.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Protocol

import cv2
import httpx
import numpy as np

logger = logging.getLogger("touchsight.pipeline.sam2")


class SamError(RuntimeError):
    """The hosted segmenter did not return usable masks."""


class Sam2Client(Protocol):
    @property
    def available(self) -> bool: ...

    def segment_boxes(
        self,
        image_bgr: np.ndarray,
        boxes_xyxy: Sequence[tuple[int, int, int, int]],
    ) -> list[np.ndarray]:
        """Return one uint8 mask (0 or 255) per box, same height and width as `image_bgr`."""


def encode_png_data_uri(image_bgr: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".png", image_bgr)
    if not ok:
        raise SamError("Could not encode the image for SAM 2")
    import base64

    payload = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def decode_mask_png(payload: bytes, shape: tuple[int, int]) -> np.ndarray:
    """Turn a mask PNG into a binary uint8 image of `shape` (height, width)."""

    array = np.frombuffer(payload, dtype=np.uint8)
    decoded = cv2.imdecode(array, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise SamError("SAM 2 returned an unreadable mask image")
    if decoded.ndim == 3:
        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
    height, width = shape
    if decoded.shape[:2] != (height, width):
        decoded = cv2.resize(decoded, (width, height), interpolation=cv2.INTER_NEAREST)
    return np.where(decoded > 127, 255, 0).astype(np.uint8)


def mask_box_iou(mask: np.ndarray, box: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = box
    window = np.zeros(mask.shape[:2], dtype=bool)
    window[y1:y2, x1:x2] = True
    foreground = mask > 0
    intersection = np.logical_and(foreground, window).sum()
    union = np.logical_or(foreground, window).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def match_masks_to_boxes(
    masks: Sequence[np.ndarray],
    boxes_xyxy: Sequence[tuple[int, int, int, int]],
    *,
    min_iou: float = 0.12,
) -> list[np.ndarray]:
    """Greedy one-to-one assignment of masks to boxes. Raises if a box is unmatched."""

    if len(boxes_xyxy) == 0:
        raise SamError("No boxes to prompt SAM 2")
    pairs: list[tuple[float, int, int]] = []
    for mask_index, mask in enumerate(masks):
        for box_index, box in enumerate(boxes_xyxy):
            score = mask_box_iou(mask, box)
            if score >= min_iou:
                pairs.append((score, mask_index, box_index))
    pairs.sort(reverse=True)
    used_masks: set[int] = set()
    used_boxes: set[int] = set()
    chosen: dict[int, np.ndarray] = {}
    for _score, mask_index, box_index in pairs:
        if mask_index in used_masks or box_index in used_boxes:
            continue
        used_masks.add(mask_index)
        used_boxes.add(box_index)
        chosen[box_index] = masks[mask_index]
    if len(chosen) != len(boxes_xyxy):
        raise SamError(
            f"SAM 2 matched {len(chosen)} of {len(boxes_xyxy)} part boxes"
        )
    return [chosen[index] for index in range(len(boxes_xyxy))]


class ReplicateSam2:
    """`meta/sam-2` on Replicate. Automatic masks, then box matching."""

    def __init__(
        self,
        token: str,
        *,
        model: str = "meta/sam-2",
        api_base: str = "https://api.replicate.com/v1",
        timeout_s: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._token = token.strip()
        self._model = model.strip().strip("/")
        self._api_base = api_base.rstrip("/")
        self._timeout_s = timeout_s
        self._client = client

    @property
    def available(self) -> bool:
        return bool(self._token)

    def segment_boxes(
        self,
        image_bgr: np.ndarray,
        boxes_xyxy: Sequence[tuple[int, int, int, int]],
    ) -> list[np.ndarray]:
        if not self.available:
            raise SamError("SAM 2 token is not configured")
        owns_client = self._client is None
        client = self._client or httpx.Client(timeout=httpx.Timeout(self._timeout_s))
        try:
            urls = self._predict_mask_urls(client, image_bgr)
            masks = [self._download_mask(client, url, image_bgr.shape[:2]) for url in urls]
        finally:
            if owns_client:
                client.close()
        usable = [mask for mask in masks if _mask_is_usable(mask)]
        return match_masks_to_boxes(usable, boxes_xyxy)

    def _predict_mask_urls(self, client: httpx.Client, image_bgr: np.ndarray) -> list[str]:
        deadline = time.monotonic() + self._timeout_s
        url = f"{self._api_base}/models/{self._model}/predictions"
        response = client.post(
            url,
            headers=self._headers(wait=min(20, int(self._timeout_s))),
            json={
                "input": {
                    "image": encode_png_data_uri(image_bgr),
                    "points_per_side": 16,
                    "pred_iou_thresh": 0.86,
                    "stability_score_thresh": 0.9,
                    "use_m2m": False,
                }
            },
        )
        if response.status_code >= 400:
            raise SamError(f"SAM 2 HTTP {response.status_code}")
        prediction = response.json()
        prediction = self._poll(client, prediction, deadline)
        return _individual_mask_urls(prediction)

    def _poll(self, client: httpx.Client, prediction: dict[str, object], deadline: float) -> dict[str, object]:
        while True:
            status = str(prediction.get("status", ""))
            if status == "succeeded":
                return prediction
            if status in {"failed", "canceled"}:
                detail = prediction.get("error") or status
                raise SamError(f"SAM 2 prediction {detail}")
            if time.monotonic() > deadline:
                raise SamError("SAM 2 timed out")
            get_url = _prediction_url(prediction)
            time.sleep(0.8)
            response = client.get(get_url, headers=self._headers())
            if response.status_code >= 400:
                raise SamError(f"SAM 2 poll HTTP {response.status_code}")
            prediction = response.json()

    def _download_mask(
        self,
        client: httpx.Client,
        url: str,
        shape: tuple[int, int],
    ) -> np.ndarray:
        response = client.get(url, headers=self._headers(), follow_redirects=True)
        if response.status_code >= 400:
            response = client.get(url, follow_redirects=True)
        if response.status_code >= 400:
            raise SamError(f"SAM 2 mask download HTTP {response.status_code}")
        return decode_mask_png(response.content, shape)

    def _headers(self, wait: int | None = None) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }
        if wait is not None:
            headers["Prefer"] = f"wait={wait}"
        return headers


def _prediction_url(prediction: dict[str, object]) -> str:
    urls = prediction.get("urls")
    if isinstance(urls, dict) and isinstance(urls.get("get"), str):
        return str(urls["get"])
    prediction_id = prediction.get("id")
    if not isinstance(prediction_id, str) or not prediction_id:
        raise SamError("SAM 2 response had no prediction URL")
    return f"https://api.replicate.com/v1/predictions/{prediction_id}"


def _individual_mask_urls(prediction: dict[str, object]) -> list[str]:
    output = prediction.get("output")
    urls: list[str] = []
    if isinstance(output, dict):
        masks = output.get("individual_masks")
        if isinstance(masks, list):
            urls = [item for item in masks if isinstance(item, str)]
    elif isinstance(output, list):
        urls = [item for item in output if isinstance(item, str)]
    if not urls:
        raise SamError("SAM 2 returned no individual masks")
    return urls[:16]


def _mask_is_usable(mask: np.ndarray) -> bool:
    fraction = float((mask > 0).mean())
    return 0.002 <= fraction <= 0.9
