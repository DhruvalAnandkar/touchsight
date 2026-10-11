"""Page geometry for a tactile sheet.

Stroke, gap, and area thresholds are specified in millimetres so the same
constants mean the same thing on a finger whether the page is Letter or A4.
The raster used for segmentation is that page at `dpi`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

PageSize = Literal["letter", "a4"]

PAGE_MM: dict[str, tuple[float, float]] = {
    "letter": (215.9, 279.4),
    "a4": (210.0, 297.0),
}

# Half an inch. The graphic sits inside this margin.
PAGE_MARGIN_MM = 12.7
# Empty bands so a later title and legend do not cover the drawing.
TITLE_BAND_MM = 18.0
LEGEND_BAND_MM = 32.0


@dataclass(frozen=True)
class TactileConfig:
    page: PageSize = "letter"
    dpi: int = 120
    margin_mm: float = PAGE_MARGIN_MM
    min_stroke_mm: float = 1.5
    min_gap_mm: float = 2.5
    min_area_mm2: float = 40.0
    epsilon_mm: float = 0.55

    @property
    def page_mm(self) -> tuple[float, float]:
        return PAGE_MM[self.page]


@dataclass(frozen=True)
class FittedImage:
    """Image resized to fit the graphic box, plus where that box sits on the page."""

    image: np.ndarray
    offset_px: tuple[float, float]
    page_px: tuple[int, int]
    dpi: float
    page_mm: tuple[float, float] | None = None


def mm_to_px(mm: float, dpi: float) -> float:
    return mm * dpi / 25.4


def px_to_mm(px: float, dpi: float) -> float:
    return px * 25.4 / dpi


def page_pixels(config: TactileConfig) -> tuple[int, int]:
    width_mm, height_mm = config.page_mm
    return (
        max(1, int(round(mm_to_px(width_mm, config.dpi)))),
        max(1, int(round(mm_to_px(height_mm, config.dpi)))),
    )


def graphic_box_mm(page_mm: tuple[float, float], margin_mm: float) -> tuple[float, float, float, float]:
    """Return x, y, width, height of the drawing area, in millimetres.

    Title and legend bands are left empty so later stages can use them.
    """

    page_w, page_h = page_mm
    x = margin_mm
    y = margin_mm + TITLE_BAND_MM
    width = page_w - 2 * margin_mm
    height = page_h - 2 * margin_mm - TITLE_BAND_MM - LEGEND_BAND_MM
    return x, y, max(width, 1.0), max(height, 1.0)


def _portrait_mm(page_mm: tuple[float, float]) -> tuple[float, float]:
    short, long = sorted(page_mm)
    return short, long


def fit_scale(image_wh: tuple[float, float], page_mm: tuple[float, float], margin_mm: float) -> float:
    image_w, image_h = image_wh
    _x, _y, box_w, box_h = graphic_box_mm(page_mm, margin_mm)
    return min(box_w / image_w, box_h / image_h)


def oriented_page_mm(image_shape: tuple[int, ...], config: TactileConfig) -> tuple[float, float]:
    """Pick portrait or landscape, whichever makes the drawing larger."""

    height, width = image_shape[:2]
    portrait = _portrait_mm(config.page_mm)
    landscape = (portrait[1], portrait[0])
    portrait_scale = fit_scale((width, height), portrait, config.margin_mm)
    landscape_scale = fit_scale((width, height), landscape, config.margin_mm)
    if landscape_scale > portrait_scale:
        return landscape
    return portrait


def fit_to_page(image_bgr: np.ndarray, config: TactileConfig) -> FittedImage:
    """Scale the image to the largest size that fits the graphic box.

    The graphic box is the sheet minus the margin and the reserved title and
    legend bands. Segmentation runs on the scaled image. `offset_px` is the
    top-left of that image on the full-page raster.
    """

    page_mm = oriented_page_mm(image_bgr.shape, config)
    page_w = max(1, int(round(mm_to_px(page_mm[0], config.dpi))))
    page_h = max(1, int(round(mm_to_px(page_mm[1], config.dpi))))
    box_x, box_y, box_w, box_h = graphic_box_mm(page_mm, config.margin_mm)
    max_w = max(1, int(round(mm_to_px(box_w, config.dpi))))
    max_h = max(1, int(round(mm_to_px(box_h, config.dpi))))
    height, width = image_bgr.shape[:2]
    scale = min(max_w / width, max_h / height)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    if new_w == width and new_h == height:
        resized = image_bgr
    else:
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=interp)
    offset_x = mm_to_px(box_x, config.dpi) + (max_w - new_w) / 2
    offset_y = mm_to_px(box_y, config.dpi) + (max_h - new_h) / 2
    return FittedImage(
        image=resized,
        offset_px=(offset_x, offset_y),
        page_px=(page_w, page_h),
        dpi=float(config.dpi),
        page_mm=page_mm,
    )
