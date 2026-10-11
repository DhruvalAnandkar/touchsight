"""Shared pipeline types.

Numpy-bearing records are dataclasses. JSON records that cross the API, or that
a later vision-model stage must validate, are pydantic models.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class BBox(BaseModel):
    """Normalized box: origin at the top-left, values in 0..1 of the image."""

    model_config = ConfigDict(extra="forbid")

    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    w: float = Field(gt=0, le=1)
    h: float = Field(gt=0, le=1)

    def to_xyxy(self, width: int, height: int) -> tuple[int, int, int, int]:
        x1 = int(round(self.x * width))
        y1 = int(round(self.y * height))
        x2 = int(round((self.x + self.w) * width))
        y2 = int(round((self.y + self.h) * height))
        x1 = min(max(x1, 0), max(width - 1, 0))
        y1 = min(max(y1, 0), max(height - 1, 0))
        x2 = min(max(x2, x1 + 1), width)
        y2 = min(max(y2, y1 + 1), height)
        return x1, y1, x2, y2


class Part(BaseModel):
    """One named thing the vision model (or a test) believes is in the image."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=80)
    description: str = ""
    importance: int = Field(default=3, ge=1, le=5)
    approx_bbox: BBox


@dataclass
class SegmentedRegion:
    id: str
    name: str
    description: str
    importance: int
    mask: np.ndarray  # uint8 HxW, 0 or 255
    bbox: tuple[int, int, int, int]  # x, y, w, h in pixels
    outline_only: bool = False


@dataclass
class SegmentResult:
    width: int
    height: int
    source: str  # "sam2" or "opencv"
    regions: list[SegmentedRegion] = field(default_factory=list)


@dataclass
class SimplifiedRegion:
    id: str
    name: str
    description: str
    importance: int
    polygon_mm: list[tuple[float, float]]
    area_mm2: float
    pattern: str = ""
    label_where: str = ""
    outline_only: bool = False


@dataclass(frozen=True)
class BoundarySegment:
    """One stroked edge. Shared borders appear once."""

    start_mm: tuple[float, float]
    end_mm: tuple[float, float]


@dataclass
class SimplifyResult:
    svg: str
    page_mm: tuple[float, float]
    regions: list[SimplifiedRegion]
    source: str
    dropped: int
    segments: list[BoundarySegment] = field(default_factory=list)
    subject: str = ""
    summary: str = ""
    label_source: str = ""
    inline_labels: int = 0
    key_labels: int = 0
    pdf: bytes = b""
    stl: bytes = b""
    heightmap_png: bytes = b""
    region_id_png: bytes = b""
    rectify_before: bytes = b""
    rectify_after: bytes = b""
    image_origin_mm: tuple[float, float] = (0.0, 0.0)
    image_size_mm: tuple[float, float] = (0.0, 0.0)

    def region_map(self) -> dict[str, object]:
        return {
            "subject": self.subject,
            "summary": self.summary,
            "regions": [
                {
                    "id": region.id,
                    "name": region.name,
                    "description": region.description,
                    "pattern": region.pattern,
                    "importance": region.importance,
                    "polygon": self._image_polygon(region.polygon_mm),
                }
                for region in self.regions
            ],
        }

    def _image_polygon(self, points: list[tuple[float, float]]) -> list[list[float]]:
        """Polygon in 0..1 of the source picture, so a canvas can lay it on the photo."""

        origin_x, origin_y = self.image_origin_mm
        width, height = self.image_size_mm
        if width <= 0 or height <= 0:
            page_w, page_h = self.page_mm
            width = page_w or 1.0
            height = page_h or 1.0
            origin_x, origin_y = 0.0, 0.0
        return [
            [
                round(min(1.0, max(0.0, (x - origin_x) / width)), 4),
                round(min(1.0, max(0.0, (y - origin_y) / height)), 4),
            ]
            for x, y in points
        ]


@dataclass
class StageUpdate:
    name: str
    status: str
    elapsed_ms: float | None = None
    message: str | None = None
    provider: str | None = None


STAGE_ORDER: tuple[str, ...] = (
    "rectify",
    "understand",
    "segment",
    "simplify",
    "texture",
    "braille",
    "relief",
    "export",
)
