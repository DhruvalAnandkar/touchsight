"""Assemble the tactile page: title, textured graphic, Braille, legend."""

from __future__ import annotations

from collections.abc import Sequence

import svgwrite

from pipeline.braille import (
    DOT_DIAMETER_MM,
    PRINT_MM,
    PlacedLabel,
    block_size,
    clip_lines,
    dot_centers,
    label_size,
    translate,
    wrap_lines,
)
from pipeline.page import LEGEND_BAND_MM, TactileConfig
from pipeline.texture import inset_ring
from pipeline.types import BoundarySegment, SimplifiedRegion

INK = "#141414"
PRINT = "#5e5a54"


def render_sheet(
    page_mm: tuple[float, float],
    regions: Sequence[SimplifiedRegion],
    segments: Sequence[BoundarySegment],
    labels: Sequence[PlacedLabel],
    *,
    subject: str,
    config: TactileConfig,
) -> str:
    width_mm, height_mm = page_mm
    drawing = svgwrite.Drawing(
        size=(f"{_fmt(width_mm)}mm", f"{_fmt(height_mm)}mm"),
        viewBox=f"0 0 {_fmt(width_mm)} {_fmt(height_mm)}",
        profile="full",
        debug=False,
    )
    drawing.attribs["role"] = "img"
    names = ", ".join(region.name for region in regions) or "empty sheet"
    drawing.attribs["aria-label"] = f"Tactile graphic: {subject or names}"
    drawing.set_desc(title=subject or "TouchSight tactile graphic", desc=names)
    _add_patterns(drawing)
    drawing.add(drawing.rect(insert=(0, 0), size=(width_mm, height_mm), fill="#ffffff"))
    inset = config.min_stroke_mm
    drawing.add(
        drawing.rect(
            insert=(inset, inset),
            size=(max(width_mm - 2 * inset, inset), max(height_mm - 2 * inset, inset)),
            fill="none",
            stroke=INK,
            stroke_width=config.min_stroke_mm,
        )
    )
    by_id = {label.region_id: label for label in labels}
    for region in regions:
        ring = inset_ring(region.polygon_mm) if region.pattern and region.pattern != "empty" else []
        if len(ring) >= 3:
            drawing.add(drawing.polygon(points=ring, fill=f"url(#{region.pattern})", stroke="none"))
        group = drawing.g(id=region.id)
        group.set_desc(title=region.name, desc=region.description or None)
        group.add(drawing.polygon(points=region.polygon_mm, fill="none", stroke="none"))
        drawing.add(group)
    ink = drawing.g(
        id="boundaries",
        fill="none",
        stroke=INK,
        stroke_width=config.min_stroke_mm,
        stroke_linejoin="round",
        stroke_linecap="round",
    )
    for segment in segments:
        ink.add(drawing.line(start=segment.start_mm, end=segment.end_mm))
    drawing.add(ink)
    _title(drawing, subject or "Diagram", names, width_mm, config.margin_mm)
    for region in regions:
        label = by_id.get(region.id)
        if label is None or label.origin_mm is None:
            continue
        _label_block(drawing, label.text, label.origin_mm, mask=True)
    _legend(drawing, regions, width_mm, height_mm, config.margin_mm)
    return drawing.tostring()


def _add_patterns(drawing: svgwrite.Drawing) -> None:
    dots = drawing.pattern(id="dots", size=(4, 4), patternUnits="userSpaceOnUse")
    dots.add(drawing.circle(center=(2, 2), r=0.75, fill=INK))
    diagonal = drawing.pattern(id="diagonal", size=(4, 4), patternUnits="userSpaceOnUse")
    for start, end in (((0, 4), (4, 0)), ((-1, 1), (1, -1)), ((3, 5), (5, 3))):
        diagonal.add(drawing.line(start=start, end=end, stroke=INK, stroke_width=0.8))
    cross = drawing.pattern(id="crosshatch", size=(4, 4), patternUnits="userSpaceOnUse")
    for start, end in (
        ((0, 4), (4, 0)),
        ((-1, 1), (1, -1)),
        ((3, 5), (5, 3)),
        ((0, 0), (4, 4)),
        ((-1, 3), (1, 5)),
        ((3, -1), (5, 1)),
    ):
        cross.add(drawing.line(start=start, end=end, stroke=INK, stroke_width=0.8))
    horizontal = drawing.pattern(id="horizontal", size=(5, 5), patternUnits="userSpaceOnUse")
    horizontal.add(drawing.line(start=(0, 2.5), end=(5, 2.5), stroke=INK, stroke_width=0.8))
    drawing.defs.add(dots)
    drawing.defs.add(diagonal)
    drawing.defs.add(cross)
    drawing.defs.add(horizontal)


def fit_title_lines(subject: str, names: str, max_width_mm: float) -> list[str]:
    """Braille title, at most two lines wide.

    The caption is `subject: names`. When those words do not fit on two lines
    inside the margins, the names are dropped and only the subject is wrapped.
    """

    subject_text = (subject or "").strip() or "Diagram"
    caption = subject_text
    cleaned = (names or "").strip()
    if cleaned and cleaned.lower() != "empty sheet":
        caption = f"{subject_text}: {cleaned}"
    wrapped = wrap_lines(caption, max_width_mm, max_lines=2)
    if wrapped is not None:
        return wrapped
    subject_lines = wrap_lines(subject_text, max_width_mm, max_lines=2)
    if subject_lines is not None:
        return subject_lines
    return clip_lines(subject_text, max_width_mm, max_lines=2)


def _title(
    drawing: svgwrite.Drawing,
    subject: str,
    names: str,
    page_w: float,
    margin_mm: float,
) -> None:
    max_width = page_w - 2 * margin_mm
    lines = fit_title_lines(subject, names, max_width)
    _draw_lines(drawing, lines, (margin_mm, margin_mm), mask=False)


def _legend(
    drawing: svgwrite.Drawing,
    regions: Sequence[SimplifiedRegion],
    page_w: float,
    page_h: float,
    margin_mm: float,
) -> None:
    x_mm = margin_mm
    y_mm = page_h - margin_mm - LEGEND_BAND_MM
    row_height = 15.0
    limit = page_w - margin_mm
    swatch = 8.0
    gap = 2.0
    text_budget = page_w - 2 * margin_mm - swatch - gap
    for region in regions:
        lines = clip_lines(region.name, text_budget, max_lines=2)
        text_width = max((block_size(len(translate(line)))[0] for line in lines), default=0.0)
        item_w = swatch + gap + max(text_width, 12.0)
        if x_mm + item_w > limit and x_mm > margin_mm:
            x_mm = margin_mm
            y_mm += row_height
            row_height = 15.0
        fill = "none" if region.pattern in {"", "empty"} else f"url(#{region.pattern})"
        drawing.add(
            drawing.rect(
                insert=(x_mm, y_mm),
                size=(swatch, swatch),
                fill=fill,
                stroke=INK,
                stroke_width=0.4,
            )
        )
        _width, used_h = _draw_lines(drawing, lines, (x_mm + swatch + gap, y_mm), mask=False)
        row_height = max(row_height, used_h, 15.0)
        x_mm += item_w + 4.0


def _draw_lines(
    drawing: svgwrite.Drawing,
    lines: list[str],
    origin: tuple[float, float],
    *,
    mask: bool,
) -> tuple[float, float]:
    """Stack Braille lines. Returns the ink width and the height used."""

    x_mm, y_mm = origin
    cursor = y_mm
    width = 0.0
    gap = 1.2
    drawn = 0
    for line in lines:
        if label_size(line)[0] <= 0:
            continue
        if drawn:
            cursor += gap
        _label_block(drawing, line, (x_mm, cursor), mask=mask)
        line_w, line_h = label_size(line)
        width = max(width, line_w)
        cursor += line_h
        drawn += 1
    return width, cursor - y_mm


def _label_block(
    drawing: svgwrite.Drawing,
    text: str,
    origin: tuple[float, float],
    *,
    mask: bool,
) -> None:
    cells = translate(text)
    width, braille_h = block_size(len(cells))
    if width <= 0:
        return
    x_mm, y_mm = origin
    total_h = braille_h + 0.8 + PRINT_MM
    if mask:
        drawing.add(
            drawing.rect(
                insert=(x_mm - 0.6, y_mm - 0.6),
                size=(width + 1.2, total_h + 1.2),
                fill="#ffffff",
            )
        )
    for center in dot_centers(cells, origin):
        drawing.add(drawing.circle(center=center, r=DOT_DIAMETER_MM / 2, fill=INK))
    drawing.add(
        drawing.text(
            text,
            insert=(x_mm, y_mm + braille_h + 0.6 + PRINT_MM),
            font_size=f"{PRINT_MM}px",
            font_family="Arial, sans-serif",
            fill=PRINT,
            class_="print-label",
        )
    )


def _fmt(value: float) -> str:
    rounded = round(float(value), 2)
    if rounded == int(rounded):
        return str(int(rounded))
    return f"{rounded:.2f}".rstrip("0").rstrip(".")
