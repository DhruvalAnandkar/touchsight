"""Grade 1 English Braille, drawn as SVG circles.

Dot diameter is 1.5 mm. Dot centers inside a cell are 2.5 mm apart.
The next cell starts 6.2 mm to the right of the previous cell.
"""

from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry import Polygon, box

DOT_DIAMETER_MM = 1.5
DOT_SPACING_MM = 2.5
CELL_SPACING_MM = 6.2
PRINT_MM = 3.2
CLEARANCE_MM = 3.0

_CAPITAL = (6,)
_NUMBER = (3, 4, 5, 6)
_LETTERS: dict[str, tuple[int, ...]] = {
    "a": (1,),
    "b": (1, 2),
    "c": (1, 4),
    "d": (1, 4, 5),
    "e": (1, 5),
    "f": (1, 2, 4),
    "g": (1, 2, 4, 5),
    "h": (1, 2, 5),
    "i": (2, 4),
    "j": (2, 4, 5),
    "k": (1, 3),
    "l": (1, 2, 3),
    "m": (1, 3, 4),
    "n": (1, 3, 4, 5),
    "o": (1, 3, 5),
    "p": (1, 2, 3, 4),
    "q": (1, 2, 3, 4, 5),
    "r": (1, 2, 3, 5),
    "s": (2, 3, 4),
    "t": (2, 3, 4, 5),
    "u": (1, 3, 6),
    "v": (1, 2, 3, 6),
    "w": (2, 4, 5, 6),
    "x": (1, 3, 4, 6),
    "y": (1, 3, 4, 5, 6),
    "z": (1, 3, 5, 6),
}
_DIGITS = {str((index + 1) % 10): _LETTERS["abcdefghij"[index]] for index in range(10)}


def translate(text: str) -> list[tuple[int, ...]]:
    """Cells for `text`. A space is an empty cell. Other punctuation is dropped."""

    cells: list[tuple[int, ...]] = []
    numbering = False
    for char in text:
        if char == " ":
            cells.append(())
            numbering = False
            continue
        if char.isdigit():
            if not numbering:
                cells.append(_NUMBER)
                numbering = True
            cells.append(_DIGITS[char])
            continue
        numbering = False
        lower = char.lower()
        if lower not in _LETTERS:
            continue
        if char.isupper():
            cells.append(_CAPITAL)
        cells.append(_LETTERS[lower])
    return cells


def block_size(cell_count: int) -> tuple[float, float]:
    if cell_count <= 0:
        return 0.0, 0.0
    width = (cell_count - 1) * CELL_SPACING_MM + DOT_SPACING_MM + DOT_DIAMETER_MM
    height = 2 * DOT_SPACING_MM + DOT_DIAMETER_MM
    return width, height


def label_size(text: str) -> tuple[float, float]:
    """Braille block plus the grey print line under it."""

    width, height = block_size(len(translate(text)))
    if width <= 0:
        return 0.0, 0.0
    return width, height + 0.8 + PRINT_MM


def text_width(text: str) -> float:
    return block_size(len(translate(text)))[0]


def wrap_lines(text: str, max_width_mm: float, *, max_lines: int = 2) -> list[str] | None:
    """Word-wrap `text` into at most `max_lines` that each fit in `max_width_mm`.

    Returns None when a word is wider than the line or when the words do not
    all fit. Nothing is omitted on success.
    """

    words = text.split()
    if not words:
        return []
    if max_width_mm <= 0:
        return None
    lines: list[list[str]] = [[]]
    for word in words:
        if text_width(word) > max_width_mm + 1e-6:
            return None
        trial = " ".join([*lines[-1], word])
        if text_width(trial) <= max_width_mm + 1e-6:
            lines[-1].append(word)
            continue
        if len(lines) >= max_lines:
            return None
        lines.append([word])
    return [" ".join(line) for line in lines if line]


def clip_lines(text: str, max_width_mm: float, *, max_lines: int = 2) -> list[str]:
    """Wrap like `wrap_lines`, dropping words that would need another line.

    A single word wider than the line is split so the kept pieces still fit.
    """

    wrapped = wrap_lines(text, max_width_mm, max_lines=max_lines)
    if wrapped is not None:
        return wrapped
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        if text_width(word) > max_width_mm + 1e-6:
            if current:
                lines.append(" ".join(current))
                current = []
            if len(lines) >= max_lines:
                break
            for piece in _split_word(word, max_width_mm):
                if len(lines) >= max_lines:
                    break
                lines.append(piece)
            continue
        if current and text_width(" ".join([*current, word])) > max_width_mm + 1e-6:
            lines.append(" ".join(current))
            current = []
            if len(lines) >= max_lines:
                break
        if len(lines) >= max_lines:
            break
        current.append(word)
    if current and len(lines) < max_lines:
        lines.append(" ".join(current))
    return lines[:max_lines]


def _split_word(word: str, max_width_mm: float) -> list[str]:
    pieces: list[str] = []
    current = ""
    for char in word:
        trial = f"{current}{char}"
        if current and text_width(trial) > max_width_mm + 1e-6:
            pieces.append(current)
            current = char
            continue
        current = trial
    if current:
        pieces.append(current)
    return pieces


def dot_centers(cells: list[tuple[int, ...]], origin: tuple[float, float]) -> list[tuple[float, float]]:
    """Circle centers. `origin` is the top-left of the ink, not the first center."""

    radius = DOT_DIAMETER_MM / 2
    origin_x, origin_y = origin
    centers: list[tuple[float, float]] = []
    for index, dots in enumerate(cells):
        for dot in dots:
            column = 0 if dot <= 3 else 1
            row = (dot - 1) % 3
            centers.append(
                (
                    round(origin_x + radius + index * CELL_SPACING_MM + column * DOT_SPACING_MM, 2),
                    round(origin_y + radius + row * DOT_SPACING_MM, 2),
                )
            )
    return centers


@dataclass(frozen=True)
class PlacedLabel:
    region_id: str
    text: str
    where: str  # "inline" or "key"
    origin_mm: tuple[float, float] | None


def place_labels(
    regions: list[tuple[str, str, list[tuple[float, float]]]],
) -> list[PlacedLabel]:
    """Put a name inside a region when it fits. Otherwise put a Braille number.

    `regions` is (id, name, polygon_mm) in reading order. Every name also goes
    in the legend; `where` says whether the region itself holds the name.
    """

    placed: list[PlacedLabel] = []
    occupied: list[Polygon] = []
    for index, (region_id, name, points) in enumerate(regions, start=1):
        polygon = _polygon(points)
        origin = _find_spot(polygon, label_size(name), occupied) if polygon is not None else None
        if origin is not None:
            width, height = label_size(name)
            occupied.append(box(origin[0], origin[1], origin[0] + width, origin[1] + height))
            placed.append(PlacedLabel(region_id, name, "inline", origin))
            continue
        number = str(index)
        origin = _find_spot(polygon, label_size(number), occupied) if polygon is not None else None
        if origin is not None:
            width, height = label_size(number)
            occupied.append(box(origin[0], origin[1], origin[0] + width, origin[1] + height))
        placed.append(PlacedLabel(region_id, number, "key", origin))
    return placed


def boxes_overlap(labels: list[PlacedLabel]) -> bool:
    boxes = []
    for label in labels:
        if label.origin_mm is None:
            continue
        width, height = label_size(label.text)
        boxes.append(box(label.origin_mm[0], label.origin_mm[1], label.origin_mm[0] + width, label.origin_mm[1] + height))
    for index, left in enumerate(boxes):
        for right in boxes[index + 1 :]:
            if left.intersects(right) and left.intersection(right).area > 0.01:
                return True
    return False


def _find_spot(
    polygon: Polygon,
    size: tuple[float, float],
    occupied: list[Polygon],
) -> tuple[float, float] | None:
    width, height = size
    if width <= 0 or height <= 0:
        return None
    inset = polygon.buffer(-CLEARANCE_MM)
    if inset.is_empty:
        return None
    min_x, min_y, max_x, max_y = inset.bounds
    if max_x - min_x < width or max_y - min_y < height:
        return None
    y_mm = min_y
    while y_mm + height <= max_y + 0.05:
        x_mm = min_x
        while x_mm + width <= max_x + 0.05:
            rect = box(x_mm, y_mm, x_mm + width, y_mm + height)
            blocked = any(rect.intersects(item) and rect.intersection(item).area > 0.01 for item in occupied)
            if not blocked and inset.contains(rect):
                return (round(x_mm, 2), round(y_mm, 2))
            x_mm += 5.0
        y_mm += 5.0
    return None


def _polygon(points: list[tuple[float, float]]) -> Polygon | None:
    if len(points) < 3:
        return None
    polygon = Polygon(points)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty or polygon.geom_type != "Polygon":
        return None
    return polygon
