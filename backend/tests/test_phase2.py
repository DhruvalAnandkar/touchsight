"""Braille, textures, and vision-label validation."""

from __future__ import annotations

import json

import numpy as np

from pipeline.braille import (
    CELL_SPACING_MM,
    DOT_DIAMETER_MM,
    DOT_SPACING_MM,
    boxes_overlap,
    dot_centers,
    label_size,
    place_labels,
    translate,
)
from pipeline.texture import assign_patterns
from pipeline.understand import VlmConfig, parse_understanding, understand


def test_braille_marks_capitals_and_numbers() -> None:
    assert translate("Ab") == [(6,), (1,), (1, 2)]
    assert translate("10") == [(3, 4, 5, 6), (1,), (2, 4, 5)]
    assert translate("A1") == [(6,), (1,), (3, 4, 5, 6), (1,)]


def test_braille_dot_geometry() -> None:
    centers = dot_centers([(1,), (1, 2)], origin=(0.0, 0.0))
    radius = DOT_DIAMETER_MM / 2
    assert centers[0] == (radius, radius)
    assert (round(radius + CELL_SPACING_MM, 2), radius) in centers
    assert (round(radius + CELL_SPACING_MM, 2), round(radius + DOT_SPACING_MM, 2)) in centers
    assert DOT_DIAMETER_MM == 1.5
    assert DOT_SPACING_MM == 2.5
    assert CELL_SPACING_MM == 6.2


def test_adjacent_regions_do_not_share_a_pattern() -> None:
    left = [(0.0, 0.0), (30.0, 0.0), (30.0, 30.0), (0.0, 30.0)]
    right = [(30.0, 0.0), (60.0, 0.0), (60.0, 30.0), (30.0, 30.0)]
    far = [(80.0, 0.0), (100.0, 0.0), (100.0, 20.0), (80.0, 20.0)]
    patterns = assign_patterns([("left", left), ("right", right), ("far", far)])
    assert patterns["left"] != patterns["right"]
    assert set(patterns.values()) <= {"dots", "diagonal", "crosshatch", "horizontal", "empty"}


def test_inline_labels_do_not_collide_and_a_tiny_region_uses_the_key() -> None:
    big = [(0.0, 0.0), (80.0, 0.0), (80.0, 50.0), (0.0, 50.0)]
    other = [(90.0, 0.0), (170.0, 0.0), (170.0, 50.0), (90.0, 50.0)]
    tiny = [(0.0, 60.0), (8.0, 60.0), (8.0, 68.0), (0.0, 68.0)]
    placed = place_labels([("a", "Alpha", big), ("b", "Beta", other), ("c", "Tiny spot", tiny)])
    assert [item.where for item in placed] == ["inline", "inline", "key"]
    assert not boxes_overlap(placed)
    assert label_size("A")[0] > 0


def test_understanding_rejects_bad_json_and_retries_once() -> None:
    valid = {
        "subject": "Heart",
        "summary": "This is a heart. It has four chambers.",
        "parts": [
            {
                "name": "Left atrium",
                "description": "The upper chamber on the right of the drawing.",
                "importance": 4,
                "bbox": {"x": 0.5, "y": 0.1, "w": 0.3, "h": 0.3},
            }
        ],
    }
    parsed = parse_understanding(json.dumps(valid))
    assert parsed.parts[0].name == "Left atrium"
    fenced = parse_understanding("```json\n" + json.dumps(valid) + "\n```")
    assert fenced.subject == "Heart"
    xyxy = json.loads(json.dumps(valid))
    xyxy["parts"][0]["bbox"] = [0.5, 0.1, 0.8, 0.4]
    assert parse_understanding(json.dumps(xyxy)).parts[0].bbox.model_dump() == {
        "x": 0.5,
        "y": 0.1,
        "w": 0.3,
        "h": 0.3,
    }
    try:
        parse_understanding(json.dumps({**valid, "extra": True}))
    except Exception as exc:
        assert exc.__class__.__name__ == "ValidationError"
    else:
        raise AssertionError("extra fields should be rejected")

    calls: list[str] = []

    def fetch(prompt: str, png: bytes) -> str:
        calls.append(prompt)
        if len(calls) == 1:
            return "not json"
        return json.dumps(valid)

    image = np.zeros((16, 16, 3), dtype=np.uint8)
    result = understand(image, "hash-1", VlmConfig(provider="openai", api_key="test", fetcher=fetch))
    assert len(calls) == 2
    assert result.source == "vlm"
    assert result.parts[0].name == "Left atrium"


def test_sample_labels_are_used_when_there_is_no_key() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    heart = "d040339e623938d87467913f116bac22cab660b0485e2433439cedd798e498f6"
    labeled = understand(image, heart, VlmConfig())
    assert labeled.source == "sample"
    assert labeled.subject == "Heart"
    assert len(labeled.parts) == 4
    generic = understand(image, "not-a-built-in-sample", VlmConfig())
    assert generic.source == "generic"
    assert generic.parts == []
