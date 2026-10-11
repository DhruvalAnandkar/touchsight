# Pipeline

TouchSight is a chain of pure functions. Each stage takes a typed value and returns a typed value. Nothing in `pipeline/` reads the database or the HTTP request. `pipeline/run.py` is the only place that knows the order.

A stage is wrapped with `timed()`, which logs the start, the failure, and the elapsed milliseconds. The job API records the same timing so the upload page can show it.

Page geometry lives in `pipeline/page.py`. Stroke width, gap, and minimum area are millimetres. The raster those stages see is the printed page at `DPI` (120 by default), so a 1.5 mm line is the same physical line on Letter and on A4.

## 1. Understand — implemented

**Job:** ask a vision model what the picture is. `VLM_PROVIDER` selects Anthropic or an OpenAI-compatible client, and `VLM_MODEL` selects the model. HTTP uses one client: 5 s connect, 45 s read, one retry.

**Output:** `{ subject, summary (two sentences), parts: [{ name (at most 3 words), description, importance 1–5, bbox }] }`, at most 8 parts, validated with pydantic. One retry on bad JSON. A valid reply is cached by image hash.

**Fallback:** no key, or a failed call, uses `samples/labels.json` for the seven built-in diagrams. Any other upload is left unnamed, and segment names those masks "Region N".

Parts are matched to OpenCV masks by box overlap. SAM 2 still runs only when `SAM_API_TOKEN` is set and the call returns in time.

## 2. Segment — implemented

**Input:** a BGR image, optional parts, an optional SAM 2 client.

**Output:** at most 8 binary masks, each tied to a name.

### Hosted SAM 2

`pipeline/providers/sam2.py` defines a `Sam2Client` protocol: `segment_boxes(image, boxes) -> one mask per box`. The Replicate implementation calls `meta/sam-2`. That hosted model generates masks automatically (image URL in, mask URLs out). It does not accept box prompts, so the boxes are applied after the call: each part keeps the mask with the best overlap, one-to-one. A timeout, HTTP error, unreadable mask, or unmatched box raises `SamError`.

The client is constructed only when `SAM_API_TOKEN` is set. It is called only when parts exist, because a box-less call would ignore the prompt contract. Phase 1 uploads have no parts, so the demo path does not touch the network.

### OpenCV fallback

This is the path the demo actually runs.

1. Canny edges are taken on each color channel and OR-ed together. A red region and a green region can share a luminance; a gray-only edge detector would glue them together. Thresholds come from Otsu on the gradient, not from the median brightness. On a white worksheet the median is white, and a median-based Canny threshold finds no edges at all.
2. Seeds are the non-edge pixels eroded by a small ellipse (about 1.5% of the short side, clamped to 4–9 px). A seed has to sit *inside* a region. Thresholding the distance transform at a fraction of its global maximum keeps only the big white background and swallows every bar.
3. `cv2.watershed` grows those seeds up to the edges.
4. The label that touches the page border is dropped when it looks like paper (bright, low saturation) or when it covers more than about 70% of the image. Interior white, such as the cytoplasm inside a cell wall, does not touch the border, so it stays. A colored region that happens to touch the border, such as sky, also stays.
5. Specks, cracks, and masks with a short side under 8 px are dropped. If watershed returns nothing, a second pass treats dark strokes as walls and flood-fills the enclosed rooms.
6. If more than 8 masks remain, extras are merged into a neighbor they touch, or dropped.

Parts, when present, are matched greedily by box overlap. Unmatched masks get an importance from their area rank (largest is 5).

## 3. Simplify — implemented

**Input:** the masks, the page fit (offset and DPI), and a `TactileConfig`.

**Output:** polygons in millimetres (kept for later stages), a deduplicated boundary graph, and an SVG.

A finger cannot separate two lines that touch, and a swell-paper line thinner than about a millimetre disappears. Defaults:

| Constant | Default | Why |
| --- | --- | --- |
| `min_stroke_mm` | 1.5 | Thin enough to look like a line, thick enough to emboss. |
| `min_gap_mm` | 2.5 | Clear paper between two strokes that do not share a wall. |
| `min_area_mm2` | 40 | Drops crumbs that would only feel like noise. |
| `epsilon_mm` | 0.55 | Douglas–Peucker tolerance. Corners stay; pixel jitter goes. |
| `margin_mm` | 12.7 | Half an inch. The drawing sits inside this margin. |

The sheet is Letter or A4. Portrait or landscape is whichever fit is larger. The graphic box is the page minus the margin, minus an 18 mm title band and a 32 mm legend band. The image is scaled to the largest size that fits that box, keeping its aspect ratio.

Masks are painted into one label image. Hairline cracks (about 1.5 mm, the kind watershed leaves between neighbors) are closed so two regions meet. Each grid edge where the label changes, including the step to the background, is traced once. Paths run from junction to junction; a loop with no junction is traced on its own. Douglas–Peucker (`cv2.approxPolyDP`, exposed as `douglas_peucker`) runs in pixel space, then vertices within 1.5 mm snap together. The SVG strokes those segments as `<line>` elements at 1.5 mm, each wall once. Region polygons stay in the SVG with no stroke, and in `region_map.json`, for later hit-testing.

`enforce_clearance` still erodes masks by `(gap + stroke) / 2` for the gap tests. The rendered sheet does not use it: a shared wall is one stroke, and separated shapes keep the paper that was already between them.

The contour of each region still comes from `findContours`. The ring is turned into millimetres with the page offset, checked with shapely, and dropped if the area is under the minimum. Invalid rings are repaired with `make_valid`; if that splits the shape, the largest piece is kept.

The SVG is a white sheet and a frame at the minimum stroke. Units are millimetres, with a `viewBox` in the same space, so a browser and a printer agree. Region ids (`r1`, `r2`, …) are on the groups for the later explore mode.

## 4. Texture — implemented

Five fills: dots (1.5 mm on a 4 mm pitch), diagonal (0.8 mm / 4 mm), crosshatch, horizontal (0.8 mm / 5 mm), and empty. Greedy coloring keeps a pattern off any neighbor. The fill is inset 2 mm from the outline.

## 5. Braille — implemented

Grade 1 English: letters, a capital indicator, and a number sign before digits. Dots are SVG circles, 1.5 mm across, 2.5 mm apart, with 6.2 mm between cells. A label is placed inside its region when it fits with 3 mm of clearance, and the pattern under it is masked white. Otherwise the region gets a Braille number and the words sit in the legend. Grey print sits under each Braille label; the page can hide it. The legend lists every region as a swatch, Braille, and print.

## 6. Relief — next

A heightmap, then a watertight STL no larger than 150 mm. Depth Anything V2 (hosted, behind the same kind of interface as SAM) supplies the photo relief. If that call fails, the height is the distance transform of the masks: borders high, centers lower, sitting on a base plate. Outlines from stage 3 are added as raised ridges so the print matches the sheet.

## 7. Export — partly implemented

This build writes `tactile.svg`, `tactile.pdf` (when PyMuPDF can render the sheet), and `region_map.json` (`subject`, `summary`, `regions` with `id`, `name`, `description`, `pattern`, and a normalized polygon). `outline.svg` is the same sheet, kept so earlier clients still resolve. The STL from stage 6 is still to come.

## Cache

The cache key is the SHA-256 of the uploaded bytes, taken before any resize. `backend/demo_cache/by_hash/` holds the hero sheets. The runtime cache under `backend/data/cache/` holds everything else. A hit copies the SVG and region map into a new job and marks every stage with the timings from the run that produced it, plus `cached: true`.

Hero images are drawn by `backend/scripts/build_demo_cache.py`: a heart, an animal cell, a bar chart, a world map, and three earlier diagrams (plant cell, neighborhood, park). Sources and the CC0 dedication are in `backend/samples/CREDITS.md`. They are pictures the segmenter is known to split, so a demo click does not depend on a network call.
