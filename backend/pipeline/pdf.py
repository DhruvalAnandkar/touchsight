"""Turn the tactile SVG into a PDF.

PyMuPDF renders the SVG. If it is not installed, the caller still has the SVG.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("touchsight.pipeline.pdf")


def svg_to_pdf(svg: str) -> bytes:
    import pymupdf

    document = pymupdf.open(stream=svg.encode("utf-8"), filetype="svg")
    try:
        return document.convert_to_pdf()
    finally:
        document.close()


def try_svg_to_pdf(svg: str) -> bytes:
    try:
        return svg_to_pdf(svg)
    except Exception:
        logger.warning("could not build a PDF from the tactile sheet", exc_info=True)
        return b""
