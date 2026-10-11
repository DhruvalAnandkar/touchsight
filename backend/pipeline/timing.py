"""Timing and logging for pipeline stages."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from functools import wraps
from typing import TypeVar

logger = logging.getLogger("touchsight.pipeline")

F = TypeVar("F", bound=Callable[..., object])


def timed(stage: str) -> Callable[[F], F]:
    """Log start, failure, and elapsed milliseconds for a stage function."""

    def decorate(fn: F) -> F:
        @wraps(fn)
        def wrapper(*args: object, **kwargs: object) -> object:
            started = time.perf_counter()
            logger.info("stage %s start", stage)
            try:
                result = fn(*args, **kwargs)
            except Exception:
                elapsed_ms = (time.perf_counter() - started) * 1000
                logger.exception("stage %s failed after %.0fms", stage, elapsed_ms)
                raise
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.info("stage %s done in %.0fms", stage, elapsed_ms)
            return result

        return wrapper  # type: ignore[return-value]

    return decorate
