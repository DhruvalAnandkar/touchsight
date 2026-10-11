"""The one HTTP client used for hosted model calls.

Connect timeout is 5 seconds, read timeout is 45 seconds, and a failed
transport or a 5xx response is tried once more. HTTP 429 and a quota error
are not retried.
"""

from __future__ import annotations

import httpx

CONNECT_TIMEOUT_S = 5.0
READ_TIMEOUT_S = 45.0


def is_quota_response(status_code: int, body: str) -> bool:
    """True for HTTP 429 or an error body that says the quota is exhausted."""

    if status_code == 429:
        return True
    if status_code < 400:
        return False
    lowered = body.lower()
    return "quota" in lowered or "resource_exhausted" in lowered


def request(method: str, url: str, **kwargs: object) -> httpx.Response:
    timeout = httpx.Timeout(READ_TIMEOUT_S, connect=CONNECT_TIMEOUT_S)
    last_error: Exception | None = None
    for _attempt in range(2):
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.request(method, url, **kwargs)  # type: ignore[arg-type]
        except httpx.TransportError as exc:
            last_error = exc
            continue
        if is_quota_response(response.status_code, response.text):
            response.raise_for_status()
        if response.status_code >= 500:
            last_error = httpx.HTTPStatusError(
                f"HTTP {response.status_code}",
                request=response.request,
                response=response,
            )
            continue
        response.raise_for_status()
        return response
    assert last_error is not None
    raise last_error
