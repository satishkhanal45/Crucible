"""ASGI middleware: request-id propagation and access logging."""

from __future__ import annotations

import time
from collections.abc import MutableMapping
from typing import Any
from uuid import uuid4

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from crucible.logging import REQUEST_ID_HEADER, bind_request_id, get_logger

logger = get_logger(__name__)

_MAX_REQUEST_ID_LENGTH = 200


def _clean(candidate: str | None) -> str | None:
    """Accept a caller-supplied id only if it is short, printable, single-line."""
    if candidate is None:
        return None
    value = candidate.strip()
    if not value or len(value) > _MAX_REQUEST_ID_LENGTH:
        return None
    if not value.isprintable():
        return None
    return value


class RequestIdMiddleware:
    """Bind a request id to the log context and echo it back to the caller."""

    def __init__(self, app: ASGIApp, header_name: str = REQUEST_ID_HEADER) -> None:
        self.app = app
        self.header_name = header_name

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = Headers(scope=scope).get(self.header_name)
        request_id = _clean(incoming) or uuid4().hex
        status_code = 500
        started = time.perf_counter()

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
                MutableHeaders(scope=message).append(self.header_name, request_id)
            await send(message)

        with bind_request_id(request_id):
            try:
                await self.app(scope, receive, send_with_request_id)
            finally:
                elapsed_ms = (time.perf_counter() - started) * 1000
                details: MutableMapping[str, Any] = {
                    "http_method": scope.get("method"),
                    "http_path": scope.get("path"),
                    "http_status": status_code,
                    "duration_ms": round(elapsed_ms, 3),
                }
                logger.info("request.completed", extra=details)
