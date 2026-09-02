"""ASGI entrypoint: `uvicorn crucible.api.main:app`."""

from __future__ import annotations

from crucible.api.app import create_app

app = create_app()
