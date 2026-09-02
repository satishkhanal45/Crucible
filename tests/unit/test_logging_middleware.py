"""Verification test 2: request-id propagation into logs and response headers."""

from __future__ import annotations

import io
import json
import logging
from typing import Any

import pytest
from fastapi import FastAPI

from crucible.api.app import create_app
from crucible.config import Settings
from crucible.db.session import Database
from crucible.logging import REQUEST_ID_HEADER, configure_logging, get_logger
from tests.fixtures.asgi import running

route_logger = get_logger("tests.route")


@pytest.fixture
def log_stream() -> io.StringIO:
    return io.StringIO()


@pytest.fixture
def app(settings: Settings, log_stream: io.StringIO) -> FastAPI:
    application = create_app(settings, database=Database(settings.DATABASE_URL))

    async def emit() -> dict[str, str]:
        route_logger.info("route.handled", extra={"detail": "from the route handler"})
        return {"ok": "true"}

    application.add_api_route("/_test/log", emit, methods=["GET"])
    # create_app() points logging at stdout; capture it here instead.
    configure_logging(logging.INFO, stream=log_stream)
    return application


def _records(log_stream: io.StringIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in log_stream.getvalue().splitlines() if line.strip()]


async def test_request_id_is_generated_and_echoed(app: FastAPI, log_stream: io.StringIO) -> None:
    async with running(app) as client:
        response = await client.get("/_test/log")

    assert response.status_code == 200
    request_id = response.headers[REQUEST_ID_HEADER]
    assert request_id

    records = _records(log_stream)
    messages = [record["message"] for record in records]
    assert "route.handled" in messages
    assert "request.completed" in messages
    # Every record emitted inside the request scope carries the same id;
    # records from outside it (lifespan, the test client) carry none.
    in_scope = [r for r in records if r["message"] in {"route.handled", "request.completed"}]
    assert len(in_scope) == 2
    assert all(record["request_id"] == request_id for record in in_scope)
    out_of_scope = [r for r in records if r["message"] in {"app.startup", "app.shutdown"}]
    assert all(record["request_id"] is None for record in out_of_scope)


async def test_caller_supplied_request_id_is_preserved(
    app: FastAPI, log_stream: io.StringIO
) -> None:
    supplied = "caller-supplied-2f8c1d"

    async with running(app) as client:
        response = await client.get("/_test/log", headers={REQUEST_ID_HEADER: supplied})

    assert response.headers[REQUEST_ID_HEADER] == supplied
    request_records = [
        record
        for record in _records(log_stream)
        if record["message"] in {"route.handled", "request.completed"}
    ]
    assert request_records
    assert all(record["request_id"] == supplied for record in request_records)


async def test_unusable_caller_request_id_is_replaced(
    app: FastAPI, log_stream: io.StringIO
) -> None:
    del log_stream
    async with running(app) as client:
        response = await client.get("/_test/log", headers={REQUEST_ID_HEADER: "x" * 500})

    assert response.headers[REQUEST_ID_HEADER] != "x" * 500
    assert len(response.headers[REQUEST_ID_HEADER]) == 32


def test_records_outside_a_request_have_no_request_id(log_stream: io.StringIO) -> None:
    configure_logging(logging.INFO, stream=log_stream)
    get_logger("tests.outside").info("no.request")

    records = _records(log_stream)
    assert records[-1]["message"] == "no.request"
    assert records[-1]["request_id"] is None


async def test_access_log_carries_method_path_and_status(
    app: FastAPI, log_stream: io.StringIO
) -> None:
    async with running(app) as client:
        await client.get("/_test/log")

    completed = [r for r in _records(log_stream) if r["message"] == "request.completed"][-1]
    assert completed["http_method"] == "GET"
    assert completed["http_path"] == "/_test/log"
    assert completed["http_status"] == 200
    assert completed["duration_ms"] >= 0
