"""Boundary schemas for the liveness and readiness endpoints."""

from __future__ import annotations

from pydantic import BaseModel


class Liveness(BaseModel):
    status: str
    env: str


class Readiness(BaseModel):
    status: str
    database: bool
    pgvector: bool
    detail: str | None = None
