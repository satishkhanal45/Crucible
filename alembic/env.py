"""Alembic environment, wired to the app's metadata and DATABASE_URL."""

from __future__ import annotations

import asyncio
from typing import Any

from alembic import context
from sqlalchemy import Connection, pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from crucible.config import get_settings
from crucible.db import models  # noqa: F401 - imported so metadata is populated
from crucible.db.base import Base

config = context.config
target_metadata = Base.metadata


def database_url() -> str:
    configured = config.get_main_option("sqlalchemy.url")
    return configured or get_settings().DATABASE_URL


def render_item(type_: str, obj: Any, autogen_context: Any) -> str | bool:
    """Render pgvector columns with their real import in generated migrations."""
    if type_ == "type" and obj.__class__.__module__.startswith("pgvector"):
        autogen_context.imports.add("import pgvector.sqlalchemy")
        return f"pgvector.sqlalchemy.Vector(dim={obj.dim})"
    return False


def include_name(name: str | None, type_: str, parent_names: Any) -> bool:
    """Only manage tables this project's metadata declares.

    LangGraph's `AsyncPostgresSaver` creates and migrates its own checkpoint
    tables in the same database. They are not ours, so autogenerate must neither
    propose dropping them nor count them as drift — the empty-diff test has to
    stay meaningful.
    """
    del parent_names
    if type_ == "table":
        return name in target_metadata.tables
    return True


def configure(connection: Connection | None = None, url: str | None = None) -> None:
    context.configure(
        connection=connection,
        url=url,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        render_item=render_item,
        render_as_batch=False,
        include_schemas=False,
        include_name=include_name,
    )


def run_migrations_offline() -> None:
    configure(url=database_url())
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    configure(connection=connection)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = database_url()
    engine = async_engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)
    try:
        async with engine.connect() as connection:
            await connection.run_sync(do_run_migrations)
    finally:
        await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
