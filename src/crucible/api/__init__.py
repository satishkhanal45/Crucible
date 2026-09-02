"""HTTP layer. Depends on services, never on the ORM directly."""

from crucible.api.app import create_app

__all__ = ["create_app"]
