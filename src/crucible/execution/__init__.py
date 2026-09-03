"""Sandboxed attack execution: delivery vectors, the pool, and the egress guard."""

from crucible.execution.egress import EgressGuard, EgressViolation, GuardedTransport
from crucible.execution.executor import (
    AttemptExecutor,
    ExecutorSettings,
    ReplayResult,
)
from crucible.execution.pool import TargetPool, worker_namespace
from crucible.execution.vectors import Delivery, build_carrier_document, deliver

__all__ = [
    "AttemptExecutor",
    "Delivery",
    "EgressGuard",
    "EgressViolation",
    "ExecutorSettings",
    "GuardedTransport",
    "ReplayResult",
    "TargetPool",
    "build_carrier_document",
    "deliver",
    "worker_namespace",
]
