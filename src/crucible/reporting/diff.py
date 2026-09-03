"""Structured `DefenseConfig` diffs.

A round-over-round diff is rendered as added, removed and changed keys, not as
raw YAML text: the question a reader has is "what did the defender actually
change", and a text diff of a nested document answers it badly.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict

from crucible.defenses.config import DefenseConfig


class ChangeKind(StrEnum):
    ADDED = "added"
    REMOVED = "removed"
    CHANGED = "changed"


class ConfigChange(BaseModel):
    """One key that differs between two configs."""

    model_config = ConfigDict(frozen=True)

    path: str
    kind: ChangeKind
    before: Any = None
    after: Any = None

    def render(self) -> str:
        if self.kind is ChangeKind.ADDED:
            return f"+ {self.path} = {_value(self.after)}"
        if self.kind is ChangeKind.REMOVED:
            return f"- {self.path} (was {_value(self.before)})"
        return f"~ {self.path}: {_value(self.before)} -> {_value(self.after)}"


class ConfigDiff(BaseModel):
    """Everything that changed between D(n-1) and D(n)."""

    model_config = ConfigDict(frozen=True)

    changes: tuple[ConfigChange, ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.changes)

    def of_kind(self, kind: ChangeKind) -> tuple[ConfigChange, ...]:
        return tuple(change for change in self.changes if change.kind is kind)

    def render(self) -> str:
        if not self.changes:
            return "(no change: the defender proposed nothing better than the current config)"
        return "\n".join(change.render() for change in self.changes)


def _value(value: Any) -> str:
    if value is None:
        return "unset"
    if isinstance(value, list | tuple):
        return "[" + ", ".join(str(item) for item in value) + "]" if value else "[]"
    return str(value)


def _flatten(payload: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(_flatten(value, path))
        elif isinstance(value, list) and value and isinstance(value[0], Mapping):
            # A list of rules: address each by its name so a diff reads well.
            for item in value:
                name = str(item.get("name", "?"))
                flat.update(_flatten(item, f"{path}[{name}]"))
        else:
            flat[path] = value
    return flat


def diff_configs(before: DefenseConfig, after: DefenseConfig) -> ConfigDiff:
    """Structured diff, ordered by key so a report is reproducible."""
    left = _flatten(before.to_dict())
    right = _flatten(after.to_dict())

    changes: list[ConfigChange] = []
    for path in sorted(set(left) | set(right)):
        old, new = left.get(path), right.get(path)
        if path not in left:
            changes.append(ConfigChange(path=path, kind=ChangeKind.ADDED, after=new))
        elif path not in right:
            changes.append(ConfigChange(path=path, kind=ChangeKind.REMOVED, before=old))
        elif old != new:
            changes.append(ConfigChange(path=path, kind=ChangeKind.CHANGED, before=old, after=new))
    return ConfigDiff(changes=tuple(changes))
