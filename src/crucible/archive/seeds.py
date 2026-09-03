"""Seed attacks: the archive's starting corpus.

Hand-written, reviewed, and committed as YAML so that an experiment can be
reproduced from the repository alone. They are what makes the defender demoable
in Phase 5 before the attacker exists in Phase 4.

Ids are derived from each seed's stable name with `uuid5`, so loading the same
file twice produces the same attack ids and the archive stays idempotent.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from crucible.schemas.attack import DEFAULT_BENIGN_INPUT, Attack
from crucible.schemas.taxonomy import DeliveryVector, Objective, Technique

SEEDS_PATH = Path("data/seeds/seed_attacks.yaml")

#: Fixed namespace so a seed's id is stable across machines and runs.
SEED_NAMESPACE = uuid.UUID("6f1d6a2c-7f7e-5c2b-9d0f-3a5f0c9b1e77")


def seed_attack_id(name: str) -> uuid.UUID:
    return uuid.uuid5(SEED_NAMESPACE, name)


class SeedAttack(BaseModel):
    """One entry in the seed file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    objective: Objective
    vector: DeliveryVector
    technique: Technique
    payload: str = Field(min_length=1)
    benign_user_input: str | None = None
    carrier_title: str | None = None

    def to_attack(self) -> Attack:
        """A seed has no parent and belongs to round 0."""
        return Attack(
            attack_id=seed_attack_id(self.id),
            payload=self.payload.strip(),
            vector=self.vector,
            objective=self.objective,
            technique=self.technique,
            parent_id=None,
            round_generated=0,
            benign_user_input=self.benign_user_input or DEFAULT_BENIGN_INPUT,
            carrier_title=self.carrier_title,
        )


class SeedFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int
    attacks: tuple[SeedAttack, ...]


def load_seed_file(path: Path = SEEDS_PATH) -> SeedFile:
    """Parse and validate the seed file. Invalid axis values fail here."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return SeedFile.model_validate(raw)


def load_seed_attacks(path: Path = SEEDS_PATH) -> tuple[Attack, ...]:
    return tuple(seed.to_attack() for seed in load_seed_file(path).attacks)
