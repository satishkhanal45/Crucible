"""Archive storage, embeddings, novelty, and the MAP-Elites grid.

Only the leaf modules are re-exported here. `ArchiveService` is imported from
`crucible.archive.service` directly: it depends on the repository layer, which
depends on this package, and re-exporting it would make that a cycle.
"""

from crucible.archive.fitness import FitnessBreakdown, fitness, generality
from crucible.archive.grid import (
    Coverage,
    InvalidCellKey,
    all_cell_keys,
    coverage,
    parse_cell_key,
)
from crucible.archive.novelty import (
    K_NEIGHBOURS,
    MIN_NOVELTY,
    Neighbour,
    NoveltyRejection,
    NoveltyScore,
)

__all__ = [
    "K_NEIGHBOURS",
    "MIN_NOVELTY",
    "Coverage",
    "FitnessBreakdown",
    "InvalidCellKey",
    "Neighbour",
    "NoveltyRejection",
    "NoveltyScore",
    "all_cell_keys",
    "coverage",
    "fitness",
    "generality",
    "parse_cell_key",
]
