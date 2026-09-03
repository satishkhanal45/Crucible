"""Verification tests 6-10: collapse detection.

A halted run is a *result*. `attacker_exhausted` after nine rounds says the
search found everything this attacker can find against this target; that is a
finding, and the run's status says `halted`, never `failed`.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from crucible.loop.collapse import (
    MAX_OVERFIT_GAP,
    MIN_MEAN_NOVELTY,
    MIN_UTILITY_RATIO,
    RoundSignals,
    detect,
)
from crucible.loop.reports import HaltReason, RunStatus


def signals(
    round_number: int = 1,
    *,
    novelty: float = 0.8,
    new_cells: int = 2,
    rejection: float = 0.1,
    utility: float = 1.0,
    gap: float = 0.05,
    cost: str = "0.10",
) -> RoundSignals:
    return RoundSignals(
        round_number=round_number,
        mean_novelty=novelty,
        new_cells=new_cells,
        novelty_rejection_rate=rejection,
        utility_pass_rate=utility,
        overfit_gap=gap,
        cost_usd=Decimal(cost),
    )


# ------------------------------------------------------------------ test 6


def test_three_rounds_below_the_novelty_floor_halts() -> None:
    history = [signals(index, novelty=0.1) for index in range(1, 4)]

    assert detect(history) is HaltReason.ATTACKER_EXHAUSTED


def test_two_rounds_below_the_novelty_floor_does_not_halt() -> None:
    history = [signals(1), *[signals(index, novelty=0.1) for index in range(2, 4)]]

    assert detect(history) is None


def test_a_single_recovered_round_resets_the_novelty_streak() -> None:
    history = [
        signals(1, novelty=0.1),
        signals(2, novelty=0.1),
        signals(3, novelty=0.9),
        signals(4, novelty=0.1),
    ]

    assert detect(history) is None


def test_the_novelty_floor_is_the_documented_one() -> None:
    assert MIN_MEAN_NOVELTY == 0.2
    assert detect([signals(index, novelty=0.19) for index in range(3)]) is (
        HaltReason.ATTACKER_EXHAUSTED
    )
    assert detect([signals(index, novelty=0.2) for index in range(3)]) is None


# ------------------------------------------------------------------ test 7


def test_three_rounds_without_a_new_cell_halts_as_search_stalled() -> None:
    history = [signals(index, new_cells=0) for index in range(1, 4)]

    assert detect(history) is HaltReason.SEARCH_STALLED


def test_two_rounds_without_a_new_cell_does_not_halt() -> None:
    history = [signals(1, new_cells=1), signals(2, new_cells=0), signals(3, new_cells=0)]

    assert detect(history) is None


# ------------------------------------------------------------------ test 8


def test_utility_below_eighty_five_percent_of_baseline_halts() -> None:
    history = [signals(1, utility=0.80)]

    assert detect(history, baseline_utility=1.0) is HaltReason.UTILITY_COLLAPSE


def test_utility_just_above_the_floor_does_not_halt() -> None:
    history = [signals(1, utility=0.86)]

    assert detect(history, baseline_utility=1.0) is None
    assert MIN_UTILITY_RATIO == 0.85


def test_the_floor_is_relative_to_the_baseline_not_absolute() -> None:
    """A target whose baseline is 0.6 has not collapsed at 0.55."""
    assert detect([signals(1, utility=0.55)], baseline_utility=0.6) is None
    assert detect([signals(1, utility=0.50)], baseline_utility=0.6) is (HaltReason.UTILITY_COLLAPSE)


# ------------------------------------------------------------------ test 9


def test_an_overfit_gap_above_the_threshold_halts() -> None:
    history = [signals(1, gap=0.30)]

    assert detect(history) is HaltReason.OVERFITTING
    assert MAX_OVERFIT_GAP == 0.25


def test_a_gap_at_the_threshold_does_not_halt() -> None:
    assert detect([signals(1, gap=0.25)]) is None


def test_a_negative_gap_never_halts() -> None:
    """Holdout above archive is luck, not overfitting."""
    assert detect([signals(1, gap=-0.4)]) is None


# ------------------------------------------------- rediscovery and budget


def test_two_rounds_of_mostly_rejected_attacks_halts_as_rediscovery_only() -> None:
    history = [signals(1, rejection=0.9), signals(2, rejection=0.85)]

    assert detect(history) is HaltReason.REDISCOVERY_ONLY


def test_one_round_of_rediscovery_does_not_halt() -> None:
    assert detect([signals(1, rejection=0.95)]) is None


def test_going_over_budget_halts() -> None:
    history = [signals(1, cost="6.00")]

    assert detect(history, budget_usd=Decimal("5.00")) is HaltReason.BUDGET_EXCEEDED


def test_no_history_never_halts() -> None:
    assert detect([]) is None


# ----------------------------------------------------------------- test 10


def test_every_signal_has_its_own_named_halt_reason() -> None:
    cases = {
        HaltReason.BUDGET_EXCEEDED: ([signals(1, cost="9.00")], {"budget_usd": Decimal("1")}),
        HaltReason.UTILITY_COLLAPSE: ([signals(1, utility=0.1)], {}),
        HaltReason.OVERFITTING: ([signals(1, gap=0.9)], {}),
        HaltReason.ATTACKER_EXHAUSTED: (
            [signals(index, novelty=0.05) for index in range(3)],
            {},
        ),
        HaltReason.SEARCH_STALLED: ([signals(index, new_cells=0) for index in range(3)], {}),
        HaltReason.REDISCOVERY_ONLY: (
            [signals(index, rejection=0.95) for index in range(2)],
            {},
        ),
    }

    assert set(cases) == set(HaltReason), "every halt reason must be reachable"
    for expected, (history, kwargs) in cases.items():
        assert detect(history, **kwargs) is expected  # type: ignore[arg-type]


def test_a_halted_run_is_not_a_failed_run() -> None:
    """A run that halts on attacker_exhausted is a valid experiment."""
    assert RunStatus.HALTED.value == "halted"
    assert RunStatus.HALTED is not RunStatus.FAILED
    assert {status.value for status in RunStatus} == {
        "running",
        "completed",
        "halted",
        "failed",
    }


def test_single_round_signals_take_precedence_over_multi_round_ones() -> None:
    """A round whose own numbers are invalid is reported as such first."""
    history = [signals(index, novelty=0.01, new_cells=0, gap=0.9) for index in range(3)]

    assert detect(history) is HaltReason.OVERFITTING


@pytest.mark.parametrize("reason", list(HaltReason))
def test_every_halt_reason_is_a_known_string(reason: HaltReason) -> None:
    assert HaltReason(reason.value) is reason
