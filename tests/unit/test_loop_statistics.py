"""Verification tests 11-13: the statistics a round report is allowed to carry.

docs/spec.md section 15: Wilson intervals, never the normal approximation, and a
bare rate is not reportable. 0.72 -> 0.78 at n=90 is not an improvement.
"""

from __future__ import annotations

import math
import uuid

import pytest
from pydantic import ValidationError

from crucible.loop.reports import BareRate, HaltReason, RoundReport
from crucible.loop.statistics import (
    Proportion,
    improved,
    normal_quantile,
    two_proportion_test,
    wilson_interval,
)


def reference_wilson(
    successes: int, trials: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    """An independently arranged Wilson interval, as the quadratic's roots.

    Deliberately not the implementation's algebra: solving
    |p_hat - p| = z * sqrt(p(1-p)/n) for p gives the same interval by a
    different route, so agreement is evidence rather than a tautology.
    """
    p_hat = successes / trials
    a = 1.0 + z * z / trials
    b = -(2.0 * p_hat + z * z / trials)
    c = p_hat * p_hat
    discriminant = math.sqrt(b * b - 4.0 * a * c)
    return ((-b - discriminant) / (2.0 * a), (-b + discriminant) / (2.0 * a))


# ----------------------------------------------------------------- test 11


def test_wilson_interval_for_65_of_90_matches_a_reference_to_six_decimals() -> None:
    low, high = wilson_interval(65, 90)
    reference_low, reference_high = reference_wilson(65, 90)

    assert round(low, 6) == round(reference_low, 6)
    assert round(high, 6) == round(reference_high, 6)
    # Published value for this well-known case, to three decimals.
    assert (round(low, 3), round(high, 3)) == (0.622, 0.804)


@pytest.mark.parametrize(
    ("successes", "trials"), [(0, 10), (10, 10), (1, 3), (45, 90), (7, 13), (999, 1000)]
)
def test_wilson_agrees_with_the_reference_everywhere(successes: int, trials: int) -> None:
    low, high = wilson_interval(successes, trials)
    reference_low, reference_high = reference_wilson(successes, trials)

    assert round(low, 6) == round(max(0.0, reference_low), 6)
    assert round(high, 6) == round(min(1.0, reference_high), 6)


def test_wilson_never_leaves_the_unit_interval() -> None:
    """The normal approximation does; that is why this project does not use it."""
    for successes, trials in ((0, 5), (5, 5), (1, 100), (99, 100)):
        low, high = wilson_interval(successes, trials)
        assert 0.0 <= low <= high <= 1.0

    # The Wald interval on 5/5 runs to exactly 1.0 with zero width, which is the
    # failure mode Wilson exists to avoid.
    assert wilson_interval(5, 5)[0] < 1.0


def test_an_empty_sample_is_maximally_uncertain() -> None:
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_the_quantile_is_the_standard_two_sided_z() -> None:
    assert normal_quantile(0.95) == pytest.approx(1.959963984540054, abs=1e-12)
    assert round(normal_quantile(0.99), 4) == 2.5758


def test_a_proportion_reports_itself_with_its_interval() -> None:
    proportion = Proportion(successes=65, trials=90)

    assert str(proportion) == "0.722 [0.622, 0.804] (n=90)"
    assert proportion.rate == pytest.approx(65 / 90)


# ----------------------------------------------------------------- test 12


def test_no_significant_difference_for_072_versus_078_at_n_90() -> None:
    """The exact case docs/spec.md section 15 calls out as not an improvement."""
    before = Proportion(successes=65, trials=90)
    after = Proportion(successes=70, trials=90)

    test = two_proportion_test(before, after)

    assert test.difference > 0, "the rate did go up"
    assert test.significant is False
    assert test.p_value > 0.05
    assert before.overlaps(after), "the intervals overlap"
    assert improved(before, after) is False, "and so it may not be called an improvement"


def test_a_significant_difference_for_the_same_rates_at_n_900() -> None:
    before = Proportion(successes=648, trials=900)
    after = Proportion(successes=702, trials=900)

    test = two_proportion_test(before, after)

    assert test.significant is True
    assert test.p_value < 0.05
    assert improved(before, after) is True


def test_a_lower_rate_is_never_an_improvement_however_significant() -> None:
    before = Proportion(successes=702, trials=900)
    after = Proportion(successes=648, trials=900)

    assert two_proportion_test(before, after).significant is True
    assert improved(before, after) is False


def test_an_empty_comparison_is_not_significant() -> None:
    empty = Proportion(successes=0, trials=0)

    assert two_proportion_test(empty, empty).significant is False


# ----------------------------------------------------------------- test 13


def report(**overrides: object) -> RoundReport:
    fields: dict[str, object] = {
        "run_id": uuid.uuid4(),
        "round_number": 1,
        "attacker_mode": "black_box",
        "defense_before": "a" * 32,
        "defense_after": "b" * 32,
        "archive_block": Proportion(successes=20, trials=30),
        "holdout_block": Proportion(successes=6, trials=8),
        "utility_pass": Proportion(successes=40, trials=40),
    }
    fields.update(overrides)
    return RoundReport.model_validate(fields)


def test_every_rate_in_a_round_report_carries_an_interval() -> None:
    round_report = report()

    round_report.validate_intervals()
    for proportion in round_report.rates().values():
        low, high = proportion.interval
        assert 0.0 <= low <= high <= 1.0
    assert "[" in round_report.summary()


@pytest.mark.parametrize("field", ["archive_block", "holdout_block", "utility_pass"])
def test_a_bare_rate_cannot_be_put_in_a_report(field: str) -> None:
    """The type forbids it, so the mistake is impossible rather than discouraged."""
    with pytest.raises(ValidationError):
        report(**{field: 0.78})


def test_validate_intervals_rejects_a_smuggled_bare_rate() -> None:
    round_report = report()
    smuggled = round_report.model_copy(update={"archive_block": 0.78})

    with pytest.raises(BareRate, match="bare rate"):
        smuggled.validate_intervals()


def test_the_overfit_gap_is_archive_minus_holdout() -> None:
    round_report = report(
        archive_block=Proportion(successes=90, trials=100),
        holdout_block=Proportion(successes=60, trials=100),
    )

    assert round_report.overfit_gap == pytest.approx(0.3)


def test_a_halted_report_still_carries_its_rates() -> None:
    round_report = report(halt_reason=HaltReason.OVERFITTING)

    round_report.validate_intervals()
    assert round_report.halt_reason is HaltReason.OVERFITTING
    assert "HALT" in round_report.summary()
