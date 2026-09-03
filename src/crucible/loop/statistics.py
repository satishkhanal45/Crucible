"""Statistics for round reports.

Two rules from docs/spec.md section 15, and they are the difference between a
result and a demo:

* **Wilson score intervals, never the normal approximation.** Block rates are
  proportions from samples of tens, where the normal approximation is wrong in
  the direction that flatters the result.
* **A rate without an interval is not reportable.** `Proportion` carries its
  interval, so a bare float cannot be put in a `RoundReport` at all.

A round that moves the block rate from 0.72 to 0.78 on 90 attacks has
overlapping intervals. That is not an improvement, and this module is what stops
anyone reporting it as one.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field

#: Two-sided 95% normal quantile, to full double precision.
Z_95 = 1.959963984540054
DEFAULT_CONFIDENCE = 0.95


def normal_quantile(confidence: float = DEFAULT_CONFIDENCE) -> float:
    """Two-sided z for a confidence level, via the inverse error function."""
    if confidence == DEFAULT_CONFIDENCE:
        return Z_95
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    return math.sqrt(2.0) * _erfinv(confidence)


def _erfinv(value: float) -> float:
    """Inverse error function, Newton-refined from a rational start."""
    if not -1.0 < value < 1.0:
        raise ValueError("erfinv is defined on (-1, 1)")
    # Winitzki's approximation, then two Newton steps against erf.
    a = 0.147
    ln = math.log(1.0 - value * value)
    first = 2.0 / (math.pi * a) + ln / 2.0
    guess = math.copysign(math.sqrt(math.sqrt(first * first - ln / a) - first), value)
    for _ in range(3):
        error = math.erf(guess) - value
        guess -= error / (2.0 / math.sqrt(math.pi) * math.exp(-guess * guess))
    return guess


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def wilson_interval(
    successes: int, trials: int, confidence: float = DEFAULT_CONFIDENCE
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Not the normal approximation: at the sample sizes this project reports
    (tens of attacks per cell, ~40 in the archive) the Wald interval can run
    past 0 or 1 and understates uncertainty exactly where it matters.
    """
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("0 <= successes <= trials is required")
    if trials == 0:
        return (0.0, 1.0)

    z = normal_quantile(confidence)
    proportion = successes / trials
    z_squared = z * z
    denominator = 1.0 + z_squared / trials
    centre = (proportion + z_squared / (2.0 * trials)) / denominator
    spread = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / trials + z_squared / (4.0 * trials * trials))
        / denominator
    )
    return (max(0.0, centre - spread), min(1.0, centre + spread))


class Proportion(BaseModel):
    """A rate that cannot be reported without its interval."""

    model_config = ConfigDict(frozen=True)

    successes: int = Field(ge=0)
    trials: int = Field(ge=0)
    confidence: float = Field(default=DEFAULT_CONFIDENCE, gt=0.0, lt=1.0)

    @property
    def rate(self) -> float:
        return self.successes / self.trials if self.trials else 0.0

    @property
    def interval(self) -> tuple[float, float]:
        return wilson_interval(self.successes, self.trials, self.confidence)

    @property
    def low(self) -> float:
        return self.interval[0]

    @property
    def high(self) -> float:
        return self.interval[1]

    @property
    def width(self) -> float:
        low, high = self.interval
        return high - low

    def overlaps(self, other: Proportion) -> bool:
        low, high = self.interval
        other_low, other_high = other.interval
        return low <= other_high and other_low <= high

    def __str__(self) -> str:
        low, high = self.interval
        return f"{self.rate:.3f} [{low:.3f}, {high:.3f}] (n={self.trials})"


class ProportionTest(BaseModel):
    """A two-proportion z-test, with the honest verdict attached."""

    model_config = ConfigDict(frozen=True)

    first: Proportion
    second: Proportion
    z: float
    p_value: float
    alpha: float = 0.05

    @property
    def difference(self) -> float:
        return self.second.rate - self.first.rate

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    def __str__(self) -> str:
        verdict = "significant" if self.significant else "not significant"
        return (
            f"{self.first} -> {self.second}: difference {self.difference:+.3f}, "
            f"z={self.z:.3f}, p={self.p_value:.4f} ({verdict} at alpha={self.alpha})"
        )


def two_proportion_test(
    first: Proportion, second: Proportion, *, alpha: float = 0.05
) -> ProportionTest:
    """Pooled two-proportion z-test for a round-over-round claim."""
    total_trials = first.trials + second.trials
    if total_trials == 0:
        return ProportionTest(first=first, second=second, z=0.0, p_value=1.0, alpha=alpha)

    pooled = (first.successes + second.successes) / total_trials
    variance = pooled * (1.0 - pooled) * (1.0 / first.trials + 1.0 / second.trials)
    if variance <= 0.0:
        return ProportionTest(first=first, second=second, z=0.0, p_value=1.0, alpha=alpha)

    z = (second.rate - first.rate) / math.sqrt(variance)
    p_value = 2.0 * (1.0 - normal_cdf(abs(z)))
    return ProportionTest(first=first, second=second, z=z, p_value=p_value, alpha=alpha)


def improved(first: Proportion, second: Proportion, *, alpha: float = 0.05) -> bool:
    """Whether a round-over-round increase may be called an improvement.

    Requires both a higher rate and a significant test. A higher number with
    overlapping intervals is noise, and saying otherwise is the single easiest
    way to make this project dishonest.
    """
    test = two_proportion_test(first, second, alpha=alpha)
    return test.difference > 0 and test.significant
