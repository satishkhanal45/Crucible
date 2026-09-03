"""Hotfix verification: the CLI boundary prints one line, not sixty.

A refused database connection and an expired API key are operator errors. A
rich traceback buries the single fact that says what to change, and `--debug`
is where the traceback belongs.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
import typer
from typer.testing import CliRunner

from crucible.cli import main as cli_main
from crucible.config import ConfigurationError
from crucible.services.cost_meter import BudgetExceeded
from crucible.services.retry import AuthenticationFailed, ModelNotFound, RateLimited

runner = CliRunner()


@pytest.fixture(autouse=True)
def _reset_debug() -> None:
    cli_main._debug = False


def _app_raising(error: BaseException) -> typer.Typer:
    """A one-command app whose command fails the way a real one would."""
    app = typer.Typer()

    @app.command()
    @cli_main.handled
    def boom() -> None:
        raise error

    return app


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            ConfigurationError("Invalid configuration -> TARGET_MODEL: required"),
            "TARGET_MODEL",
        ),
        (AuthenticationFailed("groq", "Invalid API Key"), "GROQ_API_KEY"),
        (ModelNotFound("groq", "llama-3.3-70b-versatile"), "decommissioned"),
        (RateLimited("429 forever"), "rate limit"),
        (BudgetExceeded(None, Decimal("6"), Decimal("5")), "ROUND_BUDGET_USD"),
        (ConnectionRefusedError(111, "Connection refused"), "database"),
    ],
)
def test_a_known_failure_prints_one_actionable_line(error: BaseException, expected: str) -> None:
    result = runner.invoke(_app_raising(error))

    assert result.exit_code == 1
    assert expected in result.output
    assert "Traceback" not in result.output
    # A message and a hint. Counting physical lines would measure rich's
    # wrapping at the terminal width, not the shape of what was printed.
    assert len(result.output) <= 400, result.output


def test_the_model_not_found_line_names_the_model() -> None:
    result = runner.invoke(_app_raising(ModelNotFound("groq", "some-retired-id")))

    assert "some-retired-id" in result.output
    assert "console.groq.com/docs/deprecations" in result.output


def test_the_rate_limit_line_says_which_knob_to_turn() -> None:
    result = runner.invoke(_app_raising(RateLimited("429 forever")))

    assert "PROVIDER_MIN_INTERVAL_SECONDS" in result.output
    assert "PROVIDER_MAX_CONCURRENCY" in result.output


def test_debug_restores_the_traceback() -> None:
    cli_main._debug = True
    result = runner.invoke(_app_raising(AuthenticationFailed("groq")), catch_exceptions=True)

    assert isinstance(result.exception, AuthenticationFailed)


def test_an_unknown_error_is_not_swallowed() -> None:
    """The handler catches what an operator can act on, not everything."""
    result = runner.invoke(_app_raising(ZeroDivisionError("a real defect")), catch_exceptions=True)

    assert isinstance(result.exception, ZeroDivisionError)


def test_typer_exit_passes_through_unchanged() -> None:
    """Commands that exit deliberately keep their own code."""
    result = runner.invoke(_app_raising(typer.Exit(code=3)))

    assert result.exit_code == 3


def test_every_command_is_wrapped() -> None:
    """A command added later must not slip past the boundary."""
    commands = [
        cli_main.seed,
        cli_main.archive_reclassify,
        cli_main.archive_stats,
        cli_main.loop_start,
        cli_main.loop_resume,
        cli_main.loop_status,
        cli_main.eval_defense,
        cli_main.report_round,
        cli_main.report_run,
    ]

    unwrapped = [command.__name__ for command in commands if not hasattr(command, "__wrapped__")]
    assert not unwrapped, f"these commands bypass the error boundary: {unwrapped}"


def test_the_debug_flag_is_documented() -> None:
    result = runner.invoke(cli_main.app, ["--help"])

    assert "--debug" in result.output
