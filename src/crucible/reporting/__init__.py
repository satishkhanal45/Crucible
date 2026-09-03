"""Markdown and JSON reports, plus the static visuals that go with them.

Cut B1: no dashboard, no JS toolchain. What ships is the document you put in the
repository and the images you put in the README.
"""

from crucible.reporting.charts import coverage_grid, coverage_strip, three_curves
from crucible.reporting.data import GeneralAttack, ReportData, gather
from crucible.reporting.diff import ChangeKind, ConfigDiff, diff_configs
from crucible.reporting.lineage import build_lineage, render_lineage
from crucible.reporting.markdown import render_round_report, render_run_report
from crucible.reporting.redaction import (
    payload_hash,
    present_payload,
    redact_payload,
    redact_trace,
)

__all__ = [
    "ChangeKind",
    "ConfigDiff",
    "GeneralAttack",
    "ReportData",
    "build_lineage",
    "coverage_grid",
    "coverage_strip",
    "diff_configs",
    "gather",
    "payload_hash",
    "present_payload",
    "redact_payload",
    "redact_trace",
    "render_lineage",
    "render_round_report",
    "render_run_report",
    "three_curves",
]
