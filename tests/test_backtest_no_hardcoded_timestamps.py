"""Guards Milestone 1's core promise: no market-specific time constant may
be hard-coded anywhere in src/backtest — every cutoff/gate-closure/delivery
time must be read from config/markets.yaml via src.config.MarketConfig.
"""
import re
from pathlib import Path

BACKTEST_DIR = Path(__file__).resolve().parents[1] / "src" / "backtest"

# "HH:MM" string literals, e.g. "12:00" or '08:00'
TIME_STRING_RE = re.compile(r"""['"]\d{1,2}:\d{2}['"]""")
# datetime.time(12, 0) / dt.time(8, 0) style literal construction
TIME_CTOR_RE = re.compile(r"\btime\(\s*\d+\s*,\s*\d+\s*\)")


def _backtest_source_files():
    return sorted(BACKTEST_DIR.rglob("*.py"))


def test_backtest_module_has_no_hardcoded_time_strings():
    offenders = []
    for f in _backtest_source_files():
        text = f.read_text(encoding="utf-8")
        for match in TIME_STRING_RE.finditer(text):
            offenders.append(f"{f.name}: {match.group()}")
    assert not offenders, (
        "Hard-coded 'HH:MM' timestamp literal(s) found in src/backtest — "
        f"market times must come from config/markets.yaml: {offenders}"
    )


def test_backtest_module_has_no_hardcoded_time_constructors():
    offenders = []
    for f in _backtest_source_files():
        text = f.read_text(encoding="utf-8")
        for match in TIME_CTOR_RE.finditer(text):
            offenders.append(f"{f.name}: {match.group()}")
    assert not offenders, (
        "Hard-coded time(...) construction found in src/backtest — "
        f"market times must come from config/markets.yaml: {offenders}"
    )


def test_leaking_fixture_would_be_caught():
    """Sanity check the regexes actually work, using a fixture string that
    mimics a leaking hard-code (not written to src/backtest itself)."""
    leaking_snippet = 'cutoff = dt.datetime.combine(d, dt.time(12, 0))  # "12:00" hard-coded'
    assert TIME_STRING_RE.search(leaking_snippet)
    assert TIME_CTOR_RE.search(leaking_snippet)
