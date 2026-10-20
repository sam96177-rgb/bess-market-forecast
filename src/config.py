"""Loader for config/markets.yaml. This is the single place market
definitions get parsed — nothing else should read the yaml file directly.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
CONFIG_PATH = CONFIG_DIR / "markets.yaml"
STUDY_CONFIG_PATH = CONFIG_DIR / "study.yaml"


@dataclass(frozen=True)
class MarketConfig:
    name: str
    source_column: str
    target_column: str
    unit: str
    product_length_hours: float  # fractional for sub-hourly products, e.g. 0.25 = 15 minutes
    dst_convention: str  # "variable_period_count" or "fixed_period_count" — see cutoffs.py
    gate_closure_local: dt.time
    gate_closure_offset_days: int
    forecast_cutoff_local: dt.time
    delivery_start_local: dt.time
    horizons_hours: list[int]
    timezone: ZoneInfo
    results_published_local: dt.time  # PLAN_V5_REVISION.md V5-5: when a lagged auction price actually becomes known
    extra: dict


def _parse_time(s: str) -> dt.time:
    hh, mm = s.split(":")
    return dt.time(int(hh), int(mm))


def load_markets_config(path: Path | None = None) -> dict[str, MarketConfig]:
    path = path or CONFIG_PATH
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    known_fields = {
        "source_column", "target_column", "unit", "product_length_hours",
        "dst_convention", "gate_closure_local", "gate_closure_offset_days",
        "forecast_cutoff_local", "delivery_start_local", "horizons_hours", "timezone",
        "results_published_local",
    }
    valid_dst_conventions = {"variable_period_count", "fixed_period_count"}

    markets: dict[str, MarketConfig] = {}
    for name, cfg in raw.items():
        if name == "entsoe_generation_forecast_published_local":
            continue  # PLAN_V6_REVISION.md V6-3: a top-level assumption, not a market -- see load_entsoe_generation_forecast_published_local
        dst_convention = cfg["dst_convention"]
        if dst_convention not in valid_dst_conventions:
            raise ValueError(
                f"{name}: dst_convention must be one of {valid_dst_conventions}, got {dst_convention!r}"
            )
        markets[name] = MarketConfig(
            name=name,
            source_column=cfg["source_column"],
            target_column=cfg["target_column"],
            unit=cfg["unit"],
            product_length_hours=cfg["product_length_hours"],
            dst_convention=dst_convention,
            gate_closure_local=_parse_time(cfg["gate_closure_local"]),
            gate_closure_offset_days=cfg["gate_closure_offset_days"],
            forecast_cutoff_local=_parse_time(cfg["forecast_cutoff_local"]),
            delivery_start_local=_parse_time(cfg["delivery_start_local"]),
            horizons_hours=list(cfg["horizons_hours"]),
            timezone=ZoneInfo(cfg["timezone"]),
            results_published_local=_parse_time(cfg["results_published_local"]),
            extra={k: v for k, v in cfg.items() if k not in known_fields},
        )
    return markets


def load_entsoe_generation_forecast_published_local(path: Path | None = None) -> dt.time:
    """PLAN_V6_REVISION.md V6-3: the ASSUMED time ENTSO-E's day-ahead
    wind/solar generation forecast becomes available, from config/markets.yaml's
    top-level `entsoe_generation_forecast_published_local` key -- NOT the
    verified 18:00 Brussels publication deadline (src/features/availability.py),
    which is after every market's cutoff. The user chose to use the forecast
    anyway; this is what makes that choice an explicit, visible assumption
    rather than a silently different cutoff baked into the code.
    """
    path = path or CONFIG_PATH
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if "entsoe_generation_forecast_published_local" not in raw:
        raise ValueError(f"{path} missing required key: entsoe_generation_forecast_published_local")
    return _parse_time(raw["entsoe_generation_forecast_published_local"])


@dataclass(frozen=True)
class StudyWindow:
    start: dt.date
    end: dt.date

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"study window end {self.end} is before start {self.start}")


@dataclass(frozen=True)
class StudyConfig:
    """PLAN_V5_REVISION.md V5-6: one file both repos read for what a study
    is, before any data is downloaded. `optimizer:` is read only by
    bess-optimizer -- kept out of this dataclass so this repo can't
    accidentally depend on it."""

    name: str
    history_start: dt.date
    min_training_history_days: int
    test_window: dict[str, StudyWindow]
    models: list[str]
    # PLAN_V6_REVISION.md V6-4: the 60 days right before each market's
    # test_window start, scored to pick per_market_best fairly. Optional so a
    # study.yaml predating V6-4 still loads; forecast_start falls back to the
    # test window's own start when it's absent.
    validation_window: dict[str, StudyWindow] | None = None

    def max_test_window_end(self) -> dt.date:
        return max(w.end for w in self.test_window.values())

    def forecast_start(self, market: str) -> dt.date:
        """The earliest date run_backtest.py should forecast `market` for --
        the validation window's start if configured, else the test window's
        own start. The test window and every reported test-window metric are
        unaffected either way (V6-4: "the optimizer window and all scoring
        stay on the test window only")."""
        if self.validation_window is not None:
            return self.validation_window[market].start
        return self.test_window[market].start


def load_study_config(path: Path | None = None) -> StudyConfig:
    path = path or STUDY_CONFIG_PATH
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    required = {"name", "history_start", "min_training_history_days", "test_window", "models"}
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"{path} missing required keys: {sorted(missing)}")

    test_window = {
        market: StudyWindow(start=window["start"], end=window["end"])
        for market, window in raw["test_window"].items()
    }
    validation_window = None
    if "validation_window" in raw:
        validation_window = {
            market: StudyWindow(start=window["start"], end=window["end"])
            for market, window in raw["validation_window"].items()
        }
    return StudyConfig(
        name=str(raw["name"]),
        history_start=raw["history_start"],
        min_training_history_days=int(raw["min_training_history_days"]),
        test_window=test_window,
        models=list(raw["models"]),
        validation_window=validation_window,
    )
