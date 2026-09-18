"""Loader for config/markets.yaml. This is the single place market
definitions get parsed — nothing else should read the yaml file directly.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "markets.yaml"


@dataclass(frozen=True)
class MarketConfig:
    name: str
    source_column: str
    target_column: str
    unit: str
    product_length_hours: int
    gate_closure_local: dt.time
    gate_closure_offset_days: int
    forecast_cutoff_local: dt.time
    delivery_start_local: dt.time
    horizons_hours: list[int]
    timezone: ZoneInfo
    extra: dict


def _parse_time(s: str) -> dt.time:
    hh, mm = s.split(":")
    return dt.time(int(hh), int(mm))


def load_markets_config(path: Path | None = None) -> dict[str, MarketConfig]:
    path = path or CONFIG_PATH
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    known_fields = {
        "source_column", "target_column", "unit", "product_length_hours",
        "gate_closure_local", "gate_closure_offset_days", "forecast_cutoff_local",
        "delivery_start_local", "horizons_hours", "timezone",
    }

    markets: dict[str, MarketConfig] = {}
    for name, cfg in raw.items():
        markets[name] = MarketConfig(
            name=name,
            source_column=cfg["source_column"],
            target_column=cfg["target_column"],
            unit=cfg["unit"],
            product_length_hours=cfg["product_length_hours"],
            gate_closure_local=_parse_time(cfg["gate_closure_local"]),
            gate_closure_offset_days=cfg["gate_closure_offset_days"],
            forecast_cutoff_local=_parse_time(cfg["forecast_cutoff_local"]),
            delivery_start_local=_parse_time(cfg["delivery_start_local"]),
            horizons_hours=list(cfg["horizons_hours"]),
            timezone=ZoneInfo(cfg["timezone"]),
            extra={k: v for k, v in cfg.items() if k not in known_fields},
        )
    return markets
