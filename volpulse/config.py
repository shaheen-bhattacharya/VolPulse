"""Central configuration for VolPulse."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_TICKERS: tuple[str, ...] = (
    # Liquid ETFs
    "SPY", "QQQ", "IWM", "GLD", "TLT",
    # Mega-cap single names
    "AAPL", "MSFT", "NVDA",
)


@dataclass(frozen=True)
class DataConfig:
    """Settings for the historical data pipeline (Phase 1)."""

    tickers: tuple[str, ...] = DEFAULT_TICKERS
    start: str = "2019-01-01"
    interval: str = "1d"
    data_root: Path = PROJECT_ROOT / "data_store"
    # A run of more than this many consecutive missing business days is
    # flagged as a suspicious gap (single-day US market holidays are normal).
    max_gap_bdays: int = 1
    max_concurrent_fetches: int = 4
