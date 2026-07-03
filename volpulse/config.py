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


@dataclass(frozen=True)
class LiveConfig:
    """Settings for the live tick ingestion layer (Phase 2)."""

    symbols: tuple[str, ...] = DEFAULT_TICKERS
    # "iex" (free), "sip" (paid), or "test" (fake FAKEPACA trades, 24/7).
    feed: str = "iex"
    bar_interval_seconds: int = 60
    ring_buffer_size: int = 500
    vol_window: int = 20
    # 1-minute bars over US regular trading hours: 252 days x 390 minutes.
    vol_periods_per_year: int = 252 * 390
    stale_after_seconds: float = 30.0
    reconnect_initial_backoff: float = 1.0
    reconnect_max_backoff: float = 60.0
