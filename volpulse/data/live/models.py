"""Value types shared across the live data layer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Tick:
    """A single trade print."""

    symbol: str
    price: float
    size: float
    ts: datetime  # tz-aware UTC


@dataclass
class Bar:
    """An OHLCV bar being built (mutable) or closed (left alone)."""

    symbol: str
    start: datetime  # tz-aware UTC, floored to the bar interval
    open: float
    high: float
    low: float
    close: float
    volume: float
    trades: int = 1

    @classmethod
    def from_tick(cls, tick: Tick, start: datetime) -> Bar:
        return cls(
            symbol=tick.symbol,
            start=start,
            open=tick.price,
            high=tick.price,
            low=tick.price,
            close=tick.price,
            volume=tick.size,
        )

    def update(self, tick: Tick) -> None:
        self.high = max(self.high, tick.price)
        self.low = min(self.low, tick.price)
        self.close = tick.price
        self.volume += tick.size
        self.trades += 1
