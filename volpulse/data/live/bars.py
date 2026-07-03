"""Aggregate trade ticks into fixed-interval OHLCV bars.

Bars are bucketed by tick timestamp (exchange time), floored to the
interval. A bar closes when a tick arrives in a later bucket or when
``flush(now)`` is called after the bucket's interval has fully elapsed —
so a periodic flusher is needed for symbols that go quiet. Intervals with
no trades produce no bar. Closed bars are kept per symbol in a bounded
ring buffer (deque), oldest evicted first.
"""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone

from .models import Bar, Tick


class BarAggregator:
    def __init__(self, interval_seconds: int = 60, maxlen: int = 500):
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.interval = timedelta(seconds=interval_seconds)
        self._current: dict[str, Bar] = {}
        self._history: dict[str, deque[Bar]] = defaultdict(lambda: deque(maxlen=maxlen))
        self.late_ticks_dropped = 0

    def _bucket(self, ts: datetime) -> datetime:
        seconds = self.interval.total_seconds()
        epoch = ts.timestamp() // seconds * seconds
        return datetime.fromtimestamp(epoch, tz=timezone.utc)

    def add_tick(self, tick: Tick) -> Bar | None:
        """Fold a tick into its bucket. Returns the previous bar if this
        tick rolled the symbol into a new interval, else None."""
        start = self._bucket(tick.ts)
        current = self._current.get(tick.symbol)

        if current is None:
            self._current[tick.symbol] = Bar.from_tick(tick, start)
            return None
        if start == current.start:
            current.update(tick)
            return None
        if start < current.start:
            # Tick for an already-closed bucket: drop rather than rewrite
            # history that downstream consumers have already seen.
            self.late_ticks_dropped += 1
            return None

        self._history[tick.symbol].append(current)
        self._current[tick.symbol] = Bar.from_tick(tick, start)
        return current

    def flush(self, now: datetime) -> list[Bar]:
        """Close every in-progress bar whose interval ended at or before
        ``now``. Call periodically so quiet symbols still emit bars."""
        closed = []
        for symbol, bar in list(self._current.items()):
            if bar.start + self.interval <= now:
                self._history[symbol].append(bar)
                del self._current[symbol]
                closed.append(bar)
        return closed

    def current(self, symbol: str) -> Bar | None:
        return self._current.get(symbol)

    def history(self, symbol: str) -> tuple[Bar, ...]:
        """Closed bars for a symbol, oldest first."""
        return tuple(self._history[symbol])
