"""Incremental (streaming) indicator implementations.

Each accepts one observation at a time in O(1) and matches the batch
reference implementations in ``indicators.py`` — the tests enforce parity.
Phase 3 adds streaming SMA/RSI here.
"""

from __future__ import annotations

import math
from collections import deque


class IncrementalRealizedVol:
    """Sliding-window realized volatility of log returns, updated in O(1).

    Matches ``indicators.realized_vol``: std with ddof=1 over the last
    ``window`` log returns, annualized by sqrt(periods_per_year). Returns
    None until the window is full (the batch version's NaN warmup).

    Maintains running sum and sum-of-squares; to bound floating-point drift
    from repeated add/subtract, both are recomputed exactly from the stored
    window every ``_RECOMPUTE_EVERY`` updates.
    """

    _RECOMPUTE_EVERY = 4096

    def __init__(self, window: int = 20, annualize: bool = True,
                 periods_per_year: int = 252):
        if window < 2:
            raise ValueError("window must be >= 2")
        self.window = window
        self._annualizer = math.sqrt(periods_per_year) if annualize else 1.0
        self._returns: deque[float] = deque()
        self._sum = 0.0
        self._sumsq = 0.0
        self._last_close: float | None = None
        self._updates = 0

    def update(self, close: float) -> float | None:
        """Fold in a bar close; returns the current vol, or None in warmup."""
        if close <= 0:
            raise ValueError(f"close must be positive, got {close}")
        if self._last_close is None:
            self._last_close = close
            return None

        r = math.log(close / self._last_close)
        self._last_close = close

        if len(self._returns) == self.window:
            old = self._returns.popleft()
            self._sum -= old
            self._sumsq -= old * old
        self._returns.append(r)
        self._sum += r
        self._sumsq += r * r

        self._updates += 1
        if self._updates % self._RECOMPUTE_EVERY == 0:
            self._sum = sum(self._returns)
            self._sumsq = sum(x * x for x in self._returns)

        return self.value

    @property
    def value(self) -> float | None:
        """Current vol over the window, or None if not yet warm."""
        n = len(self._returns)
        if n < self.window:
            return None
        var = (self._sumsq - self._sum * self._sum / n) / (n - 1)
        return math.sqrt(max(var, 0.0)) * self._annualizer
