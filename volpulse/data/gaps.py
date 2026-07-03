"""Detection of missing trading days in a daily price series.

We compare against the business-day calendar rather than an exchange
calendar, so single-day US market holidays show up as 1-day gaps; the ETL
layer only flags runs longer than ``max_gap_bdays`` as suspicious.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Gap:
    start: pd.Timestamp  # first missing business day
    end: pd.Timestamp    # last missing business day
    n_missing: int

    def __str__(self) -> str:
        return f"{self.start.date()}..{self.end.date()} ({self.n_missing} bdays)"


def find_gaps(index: pd.DatetimeIndex) -> list[Gap]:
    """Return every run of consecutive missing business days in ``index``."""
    idx = pd.DatetimeIndex(index).normalize().unique().sort_values()
    if len(idx) < 2:
        return []

    expected = pd.bdate_range(idx[0], idx[-1])
    missing = expected.difference(idx)
    if missing.empty:
        return []

    pos = expected.get_indexer(missing)
    breaks = np.flatnonzero(np.diff(pos) > 1)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(pos) - 1]
    return [Gap(missing[s], missing[e], int(e - s + 1)) for s, e in zip(starts, ends)]


def suspicious_gaps(index: pd.DatetimeIndex, max_gap_bdays: int = 1) -> list[Gap]:
    """Gaps longer than ``max_gap_bdays`` — likely data problems, not holidays."""
    return [g for g in find_gaps(index) if g.n_missing > max_gap_bdays]
