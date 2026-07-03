"""Technical indicators, ported from trading-skeleton.py.

Implemented as pure functions on Series (rather than mutating a DataFrame in
place) so Phase 3 can wrap the same math in incremental, streaming versions
and test them against these reference implementations.

Semantics match the skeleton exactly:
- RSI uses simple rolling means of gains/losses (not Wilder smoothing), and
  is NaN where the rolling average loss is zero.
- Realized vol is the rolling std (ddof=1) of log returns, annualized.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(close: pd.Series, window: int = 20) -> pd.Series:
    return close.rolling(window).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def realized_vol(
    close: pd.Series,
    window: int = 20,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> pd.Series:
    log_returns = np.log(close / close.shift(1))
    vol = log_returns.rolling(window).std()
    if annualize:
        vol = vol * np.sqrt(periods_per_year)
    return vol


def add_indicators(
    df: pd.DataFrame,
    price_col: str = "adj_close",
    sma_window: int = 20,
    rsi_window: int = 14,
    vol_window: int = 20,
) -> pd.DataFrame:
    """Return a copy of ``df`` with the standard indicator columns.

    Defaults to adjusted close so indicators are continuous across splits
    and dividends.
    """
    out = df.copy()
    close = out[price_col]
    out[f"sma_{sma_window}"] = sma(close, sma_window)
    out[f"rsi_{rsi_window}"] = rsi(close, rsi_window)
    out[f"realized_vol_{vol_window}"] = realized_vol(close, vol_window)
    out["returns"] = close.pct_change()
    return out
