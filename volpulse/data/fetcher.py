"""Fetch raw OHLCV history from yfinance.

All network access for the historical pipeline lives here, so tests can mock
a single seam (``yfinance.download``).
"""

from __future__ import annotations

import asyncio

import pandas as pd
import yfinance as yf

RAW_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]


class DataUnavailableError(RuntimeError):
    """Raised when the provider returns no data for a request."""


def _normalize_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    # yfinance returns MultiIndex columns for multi-ticker requests and, in
    # recent versions, for single-ticker ones too.
    if isinstance(df.columns, pd.MultiIndex):
        if ticker in df.columns.get_level_values(-1):
            df = df.xs(ticker, axis=1, level=-1)
        else:
            df = df.copy()
            df.columns = df.columns.get_level_values(0)
    return df.rename(columns=lambda c: str(c).strip().lower().replace(" ", "_"))


def fetch_history(
    ticker: str,
    start: str,
    end: str | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch unadjusted OHLCV plus adj_close for one ticker.

    ``auto_adjust=False`` so we keep raw prices and the vendor's adjusted
    close side by side; adjustment factors are derived downstream.
    """
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        actions=False,
        progress=False,
    )
    if df is None or df.empty:
        raise DataUnavailableError(
            f"No data returned for {ticker} (start={start}, end={end}, interval={interval})"
        )

    df = _normalize_columns(df, ticker)
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{ticker}: provider response missing columns {missing}")
    df = df[RAW_COLUMNS].copy()

    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    if interval == "1d":
        idx = idx.normalize()
    df.index = idx
    df.index.name = "date"
    return df


async def fetch_history_async(
    ticker: str,
    start: str,
    end: str | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Async wrapper: yfinance is blocking, so run it in a worker thread."""
    return await asyncio.to_thread(fetch_history, ticker, start, end, interval)
