"""Validation, cleaning, and split/dividend adjustment of raw OHLCV data."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close"]

# Relative tolerance for OHLC consistency checks; vendors round to cents so
# exact inequalities can fail by float noise.
_REL_TOL = 1e-6


@dataclass(frozen=True)
class CleaningReport:
    ticker: str
    rows_in: int
    rows_out: int
    duplicates_dropped: int = 0
    nan_price_rows_dropped: int = 0
    nonpositive_price_rows_dropped: int = 0
    ohlc_violation_rows_dropped: int = 0
    negative_volume_rows_dropped: int = 0
    nan_volume_filled: int = 0

    @property
    def rows_dropped(self) -> int:
        return self.rows_in - self.rows_out


def clean_ohlcv(df: pd.DataFrame, ticker: str = "") -> tuple[pd.DataFrame, CleaningReport]:
    """Sort, dedupe, and drop rows that fail basic sanity checks.

    Returns the cleaned frame and a report of what was removed, so the ETL
    layer can log data-quality issues instead of silently swallowing them.
    """
    rows_in = len(df)

    # Dedupe before sorting: "last" means last in arrival order (the most
    # recent fetch wins), and sort_index is not stable across equal keys.
    deduped = df[~df.index.duplicated(keep="last")]
    duplicates = len(df) - len(deduped)
    df = deduped.sort_index()

    has_prices = df[PRICE_COLUMNS].notna().all(axis=1)
    nan_price = int((~has_prices).sum())
    df = df[has_prices]

    positive = (df[PRICE_COLUMNS] > 0).all(axis=1)
    nonpositive = int((~positive).sum())
    df = df[positive]

    body_high = df[["open", "close"]].max(axis=1)
    body_low = df[["open", "close"]].min(axis=1)
    consistent = (
        (df["high"] >= df["low"])
        & (df["high"] >= body_high * (1 - _REL_TOL))
        & (df["low"] <= body_low * (1 + _REL_TOL))
    )
    violations = int((~consistent).sum())
    df = df[consistent]

    nonneg_volume = df["volume"].isna() | (df["volume"] >= 0)
    negative_volume = int((~nonneg_volume).sum())
    df = df[nonneg_volume]

    nan_volume = int(df["volume"].isna().sum())
    if nan_volume:
        df = df.copy()
        df["volume"] = df["volume"].fillna(0)

    report = CleaningReport(
        ticker=ticker,
        rows_in=rows_in,
        rows_out=len(df),
        duplicates_dropped=duplicates,
        nan_price_rows_dropped=nan_price,
        nonpositive_price_rows_dropped=nonpositive,
        ohlc_violation_rows_dropped=violations,
        negative_volume_rows_dropped=negative_volume,
        nan_volume_filled=nan_volume,
    )
    return df, report


def apply_adjustments(df: pd.DataFrame) -> pd.DataFrame:
    """Derive split/dividend-adjusted OHLC from the vendor's adj_close.

    factor = adj_close / close reflects both splits and dividends; applying
    it to open/high/low gives a consistent adjusted bar. Volume is kept raw:
    the combined factor includes dividends, so it is not a valid share-count
    adjustment.
    """
    out = df.copy()
    factor = out["adj_close"] / out["close"]
    for col in ("open", "high", "low"):
        out[f"adj_{col}"] = out[col] * factor
    return out
