import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from volpulse.data.cleaning import apply_adjustments, clean_ohlcv


def test_clean_passes_good_data(raw_df):
    cleaned, report = clean_ohlcv(raw_df, "TEST")
    assert len(cleaned) == len(raw_df)
    assert report.rows_dropped == 0


def test_dedupes_keep_last(raw_df):
    dup = raw_df.iloc[[5]].copy()
    dup["close"] = 999.0
    dup["high"] = 1000.0
    dup["low"] = 998.0
    dup["open"] = 999.0
    dup["adj_close"] = 999.0
    df = pd.concat([raw_df, dup])

    cleaned, report = clean_ohlcv(df, "TEST")
    assert report.duplicates_dropped == 1
    assert cleaned.loc[raw_df.index[5], "close"] == 999.0


def test_drops_bad_prices(raw_df):
    df = raw_df.copy()
    df.iloc[3, df.columns.get_loc("close")] = np.nan
    df.iloc[7, df.columns.get_loc("open")] = -5.0

    cleaned, report = clean_ohlcv(df, "TEST")
    assert report.nan_price_rows_dropped == 1
    assert report.nonpositive_price_rows_dropped == 1
    assert raw_df.index[3] not in cleaned.index
    assert raw_df.index[7] not in cleaned.index


def test_drops_ohlc_violations(raw_df):
    df = raw_df.copy()
    # high well below close: impossible bar
    df.iloc[4, df.columns.get_loc("high")] = df.iloc[4]["close"] * 0.5

    cleaned, report = clean_ohlcv(df, "TEST")
    assert report.ohlc_violation_rows_dropped == 1
    assert raw_df.index[4] not in cleaned.index


def test_volume_handling(raw_df):
    df = raw_df.copy()
    df.iloc[2, df.columns.get_loc("volume")] = -100.0
    df.iloc[6, df.columns.get_loc("volume")] = np.nan

    cleaned, report = clean_ohlcv(df, "TEST")
    assert report.negative_volume_rows_dropped == 1
    assert report.nan_volume_filled == 1
    assert cleaned.loc[raw_df.index[6], "volume"] == 0.0


def test_apply_adjustments_split_math(raw_df):
    # Simulate a 2:1 split after the sample: every historical bar's
    # adjusted price is exactly half its raw price.
    df = raw_df.copy()
    df["adj_close"] = df["close"] * 0.5

    adjusted = apply_adjustments(df)
    for col in ("open", "high", "low"):
        assert_allclose(adjusted[f"adj_{col}"].values, df[col].values * 0.5)
    # Raw columns and volume are untouched.
    assert_allclose(adjusted["close"].values, df["close"].values)
    assert_allclose(adjusted["volume"].values, df["volume"].values)


def test_apply_adjustments_identity_without_actions(raw_df):
    df = raw_df.copy()
    df["adj_close"] = df["close"]

    adjusted = apply_adjustments(df)
    for col in ("open", "high", "low"):
        assert_allclose(adjusted[f"adj_{col}"].values, df[col].values, rtol=1e-12)
