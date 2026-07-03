import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from volpulse.data.cleaning import apply_adjustments
from volpulse.data.store import STORED_COLUMNS, ParquetStore

from conftest import make_raw


def stored_frame(**kwargs) -> pd.DataFrame:
    return apply_adjustments(make_raw(**kwargs))[STORED_COLUMNS]


@pytest.fixture
def store(tmp_path) -> ParquetStore:
    return ParquetStore(tmp_path / "data_store")


def test_read_missing_ticker_returns_empty(store):
    df = store.read("NOPE")
    assert df.empty
    assert list(df.columns) == STORED_COLUMNS


def test_write_read_roundtrip(store):
    df = stored_frame()
    added = store.write("SPY", df)
    assert added == len(df)

    back = store.read("SPY")
    assert_frame_equal(back, df.sort_index(), check_freq=False)


def test_incremental_write_dedupes_keep_last(store):
    full = stored_frame(periods=30)
    store.write("SPY", full.iloc[:20])

    # Second write overlaps on the last 5 stored rows with revised values.
    revised = full.iloc[15:].copy()
    revised.loc[revised.index[0], "close"] = 123.0
    added = store.write("SPY", revised)

    assert added == 10  # only genuinely new dates count
    back = store.read("SPY")
    assert len(back) == 30
    assert back.loc[revised.index[0], "close"] == 123.0  # newer row won


def test_replace_overwrites_partition(store):
    store.write("SPY", stored_frame(periods=30))
    shorter = stored_frame(periods=10, seed=1)
    store.replace("SPY", shorter)

    back = store.read("SPY")
    assert_frame_equal(back, shorter.sort_index(), check_freq=False)


def test_read_date_slicing(store):
    df = stored_frame(periods=20)
    store.write("SPY", df)

    sliced = store.read("SPY", start=df.index[5], end=df.index[10])
    assert sliced.index[0] == df.index[5]
    assert sliced.index[-1] == df.index[10]
    assert len(sliced) == 6


def test_last_date(store):
    assert store.last_date("SPY") is None
    df = stored_frame()
    store.write("SPY", df)
    assert store.last_date("SPY") == df.index.max()


def test_tickers_listing(store):
    assert store.tickers() == []
    store.write("SPY", stored_frame())
    store.write("AAPL", stored_frame(seed=2))
    assert store.tickers() == ["AAPL", "SPY"]


def test_write_missing_columns_raises(store):
    with pytest.raises(ValueError, match="missing columns"):
        store.write("SPY", make_raw())  # raw frame lacks adj_open etc.


def test_partition_layout(store, tmp_path):
    store.write("SPY", stored_frame())
    expected = tmp_path / "data_store" / "interval=1d" / "ticker=SPY" / "data.parquet"
    assert expected.exists()
