import asyncio

import pandas as pd
import pytest

from volpulse.data import fetcher
from volpulse.data.fetcher import (
    RAW_COLUMNS,
    DataUnavailableError,
    fetch_history,
    fetch_history_async,
)

YF_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def yf_style_frame(ticker: str = "SPY", periods: int = 5, multi: bool = True,
                   tz: str | None = "America/New_York") -> pd.DataFrame:
    idx = pd.date_range("2023-01-02", periods=periods, freq="B", tz=tz)
    data = {c: [float(i + 1) for i in range(periods)] for c in YF_COLUMNS}
    df = pd.DataFrame(data, index=idx)
    if multi:
        df.columns = pd.MultiIndex.from_product([YF_COLUMNS, [ticker]],
                                                 names=["Price", "Ticker"])
    return df


def install_download(monkeypatch, frame):
    calls = []

    def fake_download(ticker, **kwargs):
        calls.append((ticker, kwargs))
        return frame

    monkeypatch.setattr(fetcher.yf, "download", fake_download)
    return calls


def test_normalizes_multiindex_and_tz(monkeypatch):
    calls = install_download(monkeypatch, yf_style_frame(multi=True))
    df = fetch_history("SPY", start="2023-01-01")

    assert list(df.columns) == RAW_COLUMNS
    assert df.index.tz is None
    assert (df.index == df.index.normalize()).all()
    assert df.index.name == "date"
    # No live-network kwargs drift: adjustment must stay off.
    assert calls[0][1]["auto_adjust"] is False


def test_flat_columns(monkeypatch):
    install_download(monkeypatch, yf_style_frame(multi=False, tz=None))
    df = fetch_history("SPY", start="2023-01-01")
    assert list(df.columns) == RAW_COLUMNS
    assert len(df) == 5


def test_empty_response_raises(monkeypatch):
    install_download(monkeypatch, pd.DataFrame())
    with pytest.raises(DataUnavailableError):
        fetch_history("SPY", start="2023-01-01")


def test_missing_column_raises(monkeypatch):
    frame = yf_style_frame(multi=False, tz=None).drop(columns=["Volume"])
    install_download(monkeypatch, frame)
    with pytest.raises(ValueError, match="volume"):
        fetch_history("SPY", start="2023-01-01")


def test_async_wrapper(monkeypatch):
    install_download(monkeypatch, yf_style_frame())
    df = asyncio.run(fetch_history_async("SPY", start="2023-01-01"))
    assert list(df.columns) == RAW_COLUMNS
