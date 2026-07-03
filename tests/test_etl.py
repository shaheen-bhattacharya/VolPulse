import asyncio

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from volpulse.config import DataConfig
from volpulse.data import etl
from volpulse.data.fetcher import DataUnavailableError
from volpulse.data.store import ParquetStore

from conftest import make_raw


def install_fake_fetch(monkeypatch, histories: dict[str, pd.DataFrame]):
    """Replace the network fetch with a slice of a canned per-ticker history,
    mimicking yfinance's start-date filtering."""
    calls = []

    async def fake(ticker, start, end=None, interval="1d"):
        calls.append((ticker, start))
        if ticker not in histories:
            raise ValueError(f"unknown ticker {ticker}")
        df = histories[ticker]
        df = df[df.index >= pd.Timestamp(start)]
        if df.empty:
            raise DataUnavailableError(ticker)
        return df.copy()

    monkeypatch.setattr(etl, "fetch_history_async", fake)
    return calls


def run(cfg):
    return asyncio.run(etl.run_etl(cfg))


def make_cfg(tmp_path, tickers=("TEST",)) -> DataConfig:
    return DataConfig(tickers=tickers, start="2023-01-01",
                      data_root=tmp_path / "store", max_concurrent_fetches=2)


def test_initial_backfill(tmp_path, monkeypatch):
    full = make_raw(periods=30)
    install_fake_fetch(monkeypatch, {"TEST": full})
    cfg = make_cfg(tmp_path)

    (result,) = run(cfg)

    assert result.error is None
    assert result.rows_added == 30
    assert result.total_rows == 30
    stored = ParquetStore(cfg.data_root).read("TEST")
    assert_allclose(stored["close"].values, full["close"].values)
    # Adjustment columns were derived on the way in.
    assert_allclose(stored["adj_open"].values,
                    (full["open"] * full["adj_close"] / full["close"]).values)


def test_incremental_update_fetches_from_last_date(tmp_path, monkeypatch):
    full = make_raw(periods=30)
    cfg = make_cfg(tmp_path)

    install_fake_fetch(monkeypatch, {"TEST": full.iloc[:20]})
    run(cfg)

    calls = install_fake_fetch(monkeypatch, {"TEST": full})
    (result,) = run(cfg)

    # Refetch starts at the last stored date (overlap of one bar).
    assert calls[0] == ("TEST", full.index[19].strftime("%Y-%m-%d"))
    assert result.rows_added == 10
    assert result.total_rows == 30
    assert not result.full_refresh


def test_no_new_data_is_not_an_error(tmp_path, monkeypatch):
    full = make_raw(periods=10)
    cfg = make_cfg(tmp_path)

    install_fake_fetch(monkeypatch, {"TEST": full})
    run(cfg)

    # Second run: nothing on/after the last stored date -> provider empty.
    install_fake_fetch(monkeypatch, {"TEST": full.iloc[:0]})
    (result,) = run(cfg)

    assert result.error is None
    assert result.rows_added == 0
    assert result.total_rows == 10


def test_corporate_action_triggers_full_refresh(tmp_path, monkeypatch):
    full = make_raw(periods=30, adj_factor=1.0)
    cfg = make_cfg(tmp_path)

    install_fake_fetch(monkeypatch, {"TEST": full.iloc[:20]})
    run(cfg)

    # A dividend changed every historical adjustment factor by 10%.
    revised = full.copy()
    revised["adj_close"] = full["adj_close"] * 0.9
    calls = install_fake_fetch(monkeypatch, {"TEST": revised})
    (result,) = run(cfg)

    assert result.full_refresh
    # Second call refetched the entire configured history.
    assert calls[-1] == ("TEST", cfg.start)
    stored = ParquetStore(cfg.data_root).read("TEST")
    assert len(stored) == 30
    assert_allclose(stored["adj_close"].values, revised["adj_close"].values)


def test_one_bad_ticker_does_not_kill_run(tmp_path, monkeypatch):
    full = make_raw(periods=10)
    install_fake_fetch(monkeypatch, {"GOOD": full})  # "BAD" raises ValueError
    cfg = make_cfg(tmp_path, tickers=("GOOD", "BAD"))

    results = {r.ticker: r for r in run(cfg)}

    assert results["GOOD"].error is None
    assert results["GOOD"].rows_added == 10
    assert results["BAD"].error is not None
    assert results["BAD"].rows_added == 0


def test_gap_reporting(tmp_path, monkeypatch):
    full = make_raw(periods=20)
    holey = full.drop(full.index[5:10])  # one missing trading week
    install_fake_fetch(monkeypatch, {"TEST": holey})
    cfg = make_cfg(tmp_path)

    (result,) = run(cfg)

    assert len(result.gaps) == 1
    assert result.gaps[0].n_missing == 5
