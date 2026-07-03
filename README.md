# VolPulse

Volatility-based algorithmic trading system, built in phases:

1. **Data pipeline** (done) — historical ETL into partitioned Parquet
2. **Async live data** — Alpaca WebSocket ticks → rolling OHLCV bars, incremental realized vol
3. **Signal + risk** — streaming SMA/RSI/vol signals, position limits, kill switch
4. **Paper execution** — Alpaca paper API limit orders, SQLite trade log

## Layout

```
trading-skeleton.py     original research/backtest skeleton (unchanged)
volpulse/
  config.py             tickers, dates, paths, gap/concurrency settings
  data/
    fetcher.py          yfinance fetch (the only network seam)
    cleaning.py         validation + split/dividend adjustment
    gaps.py             missing-trading-day detection
    store.py            partitioned Parquet store (interval=/ticker=)
    etl.py              async orchestrator (incremental, corp-action aware)
  signals/indicators.py SMA, RSI, realized vol (reference implementations)
  risk/                 (Phase 3)
  execution/            (Phase 4)
tests/                  pytest suite; all network calls mocked
data_store/             Parquet output (generated, safe to delete)
```

## Usage

```bash
uv pip install -p .venv/bin/python -e ".[dev]"

.venv/bin/python -m volpulse              # ETL for configured tickers
.venv/bin/python -m volpulse SPY QQQ      # override ticker list
.venv/bin/python -m pytest                # run tests (no network)
```

Repeat runs are incremental: each refetches from the last stored bar and
merges. If the overlap bar's `adj_close` has drifted (new split/dividend
changed historical adjustment factors), the ticker's full history is
automatically refetched.

## Data model

Each partition stores raw OHLCV plus derived adjusted columns
(`adj_open/high/low/close`, factor = vendor `adj_close` / raw `close`).
Indicators default to `adj_close` so they're continuous across corporate
actions. Volume stays raw (the combined factor includes dividends, so it's
not a valid share-count adjustment).
