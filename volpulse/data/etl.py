"""Async ETL orchestrator: fetch → clean → adjust → store, per ticker.

Incremental by default: each run refetches from the last stored date
(inclusive, so a revised final bar is overwritten on merge). If the
refetched overlap bar's adj_close no longer matches what we stored, a
corporate action (split/dividend) has changed historical adjustment factors,
so the ticker's full history is refetched and the partition replaced.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass, field

from ..config import DataConfig
from .cleaning import CleaningReport, apply_adjustments, clean_ohlcv
from .fetcher import DataUnavailableError, fetch_history_async
from .gaps import Gap, suspicious_gaps
from .store import ParquetStore

log = logging.getLogger("volpulse.etl")

# Relative adj_close drift on the overlap bar that triggers a full refresh.
_ADJ_DRIFT_TOL = 1e-6


@dataclass
class ETLResult:
    ticker: str
    rows_added: int = 0
    total_rows: int = 0
    full_refresh: bool = False
    cleaning: CleaningReport | None = None
    gaps: list[Gap] = field(default_factory=list)
    error: str | None = None


def _prepare(raw, ticker: str):
    cleaned, report = clean_ohlcv(raw, ticker)
    return apply_adjustments(cleaned), report


async def update_ticker(
    cfg: DataConfig,
    store: ParquetStore,
    ticker: str,
    sem: asyncio.Semaphore,
) -> ETLResult:
    result = ETLResult(ticker=ticker)
    last = store.last_date(ticker, cfg.interval)
    fetch_start = last.strftime("%Y-%m-%d") if last is not None else cfg.start

    async with sem:
        try:
            raw = await fetch_history_async(ticker, start=fetch_start, interval=cfg.interval)
        except DataUnavailableError:
            if last is None:
                raise
            # Nothing new since the last stored bar (weekend/holiday run).
            raw = None

        if raw is not None and last is not None:
            adjusted, _ = _prepare(raw, ticker)
            stored_last = store.read(ticker, cfg.interval).loc[last, "adj_close"]
            if last in adjusted.index:
                refetched = adjusted.loc[last, "adj_close"]
                drift = abs(refetched / stored_last - 1)
                if drift > _ADJ_DRIFT_TOL:
                    log.info(
                        "%s: adj_close drift %.2e on %s — corporate action detected, "
                        "refetching full history",
                        ticker, drift, last.date(),
                    )
                    raw = await fetch_history_async(ticker, start=cfg.start, interval=cfg.interval)
                    result.full_refresh = True

    if raw is not None:
        adjusted, result.cleaning = _prepare(raw, ticker)
        if result.full_refresh:
            result.rows_added = store.replace(ticker, adjusted, cfg.interval)
        else:
            result.rows_added = store.write(ticker, adjusted, cfg.interval)

    stored = store.read(ticker, cfg.interval)
    result.total_rows = len(stored)
    result.gaps = suspicious_gaps(stored.index, cfg.max_gap_bdays)
    return result


async def run_etl(cfg: DataConfig | None = None) -> list[ETLResult]:
    cfg = cfg or DataConfig()
    store = ParquetStore(cfg.data_root)
    sem = asyncio.Semaphore(cfg.max_concurrent_fetches)

    async def _safe(ticker: str) -> ETLResult:
        try:
            return await update_ticker(cfg, store, ticker, sem)
        except Exception as exc:  # keep one bad ticker from killing the run
            log.error("%s: ETL failed: %s", ticker, exc)
            return ETLResult(ticker=ticker, error=str(exc))

    results = await asyncio.gather(*(_safe(t) for t in cfg.tickers))

    for r in results:
        if r.error:
            log.error("%s: FAILED — %s", r.ticker, r.error)
            continue
        if r.cleaning and r.cleaning.rows_dropped:
            log.warning("%s: cleaning dropped %d rows (%s)",
                        r.ticker, r.cleaning.rows_dropped, r.cleaning)
        for gap in r.gaps:
            log.warning("%s: suspicious gap %s", r.ticker, gap)
        log.info("%s: +%d rows (total %d)%s", r.ticker, r.rows_added, r.total_rows,
                 " [full refresh]" if r.full_refresh else "")
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the VolPulse historical ETL.")
    parser.add_argument("tickers", nargs="*", help="override the configured ticker list")
    parser.add_argument("--start", default=None, help="backfill start date (YYYY-MM-DD)")
    args = parser.parse_args(argv)

    overrides = {}
    if args.tickers:
        overrides["tickers"] = tuple(t.upper() for t in args.tickers)
    if args.start:
        overrides["start"] = args.start
    cfg = DataConfig(**overrides)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    results = asyncio.run(run_etl(cfg))

    failed = [r for r in results if r.error]
    print(f"\nETL complete: {len(results) - len(failed)}/{len(results)} tickers OK, "
          f"{sum(r.rows_added for r in results)} rows added, "
          f"store at {cfg.data_root}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
