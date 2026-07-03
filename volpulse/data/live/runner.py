"""Wire the live pipeline together: WebSocket ticks → bars → realized vol.

``run_live`` takes an injectable client and an ``on_bar`` callback so tests
(and, in Phase 3, the signal engine) can consume closed bars without
touching the network.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import Callable
from datetime import datetime, timezone

from ...config import LiveConfig
from ...signals.streaming import IncrementalRealizedVol
from .bars import BarAggregator
from .client import AlpacaStreamClient
from .models import Bar

log = logging.getLogger("volpulse.live")

_FLUSH_POLL_SECONDS = 1.0


class LivePipeline:
    """Holds per-symbol aggregation and vol state for one live session."""

    def __init__(self, cfg: LiveConfig):
        self.cfg = cfg
        self.aggregator = BarAggregator(cfg.bar_interval_seconds, cfg.ring_buffer_size)
        self._vols: dict[str, IncrementalRealizedVol] = {}

    def vol(self, symbol: str) -> IncrementalRealizedVol:
        if symbol not in self._vols:
            self._vols[symbol] = IncrementalRealizedVol(
                self.cfg.vol_window, periods_per_year=self.cfg.vol_periods_per_year
            )
        return self._vols[symbol]

    def close_bar(self, bar: Bar) -> float | None:
        return self.vol(bar.symbol).update(bar.close)


def _log_bar(bar: Bar, vol: float | None) -> None:
    log.info(
        "%s %s O=%.2f H=%.2f L=%.2f C=%.2f V=%.0f n=%d vol=%s",
        bar.symbol, bar.start.strftime("%H:%M:%S"),
        bar.open, bar.high, bar.low, bar.close, bar.volume, bar.trades,
        f"{vol:.4f}" if vol is not None else "warming up",
    )


async def run_live(
    cfg: LiveConfig | None = None,
    client: AlpacaStreamClient | None = None,
    on_bar: Callable[[Bar, float | None], None] = _log_bar,
) -> LivePipeline:
    cfg = cfg or LiveConfig()
    client = client or AlpacaStreamClient.from_env(
        cfg.symbols,
        feed=cfg.feed,
        initial_backoff=cfg.reconnect_initial_backoff,
        max_backoff=cfg.reconnect_max_backoff,
    )
    pipeline = LivePipeline(cfg)

    async def flush_and_watchdog() -> None:
        while True:
            await asyncio.sleep(_FLUSH_POLL_SECONDS)
            for bar in pipeline.aggregator.flush(datetime.now(timezone.utc)):
                on_bar(bar, pipeline.close_bar(bar))
            stale = client.seconds_since_last_message()
            if stale is not None and stale > cfg.stale_after_seconds:
                # Phase 3's kill switch consumes this signal; for now, log.
                log.warning("feed stale: no messages for %.0fs", stale)

    flusher = asyncio.create_task(flush_and_watchdog())
    try:
        async for tick in client.stream():
            closed = pipeline.aggregator.add_tick(tick)
            if closed is not None:
                on_bar(closed, pipeline.close_bar(closed))
    finally:
        flusher.cancel()
    return pipeline


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Stream live ticks into OHLCV bars.")
    parser.add_argument("symbols", nargs="*", help="override the configured symbol list")
    parser.add_argument("--feed", default=None, choices=["iex", "sip", "test"],
                        help="data feed ('test' streams fake FAKEPACA trades 24/7)")
    parser.add_argument("--bar-seconds", type=int, default=None, help="bar interval")
    args = parser.parse_args(argv)

    overrides = {}
    if args.symbols:
        overrides["symbols"] = tuple(s.upper() for s in args.symbols)
    if args.feed:
        overrides["feed"] = args.feed
        if args.feed == "test" and not args.symbols:
            overrides["symbols"] = ("FAKEPACA",)
    if args.bar_seconds:
        overrides["bar_interval_seconds"] = args.bar_seconds
    cfg = LiveConfig(**overrides)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        asyncio.run(run_live(cfg))
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
