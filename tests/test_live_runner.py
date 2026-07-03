import asyncio
from datetime import datetime, timezone

from numpy.testing import assert_allclose

from volpulse.config import LiveConfig
from volpulse.data.live.models import Tick
from volpulse.data.live.runner import run_live


class FakeClient:
    """Replays canned ticks through run_live without a network."""

    def __init__(self, ticks):
        self._ticks = ticks

    async def stream(self):
        for tick in self._ticks:
            yield tick

    def seconds_since_last_message(self):
        return None


def make_tick(price, t, symbol="SPY", size=10.0):
    ts = datetime.fromisoformat(t).replace(tzinfo=timezone.utc)
    return Tick(symbol=symbol, price=price, size=size, ts=ts)


def test_run_live_pipes_ticks_to_bars_and_vol():
    ticks = [
        make_tick(100.0, "2026-07-02T14:30:05"),
        make_tick(102.0, "2026-07-02T14:30:30"),
        make_tick(101.0, "2026-07-02T14:30:55"),
        make_tick(103.0, "2026-07-02T14:31:05"),  # rolls the 14:30 bar closed
        make_tick(104.0, "2026-07-02T14:32:05"),  # rolls the 14:31 bar closed
    ]
    cfg = LiveConfig(symbols=("SPY",), vol_window=2)
    closed = []

    pipeline = asyncio.run(run_live(
        cfg,
        client=FakeClient(ticks),
        on_bar=lambda bar, vol: closed.append((bar, vol)),
    ))

    assert len(closed) == 2
    first_bar, first_vol = closed[0]
    assert_allclose(
        [first_bar.open, first_bar.high, first_bar.low, first_bar.close, first_bar.volume],
        [100.0, 102.0, 100.0, 101.0, 30.0],
    )
    assert first_vol is None  # one close is not enough for a 2-return window

    # Ring buffer holds the closed bars; the 14:32 bar is still in progress.
    assert [b.close for b in pipeline.aggregator.history("SPY")] == [101.0, 103.0]
    assert pipeline.aggregator.current("SPY").close == 104.0
