from datetime import datetime, timezone

from numpy.testing import assert_allclose

from volpulse.data.live.bars import BarAggregator
from volpulse.data.live.models import Tick


def make_tick(symbol="SPY", price=100.0, size=10.0, t="2026-07-02T14:30:05") -> Tick:
    ts = datetime.fromisoformat(t).replace(tzinfo=timezone.utc)
    return Tick(symbol=symbol, price=price, size=size, ts=ts)


def test_ticks_in_one_interval_build_single_bar():
    agg = BarAggregator(interval_seconds=60)
    prices = [100.0, 102.0, 99.0, 101.0]
    sizes = [10.0, 5.0, 20.0, 7.0]
    for i, (p, s) in enumerate(zip(prices, sizes)):
        closed = agg.add_tick(make_tick(price=p, size=s, t=f"2026-07-02T14:30:{10 + i:02d}"))
        assert closed is None

    bar = agg.current("SPY")
    assert_allclose([bar.open, bar.high, bar.low, bar.close], [100.0, 102.0, 99.0, 101.0])
    assert_allclose(bar.volume, sum(sizes))
    assert bar.trades == 4
    assert bar.start == datetime(2026, 7, 2, 14, 30, tzinfo=timezone.utc)


def test_new_interval_closes_previous_bar():
    agg = BarAggregator(interval_seconds=60)
    agg.add_tick(make_tick(price=100.0, t="2026-07-02T14:30:59"))
    closed = agg.add_tick(make_tick(price=105.0, t="2026-07-02T14:31:00"))

    assert closed is not None
    assert closed.close == 100.0
    assert closed.start == datetime(2026, 7, 2, 14, 30, tzinfo=timezone.utc)
    assert agg.current("SPY").open == 105.0
    assert agg.history("SPY") == (closed,)


def test_symbols_are_independent():
    agg = BarAggregator(interval_seconds=60)
    agg.add_tick(make_tick("SPY", price=100.0, t="2026-07-02T14:30:10"))
    agg.add_tick(make_tick("AAPL", price=200.0, t="2026-07-02T14:30:20"))

    # SPY rolls to the next minute; AAPL must stay untouched.
    closed = agg.add_tick(make_tick("SPY", price=101.0, t="2026-07-02T14:31:10"))
    assert closed.symbol == "SPY"
    assert agg.current("AAPL").close == 200.0
    assert agg.history("AAPL") == ()


def test_late_tick_is_dropped_not_rewritten():
    agg = BarAggregator(interval_seconds=60)
    agg.add_tick(make_tick(price=100.0, t="2026-07-02T14:30:10"))
    agg.add_tick(make_tick(price=101.0, t="2026-07-02T14:31:10"))
    closed_bar = agg.history("SPY")[0]

    late = agg.add_tick(make_tick(price=999.0, t="2026-07-02T14:30:30"))
    assert late is None
    assert agg.late_ticks_dropped == 1
    assert agg.history("SPY")[0] is closed_bar
    assert agg.current("SPY").close == 101.0


def test_ring_buffer_evicts_oldest():
    agg = BarAggregator(interval_seconds=60, maxlen=3)
    for minute in range(5):
        agg.add_tick(make_tick(price=100.0 + minute, t=f"2026-07-02T14:{30 + minute:02d}:01"))

    history = agg.history("SPY")
    assert len(history) == 3  # bars for minutes 31..33; 30 evicted, 34 in progress
    assert [b.open for b in history] == [101.0, 102.0, 103.0]


def test_flush_closes_only_elapsed_bars():
    agg = BarAggregator(interval_seconds=60)
    agg.add_tick(make_tick("SPY", price=100.0, t="2026-07-02T14:30:10"))
    agg.add_tick(make_tick("AAPL", price=200.0, t="2026-07-02T14:31:10"))

    # At 14:30:59 SPY's bar interval hasn't elapsed yet.
    assert agg.flush(datetime(2026, 7, 2, 14, 30, 59, tzinfo=timezone.utc)) == []

    closed = agg.flush(datetime(2026, 7, 2, 14, 31, 0, tzinfo=timezone.utc))
    assert [b.symbol for b in closed] == ["SPY"]
    assert agg.current("SPY") is None
    assert agg.current("AAPL") is not None  # 14:31 bucket still open


def test_quiet_gap_produces_no_empty_bars():
    agg = BarAggregator(interval_seconds=60)
    agg.add_tick(make_tick(price=100.0, t="2026-07-02T14:30:10"))
    closed = agg.add_tick(make_tick(price=101.0, t="2026-07-02T14:35:10"))  # 4 quiet minutes

    assert closed.start == datetime(2026, 7, 2, 14, 30, tzinfo=timezone.utc)
    assert len(agg.history("SPY")) == 1  # no synthetic bars for 14:31..14:34
