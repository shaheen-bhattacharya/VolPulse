import asyncio
import json
from datetime import datetime, timezone

import pytest

from volpulse.data.live.client import (
    AlpacaStreamClient,
    AuthenticationError,
    parse_messages,
    parse_timestamp,
)

GREETING = '[{"T":"success","msg":"connected"}]'
AUTH_OK = '[{"T":"success","msg":"authenticated"}]'
AUTH_FAIL = '[{"T":"error","code":402,"msg":"auth failed"}]'
SUBSCRIPTION = '[{"T":"subscription","trades":["SPY"]}]'


def trade_frame(symbol="SPY", price=100.5, size=10, t="2026-07-02T14:30:05.123Z"):
    return json.dumps([{"T": "t", "S": symbol, "p": price, "s": size, "t": t}])


class FakeConnection:
    """Scripted stand-in for a websockets connection: recv()/iteration pop
    from a frame list; sent messages are recorded decoded."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.sent = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def send(self, data):
        self.sent.append(json.loads(data))

    async def recv(self):
        if not self.frames:
            raise OSError("connection closed")
        return self.frames.pop(0)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.frames:
            raise StopAsyncIteration
        return self.frames.pop(0)


def make_client(connections, symbols=("SPY",), **kwargs):
    """Client whose connect factory pops scripted connections in order."""
    pool = list(connections)

    def connect(url):
        assert pool, "client tried to reconnect more times than scripted"
        return pool.pop(0)

    return AlpacaStreamClient(
        "key", "secret", symbols, connect=connect,
        initial_backoff=0.001, max_backoff=0.01, **kwargs,
    )


async def collect(client, n):
    ticks = []
    async for tick in client.stream():
        ticks.append(tick)
        if len(ticks) == n:
            client.stop()
    return ticks


# --- protocol parsing ---------------------------------------------------------

def test_parse_timestamp_millis_z():
    ts = parse_timestamp("2026-07-02T14:30:05.123Z")
    assert ts == datetime(2026, 7, 2, 14, 30, 5, 123000, tzinfo=timezone.utc)


def test_parse_timestamp_nanoseconds():
    ts = parse_timestamp("2026-07-02T14:30:05.123456789Z")
    assert ts.microsecond == 123456


def test_parse_timestamp_no_fraction():
    ts = parse_timestamp("2026-07-02T14:30:05Z")
    assert ts == datetime(2026, 7, 2, 14, 30, 5, tzinfo=timezone.utc)


def test_parse_messages_splits_ticks_and_control():
    frame = json.dumps([
        {"T": "subscription", "trades": ["SPY"]},
        {"T": "t", "S": "SPY", "p": 100.5, "s": 10, "t": "2026-07-02T14:30:05Z"},
    ])
    ticks, control = parse_messages(frame)
    assert len(ticks) == 1
    assert ticks[0].symbol == "SPY"
    assert ticks[0].price == 100.5
    assert ticks[0].size == 10.0
    assert ticks[0].ts.tzinfo is not None
    assert control == [{"T": "subscription", "trades": ["SPY"]}]


# --- client behavior ----------------------------------------------------------

def test_handshake_sends_auth_then_subscribe():
    conn = FakeConnection([GREETING, AUTH_OK, SUBSCRIPTION, trade_frame()])
    client = make_client([conn], symbols=("spy", "aapl"))

    ticks = asyncio.run(collect(client, 1))

    assert conn.sent[0] == {"action": "auth", "key": "key", "secret": "secret"}
    assert conn.sent[1] == {"action": "subscribe", "trades": ["SPY", "AAPL"]}
    assert ticks[0].price == 100.5
    assert client.seconds_since_last_message() is not None


def test_auth_failure_is_fatal_no_retry():
    client = make_client([FakeConnection([GREETING, AUTH_FAIL])])
    with pytest.raises(AuthenticationError):
        asyncio.run(collect(client, 1))


def test_reconnects_after_disconnect():
    conn1 = FakeConnection([GREETING, AUTH_OK, trade_frame(price=100.0)])
    conn2 = FakeConnection([GREETING, AUTH_OK, trade_frame(price=200.0)])
    client = make_client([conn1, conn2])

    ticks = asyncio.run(collect(client, 2))

    assert [t.price for t in ticks] == [100.0, 200.0]
    assert conn2.sent[1]["action"] == "subscribe"  # resubscribed after reconnect


def test_from_env_requires_keys(monkeypatch):
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    with pytest.raises(AuthenticationError, match="APCA_API_KEY_ID"):
        AlpacaStreamClient.from_env(("SPY",))

    monkeypatch.setenv("APCA_API_KEY_ID", "k")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "s")
    client = AlpacaStreamClient.from_env(("SPY",), feed="iex")
    assert client.url.endswith("/v2/iex")
