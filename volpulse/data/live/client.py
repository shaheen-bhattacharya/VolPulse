"""Async WebSocket client for Alpaca's market data stream (v2, JSON).

Protocol (https://docs.alpaca.markets/docs/real-time-stock-pricing-data):
    server: [{"T":"success","msg":"connected"}]
    client: {"action":"auth","key":...,"secret":...}
    server: [{"T":"success","msg":"authenticated"}]
    client: {"action":"subscribe","trades":[...]}
    server: [{"T":"subscription",...}], then trade messages {"T":"t",...}

The connection factory is injectable so tests can drive the full protocol
with a scripted fake instead of a network socket.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from datetime import datetime, timezone

import websockets

from .models import Tick

log = logging.getLogger("volpulse.live.client")

BASE_URL = "wss://stream.data.alpaca.markets/v2/{feed}"
KEY_ENV = "APCA_API_KEY_ID"
SECRET_ENV = "APCA_API_SECRET_KEY"

# Server error codes that mean retrying is pointless.
_FATAL_CODES = {400, 401, 402, 404, 408, 409}
_HANDSHAKE_TIMEOUT = 10.0


class AuthenticationError(RuntimeError):
    """Bad/missing credentials or subscription — do not retry."""


class StreamProtocolError(RuntimeError):
    """Unexpected server response — connection will be retried."""


def parse_timestamp(raw: str) -> datetime:
    """Parse an RFC-3339 timestamp; Alpaca sends up to nanosecond precision,
    which ``datetime.fromisoformat`` cannot digest, so trim to microseconds."""
    if raw.endswith(("Z", "z")):
        raw = raw[:-1] + "+00:00"
    if "." in raw:
        head, rest = raw.split(".", 1)
        digits = 0
        while digits < len(rest) and rest[digits].isdigit():
            digits += 1
        frac, suffix = rest[:digits], rest[digits:]
        raw = f"{head}.{frac[:6]}{suffix}"
    ts = datetime.fromisoformat(raw)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def parse_messages(raw: str | bytes) -> tuple[list[Tick], list[dict]]:
    """Split a frame into trade ticks and control messages."""
    payload = json.loads(raw)
    ticks: list[Tick] = []
    control: list[dict] = []
    for msg in payload:
        if msg.get("T") == "t":
            ticks.append(
                Tick(
                    symbol=msg["S"],
                    price=float(msg["p"]),
                    size=float(msg.get("s", 0)),
                    ts=parse_timestamp(msg["t"]),
                )
            )
        else:
            control.append(msg)
    return ticks, control


class AlpacaStreamClient:
    def __init__(
        self,
        key_id: str,
        secret_key: str,
        symbols: tuple[str, ...],
        feed: str = "iex",
        url: str | None = None,
        connect=websockets.connect,
        initial_backoff: float = 1.0,
        max_backoff: float = 60.0,
    ):
        self._key_id = key_id
        self._secret_key = secret_key
        self.symbols = tuple(s.upper() for s in symbols)
        self.url = url or BASE_URL.format(feed=feed)
        self._connect = connect
        self._initial_backoff = initial_backoff
        self._max_backoff = max_backoff
        self._stopped = False
        self._last_message_monotonic: float | None = None

    @classmethod
    def from_env(cls, symbols: tuple[str, ...], feed: str = "iex", **kwargs) -> AlpacaStreamClient:
        key = os.environ.get(KEY_ENV)
        secret = os.environ.get(SECRET_ENV)
        if not key or not secret:
            raise AuthenticationError(
                f"Set {KEY_ENV} and {SECRET_ENV} (Alpaca paper account keys work: "
                "https://app.alpaca.markets — API Keys)"
            )
        return cls(key, secret, symbols, feed=feed, **kwargs)

    def seconds_since_last_message(self) -> float | None:
        """Feed staleness signal for the Phase 3 kill switch."""
        if self._last_message_monotonic is None:
            return None
        return time.monotonic() - self._last_message_monotonic

    def stop(self) -> None:
        self._stopped = True

    async def stream(self) -> AsyncIterator[Tick]:
        """Yield ticks forever, reconnecting with exponential backoff.

        Raises AuthenticationError immediately on fatal auth problems; all
        other connection failures are retried.
        """
        backoff = self._initial_backoff
        while not self._stopped:
            try:
                async with self._connect(self.url) as ws:
                    await self._handshake(ws)
                    log.info("connected to %s, subscribed to %s", self.url, list(self.symbols))
                    backoff = self._initial_backoff
                    async for raw in ws:
                        self._last_message_monotonic = time.monotonic()
                        ticks, control = parse_messages(raw)
                        self._handle_control(control)
                        for tick in ticks:
                            yield tick
                        if self._stopped:
                            return
            except AuthenticationError:
                raise
            except (OSError, asyncio.TimeoutError, websockets.WebSocketException,
                    StreamProtocolError) as exc:
                log.warning("stream disconnected (%s); reconnecting in %.1fs", exc, backoff)
            else:
                log.warning("stream closed by server; reconnecting in %.1fs", backoff)
            if self._stopped:
                return
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self._max_backoff)

    async def _handshake(self, ws) -> None:
        await self._expect_success(ws, "connected")
        await ws.send(json.dumps(
            {"action": "auth", "key": self._key_id, "secret": self._secret_key}
        ))
        await self._expect_success(ws, "authenticated")
        await ws.send(json.dumps(
            {"action": "subscribe", "trades": list(self.symbols)}
        ))

    async def _expect_success(self, ws, expected: str) -> None:
        raw = await asyncio.wait_for(ws.recv(), timeout=_HANDSHAKE_TIMEOUT)
        for msg in json.loads(raw):
            if msg.get("T") == "error":
                code = msg.get("code")
                if code in _FATAL_CODES:
                    raise AuthenticationError(f"server rejected connection: {msg}")
                raise StreamProtocolError(f"server error during handshake: {msg}")
            if msg.get("T") == "success" and msg.get("msg") == expected:
                return
        raise StreamProtocolError(f"expected success '{expected}', got {raw!r}")

    def _handle_control(self, control: list[dict]) -> None:
        for msg in control:
            if msg.get("T") == "error":
                if msg.get("code") in _FATAL_CODES:
                    raise AuthenticationError(f"server error: {msg}")
                log.warning("server error: %s", msg)
            elif msg.get("T") == "subscription":
                log.debug("subscription state: %s", msg)
