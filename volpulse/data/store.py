"""Partitioned Parquet storage for historical bars.

Layout (hive-style, one file per ticker — daily bars are small enough that a
single file per partition beats many row-group files):

    <root>/interval=1d/ticker=SPY/data.parquet
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd

STORED_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "adj_open", "adj_high", "adj_low", "adj_close",
]


class ParquetStore:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    def _dir(self, ticker: str, interval: str) -> Path:
        return self.root / f"interval={interval}" / f"ticker={ticker}"

    def _path(self, ticker: str, interval: str) -> Path:
        return self._dir(ticker, interval) / "data.parquet"

    def _empty(self) -> pd.DataFrame:
        return pd.DataFrame(
            columns=STORED_COLUMNS,
            index=pd.DatetimeIndex([], name="date"),
            dtype=float,
        )

    def read(
        self,
        ticker: str,
        interval: str = "1d",
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        path = self._path(ticker, interval)
        if not path.exists():
            return self._empty()
        df = pd.read_parquet(path)
        if start is not None:
            df = df[df.index >= pd.Timestamp(start)]
        if end is not None:
            df = df[df.index <= pd.Timestamp(end)]
        return df

    def write(self, ticker: str, df: pd.DataFrame, interval: str = "1d") -> int:
        """Merge ``df`` into the ticker's partition; newer rows win on
        duplicate dates. Returns the number of net new rows."""
        if df.empty:
            return 0
        missing = [c for c in STORED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{ticker}: frame missing columns {missing}")
        df = df[STORED_COLUMNS]

        existing = self.read(ticker, interval)
        if existing.empty:
            combined = df.sort_index()
        else:
            combined = pd.concat([existing, df])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()

        self._atomic_write(ticker, combined, interval)
        return len(combined) - len(existing)

    def replace(self, ticker: str, df: pd.DataFrame, interval: str = "1d") -> int:
        """Overwrite the ticker's partition entirely (full refresh)."""
        missing = [c for c in STORED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{ticker}: frame missing columns {missing}")
        previous = len(self.read(ticker, interval))
        self._atomic_write(ticker, df[STORED_COLUMNS].sort_index(), interval)
        return len(df) - previous

    def _atomic_write(self, ticker: str, df: pd.DataFrame, interval: str) -> None:
        target = self._path(ticker, interval)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(suffix=".parquet", dir=target.parent)
        os.close(fd)
        try:
            df.to_parquet(tmp)
            os.replace(tmp, target)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def last_date(self, ticker: str, interval: str = "1d") -> pd.Timestamp | None:
        df = self.read(ticker, interval)
        return None if df.empty else df.index.max()

    def tickers(self, interval: str = "1d") -> list[str]:
        base = self.root / f"interval={interval}"
        if not base.exists():
            return []
        return sorted(
            p.name.split("=", 1)[1]
            for p in base.iterdir()
            if p.is_dir() and p.name.startswith("ticker=")
        )
