import numpy as np
import pandas as pd
import pytest


def make_raw(
    start: str = "2023-01-02",
    periods: int = 30,
    seed: int = 0,
    base_price: float = 100.0,
    adj_factor: float = 0.98,
) -> pd.DataFrame:
    """Synthetic raw OHLCV frame in the shape fetcher.fetch_history returns."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=periods, name="date")
    close = base_price * np.exp(np.cumsum(rng.normal(0, 0.01, periods)))
    open_ = close * (1 + rng.normal(0, 0.002, periods))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, periods)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, periods)))
    volume = rng.integers(100_000, 1_000_000, periods).astype(float)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close * adj_factor,
            "volume": volume,
        },
        index=idx,
    )


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return make_raw()


@pytest.fixture
def make_raw_frame():
    """Factory fixture so tests can build custom synthetic histories."""
    return make_raw
