import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from volpulse.signals.indicators import realized_vol
from volpulse.signals.streaming import IncrementalRealizedVol


def random_closes(n, seed=7):
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))


def stream(closes, **kwargs):
    inc = IncrementalRealizedVol(**kwargs)
    return np.array([np.nan if (v := inc.update(c)) is None else v for c in closes])


def test_matches_batch_reference():
    closes = random_closes(200)
    got = stream(closes, window=20)
    expected = realized_vol(pd.Series(closes), window=20).to_numpy()
    assert_allclose(got, expected, equal_nan=True)


def test_matches_batch_reference_unannualized():
    closes = random_closes(100)
    got = stream(closes, window=10, annualize=False)
    expected = realized_vol(pd.Series(closes), window=10, annualize=False).to_numpy()
    assert_allclose(got, expected, equal_nan=True)


def test_intraday_annualization():
    closes = random_closes(50)
    ppy = 252 * 390
    got = stream(closes, window=20, periods_per_year=ppy)
    expected = realized_vol(pd.Series(closes), window=20, periods_per_year=ppy).to_numpy()
    assert_allclose(got, expected, equal_nan=True)


def test_no_drift_over_long_streams():
    # 10k updates crosses the periodic exact-recompute threshold and gives
    # add/subtract error time to accumulate if it were going to.
    closes = random_closes(10_000)
    got = stream(closes, window=20)
    expected = realized_vol(pd.Series(closes), window=20).to_numpy()
    assert_allclose(got[-500:], expected[-500:], rtol=1e-9)


def test_warmup_returns_none():
    inc = IncrementalRealizedVol(window=5)
    closes = random_closes(10)
    results = [inc.update(c) for c in closes]
    # First update has no prior close; next 4 accumulate returns 1..4.
    assert results[:5] == [None] * 5
    assert all(v is not None for v in results[5:])
    assert inc.value == results[-1]


def test_constant_prices_give_zero_vol():
    inc = IncrementalRealizedVol(window=5)
    results = [inc.update(100.0) for _ in range(10)]
    assert_allclose([v for v in results if v is not None], 0.0, atol=1e-12)


def test_input_validation():
    with pytest.raises(ValueError):
        IncrementalRealizedVol(window=1)
    inc = IncrementalRealizedVol(window=5)
    with pytest.raises(ValueError):
        inc.update(-1.0)
