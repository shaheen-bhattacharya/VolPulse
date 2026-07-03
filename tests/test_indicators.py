import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from volpulse.signals.indicators import add_indicators, realized_vol, rsi, sma


def random_walk(n=100, seed=42) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))))


# --- independent reference implementations (plain loops, no pandas) ---------

def manual_sma(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    for i in range(window - 1, len(values)):
        out[i] = values[i - window + 1 : i + 1].mean()
    return out


def manual_rsi(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    delta = np.diff(values)  # delta[k] belongs to bar k+1
    for i in range(window, len(values)):
        d = delta[i - window : i]
        avg_gain = d[d > 0].sum() / window
        avg_loss = -d[d < 0].sum() / window
        if avg_loss == 0:
            continue  # skeleton semantics: NaN when no losses in window
        out[i] = 100 - 100 / (1 + avg_gain / avg_loss)
    return out


def manual_realized_vol(values: np.ndarray, window: int,
                        periods_per_year: int = 252) -> np.ndarray:
    out = np.full(len(values), np.nan)
    log_ret = np.log(values[1:] / values[:-1])  # log_ret[k] belongs to bar k+1
    for i in range(window, len(values)):
        out[i] = log_ret[i - window : i].std(ddof=1) * np.sqrt(periods_per_year)
    return out


# --- tests -------------------------------------------------------------------

def test_sma_matches_manual():
    s = random_walk()
    assert_allclose(sma(s, 20).values, manual_sma(s.values, 20))


def test_rsi_matches_manual():
    s = random_walk()
    assert_allclose(rsi(s, 14).values, manual_rsi(s.values, 14))


def test_rsi_alternating_series_is_50():
    # Equal-magnitude alternating gains/losses over an even window -> RSI 50.
    s = pd.Series([100.0, 101.0] * 20)
    result = rsi(s, 14).dropna()
    assert_allclose(result.values, np.full(len(result), 50.0))


def test_rsi_nan_when_no_losses():
    s = pd.Series(np.linspace(100, 200, 50))  # monotonic up: avg_loss == 0
    assert rsi(s, 14).isna().all()


def test_realized_vol_matches_manual():
    s = random_walk()
    assert_allclose(realized_vol(s, 20).values, manual_realized_vol(s.values, 20))


def test_realized_vol_unannualized():
    s = random_walk()
    ann = realized_vol(s, 20, annualize=True)
    raw = realized_vol(s, 20, annualize=False)
    assert_allclose(ann.values, raw.values * np.sqrt(252))


def test_realized_vol_of_constant_series_is_zero():
    s = pd.Series(np.full(50, 100.0))
    assert_allclose(realized_vol(s, 20).dropna().values, 0.0, atol=1e-12)


def test_add_indicators_columns_and_values():
    n = 60
    s = random_walk(n)
    df = pd.DataFrame({"adj_close": s.values},
                      index=pd.bdate_range("2023-01-02", periods=n))

    out = add_indicators(df)

    for col in ("sma_20", "rsi_14", "realized_vol_20", "returns"):
        assert col in out.columns
    assert_allclose(out["sma_20"].values, manual_sma(s.values, 20))
    assert_allclose(out["rsi_14"].values, manual_rsi(s.values, 14))
    assert_allclose(out["realized_vol_20"].values, manual_realized_vol(s.values, 20))
    # Input frame is not mutated.
    assert "sma_20" not in df.columns
