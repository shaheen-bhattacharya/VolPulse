"""
trading_skeleton.py

A base skeleton for a volatility-based algorithmic trading system (VolPulse).

Pipeline:
    1. Data fetching (yfinance)
    2. Indicator computation (SMA, RSI, realized volatility)
    3. Signal generation
    4. Vectorized backtest with transaction cost modeling
    5. Performance metrics (CAGR, Sharpe, max drawdown)

Note: This is a research/backtesting skeleton. It intentionally avoids
lookahead bias by shifting signals forward one bar before applying returns.
"""

import numpy as np
import pandas as pd
import yfinance as yf


# ---------------------------------------------------------------------------
# 1. Data fetching
# ---------------------------------------------------------------------------

def fetch_data(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a single ticker.
    """
    df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker} between {start} and {end}")
    df = df.rename(columns=str.lower)
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    return df


# ---------------------------------------------------------------------------
# 2. Indicators
# ---------------------------------------------------------------------------

def add_sma(df: pd.DataFrame, window: int = 20, col: str = "close") -> pd.DataFrame:
    df[f"sma_{window}"] = df[col].rolling(window).mean()
    return df


def add_rsi(df: pd.DataFrame, window: int = 14, col: str = "close") -> pd.DataFrame:
    delta = df[col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df[f"rsi_{window}"] = 100 - (100 / (1 + rs))
    return df


def add_realized_vol(df: pd.DataFrame, window: int = 20, col: str = "close",
                      annualize: bool = True, periods_per_year: int = 252) -> pd.DataFrame:
    log_returns = np.log(df[col] / df[col].shift(1))
    vol = log_returns.rolling(window).std()
    if annualize:
        vol = vol * np.sqrt(periods_per_year)
    df[f"realized_vol_{window}"] = vol
    return df


def add_indicators(df: pd.DataFrame, sma_window: int = 20, rsi_window: int = 14,
                    vol_window: int = 20) -> pd.DataFrame:
    df = add_sma(df, sma_window)
    df = add_rsi(df, rsi_window)
    df = add_realized_vol(df, vol_window)
    df["returns"] = df["close"].pct_change()
    return df


# ---------------------------------------------------------------------------
# 3. Signal generation
# ---------------------------------------------------------------------------

def generate_vol_meanreversion_signal(df: pd.DataFrame, vol_col: str,
                                       vol_high_pct: float = 0.8,
                                       vol_low_pct: float = 0.2) -> pd.DataFrame:
    """
    Simple volatility mean-reversion signal:
        - Go long when realized vol is in a low percentile (calm regime, expect continuation)
        - Go flat/short when realized vol spikes into a high percentile (turbulent regime)

    This is a placeholder strategy for skeleton purposes — swap in your
    actual edge once defined.
    """
    vol_series = df[vol_col].dropna()
    high_thresh = vol_series.quantile(vol_high_pct)
    low_thresh = vol_series.quantile(vol_low_pct)

    signal = pd.Series(0, index=df.index)
    signal[df[vol_col] <= low_thresh] = 1   # long when vol is calm
    signal[df[vol_col] >= high_thresh] = 0  # flat when vol spikes

    df["signal"] = signal
    return df


# ---------------------------------------------------------------------------
# 4. Backtest (vectorized, with transaction costs, no lookahead bias)
# ---------------------------------------------------------------------------

def run_backtest(df: pd.DataFrame, signal_col: str = "signal",
                  returns_col: str = "returns",
                  transaction_cost_bps: float = 5.0) -> pd.DataFrame:
    """
    Vectorized backtest.

    IMPORTANT: signal is shifted forward by 1 bar before being applied to
    returns, so that today's signal only affects tomorrow's return
    (avoids lookahead bias — you can't trade on a bar's close using
    information only available at that same close).
    """
    df = df.copy()
    df["position"] = df[signal_col].shift(1).fillna(0)

    # Transaction costs applied whenever position changes
    position_change = df["position"].diff().abs().fillna(0)
    cost = position_change * (transaction_cost_bps / 10_000)

    df["strategy_returns"] = df["position"] * df[returns_col] - cost
    df["cumulative_returns"] = (1 + df["strategy_returns"]).cumprod()
    df["buy_hold_cumulative"] = (1 + df[returns_col]).cumprod()

    return df


# ---------------------------------------------------------------------------
# 5. Performance metrics
# ---------------------------------------------------------------------------

def compute_metrics(df: pd.DataFrame, returns_col: str = "strategy_returns",
                     periods_per_year: int = 252) -> dict:
    returns = df[returns_col].dropna()

    total_return = (1 + returns).prod() - 1
    n_periods = len(returns)
    years = n_periods / periods_per_year
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else np.nan

    ann_vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = (returns.mean() * periods_per_year) / ann_vol if ann_vol != 0 else np.nan

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    return {
        "total_return": total_return,
        "cagr": cagr,
        "annualized_vol": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
    }


# ---------------------------------------------------------------------------
# Main (example run)
# ---------------------------------------------------------------------------

def main():
    ticker = "SPY"
    start = "2019-01-01"
    end = "2024-01-01"

    df = fetch_data(ticker, start, end)
    df = add_indicators(df)
    df = generate_vol_meanreversion_signal(df, vol_col="realized_vol_20")
    df = run_backtest(df)

    metrics = compute_metrics(df)

    print(f"Backtest results for {ticker} ({start} to {end}):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()