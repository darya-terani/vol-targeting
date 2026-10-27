"""Load daily prices and turn them into returns, with sanity checks.

Note on the Stooq file: its prices are adjusted for dividends and splits
(SPY's Feb 2005 close shows as ~$94 here, versus ~$121 actually traded),
so returns computed from it are close to total returns, dividends included.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_prices(path: str | Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").drop_duplicates("Date").set_index("Date")
    close = df["Close"].astype(float)
    if (close <= 0).any():
        raise ValueError("Non-positive prices found")
    if close.isna().any():
        raise ValueError("Missing prices found")
    close.name = "close"
    return close


def log_returns(close: pd.Series) -> pd.Series:
    """Daily log returns. Used for volatility modeling (they add up over time
    and are closer to symmetric than simple returns)."""
    r = np.log(close).diff().dropna()
    r.name = "log_ret"
    return r


def simple_returns(close: pd.Series) -> pd.Series:
    """Daily simple returns. Used for P&L, because a portfolio's return is the
    weighted sum of its assets' simple returns, not of their log returns."""
    r = close.pct_change().dropna()
    r.name = "ret"
    return r
