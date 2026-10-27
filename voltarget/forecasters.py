"""One-day-ahead volatility forecasts.

Convention used everywhere: forecast[t] is the predicted standard deviation of
the return ON day t, built only from returns up to and including day t-1.
That is what makes the backtest honest: on day t-1 at the close you know the
forecast, set your position, and then earn day t's return.
`tests/test_no_lookahead.py` checks this by changing a future return and
confirming no earlier forecast moves.

Four forecasters, from simplest to most structured:
  rolling  - average squared return over the last 21 days (about one month)
  ewma     - exponentially weighted average, RiskMetrics lambda = 0.94
  garch    - GARCH(1,1) with Student-t errors, re-estimated every 21 days
  har      - HAR regression: tomorrow's variance from daily, weekly and
             monthly average variance, re-estimated every 21 days
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

REFIT_EVERY = 21      # re-estimate models about once a month
MIN_TRAIN = 500       # ~2 years of data before the first model fit
VAR_FLOOR = 1e-10     # forecasts never go to zero variance


def rolling_vol(r: pd.Series, window: int = 21) -> pd.Series:
    var = (r**2).rolling(window).mean().shift(1)  # shift(1): use only past days
    return np.sqrt(var).rename("rolling")


def ewma_vol(r: pd.Series, lam: float = 0.94) -> pd.Series:
    x = r.to_numpy()
    var = np.full(len(x), np.nan)
    s2 = x[0] ** 2
    for t in range(1, len(x)):
        var[t] = s2                                  # forecast for day t uses days < t
        s2 = lam * s2 + (1 - lam) * x[t] ** 2        # then update with day t
    return pd.Series(np.sqrt(var), index=r.index, name="ewma")


def garch_vol(r: pd.Series, refit_every: int = REFIT_EVERY, min_train: int = MIN_TRAIN) -> pd.Series:
    """Walk-forward GARCH(1,1).

    Every `refit_every` days we estimate the parameters on all data so far
    (an expanding window). Between refits the parameters stay fixed and the
    variance is rolled forward with the GARCH recursion:
        sigma2[t] = omega + alpha * eps[t-1]^2 + beta * sigma2[t-1]
    Returns are scaled to percent for the optimizer's numerical stability.
    """
    from arch import arch_model

    x = r.to_numpy() * 100.0
    n = len(x)
    var = np.full(n, np.nan)

    for k in range(min_train, n, refit_every):
        model = arch_model(x[:k], mean="Constant", vol="GARCH", p=1, q=1, dist="t")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = model.fit(disp="off")
        mu = res.params["mu"]
        omega, alpha, beta = res.params["omega"], res.params["alpha[1]"], res.params["beta[1]"]

        s2 = float(res.conditional_volatility[-1]) ** 2   # sigma^2 for day k-1 (in-sample)
        eps = x[k - 1] - mu
        for t in range(k, min(k + refit_every, n)):
            s2 = omega + alpha * eps**2 + beta * s2        # forecast for day t
            var[t] = s2
            eps = x[t] - mu                                # day t is now known
    return pd.Series(np.sqrt(var) / 100.0, index=r.index, name="garch")


def _har_features(r2: pd.Series) -> pd.DataFrame:
    """Features for day t, all built from days <= t-1."""
    return pd.DataFrame(
        {
            "rv_d": r2.shift(1),
            "rv_w": r2.rolling(5).mean().shift(1),
            "rv_m": r2.rolling(22).mean().shift(1),
        }
    )


def har_vol(r: pd.Series, refit_every: int = REFIT_EVERY, min_train: int = MIN_TRAIN) -> pd.Series:
    """Walk-forward HAR (heterogeneous autoregressive) model.

    The idea (Corsi, 2009): traders on different horizons react to volatility
    over different windows, so tomorrow's variance is a linear mix of the last
    day, the last week and the last month. With only daily closes, squared
    daily returns stand in for realized variance.
    """
    from sklearn.linear_model import LinearRegression

    r2 = r**2
    X = _har_features(r2)
    y = r2
    var = np.full(len(r), np.nan)

    for k in range(min_train, len(r), refit_every):
        train = X.iloc[:k].join(y.iloc[:k].rename("y")).dropna()   # rows < k only
        model = LinearRegression().fit(train[X.columns], train["y"])
        block = X.iloc[k : k + refit_every]
        ok = block.notna().all(axis=1).to_numpy()
        pred = np.full(len(block), np.nan)
        pred[ok] = model.predict(block[ok])
        var[k : k + len(block)] = np.maximum(pred, VAR_FLOOR)
    return pd.Series(np.sqrt(var), index=r.index, name="har")


FORECASTERS = {"rolling": rolling_vol, "ewma": ewma_vol, "garch": garch_vol, "har": har_vol}


def all_forecasts(r: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({name: fn(r) for name, fn in FORECASTERS.items()})
