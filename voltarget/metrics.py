"""Performance metrics, forecast-accuracy losses, and a block bootstrap for
judging whether a Sharpe ratio difference is real or luck."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

CRISES = {
    "2008 financial crisis": ("2008-09-01", "2009-03-09"),
    "2020 COVID crash": ("2020-02-19", "2020-03-23"),
    "2022 bear market": ("2022-01-03", "2022-10-12"),
}


def equity_curve(ret: pd.Series) -> pd.Series:
    return (1 + ret).cumprod()


def max_drawdown(ret: pd.Series) -> float:
    eq = equity_curve(ret)
    return float((eq / eq.cummax() - 1).min())


def sharpe(ret: pd.Series) -> float:
    """Annualized, against a 0% risk-free rate (stated in the README)."""
    sd = ret.std()
    return float(ret.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")


def summary(ret: pd.Series, weights: pd.Series | None = None, turnover: pd.Series | None = None) -> dict:
    years = len(ret) / TRADING_DAYS
    eq = equity_curve(ret)
    cagr = eq.iloc[-1] ** (1 / years) - 1
    vol = ret.std() * np.sqrt(TRADING_DAYS)
    downside = ret[ret < 0].std() * np.sqrt(TRADING_DAYS)
    mdd = max_drawdown(ret)
    rolling_vol = ret.rolling(63).std() * np.sqrt(TRADING_DAYS)
    out = {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe(ret),
        "Sortino": ret.mean() * TRADING_DAYS / downside if downside > 0 else np.nan,
        "Max drawdown": mdd,
        "Calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "Worst day": ret.min(),
        # How steady is risk? Spread of 3-month realized vol across time.
        "Vol of rolling vol": rolling_vol.std(),
    }
    if weights is not None:
        out["Avg weight"] = weights.mean()
        out["% days leveraged"] = (weights > 1).mean()
    if turnover is not None:
        out["Annual turnover"] = turnover.sum() / years
    return out


def crisis_table(strategies: dict[str, pd.Series]) -> pd.DataFrame:
    rows = {}
    for label, (start, end) in CRISES.items():
        rows[label] = {
            name: ((1 + r.loc[start:end]).prod() - 1) for name, r in strategies.items()
        }
    return pd.DataFrame(rows).T


# ---- forecast accuracy -----------------------------------------------------

def qlike(forecast_vol: pd.Series, ret: pd.Series) -> float:
    """QLIKE loss: mean of log(sigma^2) + r^2 / sigma^2. Lower is better.

    Squared returns are a very noisy stand-in for true variance. Patton (2011)
    shows QLIKE and MSE still rank forecasters correctly under such noise,
    and QLIKE is less dominated by a few extreme days than MSE.
    """
    df = pd.DataFrame({"s2": forecast_vol**2, "r2": ret**2}).dropna()
    return float((np.log(df["s2"]) + df["r2"] / df["s2"]).mean())


def mse_var(forecast_vol: pd.Series, ret: pd.Series) -> float:
    df = pd.DataFrame({"s2": forecast_vol**2, "r2": ret**2}).dropna()
    return float(((df["r2"] - df["s2"]) ** 2).mean())


# ---- uncertainty ------------------------------------------------------------

def block_bootstrap_sharpe_diff(
    a: pd.Series, b: pd.Series, block: int = 21, n: int = 2000, seed: int = 0
) -> tuple[float, float, float, float]:
    """Circular block bootstrap of Sharpe(a) - Sharpe(b).

    Daily returns are not independent: volatility clusters, so calm and wild
    days come in runs. Resampling single days would break those runs and make
    the interval too narrow. Instead we resample whole ~1-month blocks, and
    resample the two strategies on the SAME days so their correlation is kept.

    Returns (observed difference, CI low, CI high, share of resamples <= 0).
    """
    df = pd.DataFrame({"a": a, "b": b}).dropna().to_numpy()
    T = len(df)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(T / block))
    starts = rng.integers(0, T, size=(n, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n, -1)[:, :T] % T
    sample = df[idx]                                     # shape (n, T, 2)
    sr = sample.mean(axis=1) / sample.std(axis=1, ddof=1) * np.sqrt(TRADING_DAYS)
    diffs = sr[:, 0] - sr[:, 1]
    observed = sharpe(pd.Series(df[:, 0])) - sharpe(pd.Series(df[:, 1]))
    low, high = np.percentile(diffs, [2.5, 97.5])
    return observed, float(low), float(high), float((diffs <= 0).mean())
