"""Volatility-targeted strategy: hold more stock when markets are calm, less
when they are turbulent, aiming for a constant level of risk.

    weight[t] = min(target_vol / forecast_vol[t], max_leverage)

Each day's portfolio return:
    weight * stock return
  + (1 - weight) * cash rate          if weight < 1  (idle money sits in cash)
  - (weight - 1) * borrowing cost     if weight > 1  (leverage is not free)
  - cost_bps * |weight[t] - weight[t-1]|              (trading costs)

Assumptions are deliberately conservative and all live in BacktestConfig, so
the README can state them and a reader can change them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass(frozen=True)
class BacktestConfig:
    target_vol: float = 0.15       # 15% a year, close to the S&P 500's long-run average
    max_leverage: float = 1.5      # never more than 150% invested
    cost_bps: float = 5.0          # 0.05% per unit of weight traded (spread + slippage)
    cash_rate: float = 0.0         # idle cash earns nothing (conservative)
    borrow_rate: float = 0.03      # 3% a year to borrow for leverage
    rebalance_band: float = 0.0    # only trade when the target moves more than this (0 = trade daily)


def target_weights(forecast_vol: pd.Series, cfg: BacktestConfig) -> pd.Series:
    annual_forecast = forecast_vol * np.sqrt(TRADING_DAYS)
    w = (cfg.target_vol / annual_forecast).clip(upper=cfg.max_leverage, lower=0.0)
    return w.rename("weight")


def apply_band(target: pd.Series, band: float) -> pd.Series:
    """No-trade band: keep yesterday's position unless the new target differs
    by more than `band` (e.g. 0.10 = 10 percentage points of exposure). Small
    daily wiggles in the forecast then stop generating trading costs."""
    if band <= 0:
        return target
    held = np.empty(len(target))
    current = target.iloc[0]
    for i, tgt in enumerate(target.to_numpy()):
        if abs(tgt - current) > band:
            current = tgt
        held[i] = current
    return pd.Series(held, index=target.index, name=target.name)


def run(ret: pd.Series, forecast_vol: pd.Series, cfg: BacktestConfig) -> pd.DataFrame:
    """`ret` and `forecast_vol` share an index; forecast_vol[t] uses data up to t-1."""
    df = pd.DataFrame({"ret": ret, "forecast": forecast_vol}).dropna()
    w = apply_band(target_weights(df["forecast"], cfg), cfg.rebalance_band)

    cash_daily = cfg.cash_rate / TRADING_DAYS
    borrow_daily = cfg.borrow_rate / TRADING_DAYS
    financing = np.where(w < 1, (1 - w) * cash_daily, -(w - 1) * borrow_daily)

    turnover = w.diff().abs()
    turnover.iloc[0] = w.iloc[0]                    # the initial purchase is a trade too
    costs = turnover * cfg.cost_bps / 10_000

    out = pd.DataFrame(
        {
            "weight": w,
            "gross": w * df["ret"] + financing,
            "costs": costs,
        }
    )
    out["ret"] = out["gross"] - out["costs"]
    out["turnover"] = turnover
    return out


def buy_and_hold(ret: pd.Series, index: pd.Index) -> pd.DataFrame:
    r = ret.reindex(index)
    return pd.DataFrame({"weight": 1.0, "gross": r, "costs": 0.0, "ret": r, "turnover": 0.0}, index=index)
