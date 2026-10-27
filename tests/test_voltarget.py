import numpy as np
import pandas as pd
import pytest

from voltarget import metrics as M
from voltarget.backtest import BacktestConfig, apply_band, run, target_weights
from voltarget.forecasters import ewma_vol, garch_vol, har_vol, rolling_vol


@pytest.fixture
def returns():
    """Synthetic returns with a calm stretch and a volatile stretch."""
    rng = np.random.default_rng(0)
    r = np.concatenate([rng.normal(0, 0.01, 700), rng.normal(0, 0.03, 300)])
    return pd.Series(r, index=pd.bdate_range("2010-01-01", periods=len(r)))


# ---- the most important test: no forecast may use future data --------------

@pytest.mark.parametrize("fn", [rolling_vol, ewma_vol, garch_vol, har_vol])
def test_no_lookahead(returns, fn):
    """Change one return on day t. Forecasts for days <= t must not move,
    because they are only allowed to use data up to the day before."""
    t = 800
    shocked = returns.copy()
    shocked.iloc[t] = 0.25                       # a huge, made-up move
    before, after = fn(returns), fn(shocked)
    pd.testing.assert_series_equal(before.iloc[: t + 1], after.iloc[: t + 1])
    assert not np.allclose(before.iloc[t + 1 : t + 30].dropna(), after.iloc[t + 1 : t + 30].dropna())


def test_ewma_matches_hand_calculation():
    r = pd.Series([0.01, -0.02, 0.03])
    v = ewma_vol(r, lam=0.9)
    s2_day1 = 0.01**2
    s2_day2 = 0.9 * s2_day1 + 0.1 * 0.02**2
    assert v.iloc[1] == pytest.approx(np.sqrt(s2_day1))
    assert v.iloc[2] == pytest.approx(np.sqrt(s2_day2))


def test_forecasts_track_regime_change(returns):
    v = ewma_vol(returns)
    assert v.iloc[650:700].mean() < v.iloc[900:1000].mean()


# ---- position sizing and costs ---------------------------------------------

def test_weights_scale_inversely_and_respect_cap():
    cfg = BacktestConfig(target_vol=0.15, max_leverage=1.5)
    daily = pd.Series([0.30, 0.15, 0.05]) / np.sqrt(252)   # 30%, 15%, 5% annualized
    w = target_weights(daily, cfg)
    assert w.tolist() == pytest.approx([0.5, 1.0, 1.5])     # last one capped from 3.0


def test_costs_are_charged_on_turnover():
    idx = pd.bdate_range("2020-01-01", periods=3)
    ret = pd.Series([0.0, 0.0, 0.0], index=idx)
    vol = pd.Series([0.15, 0.30, 0.15], index=idx) / np.sqrt(252)   # weight 1.0 -> 0.5 -> 1.0
    out = run(ret, vol, BacktestConfig(cost_bps=10, borrow_rate=0, cash_rate=0))
    assert out["costs"].tolist() == pytest.approx([0.001, 0.0005, 0.0005])


def test_band_skips_small_trades():
    held = apply_band(pd.Series([1.0, 1.04, 0.97, 1.2, 1.18]), band=0.1)
    assert held.tolist() == [1.0, 1.0, 1.0, 1.2, 1.2]


# ---- metrics ----------------------------------------------------------------

def test_max_drawdown_known_path():
    r = pd.Series([0.10, -0.50, 0.20])          # 1.1 -> 0.55 -> 0.66
    assert M.max_drawdown(r) == pytest.approx(-0.5)


def test_bootstrap_of_identical_strategies_is_zero():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0005, 0.01, 1000))
    obs, low, high, _ = M.block_bootstrap_sharpe_diff(r, r, n=200)
    assert obs == low == high == 0
