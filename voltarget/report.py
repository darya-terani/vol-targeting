"""Run everything and write results/: a markdown report plus charts.

    python -m voltarget data/spy_us_d.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd

from . import metrics as M
from .backtest import BacktestConfig, buy_and_hold, run
from .data import load_prices, log_returns, simple_returns
from .forecasters import all_forecasts

LABELS = {
    "rolling": "Rolling 21-day",
    "ewma": "EWMA (λ=0.94)",
    "garch": "GARCH(1,1)",
    "har": "HAR",
    "lstm": "LSTM",
    "lstm_anchored": "LSTM (EWMA-anchored)",
}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # categorical slots 1-3


def chart_2008(forecasts: pd.DataFrame, lr: pd.Series, out: Path) -> None:
    """The plain LSTM's failure in the 2008 crash, next to GARCH and the fix."""
    window = slice("2008-07-01", "2009-06-30")
    realized = (lr.rolling(21).std() * (252**0.5) * 100).loc[window]
    fig, ax = plt.subplots(figsize=(9, 4), facecolor=SURFACE)
    ax.plot(realized.index, realized, color=MUTED, linewidth=1, linestyle="--", label="Realized (past 21 days)")
    series = [("garch", BLUE), ("lstm", ORANGE), ("lstm_anchored", AQUA)]
    for col, color in series:
        f = (forecasts[col] * (252**0.5) * 100).loc[window]
        ax.plot(f.index, f, color=color, linewidth=2, label=LABELS[col])
    _style(ax, "2008: the plain LSTM could not forecast volatility it had never seen",
           "Forecast volatility (% annualized)")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "lstm_2008.png", dpi=160)
    plt.close(fig)


def _style(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)


def _end_label(ax, series: pd.Series, text: str, color: str, dy: float = 0) -> None:
    ax.annotate(text, xy=(series.index[-1], series.iloc[-1]), xytext=(6, dy), textcoords="offset points",
                color=INK, fontsize=8, va="center")
    ax.plot(series.index[-1], series.iloc[-1], "o", color=color, markersize=4)


def charts(bh: pd.Series, strat: pd.Series, name: str, cfg: BacktestConfig, out: Path) -> None:
    pairs = [(bh, "Buy & hold", BLUE), (strat, f"Vol-targeted ({name})", ORANGE)]

    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=SURFACE)
    for r, label, color in pairs:
        eq = M.equity_curve(r)
        ax.plot(eq.index, eq, color=color, linewidth=2, label=label)
        _end_label(ax, eq, f"${eq.iloc[-1]:.2f}", color)
    ax.set_yscale("log")
    ticks = [0.5, 1, 2, 4, 8]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"${t:g}" for t in ticks])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    _style(ax, "Growth of $1 (log scale), after costs", "Portfolio value")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "equity_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.6), facecolor=SURFACE)
    for r, label, color in pairs:
        eq = M.equity_curve(r)
        dd = eq / eq.cummax() - 1
        ax.plot(dd.index, dd * 100, color=color, linewidth=1.5, label=f"{label} (worst {dd.min():.0%})")
    _style(ax, "Drawdown from previous peak", "Drawdown (%)")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "drawdowns.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.6), facecolor=SURFACE)
    for r, label, color in pairs:
        rv = r.rolling(63).std() * (252**0.5) * 100
        ax.plot(rv.index, rv, color=color, linewidth=1.5, label=label)
    ax.axhline(cfg.target_vol * 100, color=MUTED, linewidth=1, linestyle="--")
    ax.annotate(f"Target {cfg.target_vol:.0%}", xy=(bh.index[-1], cfg.target_vol * 100), xytext=(4, 0),
                textcoords="offset points", fontsize=8, color=MUTED, va="center")
    _style(ax, "Realized volatility, rolling 3 months (annualized)", "Volatility (%)")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "rolling_volatility.png", dpi=160)
    plt.close(fig)


def _fmt_table(df: pd.DataFrame, pct_rows: set[str]) -> str:
    cols = list(df.columns)
    lines = ["| Metric | " + " | ".join(cols) + " |", "| --- |" + " --- |" * len(cols)]
    for metric, row in df.iterrows():
        cells = []
        for v in row:
            if pd.isna(v):
                cells.append("–")
            elif metric in pct_rows:
                cells.append(f"{v:.1%}")
            else:
                cells.append(f"{v:.2f}")
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(path: str, out_dir: str = "results") -> None:
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    cfg = BacktestConfig()

    close = load_prices(path)
    lr, sr = log_returns(close), simple_returns(close)
    forecasts = all_forecasts(lr).dropna()           # common window where every model has a forecast
    idx = forecasts.index

    # 1. Forecast accuracy
    r_eval = lr.loc[idx]
    acc_rows = {}
    for c in forecasts:
        row = {"QLIKE (lower is better)": f"{M.qlike(forecasts[c], r_eval):.4f}",
               "MSE of variance (×1e-7)": f"{M.mse_var(forecasts[c], r_eval) * 1e7:.4f}"}
        if c == "garch":
            row["vs GARCH: DM t-stat (p)"] = "–"
        else:
            t, p = M.diebold_mariano(forecasts[c], forecasts["garch"], r_eval)
            row["vs GARCH: DM t-stat (p)"] = f"{t:+.2f} ({p:.3f})"
        acc_rows[LABELS[c]] = row
    acc = pd.DataFrame(acc_rows).T

    # 2. Strategies
    returns, summ = {}, {}
    for c in forecasts:
        bt = run(sr, forecasts[c], cfg)
        returns[LABELS[c]] = bt["ret"]
        summ[LABELS[c]] = M.summary(bt["ret"], bt["weight"], bt["turnover"])
    bh = buy_and_hold(sr, idx)["ret"]
    returns["Buy & hold"] = bh
    summ["Buy & hold"] = M.summary(bh)
    summary_df = pd.DataFrame(summ)

    # 3. Is the Sharpe improvement real?
    boot = {name: M.block_bootstrap_sharpe_diff(returns[name], bh) for name in summ if name != "Buy & hold"}

    # 4. Sensitivity (reported, not tuned): trading costs and rebalancing band
    sens_cost = pd.DataFrame(
        {f"{bps} bps": {LABELS[c]: M.sharpe(run(sr, forecasts[c], BacktestConfig(cost_bps=bps))["ret"]) for c in forecasts}
         for bps in (0, 5, 10, 20)}
    )
    sens_band = pd.DataFrame(
        {f"band {b:.0%}": {LABELS[c]: M.sharpe(run(sr, forecasts[c], BacktestConfig(rebalance_band=b))["ret"]) for c in forecasts}
         for b in (0.0, 0.05, 0.10, 0.20)}
    )

    charts(bh, returns[LABELS["garch"]], "GARCH", cfg, out)
    chart_2008(forecasts, lr, out)

    pct = {"CAGR", "Volatility", "Max drawdown", "Worst day", "Vol of rolling vol", "% days leveraged"}
    crisis = M.crisis_table(returns)
    md = [
        f"# Results\n\nSPY, {idx[0]:%b %d, %Y} to {idx[-1]:%b %d, %Y} ({len(idx):,} trading days), after costs.\n",
        f"Settings: target volatility {cfg.target_vol:.0%}, max leverage {cfg.max_leverage}×, "
        f"trading cost {cfg.cost_bps:g} bps, cash rate {cfg.cash_rate:.0%}, borrowing rate {cfg.borrow_rate:.0%}, "
        "Sharpe against a 0% risk-free rate.\n",
        "## Forecast accuracy\n",
        acc.to_markdown(),
        "\nDiebold-Mariano test on daily QLIKE loss differences with Newey-West standard errors. "
        "Negative t means the model's loss is lower than GARCH's.\n",
        "![LSTM in 2008](lstm_2008.png)\n",
        "\n## Strategy performance\n",
        _fmt_table(summary_df, pct),
        "\n## Crisis windows (total return over each window)\n",
        "Windows: " + "; ".join(f"{k} {a} to {b}" for k, (a, b) in M.CRISES.items()) + ".\n",
        crisis.map(lambda v: f"{v:.1%}").to_markdown(),
        "\n## Is the Sharpe ratio gain real?\n",
        "Circular block bootstrap (21-day blocks, 2,000 resamples) of Sharpe(strategy) − Sharpe(buy & hold).\n",
        "| Strategy | Sharpe gain | 95% CI | Share of resamples ≤ 0 |",
        "| --- | --- | --- | --- |",
        *[f"| {k} | {v[0]:+.3f} | [{v[1]:+.3f}, {v[2]:+.3f}] | {v[3]:.0%} |" for k, v in boot.items()],
        "\n## Sensitivity: Sharpe ratio by trading cost\n",
        sens_cost.round(3).to_markdown(),
        "\n## Sensitivity: Sharpe ratio by rebalancing band\n",
        sens_band.round(3).to_markdown(),
        "\n![Growth of $1](equity_curve.png)\n\n![Drawdowns](drawdowns.png)\n\n![Rolling volatility](rolling_volatility.png)\n",
    ]
    (out / "results.md").write_text("\n".join(md))
    print((out / "results.md").read_text())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/spy_us_d.csv")
