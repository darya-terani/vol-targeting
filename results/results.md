# Results

SPY, Feb 23, 2007 to Oct 27, 2026 (4,950 trading days), after costs.

Settings: target volatility 15%, max leverage 1.5×, trading cost 5 bps, cash rate 0%, borrowing rate 3%, Sharpe against a 0% risk-free rate.

## Forecast accuracy

|                      |   QLIKE (lower is better) |   MSE of variance (×1e-7) | vs GARCH: DM t-stat (p)   |
|:---------------------|--------------------------:|--------------------------:|:--------------------------|
| Rolling 21-day       |                   -8.1921 |                    3.1219 | +6.00 (0.000)             |
| EWMA (λ=0.94)        |                   -8.2574 |                    3.0157 | +4.66 (0.000)             |
| GARCH(1,1)           |                   -8.3157 |                    2.8762 | –                         |
| HAR                  |                   -8.2214 |                    3.5083 | +2.57 (0.010)             |
| LSTM                 |                   -7.9958 |                    3.3476 | +2.42 (0.016)             |
| LSTM (EWMA-anchored) |                   -8.3383 |                    2.8219 | -1.70 (0.089)             |

Diebold-Mariano test on daily QLIKE loss differences with Newey-West standard errors. Negative t means the model's loss is lower than GARCH's.

![LSTM in 2008](lstm_2008.png)


## Strategy performance

| Metric | Rolling 21-day | EWMA (λ=0.94) | GARCH(1,1) | HAR | LSTM | LSTM (EWMA-anchored) | Buy & hold |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CAGR | 10.0% | 9.3% | 8.9% | 8.4% | 5.4% | 7.6% | 10.3% |
| Volatility | 15.5% | 15.2% | 14.6% | 14.6% | 17.8% | 14.2% | 19.6% |
| Sharpe | 0.69 | 0.66 | 0.65 | 0.63 | 0.39 | 0.59 | 0.60 |
| Sortino | 0.90 | 0.86 | 0.86 | 0.81 | 0.47 | 0.77 | 0.73 |
| Max drawdown | -38.6% | -36.6% | -38.1% | -37.6% | -65.8% | -38.2% | -56.5% |
| Calmar | 0.26 | 0.25 | 0.23 | 0.22 | 0.08 | 0.20 | 0.18 |
| Worst day | -5.9% | -5.9% | -5.9% | -6.3% | -13.3% | -5.9% | -10.9% |
| Vol of rolling vol | 2.3% | 2.4% | 1.9% | 2.6% | 7.3% | 1.9% | 10.2% |
| Avg weight | 1.09 | 1.07 | 1.05 | 1.01 | 1.13 | 1.06 | – |
| % days leveraged | 59.7% | 58.8% | 55.6% | 53.2% | 67.3% | 56.6% | – |
| Annual turnover | 6.38 | 6.54 | 11.42 | 15.80 | 14.41 | 14.34 | – |

## Crisis windows (total return over each window)

Windows: 2008 financial crisis 2008-09-01 to 2009-03-09; 2020 COVID crash 2020-02-19 to 2020-03-23; 2022 bear market 2022-01-03 to 2022-10-12.

|                       | Rolling 21-day   | EWMA (λ=0.94)   | GARCH(1,1)   | HAR    | LSTM   | LSTM (EWMA-anchored)   | Buy & hold   |
|:----------------------|:-----------------|:----------------|:-------------|:-------|:-------|:-----------------------|:-------------|
| 2008 financial crisis | -22.4%           | -21.4%          | -21.9%       | -19.9% | -52.2% | -21.0%                 | -47.1%       |
| 2020 COVID crash      | -16.6%           | -17.7%          | -14.6%       | -16.3% | -16.2% | -14.2%                 | -33.4%       |
| 2022 bear market      | -18.5%           | -18.8%          | -17.9%       | -17.8% | -18.8% | -18.4%                 | -24.1%       |

## Is the Sharpe ratio gain real?

Circular block bootstrap (21-day blocks, 2,000 resamples) of Sharpe(strategy) − Sharpe(buy & hold).

| Strategy | Sharpe gain | 95% CI | Share of resamples ≤ 0 |
| --- | --- | --- | --- |
| Rolling 21-day | +0.093 | [-0.127, +0.294] | 23% |
| EWMA (λ=0.94) | +0.062 | [-0.151, +0.259] | 31% |
| GARCH(1,1) | +0.056 | [-0.156, +0.267] | 34% |
| HAR | +0.028 | [-0.190, +0.235] | 42% |
| LSTM | -0.212 | [-0.369, -0.054] | 99% |
| LSTM (EWMA-anchored) | -0.009 | [-0.245, +0.213] | 56% |

## Sensitivity: Sharpe ratio by trading cost

|                      |   0 bps |   5 bps |   10 bps |   20 bps |
|:---------------------|--------:|--------:|---------:|---------:|
| Rolling 21-day       |   0.711 |   0.691 |    0.67  |    0.629 |
| EWMA (λ=0.94)        |   0.682 |   0.66  |    0.639 |    0.596 |
| GARCH(1,1)           |   0.692 |   0.653 |    0.614 |    0.536 |
| HAR                  |   0.68  |   0.626 |    0.572 |    0.463 |
| LSTM                 |   0.426 |   0.386 |    0.345 |    0.264 |
| LSTM (EWMA-anchored) |   0.639 |   0.589 |    0.538 |    0.437 |

## Sensitivity: Sharpe ratio by rebalancing band

|                      |   band 0% |   band 5% |   band 10% |   band 20% |
|:---------------------|----------:|----------:|-----------:|-----------:|
| Rolling 21-day       |     0.691 |     0.696 |      0.719 |      0.708 |
| EWMA (λ=0.94)        |     0.66  |     0.662 |      0.682 |      0.681 |
| GARCH(1,1)           |     0.653 |     0.657 |      0.665 |      0.665 |
| HAR                  |     0.626 |     0.631 |      0.624 |      0.626 |
| LSTM                 |     0.386 |     0.393 |      0.412 |      0.429 |
| LSTM (EWMA-anchored) |     0.589 |     0.598 |      0.616 |      0.654 |

![Growth of $1](equity_curve.png)

![Drawdowns](drawdowns.png)

![Rolling volatility](rolling_volatility.png)
