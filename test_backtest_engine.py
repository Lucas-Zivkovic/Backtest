import os

import matplotlib
matplotlib.use("Agg")  # headless backend so the test runs without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import DataLoader
import Metrics
from BacktestEngine import BacktestEngine
from Strategies.CrossSectionalMomentumStrategy import CrossSectionalMomentumStrategy
from Strategies.MeanReversion import MeanReversion
from Strategies.PortfolioCombiner.Uniform import Uniform
from Strategies.PortfolioCombiner.InvVol import InvVol
from Strategies.PortfolioCombiner.RiskParity import RiskParity

REAL_TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM", "XOM", "JNJ", "PG", "V", "UNH"]

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")


class InMemoryDataSource:
    """Feeds BacktestEngine a growing window of price history, one date at a time."""

    def __init__(self, prices: pd.DataFrame):
        self.prices = prices

    def iterate(self, start_date, end_date):
        mask = (self.prices.index >= start_date) & (self.prices.index <= end_date)
        for i in np.where(mask)[0]:
            yield self.prices.index[i], self.prices.iloc[: i + 1]


def test_backtestEngine_on_real_data():
    ohlcv = DataLoader.getOHLCV("2015-01-01", "2023-01-01", REAL_TICKERS, "1d").dropna()
    close = ohlcv.xs("Close", axis=1, level=1)
    dates = ohlcv.index
    warmup = 280  # skip (1mo) + lookback (12mo) = 273 trading days needed before the 12-1 signal is populated

    strategy = CrossSectionalMomentumStrategy(skipMonths=1, lookbackMonth=12)
    engine = BacktestEngine(strategy, InMemoryDataSource(ohlcv), transactionCostRate=0.0005)
    results = engine.run(dates[warmup], dates[-1])

    n_steps = len(close) - warmup
    step_dates = dates[warmup:]
    assert len(results) == n_steps
    assert len(engine.costs) == n_steps
    assert all(c >= 0 for c in engine.costs)

    _plot_backtest(
        step_dates, results, engine.costs,engine.metrics,
        f"BacktestEngine — cross-sectional 12-1 momentum on {len(REAL_TICKERS)} large-caps",
        "backtestEngine_realData.png",
    )


def test_backtestEngine_on_real_data_meanReversion():
    # Same real-data setup as test_backtestEngine_on_real_data (reuses the
    # cached download), but with MeanReversion. Its signal() recalibrates an
    # OU process per ticker per step, which is expensive, so we keep the
    # universe and the number of steps small enough to run quickly.
    tickers = REAL_TICKERS
    ohlcv = DataLoader.getOHLCV("2015-01-01", "2023-01-01", REAL_TICKERS, "1d").dropna()
    close = ohlcv.xs("Close", axis=1, level=1)
    dates = ohlcv.index

    window = 120
    warmup = len(dates)-200 # only backtest the last 80 days: OU calibration is per-step, per-ticker

    strategy = MeanReversion(window=window)
    engine = BacktestEngine(strategy, InMemoryDataSource(ohlcv), transactionCostRate=0.0005)
    results = engine.run(dates[warmup], dates[-1])

    n_steps = len(close) - warmup
    step_dates = dates[warmup:]
    assert len(results) == n_steps
    assert len(engine.costs) == n_steps
    assert len(engine.positions) == n_steps
    assert all(c >= 0 for c in engine.costs)

    period_returns = close.pct_change()
    for i, date in enumerate(step_dates):
        expected_gross = float((engine.positions.iloc[i] * period_returns.loc[date]).sum())
        assert np.isclose(results.iloc[i], expected_gross - engine.costs.iloc[i])

    _plot_backtest(
        step_dates, results, engine.costs,engine.metrics,
        f"BacktestEngine — OU mean reversion on {len(tickers)} large-caps",
        "backtestEngine_meanReversion_realData.png",
    )


def test_backtestEngine_on_real_data_uniformCombiner():
    # Uniform 1/n blend of cross-sectional 12-1 momentum and OU mean reversion.
    # MeanReversion recalibrates per step, so like its own test we only backtest
    # the last 200 days; that start is well past momentum's 273-day warmup.
    ohlcv = DataLoader.getOHLCV("2015-01-01", "2023-01-01", REAL_TICKERS, "1d").dropna()
    close = ohlcv.xs("Close", axis=1, level=1)
    dates = ohlcv.index
    warmup = len(dates) - 100

    strategy = Uniform([
        CrossSectionalMomentumStrategy(skipMonths=1, lookbackMonth=12),
        MeanReversion(window=120),
    ])
    engine = BacktestEngine(strategy, InMemoryDataSource(ohlcv), transactionCostRate=0.0005)
    results = engine.run(dates[warmup], dates[-1])

    n_steps = len(close) - warmup
    step_dates = dates[warmup:]
    assert len(results) == n_steps
    assert len(engine.costs) == n_steps
    assert len(engine.positions) == n_steps
    assert all(c >= 0 for c in engine.costs)

    # each sub-book has gross <= 1 and the blend is not renormalized, so a
    # flat sub-strategy leaves its 1/2 in cash instead of levering the other
    gross = engine.positions.abs().sum(axis=1)
    assert (gross <= 1.0 + 1e-9).all()

    period_returns = close.pct_change()
    for i, date in enumerate(step_dates):
        expected_gross = float((engine.positions.iloc[i] * period_returns.loc[date]).sum())
        assert np.isclose(results.iloc[i], expected_gross - engine.costs.iloc[i])

    _plot_backtest(
        step_dates, results, engine.costs, engine.metrics,
        f"BacktestEngine — uniform momentum + mean reversion on {len(REAL_TICKERS)} large-caps",
        "backtestEngine_uniform_realData.png",
    )


def test_backtestEngine_on_real_data_invVolCombiner():
    strategy = _backtest_calibrated_combiner(InvVol, "inverse-vol", "backtestEngine_invVol_realData.png")

    vol = strategy.strategyMetrics.loc["activeVolatility"]
    traded = vol > 0
    weights = strategy.weights()
    # inverse-vol: weight * vol is the same constant for every strategy that traded
    assert np.allclose(weights[traded] * vol[traded], (weights[traded] * vol[traded]).iloc[0])


def test_backtestEngine_on_real_data_riskParityCombiner():
    strategy = _backtest_calibrated_combiner(RiskParity, "risk parity", "backtestEngine_riskParity_realData.png")

    # every strategy that traded carries the same share of the combined risk
    shares = strategy.riskContributions[strategy.weights() > 0]
    assert np.allclose(shares, 1 / len(shares), atol=1e-4)


def test_riskParity_accounts_for_correlation():
    # A and B are near-duplicates (correlation ~0.96), C is independent with
    # the same vol, D never trades. InvVol ignores correlation and gives A, B,
    # C a third each, i.e. two thirds of the risk to the A/B bet. RiskParity
    # cuts A and B so that each of the three carries a third of the risk.
    rng = np.random.default_rng(0)
    n = 2000
    common = rng.normal(0, 0.01, n)
    returns = pd.DataFrame({
        "A": common + rng.normal(0, 0.002, n),
        "B": common + rng.normal(0, 0.002, n),
        "C": rng.normal(0, 0.0102, n),
        "D": np.zeros(n),
    })

    combiner = RiskParity([], "2020-01-01", "2020-12-31")
    weights = combiner.computeWeights(returns, None)

    assert np.isclose(weights.sum(), 1.0)
    assert weights["D"] == 0.0
    assert weights["C"] > weights["A"] + 0.05 and weights["C"] > weights["B"] + 0.05
    assert np.allclose(combiner.riskContributions[["A", "B", "C"]], 1 / 3, atol=1e-4)


def _backtest_calibrated_combiner(combiner_cls, label, filename):
    # Blend of cross-sectional 12-1 momentum and OU mean reversion. Weights
    # come from backtesting each sub-strategy alone on a calibration window
    # that ends strictly before the evaluated backtest starts, so the
    # combination never sees the returns it is scored on. Both windows are
    # 100 days because MeanReversion recalibrates per step.
    ohlcv = DataLoader.getOHLCV("2015-01-01", "2023-01-01", REAL_TICKERS, "1d").dropna()
    close = ohlcv.xs("Close", axis=1, level=1)
    dates = ohlcv.index
    calib_start, calib_end = dates[-200], dates[-101]
    warmup = len(dates) - 100

    data_source = InMemoryDataSource(ohlcv)
    strategy = combiner_cls(
        [CrossSectionalMomentumStrategy(skipMonths=1, lookbackMonth=12), MeanReversion(window=120)],
        calib_start, calib_end, transactionCostRate=0.0005,
    )
    strategyMetrics = strategy.calibrate(data_source)

    assert list(strategyMetrics.columns) == ["CrossSectionalMomentumStrategy", "MeanReversion"]
    assert len(strategy.strategyReturns) == 100
    assert np.isclose(strategy.weights().sum(), 1.0)
    assert (strategy.weights() >= 0).all()

    # trading inside the calibration window is refused
    with pytest.raises(ValueError, match="Look-ahead"):
        BacktestEngine(strategy, data_source).run(calib_end, calib_end)

    engine = BacktestEngine(strategy, data_source, transactionCostRate=0.0005)
    results = engine.run(dates[warmup], dates[-1])

    n_steps = len(close) - warmup
    step_dates = dates[warmup:]
    assert step_dates[0] > calib_end
    assert len(results) == n_steps
    assert len(engine.positions) == n_steps
    assert all(c >= 0 for c in engine.costs)

    gross = engine.positions.abs().sum(axis=1)
    assert (gross <= 1.0 + 1e-9).all()

    period_returns = close.pct_change()
    for i, date in enumerate(step_dates):
        expected_gross = float((engine.positions.iloc[i] * period_returns.loc[date]).sum())
        assert np.isclose(results.iloc[i], expected_gross - engine.costs.iloc[i])

    _plot_backtest(
        step_dates, results, engine.costs, engine.metrics,
        f"BacktestEngine — {label} momentum + mean reversion on {len(REAL_TICKERS)} large-caps",
        filename,
    )
    return strategy

def _plot_backtest(dates, results, costs,metrics:pd.DataFrame, title, filename):
    results = pd.Series(results, index=dates)
    costs = pd.Series(costs, index=dates)
    cumulative = (1 + results).cumprod()

    os.makedirs(PLOT_DIR, exist_ok=True)
    fig, (ax_cum, ax_ret, ax_cost) = plt.subplots(
        3, 1, figsize=(12, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]}
    )

    ax_cum.plot(cumulative.index, cumulative.values, color="tab:blue")
    ax_cum.set_title(title)
    ax_cum.set_ylabel("Growth of $1")
    ax_cum.text(
        0.01, 0.98,
        "\n".join(f"{k}: {v:.4f}" for k, v in metrics["value"].items()),
        transform=ax_cum.transAxes, va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax_ret.bar(results.index, results.values, color="tab:green", width=1.0)
    ax_ret.set_ylabel("Step result")

    ax_cost.bar(costs.index, costs.values, color="tab:red", width=1.0)
    ax_cost.set_ylabel("Step cost")
    ax_cost.set_xlabel("Date")

    fig.tight_layout()
    out_path = os.path.join(PLOT_DIR, filename)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    assert os.path.exists(out_path)


if __name__ == "__main__":
    test_backtestEngine_on_real_data()
    test_backtestEngine_on_real_data_meanReversion()
    test_backtestEngine_on_real_data_uniformCombiner()
    test_backtestEngine_on_real_data_invVolCombiner()
    test_backtestEngine_on_real_data_riskParityCombiner()
    test_riskParity_accounts_for_correlation()
    print("ok")