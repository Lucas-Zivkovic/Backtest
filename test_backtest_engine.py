import os

import matplotlib
matplotlib.use("Agg")  # headless backend so the test runs without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import DataLoader
import Metrics
from BacktestEngine import BacktestEngine
from Strategies.Momentum import MomentumStrategy
from Strategies.MeanReversion import MeanReversion

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


def _synthetic_ohlcv(seed=7, n_periods=300, n_tickers=12):
    """Builds an OHLCV frame shaped like DataLoader.getOHLCV: columns are
    (ticker, field), with field in {Open, High, Low, Close, Volume}."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=n_periods, freq="B")
    tickers = [f"T{i}" for i in range(n_tickers)]
    close = 100 + np.cumsum(rng.normal(0, 1, size=(n_periods, n_tickers)), axis=0)
    volume = rng.integers(1e5, 1e6, size=(n_periods, n_tickers))

    frames = {}
    for j, ticker in enumerate(tickers):
        frames[ticker] = pd.DataFrame(
            {
                "Open": close[:, j],
                "High": close[:, j],
                "Low": close[:, j],
                "Close": close[:, j],
                "Volume": volume[:, j],
            },
            index=dates,
        )
    ohlcv = pd.concat(frames, axis=1).sort_index(axis=1)
    return ohlcv, tickers


def test_backtestEngine_tracks_costs_and_results_at_each_step():
    ohlcv, tickers = _synthetic_ohlcv()
    close = ohlcv.xs("Close", axis=1, level=1)
    warmup = 40
    dates = ohlcv.index

    strategy = MomentumStrategy(skipMonths=0, lookbackMonth=1)
    engine = BacktestEngine(strategy, InMemoryDataSource(ohlcv), transactionCostRate=0.0005)
    results = engine.run(dates[warmup], dates[-1])

    n_steps = len(ohlcv) - warmup
    step_dates = dates[warmup:]
    assert len(results) == n_steps
    assert len(engine.costs) == n_steps
    assert len(engine.positions) == n_steps
    assert len(engine.signal) == n_steps

    # costs are never negative, and the first step pays to enter from flat
    assert all(c >= 0 for c in engine.costs)
    assert engine.costs[0] > 0

    # MomentumStrategy's default positions() is dollar-neutral at every step
    for weights in engine.positions:
        assert np.isclose(weights.sum(), 0.0, atol=1e-8)

    # each step's result = (today's weights, decided from data before today . today's return) - today's cost
    period_returns = close.pct_change()
    for i, date in enumerate(step_dates):
        expected_gross = float((engine.positions[i] * period_returns.loc[date]).sum())
        assert np.isclose(results[i], expected_gross - engine.costs[i])

    _plot_backtest(step_dates, results, engine.costs, "BacktestEngine — synthetic prices", "backtestEngine.png")


def test_backtestEngine_on_real_data():
    ohlcv = DataLoader.getOHLCV("2015-01-01", "2023-01-01", REAL_TICKERS, "1d").dropna()
    close = ohlcv.xs("Close", axis=1, level=1)
    dates = ohlcv.index
    warmup = 280  # skip (1mo) + lookback (12mo) = 273 trading days needed before the 12-1 signal is populated

    strategy = MomentumStrategy(skipMonths=1, lookbackMonth=12)
    engine = BacktestEngine(strategy, InMemoryDataSource(ohlcv), transactionCostRate=0.0005)
    results = engine.run(dates[warmup], dates[-1])

    n_steps = len(close) - warmup
    step_dates = dates[warmup:]
    assert len(results) == n_steps
    assert len(engine.costs) == n_steps
    assert all(c >= 0 for c in engine.costs)

    _plot_backtest(
        step_dates, results, engine.costs,
        f"BacktestEngine — 12-1 momentum on {len(REAL_TICKERS)} large-caps",
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
        expected_gross = float((engine.positions[i] * period_returns.loc[date]).sum())
        assert np.isclose(results[i], expected_gross - engine.costs[i])

    _plot_backtest(
        step_dates, results, engine.costs,
        f"BacktestEngine — OU mean reversion on {len(tickers)} large-caps",
        "backtestEngine_meanReversion_realData.png",
    )


def _plot_backtest(dates, results, costs, title, filename):
    results = pd.Series(results, index=dates)
    costs = pd.Series(costs, index=dates)
    cumulative = (1 + results).cumprod()

    metrics = {
        "Sharpe": Metrics.sharpRatio(results),
        "Sortino": Metrics.sortinoRatio(results, targetReturn=0.0),
        "Max drawdown": Metrics.maxDrawdown(cumulative),
        "Drawdown duration (days)": Metrics.drawdownDuration(cumulative),
        "Hit ratio": Metrics.hitRatio(results),
        "Total return": cumulative.iloc[-1] - 1,
        "Total cost drag": costs.sum(),
    }

    os.makedirs(PLOT_DIR, exist_ok=True)
    fig, (ax_cum, ax_ret, ax_cost) = plt.subplots(
        3, 1, figsize=(12, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]}
    )

    ax_cum.plot(cumulative.index, cumulative.values, color="tab:blue")
    ax_cum.set_title(title)
    ax_cum.set_ylabel("Growth of $1")
    ax_cum.text(
        0.01, 0.98,
        "\n".join(f"{k}: {v:.4f}" for k, v in metrics.items()),
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
    test_backtestEngine_tracks_costs_and_results_at_each_step()
    test_backtestEngine_on_real_data()
    test_backtestEngine_on_real_data_meanReversion()
    print("ok")