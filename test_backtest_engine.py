import os

import matplotlib
matplotlib.use("Agg")  # headless backend so the test runs without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import DataLoader
from BacktestEngine import BacktestEngine
from Strategies.CrossSectionalMomentumStrategy import CrossSectionalMomentumStrategy
from Strategies.MeanReversion import MeanReversion
from Strategies.PairFinding import PairFinding, findPairs, findParameters, johansen
from Strategies.PairTrading import PairTrading
from Strategies.VolatilityTarget import VolatilityTarget
from Strategies.ZScorePositions import ZScorePositions
from Strategies.PortfolioCombiner.Uniform import Uniform
from Strategies.PortfolioCombiner.InvVol import InvVol
from Strategies.PortfolioCombiner.RiskParity import RiskParity

REAL_TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM", "XOM", "JNJ", "PG", "V", "UNH"]

# Wider universe for pair finding: several names per sector so DBSCAN has
# groups to find. All listed before 2013, so dropna() keeps the full period.
PAIR_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD", "INTC", "ORCL", "CSCO",  # tech
    "JPM", "BAC", "C", "WFC", "GS", "MS",                                    # banks
    "XOM", "CVX", "COP", "SLB", "EOG",                                       # energy
    "JNJ", "PFE", "MRK", "ABT", "LLY",                                       # pharma
    "PG", "KO", "PEP", "CL", "WMT",                                          # staples
    "DUK", "SO", "NEE", "AEP", "D",                                          # utilities
]

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
    warmup = 200 # only backtest the last 80 days: OU calibration is per-step, per-ticker

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


def test_meanReversion_window_search():
    # Backtests MeanReversion for several OU calibration windows over the same
    # evaluation period and picks the window with the best Sharpe ratio. The
    # choice is in-sample: the best window is selected on the returns it is
    # scored on. OU calibration is per-step, per-ticker, so the period is kept
    # short; it must start after the largest window has enough history.
    ohlcv = DataLoader.getOHLCV("2015-01-01", "2023-01-01", REAL_TICKERS, "1d").dropna()
    dates = ohlcv.index
    data_source = InMemoryDataSource(ohlcv)

    windows = [30, 60, 90, 120, 180, 252]
    warmup = len(dates) - 200
    assert warmup > max(windows)
    step_dates = dates[warmup:]

    results, metrics = {}, {}
    for window in windows:
        engine = BacktestEngine(MeanReversion(window=window), data_source, transactionCostRate=0.0005)
        results[window] = engine.run(dates[warmup], dates[-1])
        metrics[window] = engine.metrics["value"]
        assert len(results[window]) == len(step_dates)
        assert all(c >= 0 for c in engine.costs)

    metrics = pd.DataFrame(metrics)  # rows: metric, columns: window
    best = metrics.loc["sharpeRatio"].idxmax()
    print(f"\n{metrics.to_string()}\nbest window (Sharpe): {best}")

    _plot_window_search(
        step_dates, results, metrics, best,
        f"MeanReversion window search on {len(REAL_TICKERS)} large-caps (best by Sharpe: {best})",
        "meanReversion_windowSearch.png",
    )


def test_pairFinding_on_real_data():
    # Clusters PAIR_TICKERS with PairFinding (correlation distance -> PCA ->
    # DBSCAN) over 2013-2025, then findPairs tests every pair inside a cluster
    # with the Johansen trace test on log prices, over the whole period and
    # each half. Correlation is measured on daily returns: prices trend, so
    # their correlation is spurious.
    ohlcv = DataLoader.getOHLCV("2013-01-01", "2025-01-01", PAIR_TICKERS, "1d").dropna()
    close = ohlcv.xs("Close", axis=1, level=1)[PAIR_TICKERS]
    returns = close.pct_change().dropna()

    clusters = PairFinding(returns)

    assert list(clusters.index) == PAIR_TICKERS
    assert (clusters >= -1).all()
    assert (clusters >= 0).any(), "DBSCAN put every ticker in noise (-1); tune eps/min_samples"

    pairs = findPairs(close)
    assert not pairs.empty
    assert (clusters[pairs["a"]].values == clusters[pairs["b"]].values).all()
    weakest = pairs[["strength", "strengthFirst", "strengthSecond"]].min(axis=1)
    assert (pairs["stable"] == ((weakest > 1) & (pairs["beta"] > 0))).all()
    assert pairs["stable"].is_monotonic_decreasing  # stable pairs first

    # Johansen does not depend on which leg comes first, unlike Engle-Granger
    a, b = pairs.loc[0, "a"], pairs.loc[0, "b"]
    logClose = np.log(close)
    strengthAB, betaAB = johansen(logClose[a], logClose[b])
    strengthBA, betaBA = johansen(logClose[b], logClose[a])
    assert np.isclose(strengthAB, strengthBA) and np.isclose(betaAB * betaBA, 1.0)

    print(f"\nclusters:\n{clusters.sort_values().to_string()}\n\npairs:\n{pairs.to_string()}")

    _plot_pairs(
        close, returns, clusters, pairs,
        f"PairFinding on {len(PAIR_TICKERS)} large-caps, 2013-2025",
        "pairFinding_realData.png",
    )


def test_zScorePositions_holds_until_exit_or_stop():
    # The trade rules shared by MeanReversion and PairTrading, no data: a
    # position opened past entryZ is held through the 0.5-1.5 band, closed
    # under exitZ, stopped past stopZ, and the ADF p-value only gates entries.
    # The fit is mu = 0, std = 1 every day, so the value is the z-score.
    strategy = ZScorePositions(entryZ=1.5, exitZ=0.5, stopZ=3.0, pValueMax=0.05)

    def run(zs, pValues, mus=None, stds=None):
        mus = mus or [0.0] * len(zs)
        stds = stds or [1.0] * len(zs)
        state, path = (0, False, np.nan, np.nan), []
        for z, p, mu, std in zip(zs, pValues, mus, stds):
            state, _ = strategy.nextPosition(state, z, mu, std, p)
            path.append(state[0])
        return path

    ok = [0.01] * 6
    # z rises past 1.5: short the spread, held while it reverts, closed under 0.5
    assert run([1.0, 1.6, 1.4, 1.0, 0.6, 0.4], ok) == [0, -1, -1, -1, -1, 0]
    # negative side: long the spread
    assert run([-1.6, -1.2, -0.3], ok[:3]) == [1, 1, 0]
    # no entry when the spread is not stationary
    assert run([1.6, 2.0], [0.5, 0.5]) == [0, 0]
    # a failing ADF test does not close an open trade
    assert run([1.6, 2.0, 0.4], [0.01, 0.5, 0.5]) == [-1, -1, 0]
    # stop past 3, then no re-entry until |z| came back under 0.5
    assert run([1.6, 3.2, 2.0, 1.6, 0.4, 1.6], ok) == [-1, 0, 0, 0, 0, -1]
    # no entry without a valid fit (OU not mean-reverting: NaN std)
    assert run([1.6, 1.6], ok[:2], stds=[np.nan, 1.0]) == [0, -1]

    # in a trade, z is measured against the mu/std frozen at entry:
    # today's mu drifting up to the price is not a reversion...
    assert run([1.6, 1.6, 1.6], ok[:3], mus=[0.0, 1.4, 1.5]) == [-1, -1, -1]
    # ...the stop still fires when today's fit is not mean-reverting...
    assert run([1.6, 3.2], ok[:2], stds=[1.0, np.nan]) == [-1, 0]
    # ...and today's std shrinking does not push it past the stop
    assert run([1.6, 2.0], ok[:2], stds=[1.0, 0.5]) == [-1, -1]

    # a slow fit (half-life past maxHalfLife) cannot enter, but does not
    # close a trade already open
    slow = ZScorePositions(entryZ=1.5, exitZ=0.5, stopZ=3.0, pValueMax=0.05, maxHalfLife=30)
    state, _ = slow.nextPosition((0, False, np.nan, np.nan), 1.6, 0.0, 1.0, 0.01, halfLife=60)
    assert state[0] == 0
    state, _ = slow.nextPosition(state, 1.6, 0.0, 1.0, 0.01, halfLife=10)
    assert state[0] == -1
    state, _ = slow.nextPosition(state, 1.2, 0.0, 1.0, 0.01, halfLife=60)
    assert state[0] == -1


def test_zScorePositions_carries_state_only_within_a_run():
    rules = ZScorePositions()
    dates = pd.bdate_range("2024-01-01", periods=5)
    mu, std, p = pd.Series({"A": 0.0}), pd.Series({"A": 1.0}), pd.Series({"A": 0.001})
    enter, hold = pd.Series({"A": 2.0}), pd.Series({"A": 1.0})

    state = rules.update(dates[:2], enter, mu, std, p)
    assert state.loc["A", "position"] == -1
    assert (state.loc["A", "entryMu"], state.loc["A", "entryStd"]) == (0.0, 1.0)
    # next day of the same run: the trade is held
    assert rules.update(dates[:3], hold, mu, std, p).loc["A", "position"] == -1
    # same day again (signal recomputed): same answer, not a second step
    assert rules.update(dates[:3], hold, mu, std, p).loc["A", "position"] == -1
    # history that does not continue the last date: a new run, starts flat
    assert rules.update(dates[:2], hold, mu, std, p).loc["A", "position"] == 0


@pytest.mark.parametrize("numberPair", [1,2])
def test_backtestEngine_on_real_data_pairTrading(numberPair):
    # PAIR_TICKERS, candidates paired inside the PairFinding clusters.
    ohlcv = DataLoader.getOHLCV("2013-01-01", "2025-01-01", PAIR_TICKERS, "1d").dropna()
    _backtest_pair_trading(
        ohlcv, PairTrading(numberPair=numberPair, formationDays=5 * 252, refitDays=2 * 252),
        f"OU pair trading on {len(PAIR_TICKERS)} large-caps, {numberPair} pairs re-selected every 2 years, 2020-2024",
        f"backtestEngine_pairTrading_{numberPair}pairs.png",
    )


@pytest.mark.parametrize("numberPair", [30, 50])
def test_backtestEngine_on_sp500_pairTrading(numberPair):
    # Current S&P 500 members with a full 2010-2024 history (survivorship
    # bias: companies dropped from the index are missing), candidates paired
    # inside each GICS sub-industry: ~1400 candidate pairs instead of ~35.
    # Traded on 2015-2024, 5 formations; the first uses 2010-2014.
    # Stocks trading under $5M a day before 2015 are left out: their early
    # history is not the S&P 500 company (e.g. a pre-merger listing), and the
    # square-root impact makes them very expensive. Only pre-2015 data is
    # used for this, so it does not look ahead into the backtest.
    # The book is scaled to 10% expected annual volatility, gross capped at 4.
    constituents = DataLoader.getSP500Constituents()
    ohlcv = DataLoader.getOHLCV("2010-01-01", "2025-01-01", list(constituents["Symbol"]), "1d")
    close = ohlcv.xs("Close", axis=1, level=1)
    volume = ohlcv.xs("Volume", axis=1, level=1)
    dollarVolume = (close * volume).loc[:"2014-12-31"].median()
    universe = [t for t in close.columns if close[t].notna().all() and dollarVolume[t] > 5e6]
    ohlcv = ohlcv[universe]
    print(f"\nuniverse: {len(universe)} of {len(close.columns)} tickers")
    assert len(universe) > 350

    groups = constituents.set_index("Symbol")["SubIndustry"]
    pairTrading = PairTrading(numberPair=numberPair, formationDays=5 * 252, refitDays=2 * 252, groups=groups)
    strategy = VolatilityTarget(pairTrading, targetVol=0.10, lookback=60, maxGross=4.0)
    _backtest_pair_trading(
        ohlcv, strategy,
        f"OU pair trading on {len(universe)} S&P 500 stocks by sub-industry, "
        f"{numberPair} pairs re-selected every 2 years, 10% vol target, 2015-2024",
        f"backtestEngine_pairTrading_sp500_{numberPair}pairs.png",
        start="2015-01-01",
    )
    # every pair is two stocks of the same sub-industry
    for pairs in pairTrading.formations.values():
        assert (groups[pairs["a"]].values == groups[pairs["b"]].values).all()


def _backtest_pair_trading(ohlcv, strategy, title, filename, start="2020-01-01"):
    # Traded from `start` to the end of the data. Pairs, hedge ratios and the
    # OU mean/std of each spread are re-fitted every refitDays on the
    # formationDays before, from the history the engine hands the strategy,
    # so no formation sees a price after the day it is made. Only stable
    # pairs (Johansen cointegrated on the whole formation and on each half)
    # are traded; a formation may find none, and then the strategy stays in
    # cash until the next one. `strategy` is a PairTrading, or a
    # VolatilityTarget around one (gross then capped at its maxGross).
    pairStrategy = strategy.strategy if isinstance(strategy, VolatilityTarget) else strategy
    maxGross = strategy.maxGross if isinstance(strategy, VolatilityTarget) else 1.0
    close = ohlcv.xs("Close", axis=1, level=1)
    dates = ohlcv.index
    warmup = int(np.searchsorted(dates, pd.Timestamp(start)))

    engine = BacktestEngine(strategy, InMemoryDataSource(ohlcv), transactionCostRate=0.0005)
    results = engine.run(dates[warmup], dates[-1])

    for formedOn, pairs in pairStrategy.formations.items():
        print(f"\nformation on {formedOn.date()}: {len(pairs)} stable pairs\n{pairs.to_string()}")

    n_steps = len(close) - warmup
    step_dates = dates[warmup:]
    assert len(results) == n_steps
    assert len(engine.positions) == n_steps
    assert all(c >= 0 for c in engine.costs)

    # one formation at the start, then one every refitDays steps, each on
    # history that ends before the first day it trades
    formedOn = list(pairStrategy.formations)
    assert len(formedOn) == 1 + (n_steps - 1) // pairStrategy.refitDays
    assert formedOn[0] == dates[warmup - 1]
    for pairs in pairStrategy.formations.values():
        assert len(pairs) <= pairStrategy.numberPair  # fewer, or none, if fewer pairs are stable
        assert pairs["stable"].all()
        assert {"mu", "std", "halfLife", "alpha", "beta"} <= set(pairs.columns)

    # only tickers that belong to a selected pair are ever traded
    pairTickers = set().union(*(set(p["a"]) | set(p["b"]) for p in pairStrategy.formations.values()))
    others = [t for t in engine.positions.columns if t not in pairTickers]
    assert (engine.positions[others] == 0).all().all()

    gross = engine.positions.abs().sum(axis=1)
    assert (gross <= maxGross + 1e-9).all()

    period_returns = close.pct_change()
    for i, date in enumerate(step_dates):
        expected_gross = float((engine.positions.iloc[i] * period_returns.loc[date]).sum())
        assert np.isclose(results.iloc[i], expected_gross - engine.costs.iloc[i])

    def label(pairs):
        names = [f"{a}/{b}" for a, b in zip(pairs["a"], pairs["b"])]
        if not names:
            return "no stable pair"
        return f"{len(names)} pairs: " + ", ".join(names[:4]) + (", ..." if len(names) > 4 else "")

    _plot_backtest(
        step_dates, results, engine.costs, engine.metrics, f"BacktestEngine — {title}", filename,
        events={d: label(p) for d, p in pairStrategy.formations.items()},
    )
    return pairStrategy


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

def _plot_backtest(dates, results, costs,metrics:pd.DataFrame, title, filename, events: dict = None):
    # events: optional {date: label}, drawn as dashed vertical lines
    results = pd.Series(results, index=dates)
    costs = pd.Series(costs, index=dates)
    cumulative = (1 + results).cumprod()

    os.makedirs(PLOT_DIR, exist_ok=True)
    fig, (ax_cum, ax_ret, ax_cost) = plt.subplots(
        3, 1, figsize=(12, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]}
    )

    ax_cum.plot(cumulative.index, cumulative.values, color="tab:blue")
    ax_cum.set_title(title)
    for date, label in (events or {}).items():
        ax_cum.axvline(date, color="gray", linestyle="--", linewidth=0.8)
        ax_cum.annotate(label, xy=(date, 0.02), xycoords=("data", "axes fraction"),
                        rotation=90, fontsize=7, color="gray", ha="right", va="bottom")
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


def _plot_window_search(dates, results: dict, metrics: pd.DataFrame, best, title, filename):
    # Top: growth of $1 for every window. Below: one bar chart per metric,
    # windows on the x axis, the best window highlighted.
    n_cols = 4
    n_rows = -(-len(metrics.index) // n_cols)
    os.makedirs(PLOT_DIR, exist_ok=True)
    fig = plt.figure(figsize=(16, 4 + 2.6 * n_rows))
    grid = fig.add_gridspec(n_rows + 1, n_cols, height_ratios=[2.2] + [1] * n_rows)

    ax_cum = fig.add_subplot(grid[0, :])
    for window, result in results.items():
        cumulative = (1 + pd.Series(result, index=dates)).cumprod()
        ax_cum.plot(cumulative.index, cumulative.values,
                    linewidth=2.5 if window == best else 1.2, label=f"window={window}")
    ax_cum.set_title(title)
    ax_cum.set_ylabel("Growth of $1")
    ax_cum.legend(loc="upper left", fontsize=9)

    labels = [str(w) for w in metrics.columns]
    colors = ["tab:orange" if w == best else "tab:blue" for w in metrics.columns]
    for i, name in enumerate(metrics.index):
        ax = fig.add_subplot(grid[1 + i // n_cols, i % n_cols])
        ax.bar(labels, metrics.loc[name].values, color=colors)
        ax.set_title(name, fontsize=10)
        ax.tick_params(labelsize=8)
        ax.axhline(0, color="black", linewidth=0.6)
    fig.supxlabel("OU calibration window (days)")

    fig.tight_layout()
    out_path = os.path.join(PLOT_DIR, filename)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    assert os.path.exists(out_path)


def _plot_pairs(close: pd.DataFrame, returns: pd.DataFrame, clusters: pd.Series, pairs: pd.DataFrame, title, filename):
    # Left: return correlation matrix, tickers sorted by cluster, with the
    # cluster boundaries drawn. Right: the most cointegrated pair, its
    # normalized prices and the z-score of its log-price spread.
    order = clusters.sort_values(kind="stable").index
    corr = returns[order].corr()

    best = pairs.iloc[0]
    a, b = best["a"], best["b"]
    alpha, beta = findParameters(close, best)
    spread = np.log(close[a]) - beta * np.log(close[b]) - alpha
    zscore = (spread - spread.mean()) / spread.std()

    os.makedirs(PLOT_DIR, exist_ok=True)
    fig = plt.figure(figsize=(18, 10))
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.3])
    fig.suptitle(title)

    ax_corr = fig.add_subplot(grid[:, 0])
    im = ax_corr.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    labels = [f"{t} ({clusters[t]})" for t in order]
    ax_corr.set_xticks(range(len(order)), labels, rotation=90, fontsize=7)
    ax_corr.set_yticks(range(len(order)), labels, fontsize=7)
    sorted_labels = clusters[order].values
    for i in np.where(sorted_labels[1:] != sorted_labels[:-1])[0]:
        ax_corr.axhline(i + 0.5, color="black", linewidth=0.8)
        ax_corr.axvline(i + 0.5, color="black", linewidth=0.8)
    ax_corr.set_title("Return correlation by cluster (-1 = noise)")
    fig.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)

    ax_price = fig.add_subplot(grid[0, 1])
    ax_price.plot(close.index, close[a] / close[a].iloc[0], label=a)
    ax_price.plot(close.index, close[b] / close[b].iloc[0], label=b)
    ax_price.set_title(f"Best pair {a} / {b}: cluster {best['cluster']}, "
                       f"{'stable' if best['stable'] else 'NOT stable'}, Johansen trace / crit "
                       f"{best['strength']:.2f} (halves {best['strengthFirst']:.2f}, {best['strengthSecond']:.2f})")
    ax_price.set_ylabel("Price / first price")
    ax_price.legend(loc="upper left")
    ax_price.text(
        0.99, 0.98,
        "Top pairs: trace / crit (whole, 1st half, 2nd half)\n" + "\n".join(
            f"{'*' if r.stable else ' '} {r.a}/{r.b}: {r.strength:.2f}, {r.strengthFirst:.2f}, {r.strengthSecond:.2f}"
            for r in pairs.head(10).itertuples()
        ) + "\n* stable: > 1 on all three",
        transform=ax_price.transAxes, va="top", ha="right", fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax_spread = fig.add_subplot(grid[1, 1], sharex=ax_price)
    ax_spread.plot(zscore.index, zscore.values, color="tab:purple", linewidth=0.8)
    for level, style in [(0, "-"), (2, "--"), (-2, "--")]:
        ax_spread.axhline(level, color="black", linestyle=style, linewidth=0.7)
    ax_spread.set_title(f"Spread z-score: log {a} - {beta:.3f} log {b}")
    ax_spread.set_ylabel("z-score")
    ax_spread.set_xlabel("Date")

    fig.tight_layout()
    out_path = os.path.join(PLOT_DIR, filename)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    assert os.path.exists(out_path)


if __name__ == "__main__":
    test_backtestEngine_on_real_data()
    test_backtestEngine_on_real_data_meanReversion()
    test_meanReversion_window_search()
    test_pairFinding_on_real_data()
    test_zScorePositions_holds_until_exit_or_stop()
    test_zScorePositions_carries_state_only_within_a_run()
    test_backtestEngine_on_real_data_pairTrading(5)
    test_backtestEngine_on_real_data_pairTrading(10)
    test_backtestEngine_on_real_data_uniformCombiner()
    test_backtestEngine_on_real_data_invVolCombiner()
    test_backtestEngine_on_real_data_riskParityCombiner()
    test_riskParity_accounts_for_correlation()
    print("ok")