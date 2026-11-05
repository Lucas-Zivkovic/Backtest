import io
import os
import zipfile

import matplotlib

from DataLoaders.IBKRLoader import IBKRLoader
from DataLoaders.KrakenHistoryLoader import KrakenHistoryLoader, _RemoteFile
from DataLoaders.KrakenLoader import KrakenLoader

matplotlib.use("Agg")  # headless backend so the test runs without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import DataLoader
import strategies
import Metrics
from BacktestEngine import BacktestEngine
from Strategies.Strategy import Strategy
from PerformanceAnalyzer import PerformanceAnalyzer
from Strategies.CrossSectionalMomentumStrategy import CrossSectionalMomentumStrategy
from Strategies.CrossSectionalReversalStrategy import CrossSectionalReversalStrategy
from Strategies.ResidualReversalStrategy import ResidualReversalStrategy
from Strategies.MeanReversion import MeanReversion
from Strategies.PairFinding import PairFinding, findPairs, findParameters, johansen
from Strategies.PairTrading import PairTrading
from Strategies.VolatilityTarget import VolatilityTarget
from Strategies.ZScorePositions import ZScorePositions
from Strategies.PortfolioCombiner.Uniform import Uniform
from Strategies.PortfolioCombiner.InvVol import InvVol
from Strategies.PortfolioCombiner.RiskParity import RiskParity
from Strategies.PortfolioCombiner.CalibratedCombiner import CalibratedCombiner

FIELDS_OHLCV = ["Open", "High", "Low", "Close", "Volume"]

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

# Listed AI companies, by segment (Anthropic, OpenAI... are not listed),
# with the semiconductor ETFs SMH and SOXX and both Alphabet share classes.
# Some listed during the period (ALAB 2024-03, NBIS 2024-10, CRWV 2025-03):
# a formation only uses the tickers with a full history over its window.
AI_SEGMENTS = {
    "NVDA": "chips", "AMD": "chips", "AVGO": "chips", "INTC": "chips", "MU": "chips", "QCOM": "chips",
    "MRVL": "chips", "TSM": "chips", "ARM": "chips", "ALAB": "chips", "CRDO": "chips",
    "SMH": "chips", "SOXX": "chips",
    "ASML": "chip equipment", "AMAT": "chip equipment", "LRCX": "chip equipment", "KLAC": "chip equipment",
    "MSFT": "cloud & AI platforms", "GOOGL": "cloud & AI platforms", "GOOG": "cloud & AI platforms",
    "AMZN": "cloud & AI platforms", "META": "cloud & AI platforms", "ORCL": "cloud & AI platforms",
    "IBM": "cloud & AI platforms", "CRWV": "cloud & AI platforms", "NBIS": "cloud & AI platforms",
    "CRM": "AI software", "ADBE": "AI software", "NOW": "AI software", "PLTR": "AI software",
    "SNOW": "AI software", "AI": "AI software", "PATH": "AI software", "MDB": "AI software",
    "DDOG": "AI software",
    "ANET": "AI infrastructure", "SMCI": "AI infrastructure", "EQIX": "AI infrastructure",
    "DELL": "AI infrastructure", "VRT": "AI infrastructure", "HPE": "AI infrastructure",
    "CEG": "AI power", "VST": "AI power", "NRG": "AI power", "TLN": "AI power",
}
AI_TICKERS = list(AI_SEGMENTS)
AI_BARS_PER_DAY = 7

# Crypto pairs on Kraken (XDG is Dogecoin). Kraken's API only serves the
# last 720 candles, so "1d" and "1h" use the most recent ones, dated from
# today: ~2 years of daily bars, ~29 days of hourly bars. "1h-history" reads
# a year of hourly bars from Kraken's downloadable archive instead
# (KrakenHistoryLoader, which ends with the last full quarter). Crypto
# trades every day, around the clock: 30-day months, 365-day years.
CRYPTO_PAIRS = [
    "XBTUSD", "ETHUSD", "SOLUSD", "XRPUSD", "ADAUSD", "XDGUSD", "LTCUSD", "BCHUSD", "LINKUSD", "XLMUSD",
    "ETCUSD", "ATOMUSD", "AVAXUSD", "DOTUSD", "TRXUSD", "UNIUSD", "AAVEUSD", "NEARUSD", "FILUSD", "XMRUSD",
    "ALGOUSD", "SUIUSD", "APTUSD",
]
# Per interval: days of data, bars a day, and the strategies' settings in
# bars. Momentum/reversal: name -> (bars per unit, skipped units, lookback
# units); pairs: formation and refit bars.
CRYPTO_INTERVALS = {
    "1d": {
        "days": 715, "barsPerDay": 1, "momentumWarmup": 215,
        "momentum": {"1w": (7, 0, 1), "1m": (30, 0, 1), "3m": (30, 0, 3), "6-1m": (30, 1, 6)},
        "reversal": {"1w": (7, 0, 1), "1m": (30, 0, 1)},
        "formation": 180, "refit": 30,
    },
    "1h": {
        "days": 29, "barsPerDay": 24, "momentumWarmup": 200,
        "momentum": {"6h": (6, 0, 1), "1d": (24, 0, 1), "3d": (24, 0, 3), "1w-1d": (24, 1, 7)},
        "reversal": {"6h": (6, 0, 1), "1d": (24, 0, 1)},
        "formation": 168, "refit": 24,
    },
}
CRYPTO_INTERVALS["1h-history"] = {**CRYPTO_INTERVALS["1h"], "interval": "1h", "history": ("2025-10-01", "2026-10-01")}
# Kraken's taker fee for small accounts, and a retail-sized book: most of
# these pairs trade a few $M a day on Kraken
CRYPTO_ENGINE = {"transactionCostRate": 0.0026, "portfolioValue": 100_000}

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")

# Date range with Fama-French factors, AAPL and ^GSPC already in the data
# cache, so the Fama-French tests on real data do not need the network.
FF_START, FF_END = "2011-01-01", "2026-08-01"
FF_FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]


class InMemoryDataSource:
    """Feeds BacktestEngine a growing window of price history, one date at a time."""

    def __init__(self, prices: pd.DataFrame):
        self.prices = prices

    def iterate(self, start_date, end_date):
        mask = (self.prices.index >= start_date) & (self.prices.index <= end_date)
        for i in np.where(mask)[0]:
            yield self.prices.index[i], self.prices.iloc[: i + 1]


@pytest.mark.realdata
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


@pytest.mark.realdata
def test_backtestEngine_on_real_data_ibkr():
    ibkr = IBKRLoader()
    ohlcv = ibkr.getOHLCV("2015-01-01", "2023-01-01", REAL_TICKERS, "4h").dropna()
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

@pytest.mark.realdata
def test_momentum_interval_search_ibkr():
    # Cross-sectional 12-1 momentum on IBKR bars of several sizes, best bar
    # size by Sharpe (see _interval_search). The lookback is the same 12-1
    # months for every interval (21 trading days a month, times the bars per
    # day); only how often the book is rebalanced changes.
    bars30m = IBKRLoader().getOHLCV("2015-01-01", "2023-01-01", REAL_TICKERS, "30m").dropna()
    days = bars30m.index.normalize().unique()
    _interval_search(
        bars30m, days[280],  # 12-1 signal needs 273 trading days of history
        lambda barsPerDay: CrossSectionalMomentumStrategy(
            skipMonths=1, lookbackMonth=12, tradingDayPerMonth=21 * barsPerDay),
        "12-1 momentum", "momentum_intervalSearch_ibkr.png",
    )


@pytest.mark.realdata
def test_meanReversion_interval_search_ibkr():
    # OU mean reversion on the same IBKR bars, best bar size by Sharpe (see
    # _interval_search). The OU window is the same 120 trading days for every
    # interval (120 times the bars per day), so the max half-life is 60 days
    # for all; only how often the fit is redone and the z-score checked
    # changes. OU calibration is per-step, per-ticker (~0.2 s a step), so
    # only the last 80 trading days are backtested: 2,160 steps in all.
    bars30m = IBKRLoader().getOHLCV("2015-01-01", "2023-01-01", REAL_TICKERS, "30m").dropna()
    days = bars30m.index.normalize().unique()
    _interval_search(
        bars30m, days[200],
        lambda barsPerDay: MeanReversion(window=120 * barsPerDay),
        "OU mean reversion", "meanReversion_intervalSearch_ibkr.png",
    )


def _interval_search(bars30m: pd.DataFrame, start, makeStrategy, label, filename):
    # Backtests makeStrategy(barsPerDay) on bars of 30m, 1h, 2h, 4h and 1d
    # from `start` to the end of the data, and picks the bar size with the
    # best Sharpe ratio. Every interval is built from the same 30m bars
    # (regular hours, 13 a day), so they cover the same prices; a bar of
    # k*30m groups k consecutive 30m bars of a day, starting at the open.
    # Bar returns are compounded into daily returns before scoring, so
    # Sharpe and volatility are annualized the same way for every interval.
    # The choice is in-sample, like test_meanReversion_window_search.
    intervals = {"30m": 1, "1h": 2, "2h": 4, "4h": 8, "1d": 13}  # 30m bars per bar
    results, metrics = {}, {}
    for name, k in intervals.items():
        ohlcv = _group_bars(bars30m, k)
        barsPerDay = -(-13 // k)
        engine = BacktestEngine(makeStrategy(barsPerDay), InMemoryDataSource(ohlcv), transactionCostRate=0.0005)
        barReturns = engine.run(start, ohlcv.index[-1])

        assert barReturns.index[0].normalize() == start
        assert all(c >= 0 for c in engine.costs)
        assert (engine.positions.abs().sum(axis=1) <= 1.0 + 1e-9).all()

        daily = (1 + barReturns).groupby(barReturns.index.normalize()).prod() - 1
        endOfDayPositions = engine.positions.groupby(engine.positions.index.normalize()).last()
        results[name] = daily
        metrics[name] = PerformanceAnalyzer().metrics(daily, endOfDayPositions)["value"]
        # turnover counts every rebalance, intraday ones included
        metrics[name]["annualizedTurnover"] = Metrics.annualizedTurnover(engine.positions.values, 252 * barsPerDay)
        metrics[name]["annualizedCost"] = engine.costs.sum() / len(daily) * 252

    step_dates = results["1d"].index
    for name, daily in results.items():
        assert daily.index.equals(step_dates), name

    metrics = pd.DataFrame(metrics)  # rows: metric, columns: interval
    best = metrics.loc["sharpeRatio"].idxmax()
    print(f"\n{metrics.to_string()}\nbest interval (Sharpe): {best}")

    _plot_window_search(
        step_dates, results, metrics, best,
        f"{label} by bar interval on {len(REAL_TICKERS)} large-caps, IBKR (best by Sharpe: {best})",
        filename, legendPrefix="interval=", xlabel="Bar interval (rebalancing frequency)",
    )
    return metrics


def _group_bars(ohlcv: pd.DataFrame, k: int) -> pd.DataFrame:
    """OHLCV bars of k consecutive bars of the same day, starting at the
    open: the last bar of a day may be shorter. Indexed by the timestamp of
    the last bar in each group, i.e. when the grouped bar is complete."""
    day = ohlcv.index.normalize()
    group = pd.Series(ohlcv.groupby(day).cumcount().values // k, index=ohlcv.index)
    keys = [day, group.values]
    grouped = {}
    for ticker in ohlcv.columns.get_level_values(0).unique():
        bars = ohlcv[ticker].groupby(keys)
        grouped[ticker] = pd.DataFrame({
            "Open": bars["Open"].first(), "High": bars["High"].max(), "Low": bars["Low"].min(),
            "Close": bars["Close"].last(), "Volume": bars["Volume"].sum(),
        })
    result = pd.concat(grouped, axis=1)
    result.index = pd.Series(ohlcv.index, index=ohlcv.index).groupby(keys).last().values
    return result


@pytest.mark.realdata
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


@pytest.mark.realdata
def test_backtestEngine_on_real_data_meanReversion_ibkr():
    # Same real-data setup as test_backtestEngine_on_real_data (reuses the
    # cached download), but with MeanReversion. Its signal() recalibrates an
    # OU process per ticker per step, which is expensive, so we keep the
    # universe and the number of steps small enough to run quickly.
    tickers = REAL_TICKERS
    ibkr = IBKRLoader()
    ohlcv = ibkr.getOHLCV("2023-01-01", "2026-01-01", REAL_TICKERS, "1h").dropna()
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

@pytest.mark.realdata
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


@pytest.mark.realdata
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


@pytest.mark.realdata
def test_pairFinding_on_real_data_ibkr():
    # Clusters PAIR_TICKERS with PairFinding (correlation distance -> PCA ->
    # DBSCAN) over 2013-2025, then findPairs tests every pair inside a cluster
    # with the Johansen trace test on log prices, over the whole period and
    # each half. Correlation is measured on daily returns: prices trend, so
    # their correlation is spurious.
    ibkr = IBKRLoader()
    ohlcv = ibkr.getOHLCV("2020-01-01", "2025-01-01", PAIR_TICKERS, "30m").dropna()
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

def _ai_prices():
    """IBKR 1h bars of AI_TICKERS, January 2024 to September 2026: NaN
    before a ticker's first bar (listed during the period)."""
    ohlcv = IBKRLoader().getOHLCV("2024-01-01", "2026-10-01", AI_TICKERS, "1h")
    close = ohlcv.xs("Close", axis=1, level=1)
    # a bar most tickers do not have is not a regular trading hour (a stray
    # bar of one ticker): it would leave the others out of a formation
    return ohlcv[close.notna().sum(axis=1) > len(AI_TICKERS) / 2]


@pytest.mark.realdata
def test_pairFinding_on_ai_stocks_ibkr():
    # The pairs PairTrading would pick now: PairFinding clusters (correlation
    # of returns -> PCA -> DBSCAN) and the Johansen test on every pair of a
    # segment of AI_SEGMENTS, on the last 6 months of 1h bars, over the whole
    # window and each half. Tickers listed during the window are left out.
    ohlcv = _ai_prices()
    close = ohlcv.xs("Close", axis=1, level=1)[AI_TICKERS]
    close = close.tail(126 * AI_BARS_PER_DAY).dropna(axis=1)
    returns = close.pct_change().dropna()
    segments = pd.Series(AI_SEGMENTS)[close.columns]

    clusters = PairFinding(returns)
    assert list(clusters.index) == list(close.columns)
    assert (clusters >= -1).all()

    pairs = findPairs(close, groups=segments)
    assert not pairs.empty
    assert (segments[pairs["a"]].values == segments[pairs["b"]].values).all()
    weakest = pairs[["strength", "strengthFirst", "strengthSecond"]].min(axis=1)
    assert (pairs["stable"] == ((weakest > 1) & (pairs["beta"] > 0))).all()

    print(f"\n{close.index[0]} to {close.index[-1]}, {len(close)} bars, {len(close.columns)} tickers")
    table = pd.DataFrame({"cluster": clusters, "segment": segments}).sort_values(["cluster", "segment"])
    print(f"\nclusters (-1 = noise):\n{table.to_string()}")
    print(f"\nstable pairs: {int(pairs['stable'].sum())} of {len(pairs)}\n{pairs.head(20).to_string()}")

    _plot_pairs(
        close, returns, clusters, pairs,
        f"PairFinding on {len(close.columns)} AI stocks, IBKR 1h bars, "
        f"{close.index[0].date()} to {close.index[-1].date()}",
        "pairFinding_ai_ibkr.png",
    )


@pytest.mark.realdata
@pytest.mark.parametrize("numberPair", [3, 5])
def test_backtestEngine_on_ai_stocks_pairTrading_ibkr(numberPair):
    # OU pair trading on the AI stocks with 1h bars, candidates paired
    # inside the segments of AI_SEGMENTS. The pairs are re-selected every
    # month on the 3 months before (a recent theme: relationships are short
    # lived), traded from April 2024 to September 2026, the first formation
    # on January-March 2024. The book is scaled to 10% expected annual
    # volatility, gross capped at 4; the volatility target and the metrics
    # are annualized over 252 * 7 bars.
    ohlcv = _ai_prices()
    segments = pd.Series(AI_SEGMENTS)
    periodsPerYear = 252 * AI_BARS_PER_DAY
    pairTrading = PairTrading(numberPair=numberPair, formationDays=63 * AI_BARS_PER_DAY,
                              refitDays=21 * AI_BARS_PER_DAY, groups=segments)
    strategy = VolatilityTarget(pairTrading, targetVol=0.10, lookback=60, maxGross=4.0,
                                periodsPerYear=periodsPerYear)
    _backtest_pair_trading(
        ohlcv, strategy,
        f"OU pair trading on {len(AI_TICKERS)} AI stocks by segment, IBKR 1h bars,\n"
        f"{numberPair} pairs re-selected every month on the 3 months before, 10% vol target, Apr 2024-Sep 2026",
        f"backtestEngine_pairTrading_ai_{numberPair}pairs_ibkr.png",
        start="2024-04-01", periodsPerYear=periodsPerYear,
    )
    for pairs in pairTrading.formations.values():
        assert (segments[pairs["a"]].values == segments[pairs["b"]].values).all()


def test_reversal_buysTheLosers_sellsTheWinners():
    # 20 tickers with steady trends, ticker i growing i% a day: the reversal
    # strategy is long the 2 slowest (10%), short the 2 fastest, at +-0.25
    # each, the exact opposite of the momentum strategy.
    dates = pd.bdate_range("2024-01-01", periods=60)
    growth = {f"T{i:02d}": (1 + i / 100) ** np.arange(60) for i in range(20)}
    prices = pd.DataFrame(growth, index=dates)

    reversal = CrossSectionalReversalStrategy(lookbackMonth=1).positions(prices).iloc[-1]
    momentum = CrossSectionalMomentumStrategy(skipMonths=0, lookbackMonth=1).positions(prices).iloc[-1]
    assert list(reversal[reversal > 0].index) == ["T00", "T01"]
    assert list(reversal[reversal < 0].index) == ["T18", "T19"]
    assert np.allclose(reversal[reversal != 0].abs(), 0.25)
    assert np.allclose(reversal, -momentum)


def test_residualReversal_ignoresTheMarketMove():
    # The market falls 1% a day for the last month. Ticker i has beta
    # 0.5 + i/10 and its own drift: the 2 high-beta names fell most, but
    # their residual is 0; T03 and T07 are the ones that did worse than
    # their beta explains, T05 and T09 better. Residual reversal is long
    # T03/T07 and short T05/T09; the raw reversal buys the high betas.
    rng = np.random.default_rng(0)
    n, tickers = 300, [f"T{i:02d}" for i in range(20)]
    market = np.r_[rng.normal(0, 0.01, n - 21), np.full(21, -0.01)]
    drift = pd.Series(0.0, index=tickers)
    drift[["T03", "T07"]] = -0.004
    drift[["T05", "T09"]] = 0.004
    returns = {}
    for i, t in enumerate(tickers):
        own = rng.normal(0, 0.002, n)
        own[-21:] += drift[t]
        returns[t] = (0.5 + i / 10) * market + own
    prices = 100 * (1 + pd.DataFrame(returns, index=pd.bdate_range("2023-01-02", periods=n))).cumprod()

    residual = ResidualReversalStrategy(lookbackDays=21, betaDays=252).positions(prices).iloc[-1]
    assert set(residual[residual > 0].index) == {"T03", "T07"}
    assert set(residual[residual < 0].index) == {"T05", "T09"}
    raw = CrossSectionalReversalStrategy(lookbackMonth=1).positions(prices).iloc[-1]
    assert set(raw[raw > 0].index) == {"T18", "T19"}


@pytest.mark.realdata
def test_residualReversal_lookback_search_sp500_ibkr():
    # Residual reversal on the same S&P 500 universe and period as
    # test_reversal_lookback_search_sp500_ibkr (IBKR daily, 2015-2024,
    # rebalanced every day): long the 10% worst residual returns over the
    # lookback, short the 10% best, betas on the year before. The raw 1m
    # reversal is the reference. `annualizedGrossMean` is the return before
    # trading costs, to tell the signal from what costs take.
    constituents = DataLoader.getSP500Constituents()
    ohlcv = IBKRLoader().getOHLCV("2010-01-01", "2025-01-01", list(constituents["Symbol"]), "1d")
    ohlcv = _liquid_universe(ohlcv, "2015-01-01")
    dates = ohlcv.index
    start = dates[np.searchsorted(dates, pd.Timestamp("2015-01-01"))]

    candidates = {
        "residual 1w": ResidualReversalStrategy(lookbackDays=5),
        "residual 1m": ResidualReversalStrategy(lookbackDays=21),
        "residual 3m": ResidualReversalStrategy(lookbackDays=63),
        "raw reversal 1m": CrossSectionalReversalStrategy(lookbackMonth=1),
    }
    results, metrics = {}, {}
    for name, strategy in candidates.items():
        engine = BacktestEngine(strategy, InMemoryDataSource(ohlcv), transactionCostRate=0.0005)
        results[name] = engine.run(start, dates[-1])
        metrics[name] = engine.metrics["value"]
        metrics[name]["annualizedCost"] = engine.costs.sum() / len(engine.costs) * 252
        metrics[name]["annualizedGrossMean"] = metrics[name]["annualizedMean"] + metrics[name]["annualizedCost"]
        assert all(c >= 0 for c in engine.costs)
        last = engine.positions.iloc[-1]
        assert np.isclose(last.sum(), 0) and np.isclose(last.abs().sum(), 1)

    metrics = pd.DataFrame(metrics)
    best = metrics.loc["sharpeRatio"].idxmax()
    print(f"\n{metrics.to_string()}\nbest (Sharpe): {best}")

    _plot_window_search(
        results["raw reversal 1m"].index, results, metrics, best,
        f"Residual reversal on {len(ohlcv.columns.get_level_values(0).unique())} S&P 500 stocks, "
        f"IBKR daily, 2015-2024 (best by Sharpe: {best})",
        "residualReversal_lookbackSearch_sp500_ibkr.png",
        legendPrefix="", xlabel="Strategy (lookback)",
    )


@pytest.mark.realdata
def test_reversal_lookback_search_sp500_ibkr():
    # Cross-sectional reversal on the S&P 500 (IBKR daily prices, the
    # tickers of test_backtestEngine_on_sp500_pairTrading_ibkr's universe):
    # long the 10% worst performers over the lookback, short the 10% best,
    # rebalanced every day, 2015-2024. Several lookbacks, with 12-1 momentum
    # (the opposite bet, on the year) as a reference; the best by Sharpe is
    # picked in-sample, like test_meanReversion_window_search.
    constituents = DataLoader.getSP500Constituents()
    ohlcv = IBKRLoader().getOHLCV("2010-01-01", "2025-01-01", list(constituents["Symbol"]), "1d")
    ohlcv = _liquid_universe(ohlcv, "2015-01-01")
    dates = ohlcv.index
    start = dates[np.searchsorted(dates, pd.Timestamp("2015-01-01"))]

    candidates = {
        "reversal 1w": CrossSectionalReversalStrategy(lookbackMonth=1, tradingDayPerMonth=5),
        "reversal 1m": CrossSectionalReversalStrategy(lookbackMonth=1),
        "reversal 3m": CrossSectionalReversalStrategy(lookbackMonth=3),
        "momentum 12-1": CrossSectionalMomentumStrategy(skipMonths=1, lookbackMonth=12),
    }
    results, metrics = {}, {}
    for name, strategy in candidates.items():
        engine = BacktestEngine(strategy, InMemoryDataSource(ohlcv), transactionCostRate=0.0005)
        results[name] = engine.run(start, dates[-1])
        metrics[name] = engine.metrics["value"]
        metrics[name]["annualizedCost"] = engine.costs.sum() / len(engine.costs) * 252
        assert all(c >= 0 for c in engine.costs)
        # dollar neutral: as much long as short, unit gross
        last = engine.positions.iloc[-1]
        assert np.isclose(last.sum(), 0) and np.isclose(last.abs().sum(), 1)

    metrics = pd.DataFrame(metrics)  # rows: metric, columns: strategy
    best = metrics.loc["sharpeRatio"].idxmax()
    print(f"\n{metrics.to_string()}\nbest (Sharpe): {best}")

    _plot_window_search(
        results["momentum 12-1"].index, results, metrics, best,
        f"Cross-sectional reversal on {len(ohlcv.columns.get_level_values(0).unique())} S&P 500 stocks, "
        f"IBKR daily, 2015-2024 (best by Sharpe: {best})",
        "reversal_lookbackSearch_sp500_ibkr.png",
        legendPrefix="", xlabel="Strategy (lookback)",
    )


def _kraken_prices(key: str, warmupBars: int):
    """Kraken OHLCV of CRYPTO_PAIRS for CRYPTO_INTERVALS[key] (the API's last
    `days` up to yesterday, or the archive's `history` period), and the first
    traded bar, `warmupBars` in. Pairs missing a bar (an hour without trades,
    from the API), or trading under $0.5M a day (median, over the warmup
    only: no look-ahead), are left out."""
    settings = CRYPTO_INTERVALS[key]
    interval = settings.get("interval", key)
    if "history" in settings:
        ohlcv = KrakenHistoryLoader().getOHLCV(*settings["history"], CRYPTO_PAIRS, interval)
    else:
        today = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
        start = (today - pd.Timedelta(days=settings["days"])).strftime("%Y-%m-%d")
        ohlcv = KrakenLoader().getOHLCV(start, today.strftime("%Y-%m-%d"), CRYPTO_PAIRS, interval)
    close = ohlcv.xs("Close", axis=1, level=1)
    volume = ohlcv.xs("Volume", axis=1, level=1)  # in coins on Kraken
    first = close.index[warmupBars]
    dollarVolume = (close * volume)[close.index < first]
    dailyDollarVolume = dollarVolume.groupby(dollarVolume.index.normalize()).sum(min_count=1).median()
    universe = [p for p in CRYPTO_PAIRS if close[p].notna().all() and dailyDollarVolume[p] > 5e5]
    print(f"\nuniverse: {len(universe)} of {len(CRYPTO_PAIRS)} pairs, {key} bars from {close.index[0]}, "
          f"traded from {first}")
    assert len(universe) > 10
    return ohlcv[universe], first


@pytest.mark.realdata
@pytest.mark.parametrize("interval", list(CRYPTO_INTERVALS))
def test_crypto_momentum_reversal_search_kraken(interval):
    # Cross-sectional momentum and reversal on Kraken crypto pairs,
    # rebalanced every bar: long the top 25% by trailing return, short the
    # bottom 25% (reversal: the other way round). Daily: ~500 days traded
    # after a 215-day warmup (6-1 momentum needs 7 months of 30 days).
    # Hourly: ~20 days traded after a 200-hour warmup (1w-1d needs 8 days).
    # The best by Sharpe is picked in-sample. `annualizedGrossMean` is the
    # return before costs.
    settings = CRYPTO_INTERVALS[interval]
    ohlcv, start = _kraken_prices(interval, settings["momentumWarmup"])
    periodsPerYear = 365 * settings["barsPerDay"]
    candidates = {}
    for name, (unit, skip, lookback) in settings["momentum"].items():
        candidates[f"momentum {name}"] = CrossSectionalMomentumStrategy(
            0.25, skipMonths=skip, lookbackMonth=lookback, tradingDayPerMonth=unit)
    for name, (unit, skip, lookback) in settings["reversal"].items():
        candidates[f"reversal {name}"] = CrossSectionalReversalStrategy(
            0.25, skipMonths=skip, lookbackMonth=lookback, tradingDayPerMonth=unit)

    results, metrics = {}, {}
    for name, strategy in candidates.items():
        engine = BacktestEngine(strategy, InMemoryDataSource(ohlcv), periodsPerYear=periodsPerYear, **CRYPTO_ENGINE)
        results[name] = engine.run(start, ohlcv.index[-1])
        metrics[name] = engine.metrics["value"]
        metrics[name]["annualizedCost"] = engine.costs.sum() / len(engine.costs) * periodsPerYear
        metrics[name]["annualizedGrossMean"] = metrics[name]["annualizedMean"] + metrics[name]["annualizedCost"]
        assert results[name].index[0] == start
        assert all(c >= 0 for c in engine.costs)
        last = engine.positions.iloc[-1]
        assert np.isclose(last.sum(), 0) and np.isclose(last.abs().sum(), 1)

    metrics = pd.DataFrame(metrics)
    best = metrics.loc["sharpeRatio"].idxmax()
    print(f"\n{metrics.to_string()}\nbest (Sharpe): {best}")

    first = next(iter(results.values()))
    _plot_window_search(
        first.index, results, metrics, best,
        f"Crypto momentum and reversal on {len(ohlcv.columns.get_level_values(0).unique())} Kraken pairs, "
        f"{interval} bars (best by Sharpe: {best})",
        f"crypto_momentumReversal_kraken_{interval}.png",
        legendPrefix="", xlabel="Strategy (lookback)",
    )


@pytest.mark.realdata
@pytest.mark.parametrize("numberPair", [2, 4])
@pytest.mark.parametrize("interval", list(CRYPTO_INTERVALS))
def test_crypto_pairTrading_kraken(interval, numberPair):
    # OU pair trading on Kraken crypto pairs: every two coins of the
    # universe are candidates. Daily: re-selected every 30 days on the 180
    # days before, ~535 days traded. Hourly: re-selected every day on the
    # week before, ~22 days traded. The book is scaled to 10% expected
    # annual volatility (365-day years), gross capped at 4.
    settings = CRYPTO_INTERVALS[interval]
    ohlcv, start = _kraken_prices(interval, settings["formation"])
    pairs = ohlcv.columns.get_level_values(0).unique()
    periodsPerYear = 365 * settings["barsPerDay"]
    pairTrading = PairTrading(numberPair=numberPair, formationDays=settings["formation"],
                              refitDays=settings["refit"], groups=pd.Series("crypto", index=pairs))
    strategy = VolatilityTarget(pairTrading, targetVol=0.10, lookback=60, maxGross=4.0, periodsPerYear=periodsPerYear)
    _backtest_pair_trading(
        ohlcv, strategy,
        f"OU pair trading on {len(pairs)} Kraken crypto pairs, {interval} bars,\n"
        f"{numberPair} pairs re-selected every {settings['refit']} bars on the {settings['formation']} before, "
        f"10% vol target",
        f"backtestEngine_pairTrading_crypto_{numberPair}pairs_kraken_{interval}.png",
        start=start, periodsPerYear=periodsPerYear, **CRYPTO_ENGINE,
    )


def test_backtestEngine_rebalanceEvery_and_band():
    # Flip: the target swaps between +0.5/-0.5 and -0.5/+0.5 every bar.
    # Rebalancing every 3 bars, the book trades on bars 0, 3, 6, 9 only and
    # pays costs only then. Wiggle: the target on A moves between 0.50 and
    # 0.51 every bar; with a band of 0.05 those moves are not traded, but
    # the first bar (from flat) is.
    
    class Flip(Strategy):
        def signal(self, prices):
            return prices.tail(1)

        def positions(self, prices):
            sign = 1 if len(prices) % 2 else -1
            return pd.DataFrame([[0.5 * sign, -0.5 * sign]], index=prices.index[-1:], columns=["A", "B"])

    class Wiggle(Flip):
        def positions(self, prices):
            a = 0.51 if len(prices) % 2 else 0.50
            return pd.DataFrame([[a, -0.5]], index=prices.index[-1:], columns=["A", "B"])

    dates = pd.bdate_range("2024-01-01", periods=12)
    prices = {(t, f): 100.0 + np.arange(12) * (1 if t == "A" else -1) for t in ["A", "B"] for f in FIELDS_OHLCV}
    ohlcv = pd.DataFrame(prices, index=dates)
    ohlcv.columns = pd.MultiIndex.from_tuples(ohlcv.columns)
    source = InMemoryDataSource(ohlcv)

    every = BacktestEngine(Flip(), source, rebalanceEvery=3)
    every.run(dates[2], dates[-1])
    traded = every.positions.diff().abs().sum(axis=1) > 0
    traded.iloc[0] = True
    assert list(np.where(traded)[0]) == [0, 3, 6, 9]
    assert list(np.where(every.costs > 0)[0]) == [0, 3, 6, 9]

    band = BacktestEngine(Wiggle(), source, rebalanceBand=0.05)
    band.run(dates[2], dates[-1])
    assert (band.positions["A"] == band.positions["A"].iloc[0]).all()
    assert list(np.where(band.costs > 0)[0]) == [0]
    daily = BacktestEngine(Wiggle(), source)
    daily.run(dates[2], dates[-1])
    assert (daily.costs > 0).all()

    # a position the strategy closes is closed, even when it is within the
    # band: 0.10 (traded), cut to 0.03 (a 0.07 move, traded), then 0 (a
    # 0.03 move, within the band, but a close)
    class Close(Flip):
        def positions(self, prices):
            a = 0.10 if len(prices) < 5 else 0.03 if len(prices) < 7 else 0.0
            return pd.DataFrame([[a, -a]], index=prices.index[-1:], columns=["A", "B"])

    closing = BacktestEngine(Close(), source, rebalanceBand=0.05)
    closing.run(dates[2], dates[-1])
    assert np.allclose(closing.positions["A"], [0.10] * 3 + [0.03] * 2 + [0.0] * 5)


def _rebalancing_search(ohlcv, start, candidates, settings, periodsPerYear, title, filename, **engineKwargs):
    # Backtests every strategy of `candidates` (name -> a function making a
    # fresh one) with each trading rule of `settings` (label -> engine
    # settings, e.g. {"rebalanceEvery": 5} or {"rebalanceBand": 0.01}), from
    # `start`. Prints and plots the metrics, with the return before costs
    # (`annualizedGrossMean`), and returns them.
    results, metrics = {}, {}
    for name, makeStrategy in candidates.items():
        for label, rule in settings.items():
            engine = BacktestEngine(makeStrategy(), InMemoryDataSource(ohlcv), periodsPerYear=periodsPerYear,
                                    **rule, **engineKwargs)
            every = rule.get("rebalanceEvery", 1)
            key = f"{name}, {label}"
            results[key] = engine.run(start, ohlcv.index[-1])
            m = engine.metrics["value"]
            m["annualizedCost"] = engine.costs.sum() / len(engine.costs) * periodsPerYear
            m["annualizedGrossMean"] = m["annualizedMean"] + m["annualizedCost"]
            metrics[key] = m
            assert all(c >= 0 for c in engine.costs)
            # trades only on the rebalancing bars
            traded = np.where(engine.costs > 0)[0]
            assert (traded % every == 0).all()

    metrics = pd.DataFrame(metrics)
    best = metrics.loc["sharpeRatio"].idxmax()
    rows = ["totalReturn", "sharpeRatio", "annualizedGrossMean", "annualizedCost", "annualizedMean",
            "annualizedTurnover", "maxDrawdown"]
    print(f"\n{metrics.loc[rows].T.to_string(float_format=lambda x: f'{x:.3f}')}\nbest (Sharpe): {best}")
    first = next(iter(results.values()))
    _plot_window_search(first.index, results, metrics.loc[rows], best, f"{title} (best by Sharpe: {best})",
                        filename, legendPrefix="", xlabel="Strategy, rebalancing frequency")
    return metrics


def _every(frequencies: dict) -> dict:
    """{"1d": 1, ...} -> {"every 1d": {"rebalanceEvery": 1}, ...}"""
    return {f"every {label}": {"rebalanceEvery": n} for label, n in frequencies.items()}


BANDS = {f"band {band:.1%}": {"rebalanceBand": band} for band in [0.0, 0.005, 0.01, 0.02, 0.05]}


@pytest.mark.realdata
def test_rebalanceBand_crypto_pairTrading_kraken():
    # Pair trading on Kraken daily bars, as in test_crypto_pairTrading_kraken
    # (4 pairs, 10% vol target): the vol target resizes the book a little
    # every day. With a band, a leg is only traded when its target moved by
    # more than the band (closing a position is always traded).
    settings = CRYPTO_INTERVALS["1d"]
    ohlcv, start = _kraken_prices("1d", settings["formation"])
    pairs = ohlcv.columns.get_level_values(0).unique()

    def pairTrading():
        strategy = PairTrading(numberPair=4, formationDays=settings["formation"], refitDays=settings["refit"],
                               groups=pd.Series("crypto", index=pairs))
        return VolatilityTarget(strategy, targetVol=0.10, lookback=60, maxGross=4.0, periodsPerYear=365)

    _rebalancing_search(ohlcv, start, {"pairs 4": pairTrading}, BANDS, 365,
                        "Rebalancing band, crypto pair trading, Kraken daily",
                        "rebalanceBand_crypto_pairTrading_kraken_1d.png", **CRYPTO_ENGINE)


@pytest.mark.realdata
def test_rebalanceBand_sp500_pairTrading_ibkr_4h():
    # Pair trading on the S&P 500 with IBKR 4h bars of 2025, as in
    # test_backtestEngine_on_sp500_pairTrading_ibkr (3 pairs re-selected every
    # quarter, 10% vol target, traded April-December), with each band.
    constituents = DataLoader.getSP500Constituents()
    ohlcv = IBKRLoader().getOHLCV("2025-01-01", "2026-01-01", list(constituents["Symbol"]), "4h")
    ohlcv = _liquid_universe(ohlcv, "2025-04-01")
    start = ohlcv.index[np.searchsorted(ohlcv.index, pd.Timestamp("2025-04-01"))]
    quarter = 63 * 2
    groups = constituents.set_index("Symbol")["SubIndustry"]

    def pairTrading():
        strategy = PairTrading(numberPair=3, formationDays=quarter, refitDays=quarter, groups=groups)
        return VolatilityTarget(strategy, targetVol=0.10, lookback=60, maxGross=4.0, periodsPerYear=504)

    _rebalancing_search(ohlcv, start, {"pairs 3": pairTrading}, BANDS, 504,
                        "Rebalancing band, S&P 500 pair trading, IBKR 4h, Apr-Dec 2025",
                        "rebalanceBand_sp500_pairTrading_ibkr_4h.png", transactionCostRate=0.0005)


@pytest.mark.realdata
def test_rebalancing_crypto_daily_kraken():
    # The best daily crypto strategies of test_crypto_momentum_reversal_search_kraken
    # (Kraken API, last ~2 years), traded every day, week or month instead
    # of every day only. The signal is the same; only the trading changes.
    ohlcv, start = _kraken_prices("1d", CRYPTO_INTERVALS["1d"]["momentumWarmup"])
    candidates = {
        "momentum 1w": lambda: CrossSectionalMomentumStrategy(0.25, skipMonths=0, lookbackMonth=1, tradingDayPerMonth=7),
        "momentum 6-1m": lambda: CrossSectionalMomentumStrategy(0.25, skipMonths=1, lookbackMonth=6, tradingDayPerMonth=30),
        "reversal 1m": lambda: CrossSectionalReversalStrategy(0.25, lookbackMonth=1, tradingDayPerMonth=30),
    }
    _rebalancing_search(ohlcv, start, candidates, _every({"1d": 1, "7d": 7, "30d": 30}), 365,
                        "Crypto rebalancing, Kraken daily", "rebalancing_crypto_kraken_1d.png", **CRYPTO_ENGINE)


@pytest.mark.realdata
def test_rebalancing_crypto_hourly_kraken_history():
    # The best hourly crypto strategies of the 1h-history run (a year of
    # Kraken archive hourly bars): their signal every hour, traded every
    # hour, day or week.
    ohlcv, start = _kraken_prices("1h-history", CRYPTO_INTERVALS["1h-history"]["momentumWarmup"])
    candidates = {
        "reversal 6h": lambda: CrossSectionalReversalStrategy(0.25, lookbackMonth=1, tradingDayPerMonth=6),
        "momentum 1w-1d": lambda: CrossSectionalMomentumStrategy(0.25, skipMonths=1, lookbackMonth=7, tradingDayPerMonth=24),
    }
    _rebalancing_search(ohlcv, start, candidates, _every({"1h": 1, "1d": 24, "1w": 168}), 365 * 24,
                        "Crypto rebalancing, Kraken archive hourly, Oct 2025-Sep 2026",
                        "rebalancing_crypto_kraken_1h-history.png", **CRYPTO_ENGINE)


@pytest.mark.realdata
def test_rebalancing_residualReversal_sp500_ibkr():
    # Residual reversal on the S&P 500 (as in
    # test_residualReversal_lookback_search_sp500_ibkr, IBKR daily,
    # 2015-2024), traded every day, week (5 days) or month (21 days).
    constituents = DataLoader.getSP500Constituents()
    ohlcv = IBKRLoader().getOHLCV("2010-01-01", "2025-01-01", list(constituents["Symbol"]), "1d")
    ohlcv = _liquid_universe(ohlcv, "2015-01-01")
    start = ohlcv.index[np.searchsorted(ohlcv.index, pd.Timestamp("2015-01-01"))]
    candidates = {
        "residual 1m": lambda: ResidualReversalStrategy(lookbackDays=21),
        "residual 3m": lambda: ResidualReversalStrategy(lookbackDays=63),
    }
    _rebalancing_search(ohlcv, start, candidates, _every({"1d": 1, "5d": 5, "21d": 21}), 252,
                        "Residual reversal rebalancing, S&P 500 IBKR daily 2015-2024",
                        "rebalancing_residualReversal_sp500_ibkr.png", transactionCostRate=0.0005)


@pytest.mark.realdata
def test_rebalancing_momentum_sp500_ibkr():
    # Cross-sectional momentum on the S&P 500 (IBKR daily, 2015-2024): long
    # the top 10%, short the bottom 10%, traded every day, week or month.
    constituents = DataLoader.getSP500Constituents()
    ohlcv = IBKRLoader().getOHLCV("2010-01-01", "2025-01-01", list(constituents["Symbol"]), "1d")
    ohlcv = _liquid_universe(ohlcv, "2015-01-01")
    start = ohlcv.index[np.searchsorted(ohlcv.index, pd.Timestamp("2015-01-01"))]
    candidates = {
        "momentum 12-1m": lambda: CrossSectionalMomentumStrategy(skipMonths=1, lookbackMonth=12),
        "momentum 6-1m": lambda: CrossSectionalMomentumStrategy(skipMonths=1, lookbackMonth=6),
    }
    _rebalancing_search(ohlcv, start, candidates, _every({"1d": 1, "5d": 5, "21d": 21}), 252,
                        "Momentum rebalancing, S&P 500 IBKR daily 2015-2024",
                        "rebalancing_momentum_sp500_ibkr.png", transactionCostRate=0.0005)


@pytest.mark.realdata
def test_rebalancing_meanReversion_ibkr():
    # OU mean reversion (120-day window) on the 12 large caps, IBKR daily,
    # 2023-2024 only: the OU fit is redone per ticker on every bar. Its
    # inverse-vol weights move a little every day, so bands are tried too.
    ohlcv = IBKRLoader().getOHLCV("2010-01-01", "2025-01-01", REAL_TICKERS, "1d").dropna()
    start = ohlcv.index[np.searchsorted(ohlcv.index, pd.Timestamp("2023-01-01"))]
    ohlcv = ohlcv.loc[ohlcv.index >= ohlcv.index[ohlcv.index.get_loc(start) - 130]]
    settings = {**_every({"1d": 1, "5d": 5, "21d": 21}),
                "band 1%": {"rebalanceBand": 0.01}, "band 2%": {"rebalanceBand": 0.02}}
    _rebalancing_search(ohlcv, start, {"OU 120d": lambda: MeanReversion(window=120)}, settings, 252,
                        "Mean reversion rebalancing, 12 large caps IBKR daily 2023-2024",
                        "rebalancing_meanReversion_ibkr.png", transactionCostRate=0.0005)


class _PointInTimeUniverse(Strategy):
    """Runs `inner` each day on that day's `members` only (a date x ticker
    boolean table, known as of that date) with full price history over the
    last `historyBars`; every other ticker is held at 0."""

    def __init__(self, inner: Strategy, members: pd.DataFrame, historyBars: int):
        self.inner, self.members, self.historyBars = inner, members, historyBars

    def _universe(self, prices):
        members = self.members.loc[prices.index[-1]].reindex(prices.columns, fill_value=False)
        recent = prices.loc[:, members.to_numpy()].tail(self.historyBars)
        return recent.loc[:, recent.notna().all() & (len(recent) == self.historyBars)]

    def signal(self, prices):
        return self.inner.cachedSignal(self._universe(prices))

    def positions(self, prices):
        recent = self._universe(prices)
        if recent.shape[1] <= 10:  # too few for a cross-section
            return prices.iloc[[-1]].notna() * 0.0
        return self.inner.positions(recent).iloc[[-1]].reindex(columns=prices.columns, fill_value=0.0)


# stablecoins, fiat, gold and wrapped coins: not a cross-section of crypto
KRAKEN_NOT_CRYPTO = {
    "USDT", "USDC", "DAI", "TUSD", "PYUSD", "USDG", "USDD", "FDUSD", "USDE", "USDS", "RLUSD", "UST", "USDQ",
    "USDR", "EURT", "EURQ", "EURR", "EUROP", "EURC", "AUSD", "USD1", "EUR", "GBP", "AUD", "CAD", "CHF", "JPY",
    "PAXG", "XAUT", "TBTC", "WBTC", "CBETH", "ETH2", "ETH2.S", "LSETH", "METH", "WETH", "CBBTC",
}


def _kraken_history_daily(start, end, top=30):
    """Daily OHLCV of every USD pair of Kraken's archive (dead ones included,
    so no survivorship bias), and each day's universe: the `top` pairs by
    median dollar volume over the 30 days up to that day, among those
    moving like crypto (30-day daily volatility over 1%)."""
    loader = KrakenHistoryLoader()
    remote = _RemoteFile([loader.URL + part for part in loader.PARTS])
    with zipfile.ZipFile(io.BufferedReader(remote, buffer_size=1 << 20)) as archive:
        names = [n[:-len("_1440.csv")] for n in archive.namelist() if n.endswith("USD_1440.csv")]
    pairs = sorted(p for p in names if p[:-3] not in KRAKEN_NOT_CRYPTO)
    # the whole archive, so every pair has data; sliced below
    ohlcv = loader.getOHLCV("2013-01-01", "2026-10-01", pairs, "1d")
    ohlcv = ohlcv[(ohlcv.index >= pd.Timestamp(start) - pd.Timedelta(days=400)) & (ohlcv.index < pd.Timestamp(end))]
    close = ohlcv.xs("Close", axis=1, level=1)
    dollarVolume = (close * ohlcv.xs("Volume", axis=1, level=1)).rolling(30, min_periods=30).median()
    volatility = close.pct_change(fill_method=None).rolling(30, min_periods=30).std()
    eligible = dollarVolume.where(volatility > 0.01)
    members = eligible.rank(axis=1, ascending=False, method="first") <= top
    ever = members.columns[members.any()]
    print(f"\n{len(pairs)} USD pairs in the archive, {len(ever)} ever in the top {top}")
    return ohlcv[ever], members[ever]


@pytest.mark.realdata
def test_crypto_reversal_full_history_kraken():
    # Crypto 1-month reversal (long the worst 25% of the last 30 days, short
    # the best 25%), traded every 30 days, on Kraken's daily archive
    # 2018-Sep 2026 with a point-in-time top-30 universe. Three rebalancing
    # days (offset 0, 10, 20 days) check it isn't luck of the calendar; daily
    # trading and 1-month momentum are there for comparison.
    ohlcv, members = _kraken_history_daily("2018-01-01", "2026-10-01")
    close = ohlcv.xs("Close", axis=1, level=1)
    first = close.index[np.searchsorted(close.index, pd.Timestamp("2018-01-01"))]
    history = 32

    def reversal():
        return _PointInTimeUniverse(CrossSectionalReversalStrategy(0.25, lookbackMonth=1, tradingDayPerMonth=30),
                                    members, history)

    def momentum():
        return _PointInTimeUniverse(CrossSectionalMomentumStrategy(0.25, skipMonths=0, lookbackMonth=1,
                                                                   tradingDayPerMonth=30), members, history)

    runs = {f"reversal 1m, every 30d, offset {o}d": (reversal, 30, o) for o in (0, 10, 20)}
    runs["reversal 1m, every 1d"] = (reversal, 1, 0)
    runs["momentum 1m, every 30d"] = (momentum, 30, 0)
    results, metrics = {}, {}
    for label, (make, every, offset) in runs.items():
        start = close.index[close.index.get_loc(first) + offset]
        engine = BacktestEngine(make(), InMemoryDataSource(ohlcv), periodsPerYear=365, rebalanceEvery=every,
                                **CRYPTO_ENGINE)
        results[label] = engine.run(start, close.index[-1]).loc[close.index[close.index.get_loc(first) + 20]:]
        m = engine.metrics["value"]
        m["annualizedCost"] = engine.costs.sum() / len(engine.costs) * 365
        m["annualizedGrossMean"] = m["annualizedMean"] + m["annualizedCost"]
        years = len(engine.result) / 365
        m["tStat"] = m["sharpeRatio"] * np.sqrt(years)
        metrics[label] = m
        assert all(c >= 0 for c in engine.costs)

    metrics = pd.DataFrame(metrics)
    rows = ["totalReturn", "sharpeRatio", "tStat", "annualizedGrossMean", "annualizedCost", "annualizedMean",
            "annualizedTurnover", "maxDrawdown"]
    yearly = pd.DataFrame({k: (1 + r).groupby(r.index.year).prod() - 1 for k, r in results.items()})
    print(f"\n{metrics.loc[rows].T.to_string(float_format=lambda x: f'{x:.3f}')}")
    print(f"\nreturn by year (same period for all):\n{yearly.to_string(float_format=lambda x: f'{x:+.1%}')}")
    best = metrics.loc["sharpeRatio"].idxmax()
    first_result = next(iter(results.values()))
    _plot_window_search(first_result.index, results, metrics.loc[rows], best,
                        f"Crypto reversal 1m, Kraken archive daily 2018-2026, top-30 universe (best: {best})",
                        "reversal_fullHistory_crypto_kraken.png", legendPrefix="", xlabel="Strategy")


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


def test_pairTrading_formationWithoutCandidates_staysInCash():
    # Two tickers in different groups: no candidate pair at all. The
    # formation keeps an empty pair table and the book stays flat.
    dates = pd.bdate_range("2024-01-01", periods=60)
    prices = pd.DataFrame({"A": np.linspace(10, 20, 60), "B": np.linspace(30, 25, 60)}, index=dates)
    strategy = PairTrading(numberPair=2, formationDays=40, refitDays=40, groups=pd.Series({"A": "x", "B": "y"}))
    positions = strategy.positions(prices)
    assert strategy.pairs.empty and {"a", "b", "stable"} <= set(strategy.pairs.columns)
    assert (positions.iloc[-1] == 0).all()
    # per-ticker score, so a combiner can blend it with other strategies
    assert list(strategy.score(prices).columns) == ["A", "B"]


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


@pytest.mark.realdata
@pytest.mark.parametrize("numberPair", [1,2])
def test_backtestEngine_on_real_data_pairTrading(numberPair):
    # PAIR_TICKERS, candidates paired inside the PairFinding clusters.
    ohlcv = DataLoader.getOHLCV("2013-01-01", "2025-01-01", PAIR_TICKERS, "1d").dropna()
    _backtest_pair_trading(
        ohlcv, PairTrading(numberPair=numberPair, formationDays=5 * 252, refitDays=2 * 252),
        f"OU pair trading on {len(PAIR_TICKERS)} large-caps, {numberPair} pairs re-selected every 2 years, 2020-2024",
        f"backtestEngine_pairTrading_{numberPair}pairs.png",
    )


@pytest.mark.realdata
@pytest.mark.parametrize("numberPair", [1,2])
def test_backtestEngine_on_real_data_pairTrading_ibkr(numberPair):
    # PAIR_TICKERS, candidates paired inside the PairFinding clusters.
    ibkr = IBKRLoader()
    ohlcv = ibkr.getOHLCV("2020-01-01", "2025-01-01", PAIR_TICKERS, "30m").dropna()
    _backtest_pair_trading(
        ohlcv, PairTrading(numberPair=numberPair, formationDays=5 * 252, refitDays=2 * 252),
        f"OU pair trading on {len(PAIR_TICKERS)} large-caps, {numberPair} pairs re-selected every 2 years, 2020-2024",
        f"backtestEngine_pairTrading_{numberPair}pairs.png",
    )

@pytest.mark.realdata
@pytest.mark.parametrize("numberPair", [30, 50])
def test_backtestEngine_on_sp500_pairTrading(numberPair):
    # Yahoo Finance daily prices, 2010-2024: traded on 2015-2024, 5
    # formations on the 5 years before, the first on 2010-2014.
    constituents = DataLoader.getSP500Constituents()
    ohlcv = DataLoader.getOHLCV("2010-01-01", "2025-01-01", list(constituents["Symbol"]), "1d")
    _sp500_pair_trading(
        ohlcv, constituents, numberPair, start="2015-01-01", formationDays=5 * 252, refitDays=2 * 252,
        title=f"{numberPair} pairs re-selected every 2 years on the 5 years before, 10% vol target, 2015-2024",
        filename=f"backtestEngine_pairTrading_sp500_{numberPair}pairs.png",
    )


@pytest.mark.realdata
@pytest.mark.parametrize("numberPair", [1, 2, 3])
def test_backtestEngine_on_sp500_pairTrading_ibkr(numberPair):
    # IBKR 4h bars of 2025 (2 a day in regular hours: 9:30-13:30 and
    # 13:30-16:00 New York; intraday bars are split adjusted, not dividend
    # adjusted). Traded from April to December, pairs re-selected every
    # quarter on the quarter before, the first on January-March.
    # TWS or IB Gateway must be running for the first run: about 1 hour to
    # download the ~500 tickers (one request each, ~7 s), each ticker cached
    # as soon as it is downloaded; then ~5 min for the 3 backtests.
    constituents = DataLoader.getSP500Constituents()
    ohlcv = IBKRLoader().getOHLCV("2025-01-01", "2026-01-01", list(constituents["Symbol"]), "4h")
    quarter = 63 * 2  # trading days in a quarter, 2 bars a day
    _sp500_pair_trading(
        ohlcv, constituents, numberPair, start="2025-04-01", formationDays=quarter, refitDays=quarter,
        barsPerDay=2,
        title=f"IBKR 4h bars, {numberPair} pairs re-selected every quarter on the quarter before, "
              f"10% vol target, Apr-Dec 2025",
        filename=f"backtestEngine_pairTrading_sp500_{numberPair}pairs_ibkr4h.png",
    )


def _sp500_pair_trading(ohlcv, constituents, numberPair, start, formationDays, refitDays, title, filename,
                        barsPerDay=1):
    # Current S&P 500 members with a full history over the data (survivorship
    # bias: companies dropped from the index are missing), candidates paired
    # inside each GICS sub-industry: ~1400 candidate pairs instead of ~35.
    # Traded from `start`; formationDays and refitDays count bars.
    # Stocks trading under $5M a day before `start` are left out: their early
    # history is not the S&P 500 company (e.g. a pre-merger listing), and the
    # square-root impact makes them very expensive. Only data before `start`
    # is used for this, so it does not look ahead into the backtest.
    # The book is scaled to 10% expected annual volatility, gross capped at 4.
    # With barsPerDay bars a day, the volatility target and the metrics are
    # annualized over 252 * barsPerDay bars.
    ohlcv = _liquid_universe(ohlcv, start)
    universe = list(ohlcv.columns.get_level_values(0).unique())

    periodsPerYear = 252 * barsPerDay
    groups = constituents.set_index("Symbol")["SubIndustry"]
    pairTrading = PairTrading(numberPair=numberPair, formationDays=formationDays, refitDays=refitDays,
                              groups=groups)
    strategy = VolatilityTarget(pairTrading, targetVol=0.10, lookback=60, maxGross=4.0,
                                periodsPerYear=periodsPerYear)
    _backtest_pair_trading(
        ohlcv, strategy,
        f"OU pair trading on {len(universe)} S&P 500 stocks by sub-industry,\n{title}",
        filename, start=start, periodsPerYear=periodsPerYear,
    )
    # every pair is two stocks of the same sub-industry
    for pairs in pairTrading.formations.values():
        assert (groups[pairs["a"]].values == groups[pairs["b"]].values).all()


def _liquid_universe(ohlcv: pd.DataFrame, before) -> pd.DataFrame:
    """Tickers with a price on every bar and a median dollar volume over
    $5M a day before `before` (only data before it, no look-ahead)."""
    close = ohlcv.xs("Close", axis=1, level=1)
    volume = ohlcv.xs("Volume", axis=1, level=1)
    dollarVolume = (close * volume)[close.index < pd.Timestamp(before)]
    dailyDollarVolume = dollarVolume.groupby(dollarVolume.index.normalize()).sum(min_count=1).median()
    universe = [t for t in close.columns if close[t].notna().all() and dailyDollarVolume[t] > 5e6]
    print(f"\nuniverse: {len(universe)} of {len(close.columns)} tickers")
    assert len(universe) > 350
    return ohlcv[universe]


def _backtest_pair_trading(ohlcv, strategy, title, filename, start="2023-01-01", periodsPerYear=252,
                           transactionCostRate=0.0005, **engineKwargs):
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

    engine = BacktestEngine(strategy, InMemoryDataSource(ohlcv), transactionCostRate=transactionCostRate,
                            periodsPerYear=periodsPerYear, **engineKwargs)
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


@pytest.mark.realdata
def test_famaFrenchData_on_real_data():
    # Daily 5 factors + momentum, in percent: complete, sorted, plausible
    # magnitudes, and the market factor is the market (^GSPC daily returns).
    ff = DataLoader.getFamaFrenchData(FF_START, FF_END)

    assert list(ff.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF", "Mom"]
    assert not ff.isna().any().any()
    assert ff.index.is_monotonic_increasing and ff.index.is_unique
    assert ff.index[0] >= pd.Timestamp(FF_START) and ff.index[-1] <= pd.Timestamp(FF_END)

    # percent per day: the market moves about 1% a day, the risk-free rate
    # is a few hundredths of a percent and never negative
    assert 0.5 < ff["Mkt-RF"].std() < 2.0
    assert ((ff["RF"] >= 0) & (ff["RF"] < 0.1)).all()
    assert (ff[FF_FACTORS].abs() < 25).all().all()

    spx = DataLoader.getData(FF_START, "2026-08-02", ["^GSPC"], "1d")["Close"].pct_change() * 100
    market = pd.concat([ff["Mkt-RF"] + ff["RF"], spx], axis=1, join="inner").dropna()
    assert len(market) > 0.95 * len(ff)
    assert market.corr().iloc[0, 1] > 0.95


def test_famaFrenchRegression_recovers_synthetic_loadings(monkeypatch):
    # Synthetic factors and a stock built from known loadings: the regression
    # must give them back. DataLoader is replaced, so no data is downloaded.
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2015-01-01", periods=1500)
    stds = {"Mkt-RF": 1.0, "SMB": 0.6, "HML": 0.7, "RMW": 0.5, "CMA": 0.4, "Mom": 0.9}
    ff = pd.DataFrame({f: rng.normal(0, sd, len(dates)) for f, sd in stds.items()}, index=dates)
    ff["RF"] = 0.01
    loadings = {"const": 0.02, "Mkt-RF": 1.2, "SMB": 0.5, "HML": -0.3, "RMW": 0.2, "CMA": -0.1, "Mom": 0.4}

    # famaFrenchRegression regresses 100 * log returns minus RF on the factors
    excess = loadings["const"] + sum(loadings[f] * ff[f] for f in FF_FACTORS) + rng.normal(0, 0.3, len(dates))
    logReturns = (excess + ff["RF"]) / 100
    logReturns.iloc[0] = 0.0  # first return is dropped by logReturn()
    prices = pd.DataFrame({"Adj Close": 100 * np.exp(logReturns.cumsum())}, index=dates)

    monkeypatch.setattr(DataLoader, "getData", lambda *args, **kwargs: prices.copy())
    monkeypatch.setattr(DataLoader, "getFamaFrenchData", lambda *args, **kwargs: ff)
    model = strategies.famaFrenchRegression(["SYNTH"], "2015-01-01", "2021-01-01")

    assert list(model.params.index) == ["const"] + FF_FACTORS
    assert model.nobs == len(dates) - 1
    for name, value in loadings.items():
        assert abs(model.params[name] - value) < 0.03, name
    assert model.rsquared > 0.9


@pytest.mark.realdata
def test_famaFrenchRegression_on_real_data():
    # AAPL on the 5 factors + momentum, 2011-2026: a large-cap growth stock
    # should load about 1 on the market, negatively on value (HML), and the
    # factors should explain a good part of its daily moves.
    model = strategies.famaFrenchRegression(["AAPL"], FF_START, FF_END)
    ff = DataLoader.getFamaFrenchData(FF_START, FF_END)
    print(f"\n{model.summary()}")

    assert list(model.params.index) == ["const"] + FF_FACTORS
    assert model.nobs > 0.95 * len(ff)
    assert 0.8 < model.params["Mkt-RF"] < 1.6
    assert model.params["HML"] < 0
    assert model.rsquared > 0.3
    assert abs(model.params["const"]) < 0.1  # daily alpha, in percent

    _plot_fama_french(ff, model, "AAPL", "famaFrench_realData.png")


@pytest.mark.realdata
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


@pytest.mark.realdata
def test_backtestEngine_on_real_data_invVolCombiner():
    strategy = _backtest_calibrated_combiner(InvVol, "inverse-vol", "backtestEngine_invVol_realData.png")

    vol = strategy.strategyMetrics.loc["activeVolatility"]
    traded = vol > 0
    weights = strategy.weights()
    # inverse-vol: weight * vol is the same constant for every strategy that traded
    assert np.allclose(weights[traded] * vol[traded], (weights[traded] * vol[traded]).iloc[0])


@pytest.mark.realdata
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


# Variants of the 4h combiner test. base: 100 calibration + 100 evaluated
# days. Mean reversion rarely enters there (its ADF test must pass at 1% on
# 240 bars), so the combiners mostly scale the momentum book: longWindows
# gives it a year each, looseADF lets it enter at 5%, pairTrading blends
# momentum with pair trading instead.
COMBINER_VARIANTS = {
    "base": {"days": 100, "second": "meanReversion", "pValueMax": 0.01},
    "longWindows": {"days": 252, "second": "meanReversion", "pValueMax": 0.01},
    "looseADF": {"days": 100, "second": "meanReversion", "pValueMax": 0.05},
    "pairTrading": {"days": 252, "second": "pairTrading"},
}


@pytest.mark.realdata
@pytest.mark.parametrize("variant", list(COMBINER_VARIANTS))
@pytest.mark.parametrize("combinerCls", [Uniform, InvVol, RiskParity], ids=["uniform", "invVol", "riskParity"])
def test_portfolioCombiner_on_ibkr_4h(combinerCls, variant):
    # The three combiners on IBKR 4h bars (2 a day), blending cross-sectional
    # 12-1 momentum with OU mean reversion (or pair trading), the daily
    # settings converted to bars: 21 * 2 bars a month, a 120-day OU window
    # of 240 bars. The calibrated combiners weight the strategies on the
    # `days` days before the evaluated last `days` days of 2022; Uniform is
    # scored on the same last `days` days. Pair trading: the 66 pairs of
    # REAL_TICKERS are candidates, re-selected every quarter on the year before.
    settings = COMBINER_VARIANTS[variant]
    barsPerDay = 2
    ohlcv = IBKRLoader().getOHLCV("2015-01-01", "2023-01-01", REAL_TICKERS, "4h").dropna()
    dates = ohlcv.index
    n = settings["days"] * barsPerDay
    if settings["second"] == "meanReversion":
        second = MeanReversion(window=120 * barsPerDay, pValueMax=settings["pValueMax"])
    else:
        second = PairTrading(numberPair=3, formationDays=252 * barsPerDay, refitDays=63 * barsPerDay,
                             groups=pd.Series("large-caps", index=REAL_TICKERS))
    strategies = [
        CrossSectionalMomentumStrategy(skipMonths=1, lookbackMonth=12, tradingDayPerMonth=21 * barsPerDay),
        second,
    ]
    periodsPerYear = 252 * barsPerDay
    if combinerCls is Uniform:
        strategy = Uniform(strategies)
    else:
        strategy = combinerCls(strategies, dates[-2 * n], dates[-n - 1],
                               transactionCostRate=0.0005, periodsPerYear=periodsPerYear)
    engine = _backtest_combiner(
        strategy, ohlcv, n, periodsPerYear,
        f"BacktestEngine — {combinerCls.__name__} momentum + {settings['second']} ({variant}) on "
        f"{len(REAL_TICKERS)} large-caps, IBKR 4h bars, last {settings['days']} days of 2022",
        f"backtestEngine_{combinerCls.__name__}_{variant}_ibkr4h.png",
    )
    metrics = engine.metrics["value"]
    weights = ", ".join(f"{k} {v:.1%}" for k, v in strategy.weights().items())
    print(f"\nRESULT {variant} {combinerCls.__name__}: weights {weights}; total return "
          f"{metrics['totalReturn']:.2%}, Sharpe {metrics['sharpeRatio']:.2f}, "
          f"vol {metrics['annualizedVolatility']:.2%}, max drawdown {metrics['maxDrawdown']:.2%}")
    if isinstance(strategy, CalibratedCombiner):
        print(f"calibration activeDays: {strategy.strategyMetrics.loc['activeDays'].round(3).to_dict()}")

    if combinerCls is InvVol:
        vol = strategy.strategyMetrics.loc["activeVolatility"]
        traded = vol > 0
        weights = strategy.weights()
        assert np.allclose(weights[traded] * vol[traded], (weights[traded] * vol[traded]).iloc[0])
    if combinerCls is RiskParity:
        shares = strategy.riskContributions[strategy.weights() > 0]
        assert np.allclose(shares, 1 / len(shares), atol=1e-4)


@pytest.mark.realdata
@pytest.mark.parametrize("rebalance", ["4h", "1d", "1w"])
@pytest.mark.parametrize("combinerCls", [Uniform, InvVol, RiskParity], ids=["uniform", "invVol", "riskParity"])
def test_portfolioCombiner_on_sp500_ibkr_4h(combinerCls, rebalance):
    # The three combiners on the S&P 500 with IBKR 4h bars of 2025 (2 a day),
    # blending cross-sectional momentum and pair trading by sub-industry.
    # One year of data, so momentum is 3-1 (3 months, skipping the last
    # one) instead of 12-1. Mean reversion is left out: its OU fit per
    # ticker per bar would take ~8 s a bar on ~500 tickers. The calibrated
    # combiners weight the strategies on July-September and are scored on
    # October-December, like Uniform. Pair trading: 10 pairs re-selected
    # every month on the quarter before. The book trades every 4h bar, every
    # day (2 bars) or every week (10 bars), in the calibration too.
    barsPerDay = 2
    every = {"4h": 1, "1d": barsPerDay, "1w": 5 * barsPerDay}[rebalance]
    constituents = DataLoader.getSP500Constituents()
    ohlcv = IBKRLoader().getOHLCV("2025-01-01", "2026-01-01", list(constituents["Symbol"]), "4h")
    dates = ohlcv.index
    n = 63 * barsPerDay
    ohlcv = _liquid_universe(ohlcv, dates[-2 * n])
    strategies = [
        CrossSectionalMomentumStrategy(skipMonths=1, lookbackMonth=3, tradingDayPerMonth=21 * barsPerDay),
        PairTrading(numberPair=10, formationDays=63 * barsPerDay, refitDays=21 * barsPerDay,
                    groups=constituents.set_index("Symbol")["SubIndustry"]),
    ]
    periodsPerYear = 252 * barsPerDay
    if combinerCls is Uniform:
        strategy = Uniform(strategies)
    else:
        strategy = combinerCls(strategies, dates[-2 * n], dates[-n - 1],
                               transactionCostRate=0.0005, periodsPerYear=periodsPerYear, rebalanceEvery=every)
    engine = _backtest_combiner(
        strategy, ohlcv, n, periodsPerYear,
        f"BacktestEngine — {combinerCls.__name__} 3-1 momentum + pair trading on "
        f"{len(ohlcv.columns.get_level_values(0).unique())} S&P 500 stocks, IBKR 4h bars, Oct-Dec 2025, "
        f"traded every {rebalance}",
        f"backtestEngine_{combinerCls.__name__}_sp500_ibkr4h_every{rebalance}.png",
        rebalanceEvery=every,
    )
    metrics = engine.metrics["value"]
    weights = ", ".join(f"{k} {v:.1%}" for k, v in strategy.weights().items())
    print(f"\nRESULT sp500 {combinerCls.__name__} every {rebalance}: weights {weights}; "
          f"costs {engine.costs.sum():.2%}, total return "
          f"{metrics['totalReturn']:.2%}, Sharpe {metrics['sharpeRatio']:.2f}, "
          f"vol {metrics['annualizedVolatility']:.2%}, max drawdown {metrics['maxDrawdown']:.2%}")
    if isinstance(strategy, CalibratedCombiner):
        print(f"calibration activeDays: {strategy.strategyMetrics.loc['activeDays'].round(3).to_dict()}")
        print(f"calibration totalReturn: {strategy.strategyMetrics.loc['totalReturn'].round(4).to_dict()}")

    if combinerCls is InvVol:
        vol = strategy.strategyMetrics.loc["activeVolatility"]
        traded = vol > 0
        weights = strategy.weights()
        assert np.allclose(weights[traded] * vol[traded], (weights[traded] * vol[traded]).iloc[0])
    if combinerCls is RiskParity:
        shares = strategy.riskContributions[strategy.weights() > 0]
        assert np.allclose(shares, 1 / len(shares), atol=1e-4)


def _backtest_calibrated_combiner(combiner_cls, label, filename):
    # Blend of cross-sectional 12-1 momentum and OU mean reversion. Weights
    # come from backtesting each sub-strategy alone on a calibration window
    # that ends strictly before the evaluated backtest starts, so the
    # combination never sees the returns it is scored on. Both windows are
    # 100 days because MeanReversion recalibrates per step.
    ibkr = IBKRLoader()
    ohlcv = ibkr.getOHLCV("2023-01-01", "2026-01-01", REAL_TICKERS, "1h").dropna()
    dates = ohlcv.index
    strategy = combiner_cls(
        [CrossSectionalMomentumStrategy(skipMonths=1, lookbackMonth=12), MeanReversion(window=120)],
        dates[-200], dates[-101], transactionCostRate=0.0005,
    )
    _backtest_combiner(
        strategy, ohlcv, 100, 252,
        f"BacktestEngine — {label} momentum + mean reversion on {len(REAL_TICKERS)} large-caps",
        filename,
    )
    return strategy


def _backtest_combiner(strategy, ohlcv, n, periodsPerYear, title, filename, **engineKwargs):
    # Backtests a combiner on the last n bars. A CalibratedCombiner is first calibrated on its
    # window, which must end before those n bars: trading inside it is refused.
    close = ohlcv.xs("Close", axis=1, level=1)
    dates = ohlcv.index
    warmup = len(dates) - n
    data_source = InMemoryDataSource(ohlcv)

    if isinstance(strategy, CalibratedCombiner):
        strategyMetrics = strategy.calibrate(data_source)
        assert list(strategyMetrics.columns) == strategy.strategyNames()
        assert len(strategy.strategyReturns) == np.searchsorted(dates, strategy.end, side="right") - \
            np.searchsorted(dates, strategy.start)
        assert strategy.end < dates[warmup]
        # trading inside the calibration window is refused
        with pytest.raises(ValueError, match="Look-ahead"):
            BacktestEngine(strategy, data_source).run(strategy.end, strategy.end)
    assert np.isclose(strategy.weights().sum(), 1.0)
    assert (strategy.weights() >= 0).all()

    engine = BacktestEngine(strategy, data_source, transactionCostRate=0.0005, periodsPerYear=periodsPerYear,
                            **engineKwargs)
    results = engine.run(dates[warmup], dates[-1])

    step_dates = dates[warmup:]
    assert len(results) == n
    assert len(engine.costs) == n
    assert len(engine.positions) == n
    assert all(c >= 0 for c in engine.costs)

    # each sub-book has gross <= 1 and the blend is not renormalized, so a
    # flat sub-strategy leaves its share in cash instead of levering the other
    gross = engine.positions.abs().sum(axis=1)
    assert (gross <= 1.0 + 1e-9).all()

    period_returns = close.pct_change()
    for i, date in enumerate(step_dates):
        expected_gross = float((engine.positions.iloc[i] * period_returns.loc[date]).sum())
        assert np.isclose(results.iloc[i], expected_gross - engine.costs.iloc[i])

    _plot_backtest(step_dates, results, engine.costs, engine.metrics, title, filename)
    return engine


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


def _plot_window_search(dates, results: dict, metrics: pd.DataFrame, best, title, filename,
                        legendPrefix="window=", xlabel="OU calibration window (days)"):
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
                    linewidth=2.5 if window == best else 1.2, label=f"{legendPrefix}{window}")
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
    fig.supxlabel(xlabel)

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


def _plot_fama_french(ff: pd.DataFrame, model, ticker, filename):
    # Left: growth of $1 in each factor portfolio. Right: the ticker's
    # loadings with their 95% confidence intervals, alpha annualized.
    os.makedirs(PLOT_DIR, exist_ok=True)
    fig, (ax_fac, ax_beta) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [1.4, 1]})

    for factor in FF_FACTORS:
        ax_fac.plot(ff.index, (1 + ff[factor] / 100).cumprod(), label=factor, linewidth=1)
    ax_fac.set_yscale("log")
    ax_fac.set_title("Fama-French factors: growth of $1 (log scale)")
    ax_fac.legend(loc="upper left", fontsize=9)

    betas = model.params[FF_FACTORS]
    errors = (model.conf_int().loc[FF_FACTORS, 1] - betas).values
    ax_beta.bar(FF_FACTORS, betas.values, yerr=errors, capsize=4,
                color=["tab:blue" if b >= 0 else "tab:red" for b in betas])
    ax_beta.axhline(0, color="black", linewidth=0.6)
    alpha = model.params["const"] * 252
    ax_beta.set_title(f"{ticker} loadings (95% CI), R² {model.rsquared:.2f}, "
                      f"alpha {alpha:.1f}%/yr (t = {model.tvalues['const']:.1f})")

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
    test_famaFrenchData_on_real_data()
    test_famaFrenchRegression_on_real_data()
    print("ok")
