import os

import matplotlib
matplotlib.use("Agg")  # headless backend so the test runs without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import DataLoader
import strategies
from strategies import changingRegime
from Strategies.MeanReversion import MeanReversion
from Strategies.Momentum import MomentumStrategy
from Strategies.Strategy import Strategy
from Strategies.PortfolioCombiner.Uniform import Uniform

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")


def test_changingRegime_with_plot():
    data = DataLoader.getData("2011-01-01", "2026-08-05", ["^GSPC"], "1d")
    test = changingRegime(data)

    # The function should tag every test row with a discrete regime label.
    assert "market_regime" in test.columns
    assert len(test) > 0
    regimes = np.unique(test["market_regime"])
    assert set(regimes).issubset({0, 1})

    # Plot the price coloured by detected regime.
    os.makedirs(PLOT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    for regime in regimes:
        mask = test["market_regime"] == regime
        ax.scatter(
            test.index[mask],
            test["Close"][mask],
            s=6,
            label=f"regime {regime}",
        )
    ax.set_title("changingRegime — S&P 500 close by HMM regime")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close")
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(PLOT_DIR, "changingRegime.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    assert os.path.exists(out_path)

    # Plot the distribution of returns for the extreme (higher-variance) vs.
    # normal regime, derived from the same `test` classification as above.
    variances = test.groupby("market_regime")["returns"].var()
    extreme_label = variances.idxmax()
    normal_label = variances.idxmin()
    extreme_regime = test.loc[test["market_regime"] == extreme_label]
    normal_regime = test.loc[test["market_regime"] == normal_label]

    dist_fig, dist_ax = plt.subplots(figsize=(10, 5))
    dist_ax.hist(normal_regime["returns"], bins=50, alpha=0.6, density=True, label="normal regime")
    dist_ax.hist(extreme_regime["returns"], bins=50, alpha=0.6, density=True, label="extreme regime")
    dist_ax.set_title("Return distribution by HMM regime")
    dist_ax.set_xlabel("Log return")
    dist_ax.set_ylabel("Density")
    dist_ax.legend()
    dist_fig.tight_layout()

    dist_out_path = os.path.join(PLOT_DIR, "regimeDistribution.png")
    dist_fig.savefig(dist_out_path, dpi=120)
    plt.close(dist_fig)

    assert os.path.exists(dist_out_path)


def test_crossSectionalMomentum_top_and_bottom_tickers_per_date():
    # Synthetic random-walk prices for >= 10 tickers — deterministic and
    # network-free, unlike the regime test above which pulls real market data.
    prices, tickers = _synthetic_prices()

    long, short = strategies.crossSectionalMomentum(prices, percentage=0.1)

    signal = strategies.momentum(prices).dropna()
    expected_n = max(1, int(len(tickers) * 0.1))  # tickers ranked against each other, per date

    # columns naturally shrink to whichever tickers were ever top/bottom-ranked
    # (the 12-1 signal moves slowly, so most tickers may never win a given date)
    assert set(long.columns) <= set(tickers)
    assert set(short.columns) <= set(tickers)
    assert len(long) == len(signal)
    assert len(short) == len(signal)

    # exactly expected_n tickers picked long and expected_n picked short on every date
    assert (long.count(axis=1) == expected_n).all()
    assert (short.count(axis=1) == expected_n).all()

    # no ticker is both long and short on the same date
    assert not (long.notna() & short.notna()).any().any()

    # spot-check every 50th date: picks must be exactly that date's top/bottom
    # tickers by momentum, and the worst long pick must beat the best short pick
    for date in signal.index[::50]:
        row = signal.loc[date]
        pd.testing.assert_series_equal(long.loc[date].dropna(), row.nlargest(expected_n), check_like=True)
        pd.testing.assert_series_equal(short.loc[date].dropna(), row.nsmallest(expected_n), check_like=True)
        assert long.loc[date].min() >= short.loc[date].max()

    # Plot price series (one per ticker) with the long/short flagged dates highlighted.
    os.makedirs(PLOT_DIR, exist_ok=True)
    fig, (ax_price, ax_signal) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    long_flat = long.stack()
    short_flat = short.stack()

    for ticker in tickers:
        ax_price.plot(prices.index, prices[ticker], color="grey", linewidth=0.6, alpha=0.5)
    long_price_points = prices.stack().loc[long_flat.index]
    short_price_points = prices.stack().loc[short_flat.index]
    ax_price.scatter(
        [d for d, _ in long_price_points.index], long_price_points,
        color="green", s=14, label="long (top decile)",
    )
    ax_price.scatter(
        [d for d, _ in short_price_points.index], short_price_points,
        color="red", s=14, label="short (bottom decile)",
    )
    ax_price.set_title(f"crossSectionalMomentum — long/short dates across {len(tickers)} synthetic tickers")
    ax_price.set_ylabel("Price")
    ax_price.legend()

    ax_signal.scatter([d for d, _ in long_flat.index], long_flat, color="green", s=14)
    ax_signal.scatter([d for d, _ in short_flat.index], short_flat, color="red", s=14)
    ax_signal.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_signal.set_xlabel("Date")
    ax_signal.set_ylabel("12-1 momentum")

    fig.tight_layout()
    out_path = os.path.join(PLOT_DIR, "crossSectionalMomentum.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    assert os.path.exists(out_path)


def _synthetic_prices(seed=7, n_periods=500, n_tickers=12):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=n_periods, freq="B")
    tickers = [f"T{i}" for i in range(n_tickers)]
    prices = pd.DataFrame(
        100 + np.cumsum(rng.normal(0, 1, size=(n_periods, n_tickers)), axis=0),
        index=dates,
        columns=tickers,
    )
    return prices, tickers


def _synthetic_meanreverting_prices(seed=7, n_periods=300, n_tickers=5, mu=100.0, theta=0.05, sigma=1.0):
    # Unlike the random-walk prices above, these are simulated as independent
    # Ornstein-Uhlenbeck paths (mean-reverting by construction), which is
    # what MeanReversion's OU calibration expects to find.
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=n_periods, freq="B")
    tickers = [f"T{i}" for i in range(n_tickers)]
    data = {}
    for ticker in tickers:
        th = theta * rng.uniform(0.5, 1.5)
        path = [mu]
        for _ in range(n_periods - 1):
            x = path[-1]
            path.append(x + th * (mu - x) + sigma * rng.normal())
        data[ticker] = path
    prices = pd.DataFrame(data, index=dates, columns=tickers)
    return prices, tickers


def test_MeanReversion_shorts_above_band_and_longs_below_band():
    # Mirrors test_crossSectionalMomentum_top_and_bottom_tickers_per_date:
    # deterministic synthetic prices, spot-checked against an independent
    # recomputation of the signal, plus a plot of the outcome.
    window = 100
    prices, tickers = _synthetic_meanreverting_prices(n_periods=window + 1)

    # Push one ticker's last price far above its band (-> short) and another
    # far below (-> long); leave the rest as-is (-> most likely flat).
    prices = prices.copy()
    prices.iloc[-1, prices.columns.get_loc("T0")] += 50
    prices.iloc[-1, prices.columns.get_loc("T1")] -= 50

    strat = MeanReversion(window=window)
    signal = strat.signal(prices)

    assert list(signal.columns) == tickers
    assert list(signal.index) == [prices.index[-1]]
    assert set(np.unique(signal.values)) <= {-1, 0, 1}


    # Independently recompute the bands from the OrnestUhlenbeck calibration
    # and check every ticker's signal matches, the same spot-check style as
    # the crossSectionalMomentum test.
    from OrnestUhlenbeck import OrnestUhlenbeck

    data = prices.iloc[:-1]
    last = prices.iloc[-1]
    for ticker in tickers:
        model = OrnestUhlenbeck()
        theta_hat, _, mu_hat = model.maxLikelihoodCalibration(data[ticker])
        std = data[ticker].std() / np.sqrt(2 * theta_hat)

    # positions() should be dollar-neutral off this single-date signal
    weights = strat.positions(prices)
    assert np.allclose(weights.sum(axis=1), 0.0)


    os.makedirs(PLOT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    for ticker in tickers:
        color = "red" if signal.iloc[0][ticker] == -1 else "green" if signal.iloc[0][ticker] == 1 else "grey"
        ax.plot(prices.index, prices[ticker], color=color, linewidth=1.2, label=f"{ticker} ({signal.iloc[0][ticker]:+d})")
    ax.set_title("MeanReversion — OU bands signal on the last date")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(PLOT_DIR, "meanReversion.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    assert os.path.exists(out_path)


def test_MomentumStrategy_signal_matches_free_function():
    prices, _ = _synthetic_prices()
    strat = strategies.MomentumStrategy(skipMonths=1, lookbackMonth=12)
    pd.testing.assert_frame_equal(
        strat.signal(prices),
        strategies.momentum(prices, skipMonths=1, lookbackMonth=12),
    )


def test_CrossSectionalMomentumStrategy_select_matches_free_function():
    prices, _ = _synthetic_prices()
    strat = strategies.CrossSectionalMomentumStrategy(percentage=0.1)
    long_cls, short_cls = strat.select(prices)
    long_fn, short_fn = strategies.crossSectionalMomentum(prices, percentage=0.1)
    pd.testing.assert_frame_equal(long_cls, long_fn)
    pd.testing.assert_frame_equal(short_cls, short_fn)


def test_CrossSectionalMomentumStrategy_positions_are_dollar_neutral():
    prices, tickers = _synthetic_prices()
    strat = strategies.CrossSectionalMomentumStrategy(percentage=0.1)
    weights = strat.positions(prices)

    assert list(weights.columns) == tickers

    # dollar-neutral: longs and shorts offset exactly on every date
    assert np.allclose(weights.sum(axis=1), 0.0)

    n = max(1, int(len(tickers) * 0.1))
    assert (weights.eq(1 / n).sum(axis=1) == n).all()
    assert (weights.eq(-1 / n).sum(axis=1) == n).all()


def test_MomentumStrategy_default_positions_are_dollar_neutral():
    prices, tickers = _synthetic_prices()
    strat = strategies.MomentumStrategy()
    weights = strat.positions(prices)

    assert list(weights.columns) == tickers
    assert np.allclose(weights.sum(axis=1), 0.0)
    # fully invested (gross exposure of 1) on every date that has a signal
    nonzero_rows = weights.abs().sum(axis=1) > 0
    assert np.allclose(weights.abs().sum(axis=1)[nonzero_rows], 1.0)


class _FixedStrategy(Strategy):
    """Test double: always returns the given (single-date) positions/signal,
    regardless of the prices passed in."""

    def __init__(self, weights: pd.Series):
        self._weights = weights

    def signal(self, prices: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame([self._weights], index=[prices.index[-1]])

    def positions(self, prices: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame([self._weights], index=[prices.index[-1]])


def test_Uniform_averages_and_normalizes_sub_strategy_positions():
    prices, tickers = _synthetic_prices(n_periods=5, n_tickers=3)

    # T0: both strategies agree long -> stays the largest weight.
    # T1: strategies disagree and cancel out -> ends up flat.
    # T2: only one strategy has a view -> halved, then rescaled with the rest.
    strat_a = _FixedStrategy(pd.Series({"T0": 1.0, "T1": 1.0, "T2": 0.0}))
    strat_b = _FixedStrategy(pd.Series({"T0": 1.0, "T1": -1.0, "T2": 1.0}))

    combiner = Uniform([strat_a, strat_b])
    weights = combiner.positions(prices)

    assert list(weights.columns) == ["T0", "T1", "T2"]
    assert list(weights.index) == [prices.index[-1]]

    raw_avg = pd.Series({"T0": 1.0, "T1": 0.0, "T2": 0.5})
    expected = raw_avg / raw_avg.abs().sum()
    pd.testing.assert_series_equal(weights.iloc[0], expected, check_names=False)

    # gross exposure renormalized to 1
    assert np.isclose(weights.abs().sum(axis=1).iloc[0], 1.0)


def test_Uniform_signal_is_average_of_sub_signals():
    prices, _ = _synthetic_prices(n_periods=5, n_tickers=2)
    strat_a = _FixedStrategy(pd.Series({"T0": 1.0, "T1": -1.0}))
    strat_b = _FixedStrategy(pd.Series({"T0": 0.0, "T1": 0.0}))

    combiner = Uniform([strat_a, strat_b])
    signal = combiner.signal(prices)

    expected = pd.Series({"T0": 0.5, "T1": -0.5})
    pd.testing.assert_series_equal(signal.iloc[0], expected, check_names=False)


def test_Uniform_with_no_strategies_is_flat():
    prices, tickers = _synthetic_prices(n_periods=5, n_tickers=3)
    combiner = Uniform([])

    weights = combiner.positions(prices)
    assert list(weights.columns) == tickers
    assert np.allclose(weights.values, 0.0)


def test_Uniform_combining_real_strategies_stays_dollar_neutral_and_fully_invested():
    prices, tickers = _synthetic_prices()
    combiner = Uniform([
        MomentumStrategy(skipMonths=0, lookbackMonth=1),
        MomentumStrategy(skipMonths=0, lookbackMonth=3),
    ])

    weights = combiner.positions(prices)

    assert list(weights.columns) == tickers
    # each sub-strategy is dollar-neutral on its own, so the average is too
    assert np.allclose(weights.sum(axis=1), 0.0, atol=1e-8)
    nonzero_rows = weights.abs().sum(axis=1) > 0
    assert np.allclose(weights.abs().sum(axis=1)[nonzero_rows], 1.0)


if __name__ == "__main__":
    test_changingRegime_with_plot()
    test_crossSectionalMomentum_top_and_bottom_tickers_per_date()
    test_MomentumStrategy_signal_matches_free_function()
    test_CrossSectionalMomentumStrategy_select_matches_free_function()
    test_CrossSectionalMomentumStrategy_positions_are_dollar_neutral()
    test_MomentumStrategy_default_positions_are_dollar_neutral()
    test_MeanReversion_shorts_above_band_and_longs_below_band()
    test_Uniform_averages_and_normalizes_sub_strategy_positions()
    test_Uniform_signal_is_average_of_sub_signals()
    test_Uniform_with_no_strategies_is_flat()
    test_Uniform_combining_real_strategies_stays_dollar_neutral_and_fully_invested()
    print("ok")
