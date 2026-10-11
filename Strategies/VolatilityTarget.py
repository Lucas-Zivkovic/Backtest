import numpy as np
from pandas import DataFrame

from Strategies.Strategy import Strategy


class VolatilityTarget(Strategy):
    """Scales another strategy's positions so the book's expected volatility
    is `targetVol` (annualized).

    The expected volatility is ex-ante: sqrt(w' Cov w * periodsPerYear), with
    w the strategy's current positions and Cov the covariance of the assets'
    daily returns over the last `lookback` days of history, so it only uses
    the past and does not depend on whether the strategy was invested before.
    Gross exposure is capped at `maxGross`; a flat book stays flat."""

    def __init__(self, strategy: Strategy, targetVol: float = 0.10, lookback: int = 60,
                 maxGross: float = 4.0, periodsPerYear: int = 252):
        self.strategy = strategy
        self.targetVol = targetVol
        self.lookback = lookback
        self.maxGross = maxGross
        self.periodsPerYear = periodsPerYear

    def signal(self, prices: DataFrame) -> DataFrame:
        return self.strategy.cachedSignal(prices)

    def scale(self, prices: DataFrame, weights) -> float:
        """Leverage applied to `weights` (a Series indexed by ticker)."""
        traded = weights.index[weights != 0]
        if len(traded) == 0:
            return 0.0
        returns = prices[traded].tail(self.lookback + 1).pct_change().dropna()
        if len(returns) < 2:
            return 0.0
        w = weights[traded].values
        vol = np.sqrt(w @ returns.cov().values @ w * self.periodsPerYear)
        if not vol > 0:
            return 0.0
        return min(self.targetVol / vol, self.maxGross / np.abs(w).sum())

    def positions(self, prices: DataFrame) -> DataFrame:
        positions = self.strategy.positions(prices)
        return positions * self.scale(prices, positions.iloc[-1])
