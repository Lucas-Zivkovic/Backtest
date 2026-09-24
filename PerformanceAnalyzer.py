import numpy as np
from pandas import DataFrame

import Metrics


class PerformanceAnalyzer:
    """Turns a strategy's positions into portfolio returns and summary metrics."""

    def __init__(self, riskFreeRate: float = 0.0025, periodsPerYear: int = 252):
        self.riskFreeRate = riskFreeRate
        self.periodsPerYear = periodsPerYear

    def portfolioReturns(self, prices: DataFrame, positions: DataFrame) -> DataFrame:
        """Daily P&L of holding yesterday's target weights into today's return."""
        returns = prices.pct_change()
        weights, returns = positions.align(returns, join="inner")
        return (weights.shift(1) * returns).sum(axis=1).dropna()

    def metrics(self, prices: DataFrame, positions: DataFrame) -> DataFrame:
        """Summary performance/risk metrics for a strategy's positions."""
        returns = self.portfolioReturns(prices, positions)
        cumulative = (1 + returns).cumprod()

        return DataFrame({"value": {
            "sharpeRatio": Metrics.sharpRatio(returns, self.riskFreeRate, self.periodsPerYear),
            "sortinoRatio": Metrics.sortinoRatio(returns, self.riskFreeRate, self.periodsPerYear),
            "maxDrawdown": Metrics.maxDrawdown(cumulative),
            "drawdownDuration": Metrics.drawdownDuration(cumulative),
            "hitRatio": Metrics.hitRatio(returns),
            "payoffRatio": Metrics.payoffRatio(returns),
            "expectancy": Metrics.expectancy(returns),
            "annualizedTurnover": Metrics.annualizedTurnover(positions.values, self.periodsPerYear),
            "totalReturn": cumulative.iloc[-1] - 1,
            "annualizedVolatility": returns.std(ddof=1) * np.sqrt(self.periodsPerYear),
        }})

