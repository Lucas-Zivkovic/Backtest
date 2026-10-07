import numpy as np
from pandas import DataFrame,Series

import Metrics


class PerformanceAnalyzer:
    """Turns a backtest's net returns and positions into summary metrics."""

    def __init__(self, riskFreeRate: float = 0.0025, periodsPerYear: int = 252):
        self.riskFreeRate = riskFreeRate
        self.periodsPerYear = periodsPerYear

    def metrics(self, returns: Series, positions: DataFrame) -> DataFrame:
        """Summary performance/risk metrics from the engine's per-step net returns and positions."""
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
            "mu":returns.mean(),
            "sigma":returns.std(),
            "annualizedVolatility": returns.std(ddof=1) * np.sqrt(self.periodsPerYear),
            "annualizedMean": returns.mean() * self.periodsPerYear,
        }})

    def covarianceMatrix(self,returns: DataFrame) -> DataFrame:
        cov=np.cov(returns, rowvar=False)
        return DataFrame(cov)

    def correlationMatrix(self,returns: DataFrame) -> DataFrame:
        corr = np.corrcoef(returns, rowvar=False)
        return DataFrame(corr)
