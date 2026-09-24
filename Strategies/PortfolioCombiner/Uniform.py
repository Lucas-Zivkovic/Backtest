from pandas import DataFrame

from PositionSizing import normalizeGrossExposure
from Strategies.PortfolioCombiner.PortfolioCombiner import PortfolioCombiner


class Uniform(PortfolioCombiner):
    """Combines sub-strategies with equal (1/n) weight."""

    def signal(self, prices: DataFrame) -> DataFrame:
        """Per (date, ticker) score: the equal-weighted average of each
        sub-strategy's own signal. Informational only — positions() is what
        actually drives the combined portfolio."""
        strategies = self.strategy
        n = len(strategies)
        if n == 0:
            return DataFrame(index=[prices.index[-1]], columns=prices.columns).fillna(0.0)

        signals = [s.signal(prices) for s in strategies]
        return sum(signals) / n

    def positions(self, prices: DataFrame) -> DataFrame:
        """Equal-weight average of each sub-strategy's own target positions,
        renormalized to unit gross exposure."""
        strategies = self.strategy
        n = len(strategies)
        if n == 0:
            return DataFrame(index=[prices.index[-1]], columns=prices.columns).fillna(0.0)

        positions = [s.positions(prices) for s in strategies]
        combined = sum(positions) / n
        return normalizeGrossExposure(combined)
