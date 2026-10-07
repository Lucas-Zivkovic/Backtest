from pandas import Series

from Strategies.PortfolioCombiner.PortfolioCombiner import PortfolioCombiner


class Uniform(PortfolioCombiner):
    """Combines sub-strategies with equal (1/n) capital weight."""

    def weights(self) -> Series:
        n = len(self.strategy)
        return Series(1.0 / n, index=self.strategyNames()) if n else Series(dtype=float)
