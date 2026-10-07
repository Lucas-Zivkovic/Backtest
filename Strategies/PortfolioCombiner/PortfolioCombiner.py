from abc import abstractmethod
import pandas as pd
from pandas import DataFrame, Series
from Strategies.Strategy import Strategy

class PortfolioCombiner(Strategy):
    """Blends sub-strategies with per-strategy capital weights.

    Each sub-strategy already sizes its own book (unit gross exposure, or
    flat). The combined book is the weighted sum of those books, *not*
    renormalized: when a sub-strategy is flat its capital share stays in
    cash instead of levering up the others, so the weights are the actual
    capital allocation. Subclasses only choose the weights."""

    def __init__(self, strategy:list[Strategy]):
        self.strategy = strategy

    @abstractmethod
    def weights(self) -> Series:
        """Capital weight of each sub-strategy, indexed by strategyNames()."""

    def strategyNames(self) -> list[str]:
        """Label for each sub-strategy: its class name, or `ClassName_<i>`
        when two sub-strategies share a class name."""
        names = [type(s).__name__ for s in self.strategy]
        if len(set(names)) < len(names):
            names = [f"{name}_{i}" for i, name in enumerate(names)]
        return names

    def signal(self, prices: DataFrame) -> DataFrame:
        """Per (date, ticker) `score`: the weighted average of each
        sub-strategy's own score, plus every sub-strategy's own score kept as
        its own field. Informational only — positions() is what actually
        drives the combined portfolio."""
        if not self.strategy:
            return DataFrame(0.0, index=[prices.index[-1]], columns=prices.columns)

        scores = [s.score(prices) for s in self.strategy]
        combined = sum(w * score for w, score in zip(self.weights(), scores))

        fields = {"score": combined}
        fields.update(zip(self.strategyNames(), scores))
        signal = pd.concat(fields, axis=1).swaplevel(axis=1)
        signal.columns.names = ["ticker", "field"]
        return signal

    def positions(self, prices: DataFrame) -> DataFrame:
        """Weighted sum of each sub-strategy's own target positions."""
        if not self.strategy:
            return DataFrame(0.0, index=[prices.index[-1]], columns=prices.columns)

        positions = [s.positions(prices) for s in self.strategy]
        return sum(w * p for w, p in zip(self.weights(), positions)).fillna(0.0)
