from abc import abstractmethod, ABC
from pandas import DataFrame

from PositionSizing import dollarNeutralWeights


class Strategy(ABC):
    """Common interface for return-predicting strategies.

    A strategy turns price history into (1) a raw `signal` — its view on
    each asset, no look-ahead — and (2) target portfolio `positions`
    derived from that signal. Subclasses only need to implement `signal`;
    `positions` has a dollar-neutral default they're free to override.
    """

    @abstractmethod
    def signal(self, prices: DataFrame) -> DataFrame:
        """Per (date, ticker) score. signal.loc[t] must use only data
        available as of t."""

    def positions(self, prices: DataFrame) -> DataFrame:
        """Dollar-neutral weights proportional to the demeaned signal."""
        signal = self.signal(prices).dropna(how="all")
        return dollarNeutralWeights(signal)
