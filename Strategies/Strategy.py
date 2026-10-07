from abc import abstractmethod, ABC
import pandas as pd
from pandas import DataFrame

from PositionSizing import dollarNeutralWeights


class Strategy(ABC):
    """Common interface for return-predicting strategies.

    A strategy turns price history into (1) a `signal` — its full view on
    each asset, no look-ahead — and (2) target portfolio `positions` derived
    from that view. `signal` may be a plain per-ticker DataFrame (the sizing
    score itself), or a richer one with MultiIndex columns (ticker, field)
    that also carries whatever the strategy computed along the way (fitted
    parameters, test statistics, raw inputs, ...), so nothing it derived is
    thrown away. Either way, the `SCORE_FIELD` is the value `positions()`
    actually sizes on.

    Subclasses only need to implement `signal`; `positions` has a
    dollar-neutral default they're free to override.
    """

    SCORE_FIELD = "score"

    @abstractmethod
    def signal(self, prices: DataFrame) -> DataFrame:
        """Per (date, ticker) diagnostics. signal.loc[t] must use only data
        available as of t. See class docstring for shape."""

    def cachedSignal(self, prices: DataFrame) -> DataFrame:
        """`signal(prices)`, computed once per `prices` object. Within a
        backtest step the same history is handed to signal, score and
        positions (and to every sub-strategy of a combiner), so this avoids
        refitting expensive signals several times per step."""
        cache = getattr(self, "_signalCache", None)
        if cache is None or cache[0] is not prices:
            cache = (prices, self.signal(prices))
            self._signalCache = cache
        return cache[1]

    def score(self, prices: DataFrame) -> DataFrame:
        """The sizing score extracted from `signal`, whichever shape it's in."""
        signal = self.cachedSignal(prices)
        if isinstance(signal.columns, pd.MultiIndex):
            return signal.xs(self.SCORE_FIELD, axis=1, level=-1)
        return signal

    def positions(self, prices: DataFrame) -> DataFrame:
        """Dollar-neutral weights proportional to the demeaned score."""
        score = self.score(prices).dropna(how="all")
        return dollarNeutralWeights(score)
