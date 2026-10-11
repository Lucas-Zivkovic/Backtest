import numpy as np
import pandas as pd
from pandas import DataFrame

from OrnestUhlenbeck import OrnestUhlenbeck
from Strategies.Strategy import Strategy
from Strategies.PairFinding import findPairs, findParameters
from Strategies.ZScorePositions import ZScorePositions


class PairTrading(Strategy):
    """Mean reversion on the spreads of the most cointegrated pairs, with the
    pairs re-selected every `refitDays` trading days.

    A formation fits everything about the pairs on the last `formationDays`
    days of the history the strategy is handed, so it never sees a price
    after the day it is made:
      - selection: the `numberPair` strongest pairs among the stable ones,
        cointegrated (Johansen, at `significance`) on the whole formation and
        on each half of it (findPairs), candidates being pairs in the same
        `groups` label (ticker -> e.g. sub-industry) when given, else in the
        same PairFinding cluster. Fewer, or none, if fewer are stable:
        then the strategy stays in cash until the next formation;
      - hedge ratio: the Johansen cointegrating vector, spread =
        log a - beta * log b - alpha;
      - OU process on the formation spread: its mean `mu`, stationary `std`
        and `halfLife`.
    The first formation happens on the first step of a run, then again every
    `refitDays` steps. A pair dropped at a re-formation is closed; a pair
    selected again keeps its open trade. Every formation is kept in
    `formations` (formation date -> pairs with their fit).

    Each step the z-score is (spread - mu) / std and positions follow the
    ZScorePositions rules (hold until |z| < exitZ, stop past stopZ). A pair
    only enters if its formation half-life < maxHalfLife."""

    def __init__(self, numberPair: int, formationDays: int = 5 * 252, refitDays: int = 2 * 252,
                 entryZ: float = 1.5, exitZ: float = 0.5, stopZ: float = 3.0,
                 significance: float = 0.05, maxHalfLife: float = np.inf, groups: pd.Series = None):
        self.numberPair = numberPair
        self.groups = groups
        self.formationDays = formationDays
        self.refitDays = refitDays
        self.significance = significance
        self.rules = ZScorePositions(entryZ, exitZ, stopZ, maxHalfLife=maxHalfLife)
        self.pairs = DataFrame(columns=["a", "b", "cluster", "strength", "strengthFirst", "strengthSecond",
                                        "beta", "stable", "mu", "std", "halfLife"])
        self.params = []
        self.formations = {}
        self._formedAt = None  # len(history) at the last formation
        self._lastDate = None

    def form(self, prices: DataFrame):
        """Select the pairs and fit their hedge ratio and OU process on
        `prices`, replacing the current pairs."""
        candidates = findPairs(prices, self.significance, self.groups)
        self.pairs = candidates[candidates["stable"]].head(self.numberPair).reset_index(drop=True)
        self.params = [findParameters(prices, pair) for pair in self.pairs.to_dict("records")]
        spreads = self.spreads(prices)
        fits = []
        for name in self.pairNames():
            theta, sigma, mu = OrnestUhlenbeck().maxLikelihoodCalibration(spreads[name])
            valid = theta > 0
            fits.append({"mu": mu,
                         "std": sigma / np.sqrt(2 * theta) if valid else np.nan,
                         "halfLife": np.log(2) / theta if valid else np.nan})
        self.pairs = pd.concat([self.pairs, DataFrame(fits, columns=["mu", "std", "halfLife"])], axis=1)
        self.formations[prices.index[-1]] = self.pairs.assign(alpha=[alpha for alpha, beta in self.params])

    def _formIfDue(self, prices: DataFrame):
        """Form on the first step of a run (history that does not continue
        the last date seen) and every refitDays steps after. Seeing the same
        date again (signal recomputed) does not form twice."""
        date = prices.index[-1]
        sameDay = date == self._lastDate
        continuing = sameDay or (len(prices) > 1 and prices.index[-2] == self._lastDate)
        if not continuing:
            self.formations = {}
        if not continuing or (not sameDay and len(prices) - self._formedAt >= self.refitDays):
            self.form(prices.tail(self.formationDays))
            self._formedAt = len(prices)
        self._lastDate = date

    def pairNames(self) -> list[str]:
        return [f"{a}/{b}" for a, b in zip(self.pairs["a"], self.pairs["b"])]

    def spreads(self, prices: DataFrame) -> DataFrame:
        """Spread of each pair (one column per pair) with the formation hedge ratio."""
        spreads = {}
        for name, pair, (alpha, beta) in zip(self.pairNames(), self.pairs.to_dict("records"), self.params):
            spreads[name] = np.log(prices[pair["a"]]) - beta * np.log(prices[pair["b"]]) - alpha
        return DataFrame(spreads, index=prices.index)

    def signal(self, prices: DataFrame) -> DataFrame:
        """Per pair: the `score` held after today (-1/0/+1), the `spread`,
        its `zScore`, `stopped`, and the formation `mu`/`std`/`halfLife`,
        Johansen `strength` (trace / critical value) and `alpha`/`beta`."""
        self._formIfDue(prices)
        date = prices.index[-1]
        if self.pairs.empty:
            return DataFrame(index=[date], columns=pd.MultiIndex.from_tuples([], names=["ticker", "field"]))

        diagnostics = DataFrame({
            "spread": self.spreads(prices.tail(1)).iloc[-1],
            "mu": self.pairs["mu"].values,
            "std": self.pairs["std"].values,
            "halfLife": self.pairs["halfLife"].values,
            "strength": self.pairs["strength"].values,
            "alpha": [alpha for alpha, beta in self.params],
            "beta": [beta for alpha, beta in self.params],
        }, index=self.pairNames()).astype(float)
        state = self.rules.update(prices.index, diagnostics["spread"], diagnostics["mu"], diagnostics["std"],
                                  halfLife=diagnostics["halfLife"])
        diagnostics.insert(0, "score", state["position"])
        diagnostics[["stopped", "zScore"]] = state[["stopped", "zScore"]]
        diagnostics.index.name = "ticker"
        signal = diagnostics.stack(future_stack=True).to_frame().T.astype(float)
        signal.index = [date]
        signal.columns.names = ["ticker", "field"]
        return signal

    def positions(self, prices: DataFrame) -> DataFrame:
        """Each pair gets 1/numberPair of gross, split between its legs as
        1 : beta, so long the spread = long a, short beta * b. A flat pair
        leaves its share in cash instead of levering the open ones. Legs are
        summed per ticker; tickers in no current pair get 0."""
        signal = self.cachedSignal(prices)
        weights = pd.Series(0.0, index=prices.columns)
        for name, pair, (alpha, beta) in zip(self.pairNames(), self.pairs.to_dict("records"), self.params):
            size = signal[(name, "score")].iloc[-1] / self.numberPair / (1 + abs(beta))
            weights[pair["a"]] += size
            weights[pair["b"]] -= size * beta
        return DataFrame([weights], index=signal.index)
