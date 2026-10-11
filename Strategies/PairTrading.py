import numpy as np
import pandas as pd
from pandas import DataFrame
from statsmodels.tsa.stattools import adfuller

from OrnestUhlenbeck import OrnestUhlenbeck
from Strategies.Strategy import Strategy
from Strategies.PairFinding import findPairs, findParameters
from Strategies.ZScorePositions import ZScorePositions


class PairTrading(Strategy):
    """OU mean reversion on the spreads of the most cointegrated pairs.

    Pairs and their hedge ratios are fixed once, in __init__, on
    `formationPrices`: pass a period that ends before the backtest starts, or
    the selection is look-ahead. The spread of a pair is
    log a - beta * log b - alpha with that fixed beta, so the spread held is
    the spread entered on.

    Each step an OU process is fitted on the last `window` days of the spread
    to get its z-score; positions follow the ZScorePositions rules, so a
    trade is held until it reverts or is stopped, and the ADF test only gates
    entries."""

    def __init__(self, window, formationPrices: DataFrame, numberPair: int,
                 entryZ: float = 1.5, exitZ: float = 0.5, stopZ: float = 3.0, pValueMax: float = 0.01,
                 maxHalfLife: float = None):
        self.window = window
        self.pairs = findPairs(formationPrices).head(numberPair)
        self.params = [findParameters(formationPrices, pair) for pair in self.pairs.to_dict("records")]
        self.rules = ZScorePositions(entryZ, exitZ, stopZ, pValueMax,
                                     window / 2 if maxHalfLife is None else maxHalfLife)

    def pairNames(self) -> list[str]:
        return [f"{a}/{b}" for a, b in zip(self.pairs["a"], self.pairs["b"])]

    def spreads(self, prices: DataFrame) -> DataFrame:
        """Spread of each pair (one column per pair) with the formation hedge ratio."""
        spreads = {}
        for name, pair, (alpha, beta) in zip(self.pairNames(), self.pairs.to_dict("records"), self.params):
            spreads[name] = np.log(prices[pair["a"]]) - beta * np.log(prices[pair["b"]]) - alpha
        return DataFrame(spreads)

    def signal(self, prices: DataFrame) -> DataFrame:
        """Per pair: the `score` held after today (-1/0/+1), the `spread`,
        today's OU `mu`/`std` and `halfLife` (days), the ADF `pValue`, `stopped`, the
        `entryMu`/`entryStd` frozen when the open trade was entered, the
        `zScore` the decision used (against entryMu/entryStd in a trade,
        today's fit when flat), and the fixed `alpha`/`beta`."""
        spreads = self.spreads(prices.tail(self.window + 1))
        rows = {}
        for name, (alpha, beta) in zip(self.pairNames(), self.params):
            spread = spreads[name]
            mu, std, halfLife, pValue = np.nan, np.nan, np.nan, np.nan
            if len(spread) == self.window + 1:
                data = spread.iloc[:-1]
                theta, sigma, mu = OrnestUhlenbeck().maxLikelihoodCalibration(data)
                if theta > 0:
                    std = sigma / np.sqrt(2 * theta)
                    halfLife = np.log(2) / theta
                pValue = adfuller(data)[1]
            rows[name] = {"spread": spread.iloc[-1], "mu": mu, "std": std, "halfLife": halfLife, "pValue": pValue,
                          "alpha": alpha, "beta": beta}

        diagnostics = DataFrame(rows, index=["spread", "mu", "std", "halfLife", "pValue", "alpha", "beta"]).T.astype(float)
        state = self.rules.update(prices.index, diagnostics["spread"], diagnostics["mu"],
                                  diagnostics["std"], diagnostics["pValue"], diagnostics["halfLife"])
        diagnostics.insert(0, "score", state["position"])
        diagnostics[["stopped", "zScore", "entryMu", "entryStd"]] = state[["stopped", "zScore", "entryMu", "entryStd"]]
        diagnostics.index.name = "ticker"
        signal = diagnostics.stack(future_stack=True).to_frame().T.astype(float)
        signal.index = [prices.index[-1]]
        signal.columns.names = ["ticker", "field"]
        return signal

    def positions(self, prices: DataFrame) -> DataFrame:
        """Each pair gets 1/numberPair of gross, split between its legs as
        1 : beta, so long the spread = long a, short beta * b. A flat pair
        leaves its share in cash instead of levering the open ones. Legs are
        summed per ticker; tickers in no pair get 0."""
        signal = self.cachedSignal(prices)
        weights = pd.Series(0.0, index=prices.columns)
        for name, pair, (alpha, beta) in zip(self.pairNames(), self.pairs.to_dict("records"), self.params):
            size = signal[(name, "score")].iloc[-1] / len(self.pairs) / (1 + abs(beta))
            weights[pair["a"]] += size
            weights[pair["b"]] -= size * beta
        return DataFrame([weights], index=signal.index)
