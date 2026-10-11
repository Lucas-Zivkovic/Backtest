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
                 entryZ: float = 1.5, exitZ: float = 0.5, stopZ: float = 3.0, pValueMax: float = 0.05):
        self.window = window
        self.pairs = findPairs(formationPrices).head(numberPair)
        self.params = [findParameters(formationPrices, pair) for pair in self.pairs.to_dict("records")]
        self.rules = ZScorePositions(entryZ, exitZ, stopZ, pValueMax)

    def pairNames(self) -> list[str]:
        return [f"{a}/{b}" for a, b in zip(self.pairs["a"], self.pairs["b"])]

    def spreads(self, prices: DataFrame) -> DataFrame:
        """Spread of each pair (one column per pair) with the formation hedge ratio."""
        spreads = {}
        for name, pair, (alpha, beta) in zip(self.pairNames(), self.pairs.to_dict("records"), self.params):
            spreads[name] = np.log(prices[pair["a"]]) - beta * np.log(prices[pair["b"]]) - alpha
        return DataFrame(spreads)

    def signal(self, prices: DataFrame) -> DataFrame:
        """Per pair: the `score` held after today (-1/0/+1), the `zScore`, the
        OU `mu`/`std` it is measured against, the ADF `pValue`, `stopped`,
        and the fixed `alpha`/`beta`."""
        spreads = self.spreads(prices.tail(self.window + 1))
        rows, zScores, pValues = {}, {}, {}
        for name, (alpha, beta) in zip(self.pairNames(), self.params):
            spread = spreads[name]
            z, mu, std, pValue = np.nan, np.nan, np.nan, np.nan
            if len(spread) == self.window + 1:
                data = spread.iloc[:-1]
                theta, sigma, mu = OrnestUhlenbeck().maxLikelihoodCalibration(data)
                if theta > 0:
                    std = abs(sigma) / np.sqrt(2 * theta)
                    z = (spread.iloc[-1] - mu) / std
                pValue = adfuller(data)[1]

            zScores[name], pValues[name] = z, pValue
            rows[name] = {"zScore": z, "mu": mu, "std": std, "pValue": pValue, "alpha": alpha, "beta": beta}

        state = self.rules.update(prices.index, pd.Series(zScores, dtype=float), pd.Series(pValues, dtype=float))
        diagnostics = DataFrame(rows, index=["zScore", "mu", "std", "pValue", "alpha", "beta"]).T
        diagnostics.insert(0, "score", state["position"])
        diagnostics["stopped"] = state["stopped"]
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
