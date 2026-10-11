import numpy as np
import pandas as pd
from pandas import DataFrame
from statsmodels.tsa.stattools import adfuller

from Strategies.Strategy import Strategy
from Strategies.ZScorePositions import ZScorePositions
from OrnestUhlenbeck import OrnestUhlenbeck

class MeanReversion(Strategy):
    """OU mean reversion per ticker on log prices. Each step an OU process is
    fitted on the last `window` log prices to get the z-score; positions
    follow the ZScorePositions rules, so a trade is held until it reverts or
    is stopped, the ADF test only gates entries, and only fits whose
    half-life is under `maxHalfLife` days (default window / 2) can enter."""

    def __init__(self, window, entryZ: float = 1.5, exitZ: float = 0.5, stopZ: float = 3.0,
                 pValueMax: float = 0.01, maxHalfLife: float = None):
        self.window = window
        self.rules = ZScorePositions(entryZ, exitZ, stopZ, pValueMax,
                                     window / 2 if maxHalfLife is None else maxHalfLife)


    def signal(self, prices: DataFrame) -> DataFrame:
        """Last date's `score` (-1 short / 0 flat / +1 long, the position held
        after today), plus every quantity it was derived from: the current
        `price`, today's OU fit on log prices `theta`/`sigma`/`mu`, the
        implied stationary `std` and `halfLife` (days), the ADF `pValue`,
        `valid` (theta > 0, i.e. actually mean-reverting), `stopped`, the
        `entryMu`/`entryStd` frozen when the open trade was entered, the
        `zScore` the decision used (against entryMu/entryStd in a trade,
        today's fit when flat), and `volatility`, the daily return std used
        for sizing. mu, std and the entry values are in log price."""
        dates = prices.index
        prices = prices.tail(self.window+1)
        logPrices = np.log(prices)
        data = logPrices[:-1]
        model = OrnestUhlenbeck()
        param = data.apply(lambda rows : model.maxLikelihoodCalibration(rows))
        param.index = ["theta", "sigma","mu"]
        mean = param.loc["mu"]
        theta = param.loc["theta"]
        sigma = param.loc["sigma"]
        valid = theta > 0
        std = sigma/np.sqrt(2*theta.where(valid))
        halfLife = np.log(2)/theta.where(valid)
        value = logPrices.iloc[-1]
        adf = data.apply(lambda row : adfuller(row))
        pValue = adf.iloc[1]
        state = self.rules.update(dates, value, mean, std, pValue, halfLife)

        diagnostics = pd.DataFrame({
            "score": state["position"],
            "price": prices.iloc[-1],
            "mu": mean,
            "theta": theta,
            "sigma": sigma,
            "std": std,
            "halfLife": halfLife,
            "zScore": state["zScore"],
            "pValue": pValue,
            "valid": valid.astype(float),
            "stopped": state["stopped"],
            "entryMu": state["entryMu"],
            "entryStd": state["entryStd"],
            "volatility": prices.pct_change().std(),
        })
        diagnostics.index.name = "ticker"
        signal = diagnostics.stack(future_stack=True).to_frame().T.astype(float)
        signal.index = [prices.index[-1]]
        signal.columns.names = ["ticker", "field"]
        return signal

    def positions(self, prices: DataFrame) -> DataFrame:
        """Inverse-volatility weights: each ticker's share of gross is
        (1 / vol) / sum over all tickers of (1 / vol), so every open trade
        carries about the same risk, and a flat ticker leaves its share in
        cash instead of levering the open ones. Gross stays <= 1."""
        signal = self.cachedSignal(prices)
        score = signal.xs("score", axis=1, level=-1)
        invVol = 1 / signal.xs("volatility", axis=1, level=-1)
        invVol = invVol.where(np.isfinite(invVol), 0.0)
        share = invVol.div(invVol.sum(axis=1), axis=0).fillna(0.0)
        return score * share
