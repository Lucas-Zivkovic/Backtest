import numpy as np
import pandas as pd
from pandas import DataFrame
from statsmodels.tsa.stattools import adfuller

from Strategies.Strategy import Strategy
from Strategies.ZScorePositions import ZScorePositions
from OrnestUhlenbeck import OrnestUhlenbeck

class MeanReversion(Strategy):
    """OU mean reversion per ticker. Each step an OU process is fitted on the
    last `window` prices to get the z-score; positions follow the
    ZScorePositions rules, so a trade is held until it reverts or is
    stopped, and the ADF test only gates entries."""

    def __init__(self, window, entryZ: float = 1.5, exitZ: float = 0.5, stopZ: float = 3.0, pValueMax: float = 0.05):
        self.window = window
        self.rules = ZScorePositions(entryZ, exitZ, stopZ, pValueMax)


    def signal(self, prices: DataFrame) -> DataFrame:
        """Last date's `score` (-1 short / 0 flat / +1 long, the position held
        after today), plus every quantity it was derived from: the fitted
        `theta`/`sigma`/`mu`, the implied stationary `std`, the current
        `price`, the `zScore` against the OU mean, the ADF `pValue`, `valid`
        (theta > 0, i.e. actually mean-reverting) and `stopped`."""
        dates = prices.index
        prices = prices.tail(self.window+1)
        data = prices[:-1]
        model = OrnestUhlenbeck()
        param = data.apply(lambda rows : model.maxLikelihoodCalibration(rows))
        param.index = ["theta", "sigma","mu"]
        mean = param.loc["mu"]
        theta = param.loc["theta"]
        sigma = param.loc["sigma"]
        valid = theta > 0
        std = sigma.abs()/np.sqrt(2*theta.where(valid))
        value = prices.iloc[-1]
        zScore = (value-mean)/std
        adf = data.apply(lambda row : adfuller(row))
        pValue = adf.iloc[1]
        state = self.rules.update(dates, zScore, pValue)

        diagnostics = pd.DataFrame({
            "score": state["position"],
            "price": value,
            "mu": mean,
            "theta": theta,
            "sigma": sigma,
            "std": std,
            "zScore": zScore,
            "pValue": pValue,
            "valid": valid.astype(float),
            "stopped": state["stopped"],
        })
        diagnostics.index.name = "ticker"
        signal = diagnostics.stack(future_stack=True).to_frame().T.astype(float)
        signal.index = [prices.index[-1]]
        signal.columns.names = ["ticker", "field"]
        return signal

    def positions(self, prices: DataFrame) -> DataFrame:
        """Each ticker gets 1/n of gross: a flat ticker leaves its share in
        cash instead of levering the open ones."""
        score = self.score(prices)
        return score / score.shape[1]
