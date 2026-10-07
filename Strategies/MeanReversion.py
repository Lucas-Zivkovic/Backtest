import numpy as np
import pandas as pd
from pandas import DataFrame
from statsmodels.tsa.stattools import adfuller

from Strategies.Strategy import Strategy
from OrnestUhlenbeck import OrnestUhlenbeck
from PositionSizing import signedWeights

class MeanReversion(Strategy):

    def __init__(self, window):
        self.window = window


    def signal(self, prices: DataFrame) -> DataFrame:
        """Last date's discrete `score` (-1 short / 0 flat / +1 long / NaN
        no view), plus every quantity the OU calibration and ADF test derived
        it from: the fitted `theta`/`sigma`/`mu`, the implied stationary
        `std`, the current `price`, the `zScore` against the OU mean, the ADF
        `pValue`, and `valid` (theta > 0, i.e. actually mean-reverting)."""
        prices = prices.tail(self.window+1)
        data = prices[:-1]
        model = OrnestUhlenbeck()
        param = data.apply(lambda rows : model.maxLikelihoodCalibration(rows))
        param.index = ["theta", "sigma","mu"]
        mean = param.loc["mu"]
        theta = param.loc["theta"]
        sigma = param.loc["sigma"]
        valid = theta > 0
        std = sigma/np.sqrt(2*theta)
        value = prices.iloc[-1]
        zScore = (value-mean)/std
        adf = data.apply(lambda row : adfuller(row))
        pValue = adf.iloc[1]
        condition = [(pValue<0.05) & (zScore > 1.5) & valid, (pValue<0.05) & (zScore < -1.5) & valid,(zScore.abs() <0.5) & valid]
        choice = [-1,1,0]
        score = pd.Series(np.select(condition, choice, default=0), index=value.index)

        diagnostics = pd.DataFrame({
            "score": score,
            "price": value,
            "mu": mean,
            "theta": theta,
            "sigma": sigma,
            "std": std,
            "zScore": zScore,
            "pValue": pValue,
            "valid": valid.astype(float),
        })
        diagnostics.index.name = "ticker"
        signal = diagnostics.stack(future_stack=True).to_frame().T.astype(float)
        signal.index = [prices.index[-1]]
        signal.columns.names = ["ticker", "field"]
        return signal

    def positions(self, prices: DataFrame) -> DataFrame:
        score = self.score(prices)
        return signedWeights(score)
