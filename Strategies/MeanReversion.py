import numpy as np
from fontTools.cu2qu.cu2qu import NAN
from pandas import DataFrame
from statsmodels.tsa.stattools import adfuller

from Strategies.Strategy import Strategy
from OrnestUhlenbeck import OrnestUhlenbeck
from PositionSizing import signedWeights

class MeanReversion(Strategy):

    def __init__(self, window):
        self.window = window


    def signal(self, prices: DataFrame) -> DataFrame:
        prices = prices.tail(self.window+1)
        data = prices[:-1]
        model = OrnestUhlenbeck()
        param = data.apply(lambda rows : model.maxLikelihoodCalibration(rows))
        param.index = ["theta", "sigma","mu"]
        mean = param.loc["mu"]
        theta = param.loc["theta"]
        valid = theta > 0
        std = param.loc["sigma"]/np.sqrt(2*theta)
        value = prices.iloc[-1]
        z_score = (value-mean)/std
        adf = data.apply(lambda row : adfuller(row))
        pValue = adf.iloc[1]
        condition = [(pValue<0.05) & (z_score > 1.5) & valid, (pValue<0.05) & (z_score < -1.5) & valid,(z_score.abs() <0.5) & valid]
        choice = [-1,1,0]
        signal = np.select(condition, choice,default=np.nan)
        return DataFrame([signal], index=[prices.index[-1]], columns=value.index)

    def positions(self, prices: DataFrame) -> DataFrame:
        signal = self.signal(prices)
        return signedWeights(signal)
