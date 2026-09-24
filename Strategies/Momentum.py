from pandas import DataFrame
from Strategies.Strategy import Strategy
from strategies import momentum

class MomentumStrategy(Strategy):
    """12-1 style trailing momentum (Jegadeesh-Titman): skip the most
    recent `skipMonths`, then look back `lookbackMonth` months."""

    def __init__(self, skipMonths: int = 1, lookbackMonth: int = 12, tradingDayPerMonth: int = 21):
        self.skipMonths = skipMonths
        self.lookbackMonth = lookbackMonth
        self.tradingDayPerMonth = tradingDayPerMonth

    def signal(self, prices: DataFrame) -> DataFrame:
        signal = momentum(prices, self.skipMonths, self.lookbackMonth, self.tradingDayPerMonth)
        signal.dropna(inplace=True)
        return signal.iloc[[-1]]