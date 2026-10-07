import pandas as pd
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
        """Last date's momentum `score`, plus the two raw prices it's built
        from: `skipPrice` (at t - skipMonths) and `tradePrice` (at
        t - skipMonths - lookbackMonth)."""
        skipdays = self.skipMonths * self.tradingDayPerMonth
        tradeDays = self.tradingDayPerMonth * self.lookbackMonth

        skipPrice = prices.shift(skipdays)
        tradePrice = prices.shift(skipdays + tradeDays)
        score = momentum(prices, self.skipMonths, self.lookbackMonth, self.tradingDayPerMonth)

        signal = pd.concat(
            {"score": score, "skipPrice": skipPrice, "tradePrice": tradePrice},
            axis=1,
        ).swaplevel(axis=1)
        signal.columns.names = ["ticker", "field"]
        signal.dropna(inplace=True)
        return signal.iloc[[-1]]