from pandas import DataFrame

from Strategies.CrossSectionalMomentumStrategy import CrossSectionalMomentumStrategy


class CrossSectionalReversalStrategy(CrossSectionalMomentumStrategy):
    """Contrarian twin of CrossSectionalMomentumStrategy: long the `percentage`
    of tickers with the worst return over the lookback, short the best, on the
    bet that recent losers bounce back against recent winners (short-term
    reversal, Jegadeesh 1990). Defaults to the last month, with no skip.

    The `score` is minus the trailing return, so positions() (long the highest
    score) and a PortfolioCombiner both see what the strategy actually bets on;
    `skipPrice` and `tradePrice` are kept as they are."""

    def __init__(self, percentage: float = 0.1, skipMonths: int = 0, lookbackMonth: int = 1,
                 tradingDayPerMonth: int = 21):
        super().__init__(percentage, skipMonths=skipMonths, lookbackMonth=lookbackMonth,
                         tradingDayPerMonth=tradingDayPerMonth)

    def signal(self, prices: DataFrame) -> DataFrame:
        signal = super().signal(prices).copy()
        scores = [column for column in signal.columns if column[1] == self.SCORE_FIELD]
        signal[scores] = -signal[scores]
        return signal
