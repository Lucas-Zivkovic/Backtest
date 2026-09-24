from abc import abstractmethod
from pandas import DataFrame
from Strategies.Strategy import Strategy

class PortfolioCombiner(Strategy):

    def __init__(self, strategy:list[Strategy]):
        self.strategy = strategy

    @abstractmethod
    def signal(self, prices: DataFrame) -> DataFrame:
        """Per (date, ticker) score. signal.loc[t] must use only data
        available as of t."""