from pandas import DataFrame
from Strategies.Momentum import MomentumStrategy
from PositionSizing import uniform

class CrossSectionalMomentumStrategy(MomentumStrategy):
    """Ranks tickers against each other on each date and goes long the
    top `percentage` / short the bottom `percentage` by momentum."""

    def __init__(self, percentage: float = 0.1, **momentum_kwargs):
        super().__init__(**momentum_kwargs)
        self.percentage = percentage

    def select(self, prices: DataFrame) -> tuple[DataFrame, DataFrame]:
        """Raw momentum values of the tickers picked long/short on each date."""
        assert len(prices.columns) > 10
        result = self.signal(prices).dropna()
        n = max(1, int(len(result.columns) * self.percentage))
        long = result.T.apply(lambda row: row.nlargest(n)).T
        short = result.T.apply(lambda row: row.nsmallest(n)).T
        return long, short

    def positions(self, prices: DataFrame) -> DataFrame:
        """Equal-weight dollar-neutral: +1/n on the long picks, -1/n on the short picks."""
        long, short = self.select(prices)
        n = int(long.notna().sum(axis=1).iloc[0]) if len(long) else 1
        long_flags = long.notna().reindex(columns=prices.columns, fill_value=False)
        short_flags = short.notna().reindex(columns=prices.columns, fill_value=False)
        weight = uniform(n, 1.0)
        return (long_flags.astype(float) - short_flags.astype(float)) * weight


