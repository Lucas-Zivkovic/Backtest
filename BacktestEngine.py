from pandas import DataFrame,Series
from PerformanceAnalyzer import PerformanceAnalyzer
from Strategies.Strategy import Strategy
import Transaction

class BacktestEngine:
    """Orchestre la boucle temporelle. Ne contient aucune logique de stratégie."""

    def __init__(self, strategy: Strategy, data_source:DataFrame, transactionCostRate: float = 0.0005,
                 spread: float = 0.0005, impactCoef: float = 1, portfolioValue: float = 1_000_000):
        self.strategy = strategy
        self.data_source = data_source
        self.transactionCostRate = transactionCostRate
        self.spread = spread
        self.impactCoef = impactCoef
        self.portfolioValue = portfolioValue
        self.signal: list[DataFrame] = []
        self.positions: DataFrame = DataFrame()
        self.costs: Series = Series()
        self.result : Series= Series()
        self.metrics=[]

    def run(self, start_date, end_date):
        costs={}
        returns={}
        positions={}
        previous_positions = None
        for date, market_data in self.data_source.iterate(start_date, end_date):
            close = market_data.xs("Close", axis=1, level=1)

            history = close.iloc[:-1]

            sigma = history.pct_change().std()
            avgVolume = market_data.xs("Volume", axis=1, level=1).mean()

            self.signal.append(self.strategy.cachedSignal(history))

            target_positions = (
                self.strategy.positions(history).iloc[-1]
                if len(history) > 0
                else close.iloc[-1] * 0
            )

            if previous_positions is None:
                previous_positions = target_positions * 0
            positions_rebalancing = target_positions - previous_positions

            order_size = positions_rebalancing.abs()
            turnover = order_size.sum()
            if turnover > 0:
                shares_traded = order_size * self.portfolioValue / close.iloc[-1]
                transactioncost = Transaction.transactionCost(turnover, minValue=0, pourcentage=self.transactionCostRate)
                spreadcost = Transaction.spreadCost(self.spread, order_size).sum()
                sqrtimpact = Transaction.sqrt_impact(shares_traded, avgVolume, sigma, self.impactCoef).sum()
                cost = transactioncost + spreadcost + sqrtimpact
            else:
                cost = 0.0
            costs[date]=cost

            period_returns = close.pct_change().iloc[-1]
            step_result = float((target_positions * period_returns).sum()) - cost
            returns[date]=step_result

            positions[date] = target_positions
            previous_positions = target_positions

        self.result = Series(returns, name="return")
        self.costs = Series(costs, name="cost")
        self.positions = DataFrame.from_dict(positions, orient="index")

        performance = PerformanceAnalyzer()

        self.metrics = performance.metrics(self.result,self.positions)
        return self.result

    def metric(self):
        return self.metrics