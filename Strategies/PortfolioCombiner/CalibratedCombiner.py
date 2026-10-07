from abc import abstractmethod

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from Strategies.PortfolioCombiner.PortfolioCombiner import PortfolioCombiner
from Strategies.Strategy import Strategy
from BacktestEngine import BacktestEngine
from PerformanceAnalyzer import PerformanceAnalyzer


class CalibratedCombiner(PortfolioCombiner):
    """Combiner whose weights are estimated by backtesting each sub-strategy
    on its own over a calibration window [start, end].

    To keep the combination out-of-sample, the calibration window must end
    before the backtest that uses this combiner starts: call `calibrate()`
    once, then run the main backtest from a date after `end`. positions()
    refuses to trade on data that ends inside the calibration window.
    Subclasses only implement `computeWeights`."""

    def __init__(self, strategy: list[Strategy], start, end, **engine_kwargs) -> None:
        super().__init__(strategy)
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)
        self.engine_kwargs = engine_kwargs
        self.strategyMetrics: DataFrame = None
        self.strategyReturns: DataFrame = None
        self._weights: Series = None

    @abstractmethod
    def computeWeights(self, returns: DataFrame, metrics: DataFrame) -> Series:
        """Capital weights from the calibration window's daily net returns
        (one column per sub-strategy) and their metrics."""

    def calibrate(self, data_source) -> DataFrame:
        """Backtest each sub-strategy alone over [start, end], store its daily
        net returns and metrics (one column per sub-strategy), and the
        weights computed from them."""
        metrics, returns = {}, {}
        for name, strategy in zip(self.strategyNames(), self.strategy):
            engine = BacktestEngine(strategy, data_source, **self.engine_kwargs)
            returns[name] = engine.run(self.start, self.end)
            strategyMetrics = engine.metrics["value"].copy()
            # Volatility over the days the strategy actually held a book. A
            # sparse strategy (flat most days) has a low average vol that
            # understates the risk it takes whenever it does trade.
            active = engine.positions.abs().sum(axis=1) > 0
            strategyMetrics["activeDays"] = float(active.mean())
            strategyMetrics["activeVolatility"] = (
                returns[name][active].std(ddof=1) * np.sqrt(PerformanceAnalyzer().periodsPerYear)
                if active.sum() > 1 else 0.0
            )
            metrics[name] = strategyMetrics
        self.strategyMetrics = DataFrame(metrics)
        self.strategyReturns = DataFrame(returns)

        self._weights = self.computeWeights(self.strategyReturns, self.strategyMetrics)
        return self.strategyMetrics

    def metrics(self) -> DataFrame:
        """Calibration-window metrics of each sub-strategy."""
        self._checkCalibrated()
        return self.strategyMetrics

    def weights(self) -> Series:
        self._checkCalibrated()
        return self._weights

    def signal(self, prices: DataFrame) -> DataFrame:
        self._checkOutOfSample(prices)
        return super().signal(prices)

    def positions(self, prices: DataFrame) -> DataFrame:
        self._checkOutOfSample(prices)
        return super().positions(prices)

    def _checkCalibrated(self):
        if self._weights is None:
            raise RuntimeError(f"{type(self).__name__}.calibrate(data_source) must be called before use")

    def _checkOutOfSample(self, prices: DataFrame):
        """The history passed in ends the day before the traded date, so
        history ending on `end` means trading the first day after it."""
        self._checkCalibrated()
        if prices.index[-1] < self.end:
            raise ValueError(
                f"Look-ahead: trading on data ending {prices.index[-1].date()}, "
                f"inside the {type(self).__name__} calibration window ending {self.end.date()}"
            )
