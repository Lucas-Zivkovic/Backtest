import numpy as np
import pandas as pd
from pandas import DataFrame

from Strategies.CrossSectionalMomentumStrategy import CrossSectionalMomentumStrategy


class ResidualReversalStrategy(CrossSectionalMomentumStrategy):
    """Cross-sectional reversal on residual returns: long the `percentage` of
    tickers that did worst over the last `lookbackDays` once the market is
    taken out, short the ones that did best.

    The market is the equal-weight average return of the tickers. Each
    ticker's beta to it is estimated on the last `betaDays` daily returns,
    and its residual return is r - beta * r_market. The score is minus the
    sum of the residuals over the lookback, divided by the residual
    volatility over `betaDays` (Blitz et al.), so a volatile stock needs a
    larger move to be picked. Unlike a raw reversal, a high-beta stock that
    fell with the market is not taken for a loser.

    Only the last `betaDays` + 1 prices are used each step, so the signal
    costs the same whatever the length of the history. Needs more than
    `betaDays` prices."""

    def __init__(self, percentage: float = 0.1, lookbackDays: int = 21, betaDays: int = 252):
        super().__init__(percentage)
        self.lookbackDays = lookbackDays
        self.betaDays = betaDays

    def signal(self, prices: DataFrame) -> DataFrame:
        """Last date's `score` (minus the standardized residual return), the
        `residualReturn` over the lookback, the `beta` and the daily
        `residualVol` it was derived from."""
        if len(prices) <= self.betaDays:
            raise ValueError(f"ResidualReversalStrategy needs more than {self.betaDays} prices, got {len(prices)}")
        returns = prices.tail(self.betaDays + 1).pct_change().iloc[1:]
        market = returns.mean(axis=1)
        demeaned = market - market.mean()
        beta = returns.sub(returns.mean()).mul(demeaned, axis=0).sum() / (demeaned ** 2).sum()
        residuals = returns - np.outer(market, beta)
        residualReturn = residuals.tail(self.lookbackDays).sum()
        residualVol = residuals.std()

        diagnostics = DataFrame({
            "score": -residualReturn / residualVol,
            "residualReturn": residualReturn,
            "beta": beta,
            "residualVol": residualVol,
        })
        diagnostics.index.name = "ticker"
        signal = diagnostics.stack(future_stack=True).to_frame().T.astype(float)
        signal.index = [prices.index[-1]]
        signal.columns.names = ["ticker", "field"]
        return signal
