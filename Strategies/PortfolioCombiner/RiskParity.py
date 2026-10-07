import numpy as np
from pandas import DataFrame, Series
from scipy.optimize import minimize

from PositionSizing import riskContribution
from Strategies.PortfolioCombiner.CalibratedCombiner import CalibratedCombiner


class RiskParity(CalibratedCombiner):
    """Equal risk contribution: long-only capital weights such that every
    sub-strategy contributes the same share of the combined book's variance,
    using the full covariance matrix of the strategies' daily returns over
    the calibration window. Unlike InvVol it accounts for correlations: two
    highly correlated strategies share one risk budget.

    The covariance uses every calibration day, flat days included: for the
    combined book's risk, a strategy's flat days genuinely carry no risk."""

    def computeWeights(self, returns: DataFrame, metrics: DataFrame) -> Series:
        """Strategies with zero variance (never traded) get weight 0."""
        weights = Series(0.0, index=returns.columns)
        live = returns.columns[returns.var(ddof=1) > 0]
        if len(live) == 0:
            raise ValueError("No sub-strategy had positive volatility over the calibration window")

        cov = returns[live].cov().values
        weights[live] = self.equalRiskContribution(cov)
        self.riskContributions = self.riskShares(weights, returns.cov())
        return weights

    @staticmethod
    def equalRiskContribution(cov: np.ndarray) -> np.ndarray:
        """Long-only, fully invested weights minimizing the dispersion of each
        asset's share of portfolio volatility around 1/n. Starts from the
        inverse-vol solution (exact when correlations are all equal)."""
        n = len(cov)
        invVol = 1 / np.sqrt(np.diag(cov))
        w0 = invVol / invVol.sum()

        def dispersion(w):
            rc = riskContribution(w, cov)
            return ((rc / rc.sum() - 1 / n) ** 2).sum()

        result = minimize(
            dispersion, w0, method="SLSQP",
            bounds=[(1e-8, 1.0)] * n,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
            options={"ftol": 1e-14, "maxiter": 1000},
        )
        if not result.success:
            raise RuntimeError(f"Equal risk contribution optimization failed: {result.message}")
        return result.x / result.x.sum()

    @staticmethod
    def riskShares(weights: Series, cov: DataFrame) -> Series:
        """Each strategy's share of the combined book's volatility."""
        live = weights.index[weights > 0]
        rc = riskContribution(weights[live].values, cov.loc[live, live].values)
        return Series(rc / rc.sum(), index=live).reindex(weights.index, fill_value=0.0)
