from pandas import DataFrame, Series

from Strategies.PortfolioCombiner.CalibratedCombiner import CalibratedCombiner


class InvVol(CalibratedCombiner):
    """Capital weights proportional to 1 / realized volatility over the
    calibration window. Ignores correlations (see RiskParity for a version
    that uses the full covariance matrix)."""

    def computeWeights(self, returns: DataFrame, metrics: DataFrame) -> Series:
        """Volatility is measured on active days only (`activeVolatility`);
        strategies that never traded get weight 0."""
        vol = metrics.loc["activeVolatility"]
        invVol = (1 / vol.where(vol > 0)).fillna(0.0)
        if invVol.sum() == 0:
            raise ValueError("No sub-strategy had positive volatility over the calibration window")
        return invVol / invVol.sum()
