import numpy as np

def uniform(n_position,capital):
    return capital/n_position

def normalizeGrossExposure(weights):
    """Scale each row so gross (absolute) exposure sums to 1, leaving
    all-zero rows at 0."""
    gross = weights.abs().sum(axis=1)
    normalized = weights.div(gross, axis=0).where(gross > 0, 0.0)
    return normalized.fillna(0.0)

def dollarNeutralWeights(signal):
    """Demean a cross-sectional signal and normalize to unit gross exposure."""
    demeaned = signal.sub(signal.mean(axis=1), axis=0)
    return normalizeGrossExposure(demeaned)

def signedWeights(signal):
    """Normalize a sparse (mostly-zero) signal to unit gross exposure, without
    demeaning. Suited to signals that already encode direction directly
    (e.g. -1/0/+1), where demeaning would spread weight into zero-signal
    names that carry no edge."""
    return normalizeGrossExposure(signal)

def invVolSizing(capital,volitities):
    invVol=1/volitities
    weight=invVol/invVol.sum()
    return weight*capital

def KellyFraction(winProb,winLossRatio):
    return winProb - (1-winProb)/winLossRatio

def riskContribution(weights,covMatrix):
    portfolioVol = np.sqrt(weights @ covMatrix @ weights)
    marginalContribution = covMatrix @ weights / portfolioVol
    return weights * marginalContribution


def portfolioLeverage(targetVol,realizedVol):
    return targetVol/realizedVol

def needRebalancing(currentWeight,targetWeight,band=0.05):
    return np.abs(currentWeight-targetWeight)>band