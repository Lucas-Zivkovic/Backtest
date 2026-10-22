from itertools import combinations

from pandas import DataFrame,Series
from PerformanceAnalyzer import PerformanceAnalyzer
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def PairFinding(returns: DataFrame) -> Series:
    performance = PerformanceAnalyzer()
    corr = performance.correlationMatrix(returns)
    dist_corr = np.sqrt(2*(1-corr))
    pca = PCA(n_components=0.85)
    X_pca=pca.fit_transform(dist_corr)

    X_pca = X_pca[:, 1:]

    dbscan = DBSCAN(eps=0.4, min_samples=3, metric='euclidean')
    labels = dbscan.fit_predict(X_pca)

    # correlationMatrix() returns a 0..n-1 index, put the tickers back
    results = Series(labels, index=returns.columns, name="Cluster")
    return results


JOHANSEN_COLUMN = {0.10: 0, 0.05: 1, 0.01: 2}


def johansen(logA: Series, logB: Series, significance: float = 0.05) -> tuple[float, float]:
    """Johansen trace test of no cointegration between two log price series.
    Returns the trace statistic over its critical value at `significance`
    (> 1: cointegrated) and the hedge ratio beta of the cointegrating vector,
    spread = log a - beta * log b. Unlike Engle-Granger, swapping a and b
    gives the same statistic (and beta becomes 1 / beta, the same spread)."""
    result = coint_johansen(np.column_stack([logA, logB]), det_order=0, k_ar_diff=1)
    vector = result.evec[:, 0]
    return result.lr1[0] / result.cvt[0, JOHANSEN_COLUMN[significance]], -vector[1] / vector[0]


def findPairs(prices: DataFrame, significance: float = 0.05, groups: Series = None) -> DataFrame:
    """Every pair inside a group, tested with the Johansen
    trace test on log prices over the whole period and over each half of it.
    Columns: a, b, cluster, strength (trace / critical value) on the whole
    period and `strengthFirst` / `strengthSecond` on each half, `beta` (whole
    period), and `stable`: cointegrated on the whole period and on both
    halves with beta > 0 (long one leg, short the other). A pair that is only
    cointegrated on part of the period is a relationship that came or went,
    or chance: with ~35 pairs tested, one or two pass at 5% even when none is
    cointegrated. Sorted stable first, then by the weakest of the three
    strengths.

    The groups are `groups` (ticker -> label, e.g. the GICS sub-industry) when
    given, else the PairFinding clusters (DBSCAN noise excluded). Tickers with
    a missing price in `prices` are left out."""
    columns = ["a", "b", "cluster", "strength", "strengthFirst", "strengthSecond", "beta", "stable"]
    prices = prices.dropna(axis=1)
    if groups is None:
        clusters = PairFinding(prices.pct_change().dropna())
        clusters = clusters[clusters >= 0]
    else:
        clusters = groups.reindex(prices.columns).dropna()
    logPrices = np.log(prices)
    half = len(logPrices) // 2
    pairs = []
    for label in sorted(clusters.unique()):
        members = clusters.index[clusters == label]
        for a, b in combinations(members, 2):
            strength, beta = johansen(logPrices[a], logPrices[b], significance)
            first, _ = johansen(logPrices[a].iloc[:half], logPrices[b].iloc[:half], significance)
            second, _ = johansen(logPrices[a].iloc[half:], logPrices[b].iloc[half:], significance)
            pairs.append({"a": a, "b": b, "cluster": label, "strength": strength,
                          "strengthFirst": first, "strengthSecond": second, "beta": beta,
                          "stable": min(strength, first, second) > 1 and beta > 0})
    if not pairs:
        return DataFrame(columns=columns)
    pairs = DataFrame(pairs, columns=columns)
    weakest = pairs[["strength", "strengthFirst", "strengthSecond"]].min(axis=1)
    order = pairs.assign(weakest=weakest).sort_values(["stable", "weakest"], ascending=False).index
    return pairs.loc[order].reset_index(drop=True)


def findParameters(prices: DataFrame, pair) -> tuple[float, float]:
    """Johansen hedge ratio on log prices and the mean of the spread:
    spread = log a - beta * log b - alpha has mean 0 over `prices`."""
    log_a, log_b = np.log(prices[pair["a"]]), np.log(prices[pair["b"]])
    _, beta = johansen(log_a, log_b)
    alpha = float((log_a - beta * log_b).mean())
    return alpha, beta
