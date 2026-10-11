from itertools import combinations

from pandas import DataFrame,Series
from PerformanceAnalyzer import PerformanceAnalyzer
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from statsmodels.tsa.stattools import coint

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


def findPairs(prices: DataFrame) -> DataFrame:
    """Every pair inside a PairFinding cluster, ranked by its Engle-Granger
    cointegration p-value on log prices (most cointegrated first). Columns:
    a, b, cluster, pValue."""
    clusters = PairFinding(prices.pct_change().dropna())
    logPrices = np.log(prices)
    pairs = []
    for label in sorted(clusters[clusters >= 0].unique()):
        members = clusters.index[clusters == label]
        for a, b in combinations(members, 2):
            pairs.append({"a": a, "b": b, "cluster": label,
                          "pValue": coint(logPrices[a], logPrices[b])[1]})
    if not pairs:
        return DataFrame(columns=["a", "b", "cluster", "pValue"])
    return DataFrame(pairs).sort_values("pValue").reset_index(drop=True)


def findParameters(prices: DataFrame, pair) -> tuple[float, float]:
    """Hedge ratio of log a on log b: log a = alpha + beta * log b."""
    a, b = pair["a"], pair["b"]
    log_a, log_b = np.log(prices[a]), np.log(prices[b])
    beta, alpha = np.polyfit(log_b, log_a, 1)
    return alpha, beta
