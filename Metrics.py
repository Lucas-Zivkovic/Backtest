import numpy as np

def sharpRatio(returns,riskFreeRate=0.0025,periodsPerYears=252):
    excessReturns = returns - riskFreeRate/periodsPerYears
    return np.sqrt(periodsPerYears)*excessReturns.mean()/excessReturns.std(ddof=1)

def sortinoRatio(returns,targetReturn,periodsPerYears=252):
    excessReturns = returns - targetReturn/periodsPerYears
    downsideReturns = np.minimum(excessReturns,0)
    downsideDeviation = np.sqrt((downsideReturns**2).mean())
    return np.sqrt(periodsPerYears) * excessReturns.mean() / downsideDeviation

def maxDrawdown(cumulativeReturns):
    runningMax = np.maximum.accumulate(cumulativeReturns)
    drawdown = cumulativeReturns / runningMax - 1
    return drawdown.min()

def drawdownDuration(cumulativeReturns):
    runningMax = np.maximum.accumulate(cumulativeReturns)
    drawdown = cumulativeReturns < runningMax
    duration = []
    count=0
    for i in drawdown:
        count = count+1 if i else 0
        duration.append(count)
    return max(duration)


def turnover(weights,weightsMinus):
    return np.abs(weights-weightsMinus).sum()/2

def annualizedTurnover(weights,periodsPerYears=252):
    turnovers = [turnover(weights[t],weights[t-1]) for t in range (1,len(weights))]
    return np.mean(turnovers)*periodsPerYears

def hitRatio(returns):
    return (returns>0).sum()/len(returns)

def payoffRatio(returns):
    wins = returns[returns > 0]
    losses = returns[returns<0]
    return wins.mean() /abs(losses.mean())

def expectancy(returns):
    hr = hitRatio(returns)
    pr = payoffRatio(returns)
    return hr*pr - (1-hr)