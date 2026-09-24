import pandas as pd
import numpy as np
import statsmodels.api as sm
from hmmlearn.hmm import GaussianHMM

import DataLoader

def momentum(prices:pd.Series,skipMonths:int=1,lookbackMonth:int=12,tradingDayPerMonth:int=21)-> pd.Series:
    skipdays = skipMonths*tradingDayPerMonth
    tradeDays = tradingDayPerMonth*lookbackMonth

    skip = prices.shift(skipdays)
    trade = prices.shift(skipdays+tradeDays)

    result = skip/trade - 1.0
    return result

def logReturn(data:pd.Series)-> pd.Series:
    data['returns'] = np.log(data['Adj Close'])-np.log(data['Adj Close'].shift(1))
    data.dropna(inplace=True)

def trainTestSet(data:pd.Series,pourcentage:float=0.80)-> tuple[pd.Series,pd.Series]:
    n = len(data)
    train = data.iloc[:int(n*pourcentage)]
    test = data.iloc[int(n*pourcentage):]
    return train,test

def changingRegime(data:pd.Series)-> pd.Series:
    logReturn(data)
    train,test = trainTestSet(data)
    hmm_model = HMM()
    hmm_model.fit(train[['returns']])
    test['market_regime'] = hmm_model.predict(np.array(test[['returns']]))
    return test

def marketRegime(data:pd.Series)-> tuple[pd.Series,pd.Series]:
    market = changingRegime(data)
    extreme_regime = market.loc[market['market_regime'] == 1]
    normal_regime = market.loc[market['market_regime'] == 0]
    return extreme_regime,normal_regime

def HMM(n_components:int=2,covariance_type:str='full')-> GaussianHMM:
    return GaussianHMM(n_components=n_components,covariance_type=covariance_type,n_iter=1000)

def famaFrenchRegression(tickers:list, startDate:str, endDate:str):
    """Regress a ticker's daily excess returns on the Fama-French 5 factors + momentum."""
    prices = DataLoader.getData(startDate, endDate, tickers, "1d")
    logReturn(prices)
    returns = prices['returns'] * 100

    ff = DataLoader.getFamaFrenchData(startDate, endDate)

    merged = pd.concat([returns, ff], axis=1, join='inner')
    excess_return = merged['returns'] - merged['RF']
    factors = sm.add_constant(merged[['Mkt-RF', 'SMB', 'HML','RMW','CMA', 'Mom']])

    model = sm.OLS(excess_return, factors).fit()
    return model