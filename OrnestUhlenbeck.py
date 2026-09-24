import numpy as np
import pandas as pd
from scipy.optimize import minimize
from statsmodels.regression.linear_model import OLS


class OrnestUhlenbeck:


    def __init__(self,dt=1):
        self.theta=None
        self.sigma=None
        self.mu=None
        self.dt=dt

    def momentCalibration(self):
        return self.theta, self.sigma, self.mu

    def llh(self,params,X):
        theta, sigma, mu = params
        dt=self.dt
        diffX = X - X.shift(1)
        X_prev = X.shift(1).reindex(diffX.index)
        resid = diffX - theta*(mu-X_prev)*dt
        logf = -0.5*np.log(2*np.pi*sigma**2*dt) - resid**2/(2*sigma**2*dt)
        return -logf.sum()

    def methodMoment(self,X:pd.Series):
        dt=self.dt
        mu=X.mean()
        diffX = X-X.shift(1)
        diffX.dropna(inplace=True)
        X_prev = X.shift(1).reindex(diffX.index)
        exog = (mu-X_prev)*dt
        model = OLS(exog=exog,endog=diffX)
        res = model.fit()
        theta = res.params.iloc[0]
        resid = diffX - theta*exog
        sigma = resid.std()/np.sqrt(dt)
        return theta, sigma, mu


    def maxLikelihoodCalibration(self,prices:pd.Series):
        X=prices.copy()
        x0 = self.methodMoment(X)
        res = minimize(self.llh,x0,args=(X),method='Nelder-Mead')
        self.theta, self.sigma, self.mu = res.x
        return self.theta, self.sigma, self.mu
