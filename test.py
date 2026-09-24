import pandas as pd
import numpy as np
import DataLoader
from Strategies.Momentum import MomentumStrategy
from Strategies.MeanReversion import MeanReversion
from BacktestEngine import BacktestEngine

REAL_TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM", "XOM", "JNJ", "PG", "V", "UNH"]

prices = DataLoader.getOHLCV("2015-01-01","2025-01-01",REAL_TICKERS,"1d")

mean_reversion = MeanReversion(20)

backtest = BacktestEngine(mean_reversion, prices)

backtest.run("2015-06-01","2023-01-01")
a=1

