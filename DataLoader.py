import os
import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
import datetime

data_path = "/Users/lucaszivkovic/ensimag/QuantResearchProjet/Project1/data/"

def getData(startDate:str, endDate:str, symbols:list, interval:str):
    path = getPath(startDate, endDate, symbols, interval)
    if os.path.exists(path):
        return _readCachedCsv(path, symbols)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    try:
        data = yf.download(
            tickers=symbols,
            start=startDate,
            end=endDate,
            interval=interval,
            auto_adjust=False,
            progress=False,
        )

    except Exception as exc:
        raise RuntimeError(
            f"Failed to download {symbols} from {startDate} to {endDate}"
        ) from exc

    if data.empty:
        raise ValueError(
            f"No data returned for {symbols} between {startDate} and {endDate}"
        )

    try:
        data.to_csv(path)
    except Exception as exc:
        raise RuntimeError(f"Unable to write data to {path}") from exc

    return data

def getOHLCV(startDate:str, endDate:str, symbols:list, interval:str)->pd.DataFrame:
    """Open/High/Low/Close/Volume data.

    For a single symbol: a flat DataFrame with those 5 columns.
    For multiple symbols: columns are (ticker, field), so data[ticker] gives
    that ticker's OHLCV DataFrame.
    """
    data = getData(startDate, endDate, symbols, interval)
    fields = ["Open", "High", "Low", "Close", "Volume"]
    if len(symbols) == 1:
        return data[fields]
    return data[fields].swaplevel(axis=1).sort_index(axis=1)

def _readCachedCsv(path:str, symbols:list):
    data = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    if len(symbols) == 1:
        data.columns = data.columns.get_level_values(0)
    return data

def getPath(startDate:str, endDate:str, symbols:list, interval:str)->str:
    datastr = '_'.join(str(s) for s in symbols)
    return f"{data_path}{datastr}_{startDate}_{endDate}_{interval}.csv".replace(' ', '')

def removeAll():
    removeList = os.listdir(data_path)
    for path in removeList: 
        removePath = data_path+path
        os.remove(removePath)

def removePath(path:str):
    path = data_path + path
    if os.path.exists(path):
        os.removePath(path)

def getFamaFrenchData(startDate:str, endDate:str):
    """Daily Fama-French 5 factors (Mkt-RF, SMB, HML, RMW, CMA, RF) plus momentum (Mom), in percent."""
    path = f"{data_path}FamaFrench_5F_Mom_{startDate}_{endDate}_daily.csv"
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    try:
        factors = web.DataReader("F-F_Research_Data_5_Factors_2x3_daily", "famafrench", start=startDate, end=endDate)[0]
        momentum = web.DataReader("F-F_Momentum_Factor_daily", "famafrench", start=startDate, end=endDate)[0]
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download Fama-French data from {startDate} to {endDate}"
        ) from exc

    momentum.columns = momentum.columns.str.strip()
    data = factors.join(momentum, how="inner")

    if data.empty:
        raise ValueError(
            f"No Fama-French data returned between {startDate} and {endDate}"
        )

    data.columns = data.columns.str.strip()

    try:
        data.to_csv(path)
    except Exception as exc:
        raise RuntimeError(f"Unable to write data to {path}") from exc

    return data

def fromDataReturnsSplit_Divident(data):
    splits = data.splits
    dividends = data.dividends
    return (splits,dividends)