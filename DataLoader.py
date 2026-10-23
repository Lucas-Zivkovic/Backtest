"""Data loaders with a local CSV cache.

Prices come from the classes in DataLoaders/, which all return the same
OHLCV format:
    YahooFinanceLoader   stocks, ETFs, indices (no account needed)
    KrakenLoader         crypto pairs (no account needed)
    IBKRLoader           Interactive Brokers (TWS or IB Gateway running)

This module keeps the shortcuts used across the project: getOHLCV (Yahoo),
the raw yfinance download getData, S&P 500 constituents (Wikipedia) and
Fama-French factors.
"""
import hashlib
import io
import os

import pandas as pd
import pandas_datareader.data as web
import requests
import yfinance as yf

from DataLoaders.YahooFinanceLoader import YahooFinanceLoader

# Download cache: data/ next to this file (git-ignored); set QUANT_DATA_PATH
# to point it somewhere else.
data_path = os.path.join(
    os.environ.get("QUANT_DATA_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")),
    "",
)


def getOHLCV(startDate:str, endDate:str, symbols:list, interval:str, adjusted:bool=True)->pd.DataFrame:
    """Open/High/Low/Close/Volume from Yahoo Finance, see YahooFinanceLoader.
    For Kraken or Interactive Brokers, use KrakenLoader / IBKRLoader."""
    return YahooFinanceLoader(adjusted=adjusted, cachePath=data_path).getOHLCV(startDate, endDate, symbols, interval)


def getData(startDate:str, endDate:str, symbols:list, interval:str):
    """Raw yfinance download (with Adj Close), cached as CSV."""
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

def _readCachedCsv(path:str, symbols:list):
    data = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    if len(symbols) == 1:
        data.columns = data.columns.get_level_values(0)
    return data

def getPath(startDate:str, endDate:str, symbols:list, interval:str)->str:
    datastr = '_'.join(str(s) for s in symbols)
    if len(datastr) > 150:
        # a file name is limited to 255 characters: name long universes by
        # their size and a hash of the symbols instead
        digest = hashlib.sha1(datastr.encode()).hexdigest()[:12]
        datastr = f"{len(symbols)}tickers_{digest}"
    return f"{data_path}{datastr}_{startDate}_{endDate}_{interval}.csv".replace(' ', '')

# --------------------------------------------------------------------------
# Other data
# --------------------------------------------------------------------------

def getSP500Constituents()->pd.DataFrame:
    """Current S&P 500 members from Wikipedia: columns Symbol (in Yahoo
    format, BRK.B -> BRK-B), Security, Sector, SubIndustry. Downloaded once
    and cached as sp500_constituents.csv in the data folder. These are
    today's members: a backtest on them has survivorship bias, it never
    holds a company that was dropped from the index or went bankrupt."""
    path = f"{data_path}sp500_constituents.csv"
    if os.path.exists(path):
        return pd.read_csv(path)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        # Wikipedia refuses requests without a browser-like user agent
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        response.raise_for_status()
        table = pd.read_html(io.StringIO(response.text), attrs={"id": "constituents"})[0]
    except Exception as exc:
        raise RuntimeError(f"Failed to download the S&P 500 constituents from {url}") from exc

    constituents = pd.DataFrame({
        "Symbol": table["Symbol"].str.replace(".", "-", regex=False),
        "Security": table["Security"],
        "Sector": table["GICS Sector"],
        "SubIndustry": table["GICS Sub-Industry"],
    })
    constituents.to_csv(path, index=False)
    return constituents

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


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------

def removeAll():
    for path in os.listdir(data_path):
        os.remove(data_path + path)

def removePath(path:str):
    path = data_path + path
    if os.path.exists(path):
        os.remove(path)
