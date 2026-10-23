import datetime as dt
import math
import os
import time
import warnings

import pandas as pd

from DataLoaders.DataLoader import DataLoader


class IBKRLoader(DataLoader):
    """Prices from Interactive Brokers. TWS or IB Gateway must be running
    with the API enabled (Settings > API > Enable ActiveX and Socket
    Clients). Needs ib_async (Python 3.10+).

    Connection: host / port / clientId, defaulting to the environment
    variables IBKR_HOST (127.0.0.1), IBKR_PORT (4001 = Gateway live; 7497 TWS paper, 7496 TWS
    live, 4002 Gateway paper) and IBKR_CLIENT_ID (17).

    Symbols are tickers built with secType/exchange/currency ("STK" or
    "ETF" stock, "CASH" forex pair like "EURUSD", "IND" index, "CRYPTO"),
    or ib_async Contract objects for anything else (futures, options...).

    `adjusted` uses IB's ADJUSTED_LAST prices (dividends and splits) for
    daily-or-longer stock bars; other bars are raw trades. `whatToShow`
    overrides this ("TRADES", "MIDPOINT", "BID", "ASK"...). `useRTH` keeps
    regular trading hours only. Without a market data subscription, IB may
    refuse the request or return delayed data.
    """

    NAME = "ibkr"
    # Yahoo-style interval -> (IB bar size, length of one request). IB limits
    # how much history one request returns, so long intraday periods are split.
    INTERVALS = {
        "1m": ("1 min", "1 D"), "2m": ("2 mins", "1 D"), "5m": ("5 mins", "1 W"),
        "15m": ("15 mins", "1 W"), "30m": ("30 mins", "1 M"), "1h": ("1 hour", "1 M"),
        "60m": ("1 hour", "1 M"), "4h": ("4 hours", "1 M"),
        "1d": ("1 day", None), "1wk": ("1 week", None), "1mo": ("1 month", None),
    }
    DURATION_UNITS = {"D": pd.Timedelta(days=1), "W": pd.Timedelta(weeks=1), "M": pd.Timedelta(days=30)}
    PACING_SLEEP = 0.5  # seconds between requests: IB allows 60 requests / 10 min

    def __init__(self, secType: str = "STK", exchange: str = "SMART", currency: str = "USD",
                 adjusted: bool = True, whatToShow: str = None, useRTH: bool = True,
                 host: str = None, port: int = None, clientId: int = None, cachePath: str = None):
        super().__init__(cachePath)
        self.secType = secType
        self.exchange = exchange
        self.currency = currency
        self.adjusted = adjusted
        self.whatToShow = whatToShow
        self.useRTH = useRTH
        self.host = host or os.environ.get("IBKR_HOST", "127.0.0.1")
        self.port = int(port or os.environ.get("IBKR_PORT", "4001"))
        self.clientId = int(clientId or os.environ.get("IBKR_CLIENT_ID", "17"))

    def cacheKey(self) -> str:
        show = self.whatToShow or ("adj" if self.adjusted else "raw")
        return f"{self.NAME}_{self.secType}_{show}_{'rth' if self.useRTH else 'all'}"

    def symbolName(self, symbol) -> str:
        if isinstance(symbol, str):
            return symbol
        return symbol.localSymbol or symbol.symbol

    def download(self, symbols: list, startDate: str, endDate: str, interval: str) -> dict:
        ib = self.connect()
        try:
            frames = {}
            for symbol in symbols:
                contract = self.contract(ib, symbol)
                show = self.whatToShow or self._defaultWhatToShow(contract.secType, interval)
                frames[self.symbolName(symbol)] = self._downloadContract(ib, contract, startDate, endDate, interval, show)
            return frames
        finally:
            ib.disconnect()

    def connect(self):
        try:
            from ib_async import IB
        except ImportError as exc:
            raise ImportError("IBKRLoader needs ib_async (Python 3.10+): uv add ib_async") from exc
        ib = IB()
        try:
            ib.connect(self.host, self.port, clientId=self.clientId, readonly=True, timeout=10)
        except Exception as exc:
            raise ConnectionError(
                f"Cannot connect to TWS / IB Gateway at {self.host}:{self.port} (clientId {self.clientId}). "
                "Is it running with the API enabled?"
            ) from exc
        return ib

    def contract(self, ib, symbol):
        from ib_async import Contract, Crypto, Forex, Index, Stock
        if isinstance(symbol, Contract):
            contract = symbol
        elif self.secType in ("STK", "ETF"):
            contract = Stock(symbol, self.exchange, self.currency)
        elif self.secType == "CASH":
            contract = Forex(symbol)
        elif self.secType == "IND":
            contract = Index(symbol, self.exchange, self.currency)
        elif self.secType == "CRYPTO":
            contract = Crypto(symbol, "PAXOS" if self.exchange == "SMART" else self.exchange, self.currency)
        else:
            raise ValueError(f"secType {self.secType!r} needs ib_async Contract objects instead of tickers")
        qualified = ib.qualifyContracts(contract)
        if not qualified or qualified[0] is None:
            raise ValueError(f"IB does not know the contract {contract}")
        return qualified[0]

    def _defaultWhatToShow(self, secType: str, interval: str) -> str:
        if secType == "CASH":
            return "MIDPOINT"  # forex has no trades
        if secType == "CRYPTO":
            return "AGGTRADES"
        if self.adjusted and secType == "STK":
            if self.INTERVALS[interval][1] is None:
                return "ADJUSTED_LAST"
            warnings.warn("IB only adjusts daily-or-longer bars: intraday bars are raw trades", stacklevel=5)
        return "TRADES"

    def _downloadContract(self, ib, contract, startDate: str, endDate: str, interval: str,
                          whatToShow: str) -> pd.DataFrame:
        from ib_async import util
        barSize, chunk = self.INTERVALS[interval]
        start, end = pd.Timestamp(startDate), pd.Timestamp(endDate)
        name = self.symbolName(contract)

        def request(endDateTime, durationStr):
            bars = ib.reqHistoricalData(
                contract, endDateTime=endDateTime, durationStr=durationStr, barSizeSetting=barSize,
                whatToShow=whatToShow, useRTH=self.useRTH, formatDate=2,
            )
            return util.df(bars) if bars else None

        pieces = []
        if chunk is None:
            # daily or longer: one request covers the whole period.
            # ADJUSTED_LAST must end now, so it asks from startDate to today.
            adjusted = whatToShow == "ADJUSTED_LAST"
            lastDay = pd.Timestamp.now().normalize() if adjusted else end
            years = max(1, math.ceil((lastDay - start).days / 365.25))
            df = request("" if adjusted else end.tz_localize("UTC").to_pydatetime(), f"{years} Y")
            if df is not None:
                pieces.append(df)
        else:
            # intraday: walk back from endDate one chunk at a time
            step = self.DURATION_UNITS[chunk[-1]] * int(chunk.split()[0])
            cursor = end
            while cursor > start:
                df = request(cursor.tz_localize("UTC").to_pydatetime(), chunk)
                if df is None:
                    cursor -= step  # holiday or weekend: no bars in this chunk
                    continue
                pieces.append(df)
                first = self._index(df["date"])[0]
                cursor = first if first < cursor else cursor - step
                time.sleep(self.PACING_SLEEP)

        if not pieces:
            raise ValueError(f"No IBKR data for {name} between {startDate} and {endDate}")
        raw = pd.concat(pieces)
        data = pd.DataFrame({
            "Open": raw["open"].values, "High": raw["high"].values, "Low": raw["low"].values,
            "Close": raw["close"].values, "Volume": raw["volume"].values,
        }, index=self._index(raw["date"]))
        data = data[~data.index.duplicated()].sort_index()
        data = data[(data.index >= start) & (data.index < end)].astype(float)
        if data.empty:
            raise ValueError(f"No IBKR data for {name} between {startDate} and {endDate}")
        return data

    @staticmethod
    def _index(dates: pd.Series) -> pd.DatetimeIndex:
        """IB dates: datetime.date for daily bars, UTC datetimes for intraday
        bars. Returns naive timestamps (UTC for intraday)."""
        if isinstance(dates.iloc[0], dt.datetime):
            return pd.DatetimeIndex(pd.to_datetime(list(dates), utc=True)).tz_localize(None)
        return pd.DatetimeIndex(pd.to_datetime(list(dates)))
