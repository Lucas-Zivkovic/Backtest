import os
from abc import ABC, abstractmethod

import pandas as pd
from pandas import DataFrame

FIELDS = ["Open", "High", "Low", "Close", "Volume"]


def defaultCachePath() -> str:
    """data/ at the repo root (git-ignored); QUANT_DATA_PATH overrides it."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.environ.get("QUANT_DATA_PATH", os.path.join(root, "data"))


class DataLoader(ABC):
    """Common interface for OHLCV price sources.

    `getOHLCV` returns the same format whatever the source: for one symbol,
    a DataFrame with columns Open/High/Low/Close/Volume; for several, columns
    are (symbol, field), so data[symbol] gives that symbol's OHLCV, aligned
    on the union of dates (NaN where a symbol has no bar). Daily bars are
    indexed by date, intraday bars by naive UTC timestamps.

    Each symbol is cached as one CSV per (source settings, period, interval),
    so only the symbols not seen yet are downloaded.

    Subclasses define `INTERVALS` (Yahoo-style interval -> the source's own
    notation) and implement `download`.
    """

    NAME = "base"
    INTERVALS: dict = {}

    def __init__(self, cachePath: str = None):
        self.cachePath = cachePath or defaultCachePath()

    @abstractmethod
    def download(self, symbols: list, startDate: str, endDate: str, interval: str) -> dict:
        """OHLCV of each symbol between startDate (included) and endDate
        (excluded), as {symbolName(symbol): DataFrame with FIELDS columns}."""

    def getOHLCV(self, startDate: str, endDate: str, symbols: list, interval: str = "1d") -> DataFrame:
        if interval not in self.INTERVALS:
            raise ValueError(f"{type(self).__name__} interval must be one of {list(self.INTERVALS)}, got {interval!r}")

        frames, missing = {}, []
        for symbol in symbols:
            path = self.cacheFile(self.symbolName(symbol), startDate, endDate, interval)
            if os.path.exists(path):
                frames[self.symbolName(symbol)] = pd.read_csv(path, index_col=0, parse_dates=True)
            else:
                missing.append(symbol)

        if missing:
            downloaded = self.download(missing, startDate, endDate, interval)
            for symbol in missing:
                name = self.symbolName(symbol)
                if name not in downloaded:
                    raise ValueError(f"{type(self).__name__} returned no data for {name}")
                data = downloaded[name][FIELDS]
                data.index.name = "Date"
                self._writeCsv(data, self.cacheFile(name, startDate, endDate, interval))
                frames[name] = data

        return self._combine([self.symbolName(s) for s in symbols], frames)

    def symbolName(self, symbol) -> str:
        """Column name of a symbol in the result (and in its cache file)."""
        return str(symbol)

    def cacheKey(self) -> str:
        """Cache file prefix: the source plus any setting that changes the
        prices (adjustment, price type...), so they never mix."""
        return self.NAME

    def cacheFile(self, name: str, startDate: str, endDate: str, interval: str) -> str:
        safe = name.replace("/", "-").replace(" ", "")
        return os.path.join(self.cachePath, f"{self.cacheKey()}_{safe}_{startDate}_{endDate}_{interval}.csv")

    @staticmethod
    def _writeCsv(data: DataFrame, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            data.to_csv(path)
        except Exception as exc:
            raise RuntimeError(f"Unable to write data to {path}") from exc

    @staticmethod
    def _combine(names: list, frames: dict) -> DataFrame:
        if len(names) == 1:
            return frames[names[0]][FIELDS]
        data = pd.concat({name: frames[name][FIELDS] for name in names}, axis=1)
        return data.sort_index().sort_index(axis=1)
