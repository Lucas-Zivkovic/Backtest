import io
import os
import zipfile

import pandas as pd
import requests

from DataLoaders.DataLoader import FIELDS, DataLoader


class KrakenHistoryLoader(DataLoader):
    """Full Kraken price history from its downloadable OHLCVT archive: every
    pair since its listing, unlike KrakenLoader whose API only serves the
    last 720 candles. Pair names as on Kraken: ["XBTUSD", "ETHUSD", ...].

    The archive is a ZIP of headerless CSVs named <PAIR>_<minutes>.csv, with
    columns timestamp (unix seconds, UTC), open, high, low, close, volume (in
    the base currency), trades. Kraken updates it every quarter, so it ends
    at the end of the last full quarter.

    `archive` is where to read it from:
      - None (default): Kraken's servers, the full archive (~9.5 GB in 5
        parts). Only the files asked for are read, with HTTP range requests,
        a few MB per pair: nothing else is downloaded;
      - the path of a local ZIP (e.g. a quarterly update), or of a folder
        holding the extracted CSVs.

    Kraken writes a candle only for intervals with trades. Missing bars are
    filled between a pair's first and last candle: Open = High = Low =
    Close = the last close, Volume 0, so every pair has a bar at every step.
    """

    NAME = "krakenhist"
    URL = "https://assets.kraken.com/marketing/institutions/"
    PARTS = [f"Kraken_OHLCVT_Full.zip.part0{i}" for i in range(5)]
    # Yahoo-style interval -> minutes in the archive's file names
    INTERVALS = {
        "1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "60m": 60,
        "4h": 240, "12h": 720, "1d": 1440,
    }

    def __init__(self, archive: str = None, cachePath: str = None):
        super().__init__(cachePath)
        self.archive = archive

    def download(self, symbols: list, startDate: str, endDate: str, interval: str) -> dict:
        minutes = self.INTERVALS[interval]
        frames = {}
        with self._open() as source:
            for pair in symbols:
                raw = source(f"{pair}_{minutes}.csv")
                frames[pair] = self._bars(raw, pair, startDate, endDate, minutes)
        return frames

    def _open(self):
        """Context manager giving a function: file name -> its bytes."""
        if self.archive is not None and os.path.isdir(self.archive):
            return _Folder(self.archive)
        if self.archive is not None:
            return _Zip(open(self.archive, "rb"))
        remote = _RemoteFile([self.URL + part for part in self.PARTS])
        return _Zip(io.BufferedReader(remote, buffer_size=1 << 20))

    @staticmethod
    def _bars(raw: bytes, pair: str, startDate: str, endDate: str, minutes: int) -> pd.DataFrame:
        columns = ["time", "Open", "High", "Low", "Close", "Volume", "Trades"]
        data = pd.read_csv(io.BytesIO(raw), header=None, names=columns)
        data.index = pd.to_datetime(data["time"], unit="s")
        data = data[FIELDS].astype(float)
        data = data[~data.index.duplicated()].sort_index()

        # fill the bars without trades, between the first and last candle
        grid = pd.date_range(data.index[0], data.index[-1], freq=pd.Timedelta(minutes=minutes))
        data = data.reindex(grid)
        close = data["Close"].ffill()
        for field in ["Open", "High", "Low"]:
            data[field] = data[field].fillna(close)
        data["Close"] = close
        data["Volume"] = data["Volume"].fillna(0.0)

        start, end = pd.Timestamp(startDate), pd.Timestamp(endDate)
        data = data[(data.index >= start) & (data.index < end)]
        if data.empty:
            raise ValueError(f"No Kraken history for {pair} between {startDate} and {endDate}")
        return data


class _Zip:
    def __init__(self, file):
        self.file = file
        self.zip = zipfile.ZipFile(file)

    def __enter__(self):
        names = set(self.zip.namelist())

        def read(name):
            if name not in names:
                raise ValueError(f"{name} is not in the Kraken archive")
            return self.zip.read(name)
        return read

    def __exit__(self, *exc):
        self.zip.close()
        self.file.close()


class _Folder:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        def read(name):
            path = os.path.join(self.path, name)
            if not os.path.exists(path):
                raise ValueError(f"{name} is not in {self.path}")
            with open(path, "rb") as f:
                return f.read()
        return read

    def __exit__(self, *exc):
        pass


class _RemoteFile(io.RawIOBase):
    """Read-only file over the concatenation of remote files (the archive's
    parts), fetched in 1 MB blocks with HTTP range requests, so zipfile can
    read one member without downloading the whole archive."""

    BLOCK = 1 << 20

    def __init__(self, urls: list):
        self.session = requests.Session()
        self.parts, offset = [], 0
        for url in urls:
            response = self.session.head(url, timeout=30)
            response.raise_for_status()
            size = int(response.headers["content-length"])
            self.parts.append((url, offset, size))
            offset += size
        self.size, self.position, self.blocks = offset, 0, {}

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        self.position = base + offset
        return self.position

    def readinto(self, buffer):
        wanted = max(0, min(len(buffer), self.size - self.position))
        done = 0
        while done < wanted:
            index, offset = divmod(self.position, self.BLOCK)
            chunk = self._block(index)[offset:offset + wanted - done]
            buffer[done:done + len(chunk)] = chunk
            done += len(chunk)
            self.position += len(chunk)
        return done

    def _block(self, index: int) -> bytes:
        if index not in self.blocks:
            first, last = index * self.BLOCK, min((index + 1) * self.BLOCK, self.size) - 1
            data = b""
            for url, offset, size in self.parts:
                low, high = max(first, offset), min(last, offset + size - 1)
                if low <= high:
                    response = self.session.get(url, headers={"Range": f"bytes={low - offset}-{high - offset}"},
                                                timeout=60)
                    response.raise_for_status()
                    data += response.content
            if len(self.blocks) > 64:
                self.blocks.clear()
            self.blocks[index] = data
        return self.blocks[index]

    def close(self):
        self.session.close()
        super().close()
