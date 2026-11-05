import warnings

import pandas as pd
import requests

from DataLoaders.DataLoader import FIELDS, DataLoader


class KrakenLoader(DataLoader):
    """Crypto pairs from Kraken's public API (no API key needed), e.g.
    ["XBTUSD", "ETHEUR", "SOLUSD"]. Volume is in the base currency.

    Kraken only serves the 720 most recent candles of each interval: about
    2 years of daily bars, 30 days of hourly bars, 12 hours of 1-minute
    bars. A warning is raised when startDate is older than that. Candles
    still in progress are dropped.
    """

    NAME = "kraken"
    URL = "https://api.kraken.com/0/public/OHLC"
    MAX_CANDLES = 720
    # Yahoo-style interval -> Kraken interval in minutes
    INTERVALS = {
        "1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "60m": 60,
        "4h": 240, "1d": 1440, "1wk": 10080, "15d": 21600,
    }

    def download(self, symbols: list, startDate: str, endDate: str, interval: str) -> dict:
        return {pair: self._downloadPair(pair, startDate, endDate, self.INTERVALS[interval]) for pair in symbols}

    def _downloadPair(self, pair: str, startDate: str, endDate: str, minutes: int) -> pd.DataFrame:
        start, end = pd.Timestamp(startDate), pd.Timestamp(endDate)
        since = int(start.tz_localize("UTC").timestamp())
        try:
            response = requests.get(self.URL, params={"pair": pair, "interval": minutes, "since": since}, timeout=30)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise RuntimeError(f"Failed to download {pair} from Kraken") from exc
        if payload.get("error"):
            raise ValueError(f"Kraken error for {pair}: {payload['error']}")

        # the result holds one entry named after Kraken's pair id (XXBTZUSD
        # for XBTUSD) plus "last"
        rows = next(v for k, v in payload["result"].items() if k != "last")
        columns = ["time", "Open", "High", "Low", "Close", "VWAP", "Volume", "Count"]
        data = pd.DataFrame(rows, columns=columns)
        data.index = pd.to_datetime(data["time"], unit="s")
        data = data[FIELDS].astype(float)

        # the last candle is still open: its close is not final
        now = pd.Timestamp.now(tz="UTC").tz_localize(None)
        data = data[data.index + pd.Timedelta(minutes=minutes) <= now]

        if not data.empty and data.index[0] > start + pd.Timedelta(minutes=minutes):
            warnings.warn(
                f"Kraken only keeps the last {self.MAX_CANDLES} candles: {pair} starts on "
                f"{data.index[0]}, not {startDate}. Use a longer interval for more history.",
                stacklevel=4,
            )
        data = data[(data.index >= start) & (data.index < end)]
        if data.empty:
            raise ValueError(f"No Kraken data for {pair} between {startDate} and {endDate}")
        return data
