import yfinance as yf

from DataLoaders.DataLoader import FIELDS, DataLoader


class YahooFinanceLoader(DataLoader):
    """Stocks, ETFs and indices from Yahoo Finance (no account needed),
    e.g. ["AAPL", "SPY", "^GSPC"].

    With `adjusted`, Open/High/Low/Close are scaled by Adj Close / Close, so
    they include dividends: an ex-dividend date is not a price drop, and
    Close returns are total returns. Volume is left as traded.

    Missing symbols are downloaded in one batch. A symbol with no data in
    the period (listed later, delisted) gets NaN columns.
    """

    NAME = "yahoo"
    INTERVALS = {i: i for i in ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h",
                                "1d", "5d", "1wk", "1mo", "3mo"]}

    def __init__(self, adjusted: bool = True, cachePath: str = None):
        super().__init__(cachePath)
        self.adjusted = adjusted

    def cacheKey(self) -> str:
        return f"{self.NAME}_{'adj' if self.adjusted else 'raw'}"

    def download(self, symbols: list, startDate: str, endDate: str, interval: str) -> dict:
        try:
            data = yf.download(
                tickers=symbols, start=startDate, end=endDate, interval=interval,
                auto_adjust=False, progress=False, multi_level_index=True,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to download {symbols} from {startDate} to {endDate}") from exc
        if data.empty:
            raise ValueError(f"No data returned for {symbols} between {startDate} and {endDate}")

        # intraday bars come with a timezone (New York for one symbol, UTC
        # for several): make them naive UTC like the other loaders
        if data.index.tz is not None:
            data.index = data.index.tz_convert("UTC").tz_localize(None)

        frames = {}
        for symbol in symbols:
            raw = data.xs(symbol, axis=1, level="Ticker")
            ohlcv = raw[FIELDS].copy()
            if self.adjusted:
                factor = raw["Adj Close"] / raw["Close"]
                for field in ["Open", "High", "Low", "Close"]:
                    ohlcv[field] = raw[field] * factor
            frames[symbol] = ohlcv
        return frames
