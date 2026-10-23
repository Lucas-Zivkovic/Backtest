"""Tests of the three price loaders (Yahoo Finance, Kraken, Interactive Brokers).

The offline tests replace each source's API with a fake one, so they check
parsing, filtering, adjustment and caching without the network. The
`realdata` tests download real prices and check they look like market data.
"""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

import DataLoaders.KrakenLoader as KrakenLoader_module
import DataLoaders.YahooFinanceLoader as YahooFinanceLoader_module
from DataLoaders.DataLoader import FIELDS
from DataLoaders.IBKRLoader import IBKRLoader
from DataLoaders.KrakenLoader import KrakenLoader
from DataLoaders.YahooFinanceLoader import YahooFinanceLoader


def assertValidOHLCV(data: pd.DataFrame):
    """Checks one symbol's bars look like prices."""
    assert sorted(data.columns) == sorted(FIELDS)
    assert isinstance(data.index, pd.DatetimeIndex)
    assert data.index.tz is None
    assert data.index.is_unique and data.index.is_monotonic_increasing
    assert not data.isna().any().any()
    assert (data[["Open", "High", "Low", "Close"]] > 0).all().all()
    assert (data["Volume"] >= 0).all()
    # small tolerance: adjusted prices are rescaled floats
    assert (data["High"] >= data[["Open", "Close"]].max(axis=1) * (1 - 1e-9)).all()
    assert (data["Low"] <= data[["Open", "Close"]].min(axis=1) * (1 + 1e-9)).all()


# --------------------------------------------------------------------------
# Base class
# --------------------------------------------------------------------------

@pytest.mark.parametrize("loader", [YahooFinanceLoader, KrakenLoader, IBKRLoader])
def test_loader_rejects_unknown_interval(loader, tmp_path):
    with pytest.raises(ValueError, match="interval"):
        loader(cachePath=str(tmp_path)).getOHLCV("2024-01-01", "2024-02-01", ["X"], "7m")


def test_cacheKey_separates_price_settings(tmp_path):
    # adjusted and raw prices of the same symbol must not share a cache file
    path = str(tmp_path)
    files = {
        YahooFinanceLoader(adjusted=True, cachePath=path).cacheFile("AAPL", "a", "b", "1d"),
        YahooFinanceLoader(adjusted=False, cachePath=path).cacheFile("AAPL", "a", "b", "1d"),
        IBKRLoader(adjusted=True, cachePath=path).cacheFile("AAPL", "a", "b", "1d"),
        IBKRLoader(whatToShow="MIDPOINT", cachePath=path).cacheFile("AAPL", "a", "b", "1d"),
        IBKRLoader(useRTH=False, cachePath=path).cacheFile("AAPL", "a", "b", "1d"),
        KrakenLoader(cachePath=path).cacheFile("AAPL", "a", "b", "1d"),
    }
    assert len(files) == 6


# --------------------------------------------------------------------------
# Yahoo Finance (yf.download replaced)
# --------------------------------------------------------------------------

def fakeYahooDownload(symbols, dates, calls=None, missing=()):
    """yf.download stand-in: Close = 100, 101, ...; Adj Close = Close / 2,
    so adjusted prices are half the raw ones. Symbols in `missing` get NaN,
    like Yahoo does for a ticker with no data."""
    def download(tickers, start, end, interval, **kwargs):
        if calls is not None:
            calls.append(list(tickers))
        index = pd.DatetimeIndex([d for d in dates if pd.Timestamp(start) <= d < pd.Timestamp(end)], name="Date")
        columns = {}
        for i, symbol in enumerate(tickers):
            close = 100.0 + np.arange(len(index)) + 10 * i
            fields = {
                "Open": close - 1, "High": close + 2, "Low": close - 2, "Close": close,
                "Adj Close": close / 2, "Volume": np.full(len(index), 1000.0),
            }
            for field, values in fields.items():
                columns[(field, symbol)] = np.nan if symbol in missing else values
        data = pd.DataFrame(columns, index=index)
        data.columns.names = ["Price", "Ticker"]
        return data
    return download


def test_yahooLoader_adjustsPrices_andCaches(monkeypatch, tmp_path):
    dates = pd.bdate_range("2024-01-01", "2024-01-31")
    calls = []
    monkeypatch.setattr(YahooFinanceLoader_module.yf, "download", fakeYahooDownload(["AAPL", "MSFT"], dates, calls))

    adjusted = YahooFinanceLoader(adjusted=True, cachePath=str(tmp_path))
    data = adjusted.getOHLCV("2024-01-01", "2024-02-01", ["MSFT", "AAPL"], "1d")

    # one batch for both symbols, (symbol, field) columns sorted by symbol
    assert calls == [["MSFT", "AAPL"]]
    assert list(data.columns.get_level_values(0).unique()) == ["AAPL", "MSFT"]
    assert len(data) == len(dates)
    for symbol in ["AAPL", "MSFT"]:
        assertValidOHLCV(data[symbol][FIELDS])
    # OHLC scaled by Adj Close / Close = 1/2, volume untouched
    first = data["MSFT"].iloc[0]
    assert first["Close"] == pytest.approx(50.0)
    assert first["Open"] == pytest.approx(49.5)
    assert first["High"] == pytest.approx(51.0)
    assert first["Volume"] == 1000.0

    # raw prices are a separate cache entry
    raw = YahooFinanceLoader(adjusted=False, cachePath=str(tmp_path)).getOHLCV("2024-01-01", "2024-02-01", ["MSFT"], "1d")
    assert list(raw.columns) == FIELDS
    assert raw["Close"].iloc[0] == pytest.approx(100.0)
    assert len(calls) == 2

    # cached symbols are not downloaded again, only the new one is
    calls.clear()
    monkeypatch.setattr(YahooFinanceLoader_module.yf, "download",
                        fakeYahooDownload(["GOOGL"], dates, calls))
    again = adjusted.getOHLCV("2024-01-01", "2024-02-01", ["AAPL", "MSFT", "GOOGL"], "1d")
    assert calls == [["GOOGL"]]
    pd.testing.assert_frame_equal(again[["AAPL", "MSFT"]], data, check_freq=False, check_names=False)


def test_yahooLoader_symbolWithoutData_getsNaN(monkeypatch, tmp_path):
    dates = pd.bdate_range("2024-01-01", "2024-01-31")
    monkeypatch.setattr(YahooFinanceLoader_module.yf, "download",
                        fakeYahooDownload(["AAPL", "DEAD"], dates, missing=("DEAD",)))
    data = YahooFinanceLoader(cachePath=str(tmp_path)).getOHLCV("2024-01-01", "2024-02-01", ["AAPL", "DEAD"], "1d")
    assert data["DEAD"].isna().all().all()
    assertValidOHLCV(data["AAPL"])


def test_yahooLoader_emptyAnswer_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(YahooFinanceLoader_module.yf, "download", lambda *a, **k: pd.DataFrame())
    with pytest.raises(ValueError, match="No data"):
        YahooFinanceLoader(cachePath=str(tmp_path)).getOHLCV("2024-01-01", "2024-02-01", ["AAPL"], "1d")


# --------------------------------------------------------------------------
# Kraken (requests.get replaced)
# --------------------------------------------------------------------------

class FakeKrakenResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def krakenRows(times, close0=100.0):
    # [time, open, high, low, close, vwap, volume, count], numbers as strings like Kraken
    return [[int(t.timestamp()), str(close0 + i - 1), str(close0 + i + 2), str(close0 + i - 2),
             str(close0 + i), str(close0 + i), "3.5", 10] for i, t in enumerate(times)]


def test_krakenLoader_parsesAndDropsOpenCandle(monkeypatch, tmp_path):
    # 3 daily candles; the last one is still open and must be dropped.
    today = pd.Timestamp.now(tz="UTC").normalize()
    days = [today - pd.Timedelta(days=n) for n in (2, 1, 0)]
    rows = krakenRows(days, close0=11)
    payload = {"error": [], "result": {"XXBTZUSD": rows, "last": rows[-1][0]}}
    monkeypatch.setattr(KrakenLoader_module.requests, "get", lambda *a, **k: FakeKrakenResponse(payload))

    loader = KrakenLoader(cachePath=str(tmp_path))
    start = (today - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    end = (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    data = loader.getOHLCV(start, end, ["XBTUSD"], "1d")

    assertValidOHLCV(data)
    assert len(data) == 2
    assert data["Close"].tolist() == [11.0, 12.0]
    assert data["Volume"].tolist() == [3.5, 3.5]
    assert data.index[0] == days[0].tz_localize(None)
    # second call reads the cache
    monkeypatch.setattr(KrakenLoader_module.requests, "get", lambda *a, **k: pytest.fail("cache not used"))
    assert loader.getOHLCV(start, end, ["XBTUSD"], "1d").equals(data)


def test_krakenLoader_requestsEachPair_andFiltersDates(monkeypatch, tmp_path):
    # hourly candles for 2 pairs; only those in [startDate, endDate) are kept
    hours = pd.date_range("2024-03-01", periods=72, freq="h", tz="UTC")
    params = []

    def get(url, params=None, timeout=None):
        sent.append(params)
        close0 = 100.0 if params["pair"] == "XBTUSD" else 5.0
        return FakeKrakenResponse({"error": [], "result": {"X" + params["pair"]: krakenRows(hours, close0), "last": 0}})

    sent = params
    monkeypatch.setattr(KrakenLoader_module.requests, "get", get)
    data = KrakenLoader(cachePath=str(tmp_path)).getOHLCV("2024-03-01", "2024-03-02", ["XBTUSD", "ETHUSD"], "1h")

    assert [p["pair"] for p in params] == ["XBTUSD", "ETHUSD"]
    assert all(p["interval"] == 60 for p in params)
    assert params[0]["since"] == int(pd.Timestamp("2024-03-01", tz="UTC").timestamp())
    assert list(data.columns.get_level_values(0).unique()) == ["ETHUSD", "XBTUSD"]
    assert len(data) == 24
    assert data.index[0] == pd.Timestamp("2024-03-01 00:00")
    assert data.index[-1] == pd.Timestamp("2024-03-01 23:00")
    for pair in ["XBTUSD", "ETHUSD"]:
        assertValidOHLCV(data[pair])
    assert data["ETHUSD"]["Close"].iloc[0] == 5.0


def test_krakenLoader_warnsWhenHistoryIsTooShort(monkeypatch, tmp_path):
    # asked from January but Kraken's candles only start in March
    hours = pd.date_range("2024-03-01", periods=24, freq="h", tz="UTC")
    payload = {"error": [], "result": {"XXBTZUSD": krakenRows(hours), "last": 0}}
    monkeypatch.setattr(KrakenLoader_module.requests, "get", lambda *a, **k: FakeKrakenResponse(payload))
    with pytest.warns(UserWarning, match="720"):
        data = KrakenLoader(cachePath=str(tmp_path)).getOHLCV("2024-01-01", "2024-03-02", ["XBTUSD"], "1h")
    assert len(data) == 24


def test_krakenLoader_apiError_raises(monkeypatch, tmp_path):
    payload = {"error": ["EQuery:Unknown asset pair"], "result": {}}
    monkeypatch.setattr(KrakenLoader_module.requests, "get", lambda *a, **k: FakeKrakenResponse(payload))
    with pytest.raises(ValueError, match="Unknown asset pair"):
        KrakenLoader(cachePath=str(tmp_path)).getOHLCV("2024-01-01", "2024-02-01", ["NOPEUSD"], "1d")


# --------------------------------------------------------------------------
# Interactive Brokers (IB client replaced)
# --------------------------------------------------------------------------

def test_ibkrLoader_dailyAdjusted_oneRequest_filteredToPeriod(monkeypatch, tmp_path):
    # Daily adjusted stock bars: one ADJUSTED_LAST request ending now (IB's
    # rule), then cut down to [startDate, endDate).
    pytest.importorskip("ib_async")
    from ib_async import BarData

    days = pd.bdate_range("2022-12-01", "2023-03-01")
    requests_made = []

    class FakeIB:
        def qualifyContracts(self, contract):
            return [contract]

        def reqHistoricalData(self, contract, endDateTime, durationStr, **kwargs):
            requests_made.append((contract.symbol, endDateTime, durationStr, kwargs["whatToShow"]))
            return [BarData(date=d.date(), open=10 + i, high=12 + i, low=9 + i, close=11 + i, volume=500)
                    for i, d in enumerate(days)]

        def disconnect(self):
            pass

    loader = IBKRLoader(cachePath=str(tmp_path))
    monkeypatch.setattr(loader, "connect", lambda: FakeIB())
    data = loader.getOHLCV("2023-01-01", "2023-02-01", ["AAPL"], "1d")

    assert len(requests_made) == 1
    symbol, endDateTime, duration, show = requests_made[0]
    assert (symbol, endDateTime, show) == ("AAPL", "", "ADJUSTED_LAST")
    assert duration.endswith(" Y")
    assertValidOHLCV(data)
    expected = days[(days >= "2023-01-01") & (days < "2023-02-01")]
    assert list(data.index) == list(expected)
    assert data["Volume"].iloc[0] == 500.0

    # cached: IB is not called again
    monkeypatch.setattr(loader, "connect", lambda: pytest.fail("cache not used"))
    assert loader.getOHLCV("2023-01-01", "2023-02-01", ["AAPL"], "1d").equals(data)


def test_ibkrLoader_forexUsesMidpoint(monkeypatch, tmp_path):
    pytest.importorskip("ib_async")
    from ib_async import BarData

    shows = []

    class FakeIB:
        def qualifyContracts(self, contract):
            return [contract]

        def reqHistoricalData(self, contract, endDateTime, durationStr, **kwargs):
            shows.append(kwargs["whatToShow"])
            return [BarData(date=dt.date(2024, 1, d), open=1.1, high=1.2, low=1.0, close=1.15, volume=-1)
                    for d in (2, 3, 4)]

        def disconnect(self):
            pass

    loader = IBKRLoader(secType="CASH", cachePath=str(tmp_path))
    monkeypatch.setattr(loader, "connect", lambda: FakeIB())
    data = loader.getOHLCV("2024-01-01", "2024-01-05", ["EURUSD"], "1d")
    assert shows == ["MIDPOINT"]
    assert len(data) == 3
    assert data["Close"].tolist() == [1.15] * 3


def test_ibkrLoader_noBars_raises(monkeypatch, tmp_path):
    pytest.importorskip("ib_async")

    class FakeIB:
        def qualifyContracts(self, contract):
            return [contract]

        def reqHistoricalData(self, *args, **kwargs):
            return []

        def disconnect(self):
            pass

    loader = IBKRLoader(adjusted=False, cachePath=str(tmp_path))
    monkeypatch.setattr(loader, "connect", lambda: FakeIB())
    with pytest.raises(ValueError, match="No IBKR data"):
        loader.getOHLCV("2024-01-01", "2024-02-01", ["AAPL"], "1d")


def test_ibkrLoader_walksBackInChunks(monkeypatch, tmp_path):
    # Hourly bars: IBKRLoader must split the period into 1-month requests,
    # walking back from endDate, and stitch them with no duplicates.
    pytest.importorskip("ib_async")
    from ib_async import BarData

    hours = pd.date_range("2024-01-01", "2024-04-01", freq="h", tz="UTC", inclusive="left")
    requests_made = []

    class FakeIB:
        def qualifyContracts(self, contract):
            return [contract]

        def reqHistoricalData(self, contract, endDateTime, durationStr, **kwargs):
            requests_made.append((endDateTime, durationStr, kwargs["whatToShow"]))
            end = pd.Timestamp(endDateTime)
            chunk = hours[(hours < end) & (hours >= end - pd.Timedelta(days=30))]
            return [BarData(date=t.to_pydatetime(), open=1, high=2, low=0.5, close=1.5, volume=10) for t in chunk]

        def disconnect(self):
            pass

    loader = IBKRLoader(cachePath=str(tmp_path))
    loader.PACING_SLEEP = 0
    monkeypatch.setattr(loader, "connect", lambda: FakeIB())
    with pytest.warns(UserWarning, match="intraday"):
        data = loader.getOHLCV("2024-01-01", "2024-04-01", ["AAPL", "MSFT"], "1h")

    assert len(requests_made) >= 6
    assert all(show == "TRADES" for _, _, show in requests_made)
    assert list(data.columns.get_level_values(0).unique()) == ["AAPL", "MSFT"]
    assert data.index.is_unique and data.index.is_monotonic_increasing
    assert len(data) == len(hours)
    assert data.index[0] == pd.Timestamp("2024-01-01 00:00")


# --------------------------------------------------------------------------
# Real data
# --------------------------------------------------------------------------

@pytest.mark.realdata
def test_yahooLoader_on_real_data(tmp_path):
    # 2023: 250 NYSE trading days; AAPL pays dividends, so adjusted closes
    # are below raw ones, more so further in the past.
    adjusted = YahooFinanceLoader(adjusted=True, cachePath=str(tmp_path)).getOHLCV(
        "2023-01-01", "2024-01-01", ["AAPL", "MSFT"], "1d")
    raw = YahooFinanceLoader(adjusted=False, cachePath=str(tmp_path)).getOHLCV(
        "2023-01-01", "2024-01-01", ["AAPL"], "1d")

    assert len(adjusted) == 250
    assert adjusted.index[0] == pd.Timestamp("2023-01-03")
    assert adjusted.index[-1] == pd.Timestamp("2023-12-29")
    assert (adjusted.index.dayofweek < 5).all()
    for symbol in ["AAPL", "MSFT"]:
        assertValidOHLCV(adjusted[symbol])
    assertValidOHLCV(raw)
    assert raw["Close"].iloc[-1] == pytest.approx(192.53, abs=0.01)  # AAPL close on 2023-12-29

    ratio = adjusted["AAPL"]["Close"] / raw["Close"]
    assert (ratio < 1).all()
    assert ratio.iloc[0] < ratio.iloc[-1]
    assert (adjusted["AAPL"]["Volume"] == raw["Volume"]).all()


@pytest.mark.realdata
def test_yahooLoader_on_real_data_splitAdjusted(tmp_path):
    # AAPL split 4-for-1 on 2020-08-31: prices must not drop by 75% that day
    data = YahooFinanceLoader(cachePath=str(tmp_path)).getOHLCV("2020-08-01", "2020-10-01", ["AAPL"], "1d")
    assert data["Close"].pct_change().abs().max() < 0.2


@pytest.mark.realdata
def test_krakenLoader_on_real_data(tmp_path):
    today = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    start = (today - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    end = (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    data = KrakenLoader(cachePath=str(tmp_path)).getOHLCV(start, end, ["XBTUSD", "ETHUSD"], "1d")

    # crypto trades every day; today's candle is still open, so it is dropped
    assert list(data.index) == list(pd.date_range(start, today - pd.Timedelta(days=1), freq="D"))
    for pair in ["XBTUSD", "ETHUSD"]:
        assertValidOHLCV(data[pair])
    assert (data["XBTUSD"]["Close"] > data["ETHUSD"]["Close"]).all()


@pytest.mark.realdata
def test_krakenLoader_matches_yahoo_bitcoin(tmp_path):
    # Two sources, same asset: Kraken XBTUSD and Yahoo BTC-USD daily closes
    # (both UTC days) should agree within about 1%.
    today = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    start = (today - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    end = (today - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    kraken = KrakenLoader(cachePath=str(tmp_path)).getOHLCV(start, end, ["XBTUSD"], "1d")["Close"]
    yahoo = YahooFinanceLoader(cachePath=str(tmp_path)).getOHLCV(start, end, ["BTC-USD"], "1d")["Close"]

    both = pd.concat({"kraken": kraken, "yahoo": yahoo}, axis=1).dropna()
    assert len(both) >= 50
    gap = (both["kraken"] / both["yahoo"] - 1).abs()
    assert gap.median() < 0.01


@pytest.mark.realdata
def test_yahooLoader_on_real_data_intraday(tmp_path):
    # Hourly bars of the last week: naive UTC timestamps (like the other
    # loaders), inside NYSE hours (13:30-20:00 UTC), same index for one or
    # several symbols. Yahoo keeps 730 days of hourly bars.
    today = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    start, end = (today - pd.Timedelta(days=10)).strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")
    one = YahooFinanceLoader(cachePath=str(tmp_path)).getOHLCV(start, end, ["AAPL"], "1h")
    both = YahooFinanceLoader(cachePath=str(tmp_path / "multi")).getOHLCV(start, end, ["AAPL", "MSFT"], "1h")

    assertValidOHLCV(one)
    assert len(one) >= 5 * 6
    assert (one.index.dayofweek < 5).all()
    minutes = one.index.hour * 60 + one.index.minute
    assert ((minutes >= 13 * 60 + 30) & (minutes < 20 * 60 + 30)).all()
    pd.testing.assert_frame_equal(both["AAPL"][FIELDS], one, check_freq=False, check_names=False)


@pytest.mark.realdata
def test_yahooLoader_on_real_data_index(tmp_path):
    # ^GSPC: no dividends, so adjusted = raw; it closed 2023 at 4769.83
    data = YahooFinanceLoader(cachePath=str(tmp_path)).getOHLCV("2023-01-01", "2024-01-01", ["^GSPC"], "1d")
    assert len(data) == 250
    assert data["Close"].iloc[-1] == pytest.approx(4769.83, abs=0.01)
    assert data[["Open", "High", "Low", "Close"]].gt(3000).all().all()


@pytest.mark.realdata
def test_krakenLoader_on_real_data_hourly(tmp_path):
    # last 2 days of hourly candles: one per hour, no gaps, open candle dropped
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    start = (now.normalize() - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    end = (now.normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    data = KrakenLoader(cachePath=str(tmp_path)).getOHLCV(start, end, ["XBTUSD"], "1h")

    assertValidOHLCV(data)
    assert list(data.index) == list(pd.date_range(start, now.floor("h") - pd.Timedelta(hours=1), freq="h"))


# Interactive Brokers: needs TWS / IB Gateway running (IBKR_PORT, default
# 4001). The tests are skipped otherwise.

@pytest.fixture(scope="module")
def ibkrAvailable():
    pytest.importorskip("ib_async")
    try:
        IBKRLoader().connect().disconnect()
    except ConnectionError:
        pytest.skip("TWS / IB Gateway not running")


@pytest.mark.realdata
def test_ibkrLoader_on_real_data(ibkrAvailable, tmp_path):
    data = IBKRLoader(cachePath=str(tmp_path)).getOHLCV("2023-01-01", "2024-01-01", ["AAPL", "MSFT"], "1d")
    assert len(data) == 250
    assert data.index[0] == pd.Timestamp("2023-01-03")
    assert data.index[-1] == pd.Timestamp("2023-12-29")
    for symbol in ["AAPL", "MSFT"]:
        assertValidOHLCV(data[symbol])

    # IB and Yahoo adjusted closes should agree closely
    yahoo = YahooFinanceLoader(cachePath=str(tmp_path)).getOHLCV("2023-01-01", "2024-01-01", ["AAPL"], "1d")
    gap = (data["AAPL"]["Close"] / yahoo["Close"] - 1).abs()
    assert gap.median() < 0.01


@pytest.mark.realdata
def test_ibkrLoader_on_real_data_raw(ibkrAvailable, tmp_path):
    # raw trades: AAPL closed 2023 at 192.53, like on Yahoo, and above the
    # dividend-adjusted price
    raw = IBKRLoader(adjusted=False, cachePath=str(tmp_path)).getOHLCV("2023-12-01", "2024-01-01", ["AAPL"], "1d")
    adjusted = IBKRLoader(cachePath=str(tmp_path)).getOHLCV("2023-12-01", "2024-01-01", ["AAPL"], "1d")
    assertValidOHLCV(raw)
    assert raw["Close"].iloc[-1] == pytest.approx(192.53, abs=0.01)
    assert (adjusted["Close"] < raw["Close"]).all()


@pytest.mark.realdata
def test_ibkrLoader_on_real_data_intraday(ibkrAvailable, tmp_path):
    # Hourly regular-hours bars of the last week: naive UTC, inside NYSE
    # hours, close to Yahoo's daily close at the end of each day.
    today = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    start, end = (today - pd.Timedelta(days=10)).strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")
    with pytest.warns(UserWarning, match="intraday"):
        data = IBKRLoader(cachePath=str(tmp_path)).getOHLCV(start, end, ["AAPL"], "1h")

    assertValidOHLCV(data)
    assert (data.index.dayofweek < 5).all()
    minutes = data.index.hour * 60 + data.index.minute
    assert ((minutes >= 13 * 60 + 30) & (minutes < 21 * 60)).all()

    lastBar = data["Close"].groupby(data.index.normalize()).last()
    yahoo = YahooFinanceLoader(adjusted=False, cachePath=str(tmp_path)).getOHLCV(start, end, ["AAPL"], "1d")["Close"]
    both = pd.concat({"ibkr": lastBar, "yahoo": yahoo}, axis=1).dropna()
    assert len(both) >= 5
    assert ((both["ibkr"] / both["yahoo"] - 1).abs() < 0.005).all()


@pytest.mark.realdata
def test_ibkrLoader_on_real_data_forex(ibkrAvailable, tmp_path):
    # EURUSD midpoints (no volume on forex: IB sends -1) against Yahoo's
    # EURUSD=X. Their days end at different hours, so allow 1%.
    today = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    start, end = (today - pd.Timedelta(days=60)).strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")
    fx = IBKRLoader(secType="CASH", cachePath=str(tmp_path)).getOHLCV(start, end, ["EURUSD"], "1d")
    yahoo = YahooFinanceLoader(cachePath=str(tmp_path)).getOHLCV(start, end, ["EURUSD=X"], "1d")["Close"]

    prices = fx[["Open", "High", "Low", "Close"]]
    assert ((prices > 0.5) & (prices < 2)).all().all()
    assert (fx["High"] >= prices.max(axis=1)).all() and (fx["Low"] <= prices.min(axis=1)).all()
    both = pd.concat({"ibkr": fx["Close"], "yahoo": yahoo}, axis=1).dropna()
    assert len(both) >= 30
    assert ((both["ibkr"] / both["yahoo"] - 1).abs() < 0.01).all()
