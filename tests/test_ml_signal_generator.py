import pandas as pd

from bot.ml_signal_generator import MLSignalGenerator


def test_crypto_fetch_prefers_ccxt(monkeypatch):
    generator = MLSignalGenerator()

    called = {"ccxt": False}

    def fake_ccxt_prices(symbol: str, timeframe: str = "1h", limit: int = 200):
        called["ccxt"] = True
        index = pd.date_range("2026-01-01", periods=60, freq="h")
        df = pd.DataFrame(
            {
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "volume": 1.0,
            },
            index=index,
        )
        generator._last_fetch_source = "ccxt"
        generator._last_fetch_timeframe = timeframe
        generator._last_fetch_rows = len(df)
        generator._last_fetch_first_ts = df.index[0].isoformat()
        generator._last_fetch_last_ts = df.index[-1].isoformat()
        return df

    def fail_yfinance(*_args, **_kwargs):
        raise AssertionError("yfinance should not be called for crypto when ccxt works")

    monkeypatch.setattr(generator, "_fetch_ccxt_prices", fake_ccxt_prices)
    monkeypatch.setattr(generator, "_fetch_yfinance_prices", fail_yfinance)

    df = generator._fetch_prices("BTC/USDT")
    assert called["ccxt"] is True
    assert df is not None
    assert len(df) == 60
    assert generator._last_fetch_source == "ccxt"
