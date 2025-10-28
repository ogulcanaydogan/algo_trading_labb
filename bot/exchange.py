from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    import ccxt  # type: ignore
except ImportError:  # pragma: no cover - optional dependency for the skeleton
    ccxt = None  # type: ignore

logger = logging.getLogger(__name__)


class ExchangeClient:
    """Thin wrapper around ccxt for fetching candles."""

    def __init__(
        self,
        exchange_id: str = "binance",
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        sandbox: bool = False,
        testnet: bool = False,
    ) -> None:
        if ccxt is None:
            raise RuntimeError(
                "ccxt not available. Install requirements or use PaperExchangeClient."
            )
        self.exchange_id = exchange_id
        self.sandbox = sandbox
        self.testnet = testnet
        exchange_class = getattr(ccxt, exchange_id)
        
        config = {
            "apiKey": api_key,
            "secret": api_secret,
        }
        
        # Binance Testnet configuration
        if testnet and exchange_id == "binance":
            config["urls"] = {
                "api": {
                    "public": "https://testnet.binance.vision/api",
                    "private": "https://testnet.binance.vision/api",
                }
            }
        
        self.client = exchange_class(config)
        
        if sandbox and hasattr(self.client, "set_sandbox_mode"):
            self.client.set_sandbox_mode(True)

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        limit: int = 250,
    ) -> pd.DataFrame:
        raw: List[List[float]] = self.client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        frame = pd.DataFrame(
            raw,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        frame.set_index("timestamp", inplace=True)
        return frame


class PaperExchangeClient:
    """
    Generates synthetic OHLCV candles for development.

    A seeded random walk keeps the structure deterministic across runs.
    """

    def __init__(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
        seed: Optional[int] = 42,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.seed = seed or random.randint(0, 1_000_000)
        self.last_price = 30_000.0
        self._rng = np.random.default_rng(self.seed)

    def fetch_ohlcv(self, limit: int = 250) -> pd.DataFrame:
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        freq = "1T" if self.timeframe.endswith("m") else "5T"
        index = pd.date_range(
            end=now,
            periods=limit,
            freq=freq,
        )
        close = self._generate_prices(limit)
        high = close + self._rng.normal(10, 5, size=limit)
        low = close - self._rng.normal(10, 5, size=limit)
        open_prices = np.roll(close, 1)
        open_prices[0] = close[0]
        volume = self._rng.normal(100, 20, size=limit).clip(min=1.0)

        frame = pd.DataFrame(
            {
                "open": open_prices,
                "high": np.maximum(high, close),
                "low": np.minimum(low, close),
                "close": close,
                "volume": volume,
            },
            index=index,
        )
        frame.index.name = "timestamp"
        return frame

    def _generate_prices(self, limit: int) -> np.ndarray:
        drift = 0.0002
        volatility = 0.002
        steps = self._rng.normal(drift, volatility, size=limit)
        prices = [self.last_price]
        for step in steps:
            prices.append(max(1.0, prices[-1] * (1 + step)))
        series = np.array(prices[1:])
        self.last_price = float(series[-1])
        return series

