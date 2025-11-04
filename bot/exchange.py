from __future__ import annotations

import importlib
import logging
import os
import random
import re
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import pandas as pd

ccxt = None  # type: ignore


def _load_ccxt():
    global ccxt  # type: ignore
    if ccxt is not None:
        return ccxt
    try:
        os.environ.setdefault("SETUPTOOLS_SCM_PRETEND_VERSION", "0.0.0")
    except Exception:  # pragma: no cover - defensive
        pass
    module = importlib.import_module("ccxt")
    ccxt = module  # type: ignore
    return module

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
        try:
            module = _load_ccxt()
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "ccxt not available. Install requirements or use PaperExchangeClient."
            ) from exc
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
        # Parse timeframe like '1m', '5m', '1h', '4h', '1d'
        tf = self.timeframe.strip().lower()
        match = re.match(r"^(\d+)([mhd])$", tf)
        freq = "1min"
        if match:
            value = int(match.group(1))
            unit = match.group(2)
            if unit == "m":
                freq = f"{value}min"
            elif unit == "h":
                freq = f"{value}H"
            elif unit == "d":
                freq = f"{value}D"
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
