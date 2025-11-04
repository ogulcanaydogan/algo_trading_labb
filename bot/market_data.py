from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

logger = logging.getLogger(__name__)


class MarketDataClient(Protocol):
    """Protocol describing the minimal surface needed by the bot."""

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame: ...


@dataclass
class MarketDataError(RuntimeError):
    """Runtime error raised when a market data client cannot serve candles."""

    message: str

    def __str__(self) -> str:  # pragma: no cover - repr helper
        return self.message


class YFinanceMarketDataClient:
    """Load OHLCV candles via the public Yahoo Finance API (yfinance wrapper)."""

    _SUPPORTED_TIMEFRAMES = {
        "1m": "1m",
        "2m": "2m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "45m": "60m",  # fetch 60m, downsample later if needed
        "60m": "60m",
        "1h": "1h",
        "1d": "1d",
        "5d": "1d",  # fetch daily and slice
        "1wk": "1wk",
        "1mo": "1mo",
        "3mo": "3mo",
    }

    def __init__(self) -> None:
        try:
            import yfinance  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise MarketDataError(
                "The 'yfinance' package is required to fetch equities or commodity data. "
                "Install it with `pip install yfinance`."
            ) from exc

        self._yfinance = yfinance

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        interval = self._map_interval(timeframe)
        period = self._select_period(limit, timeframe)
        logger.debug(
            "Downloading candles via yfinance | symbol=%s timeframe=%s interval=%s period=%s limit=%d",
            symbol,
            timeframe,
            interval,
            period,
            limit,
        )
        try:
            frame = self._yfinance.download(
                tickers=symbol,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=False,
                threads=False,
            )
        except Exception as exc:  # pragma: no cover - network/HTTP layer
            raise MarketDataError(
                f"Failed to fetch data for {symbol} via yfinance: {exc}"
            ) from exc

        if frame.empty:
            raise MarketDataError(
                f"yfinance returned no data for {symbol} using interval={interval} period={period}."
            )

        normalized = self._normalize_frame(frame).tail(limit)
        if normalized.empty:
            raise MarketDataError(
                f"No usable OHLCV rows after normalization for {symbol} ({timeframe})."
            )
        return normalized

    def _map_interval(self, timeframe: str) -> str:
        tf = timeframe.strip().lower()
        if tf in self._SUPPORTED_TIMEFRAMES:
            return self._SUPPORTED_TIMEFRAMES[tf]
        if tf.endswith("h"):
            # yfinance does not expose >1h granularities natively; fall back to 1h and resample later.
            hours = int(tf[:-1])
            if hours % 1 == 0 and hours >= 1:
                return "1h"
        raise MarketDataError(
            f"Unsupported timeframe '{timeframe}' for yfinance market data. "
            "Supported granularities: 1m,2m,5m,15m,30m,1h,1d,1wk,1mo."
        )

    def _select_period(self, limit: int, timeframe: str) -> str:
        """
        Choose a yfinance 'period' string large enough to cover `limit` candles.

        Note: For intraday intervals on equities/commodities (which yfinance commonly
        serves), markets don't trade 24h. If we naively convert minutes to days we
        under-estimate the number of calendar days needed. To compensate, when the
        timeframe is intraday (< 1d), we scale the required calendar days by a
        conservative trading-hours factor (~6.5 hours per trading day).
        """
        minutes = self._timeframe_to_minutes(timeframe)
        total_minutes = max(minutes * limit, minutes)

        # If intraday, estimate calendar days using ~6.5 trading hours per day
        if minutes < 60 * 24:
            trading_minutes_per_day = 6.5 * 60  # NYSE regular session approximation
            # Add a 25% safety buffer to account for holidays/early closes/missing bars
            effective_days = math.ceil((total_minutes / trading_minutes_per_day) * 1.25)
            total_days = max(1, effective_days)
        else:
            total_days = total_minutes / (60 * 24)
        if total_days <= 7:
            return "7d"
        if total_days <= 30:
            return "1mo"
        if total_days <= 90:
            return "3mo"
        if total_days <= 180:
            return "6mo"
        if total_days <= 365:
            return "1y"
        if total_days <= 730:
            return "2y"
        if total_days <= 1825:
            return "5y"
        years = math.ceil(total_days / 365)
        return f"{years}y"

    def _normalize_frame(self, raw: pd.DataFrame) -> pd.DataFrame:
        # yfinance sometimes returns MultiIndex columns like (TICKER, Field).
        # Flatten to single-level by taking the last level (Field) when applicable.
        frame = raw.copy()
        if isinstance(frame.columns, pd.MultiIndex):
            try:
                frame.columns = [c[-1] if isinstance(c, tuple) else c for c in frame.columns]
            except Exception:  # pragma: no cover - defensive
                frame.columns = frame.columns.get_level_values(-1)

        frame = frame.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )
        for column in ("open", "high", "low", "close", "volume"):
            if column not in frame:
                raise MarketDataError(
                    f"Missing '{column}' column in yfinance response. "
                    "Ensure you are requesting standard OHLCV data."
                )
        index = frame.index
        if index.tz is None:
            frame.index = index.tz_localize("UTC")
        else:
            frame.index = index.tz_convert("UTC")
        frame.index.name = "timestamp"
        return frame[["open", "high", "low", "close", "volume"]]

    @staticmethod
    def _timeframe_to_minutes(timeframe: str) -> int:
        tf = timeframe.strip().lower()
        if tf.endswith("m"):
            return max(1, int(tf[:-1]))
        if tf.endswith("h"):
            return max(1, int(tf[:-1]) * 60)
        if tf.endswith("d"):
            return max(1, int(tf[:-1]) * 60 * 24)
        if tf.endswith("wk"):
            return max(1, int(tf[:-2]) * 60 * 24 * 7)
        if tf.endswith("mo"):
            return max(1, int(tf[:-2]) * 60 * 24 * 30)
        raise MarketDataError(f"Cannot interpret timeframe '{timeframe}'.")


def sanitize_symbol_for_fs(symbol: str) -> str:
    """Utility to build filesystem-friendly folder names for a trading symbol."""

    return "".join(ch if ch.isalnum() else "_" for ch in symbol).strip("_")
