from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Literal

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

Decision = Literal["LONG", "SHORT", "FLAT"]


@dataclass
class StrategyConfig:
    symbol: str = "BTC/USDT"
    timeframe: str = "1m"
    ema_fast: int = 12
    ema_slow: int = 26
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    risk_per_trade_pct: float = 0.5
    stop_loss_pct: float = 0.004
    take_profit_pct: float = 0.008

    @classmethod
    def from_env(cls) -> "StrategyConfig":
        """Create a strategy config from environment variables."""

        return cls(
            symbol=os.getenv("SYMBOL", cls.symbol),
            timeframe=os.getenv("TIMEFRAME", cls.timeframe),
            ema_fast=int(os.getenv("EMA_FAST", str(cls.ema_fast))),
            ema_slow=int(os.getenv("EMA_SLOW", str(cls.ema_slow))),
            rsi_period=int(os.getenv("RSI_PERIOD", str(cls.rsi_period))),
            rsi_overbought=float(os.getenv("RSI_OVERBOUGHT", str(cls.rsi_overbought))),
            rsi_oversold=float(os.getenv("RSI_OVERSOLD", str(cls.rsi_oversold))),
            risk_per_trade_pct=float(
                os.getenv("RISK_PER_TRADE_PCT", str(cls.risk_per_trade_pct))
            ),
            stop_loss_pct=float(os.getenv("STOP_LOSS_PCT", str(cls.stop_loss_pct))),
            take_profit_pct=float(os.getenv("TAKE_PROFIT_PCT", str(cls.take_profit_pct))),
        )


def compute_indicators(
    ohlcv: pd.DataFrame,
    config: StrategyConfig,
) -> pd.DataFrame:
    """
    Enrich OHLCV dataframe with EMA and RSI indicators.

    Expected OHLCV columns: open, high, low, close, volume.
    """
    if len(ohlcv) < config.ema_slow + 5:
        raise ValueError("Not enough data for indicator calculation.")

    copy = ohlcv.copy()
    copy.sort_index(inplace=True)
    copy["ema_fast"] = EMAIndicator(
        close=copy["close"],
        window=config.ema_fast,
    ).ema_indicator()
    copy["ema_slow"] = EMAIndicator(
        close=copy["close"],
        window=config.ema_slow,
    ).ema_indicator()
    copy["rsi"] = RSIIndicator(
        close=copy["close"],
        window=config.rsi_period,
    ).rsi()

    return copy


def generate_signal(enriched: pd.DataFrame, config: StrategyConfig) -> Dict[str, float | str]:
    """
    Generate a trading signal based on EMA crossover and RSI confirmation.
    """
    last = enriched.iloc[-1]
    prev = enriched.iloc[-2]

    decision: Decision = "FLAT"
    confidence = 0.0
    reason = "No clear signal."

    ema_cross_up = prev["ema_fast"] <= prev["ema_slow"] and last["ema_fast"] > last["ema_slow"]
    ema_cross_down = prev["ema_fast"] >= prev["ema_slow"] and last["ema_fast"] < last["ema_slow"]

    if ema_cross_up and last["rsi"] < config.rsi_overbought:
        decision = "LONG"
        confidence = min(1.0, (last["ema_fast"] - last["ema_slow"]) / last["close"] * 500)
        confidence = max(confidence, 0.4)
        reason = "EMA fast crossed above EMA slow with RSI confirmation."
    elif ema_cross_down and last["rsi"] > config.rsi_oversold:
        decision = "SHORT"
        confidence = min(1.0, (last["ema_slow"] - last["ema_fast"]) / last["close"] * 500)
        confidence = max(confidence, 0.4)
        reason = "EMA fast crossed below EMA slow with RSI confirmation."
    elif last["rsi"] >= config.rsi_overbought:
        decision = "SHORT"
        confidence = 0.3
        reason = "RSI indicates overbought conditions."
    elif last["rsi"] <= config.rsi_oversold:
        decision = "LONG"
        confidence = 0.3
        reason = "RSI indicates oversold conditions."

    return {
        "decision": decision,
        "confidence": round(confidence, 4),
        "ema_fast": float(last["ema_fast"]),
        "ema_slow": float(last["ema_slow"]),
        "rsi": float(last["rsi"]),
        "close": float(last["close"]),
        "reason": reason,
    }


def calculate_position_size(
    balance: float,
    risk_pct: float,
    price: float,
    stop_loss_pct: float,
) -> float:
    """
    Basic fixed-fraction position sizing model.
    """
    risk_amount = balance * (risk_pct / 100)
    stop_distance = price * stop_loss_pct
    if stop_distance <= 0:
        return 0.0
    size = risk_amount / stop_distance
    return max(size, 0.0)
