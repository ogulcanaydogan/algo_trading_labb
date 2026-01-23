from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Literal, Optional

import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, ADXIndicator, MACD
from ta.volatility import AverageTrueRange

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
    # New parameters for improved strategy
    adx_period: int = 14
    adx_threshold: float = 20.0  # Minimum ADX for trend confirmation
    volume_ma_period: int = 20
    volume_threshold: float = 1.0  # Volume must be >= this multiple of MA
    atr_period: int = 14
    atr_stop_multiplier: float = 2.0  # Stop loss = ATR * multiplier
    atr_tp_multiplier: float = 3.0  # Take profit = ATR * multiplier
    use_atr_stops: bool = True  # Use ATR-based stops instead of percentage
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    use_macd_confirmation: bool = True
    use_volume_confirmation: bool = True
    use_adx_filter: bool = True
    min_confidence_threshold: float = 0.4  # Minimum confidence to generate signal

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
            risk_per_trade_pct=float(os.getenv("RISK_PER_TRADE_PCT", str(cls.risk_per_trade_pct))),
            stop_loss_pct=float(os.getenv("STOP_LOSS_PCT", str(cls.stop_loss_pct))),
            take_profit_pct=float(os.getenv("TAKE_PROFIT_PCT", str(cls.take_profit_pct))),
            adx_period=int(os.getenv("ADX_PERIOD", str(cls.adx_period))),
            adx_threshold=float(os.getenv("ADX_THRESHOLD", str(cls.adx_threshold))),
            volume_ma_period=int(os.getenv("VOLUME_MA_PERIOD", str(cls.volume_ma_period))),
            volume_threshold=float(os.getenv("VOLUME_THRESHOLD", str(cls.volume_threshold))),
            atr_period=int(os.getenv("ATR_PERIOD", str(cls.atr_period))),
            atr_stop_multiplier=float(
                os.getenv("ATR_STOP_MULTIPLIER", str(cls.atr_stop_multiplier))
            ),
            atr_tp_multiplier=float(os.getenv("ATR_TP_MULTIPLIER", str(cls.atr_tp_multiplier))),
            use_atr_stops=os.getenv("USE_ATR_STOPS", "true").lower() == "true",
            use_macd_confirmation=os.getenv("USE_MACD_CONFIRMATION", "true").lower() == "true",
            use_volume_confirmation=os.getenv("USE_VOLUME_CONFIRMATION", "true").lower() == "true",
            use_adx_filter=os.getenv("USE_ADX_FILTER", "true").lower() == "true",
        )


def compute_indicators(
    ohlcv: pd.DataFrame,
    config: StrategyConfig,
) -> pd.DataFrame:
    """
    Enrich OHLCV dataframe with EMA, RSI, ADX, MACD, ATR, and volume indicators.

    Expected OHLCV columns: open, high, low, close, volume.
    """
    min_required = max(config.ema_slow, config.adx_period, config.volume_ma_period) + 10
    if len(ohlcv) < min_required:
        raise ValueError(
            f"Not enough data for indicator calculation. Need at least {min_required} bars."
        )

    copy = ohlcv.copy()
    copy.sort_index(inplace=True)

    # EMA indicators
    copy["ema_fast"] = EMAIndicator(
        close=copy["close"],
        window=config.ema_fast,
    ).ema_indicator()
    copy["ema_slow"] = EMAIndicator(
        close=copy["close"],
        window=config.ema_slow,
    ).ema_indicator()

    # RSI
    copy["rsi"] = RSIIndicator(
        close=copy["close"],
        window=config.rsi_period,
    ).rsi()

    # ADX (Average Directional Index) for trend strength
    adx_indicator = ADXIndicator(
        high=copy["high"],
        low=copy["low"],
        close=copy["close"],
        window=config.adx_period,
    )
    copy["adx"] = adx_indicator.adx()
    copy["adx_pos"] = adx_indicator.adx_pos()  # +DI
    copy["adx_neg"] = adx_indicator.adx_neg()  # -DI

    # MACD
    macd_indicator = MACD(
        close=copy["close"],
        window_fast=config.macd_fast,
        window_slow=config.macd_slow,
        window_sign=config.macd_signal,
    )
    copy["macd"] = macd_indicator.macd()
    copy["macd_signal"] = macd_indicator.macd_signal()
    copy["macd_histogram"] = macd_indicator.macd_diff()

    # ATR (Average True Range) for volatility-based stops
    atr_indicator = AverageTrueRange(
        high=copy["high"],
        low=copy["low"],
        close=copy["close"],
        window=config.atr_period,
    )
    copy["atr"] = atr_indicator.average_true_range()

    # Volume indicators
    copy["volume_ma"] = copy["volume"].rolling(window=config.volume_ma_period).mean()
    copy["volume_ratio"] = copy["volume"] / copy["volume_ma"]

    # EMA spread (normalized)
    copy["ema_spread"] = (copy["ema_fast"] - copy["ema_slow"]) / copy["close"] * 100

    # Momentum (rate of change)
    copy["momentum"] = copy["close"].pct_change(5) * 100

    return copy


def generate_signal(enriched: pd.DataFrame, config: StrategyConfig) -> Dict[str, float | str]:
    """
    Generate a trading signal based on multiple confirmations:
    1. EMA crossover (primary signal)
    2. RSI confirmation (not overbought/oversold against direction)
    3. ADX filter (trend strength > threshold)
    4. Volume confirmation (volume > MA)
    5. MACD confirmation (histogram direction)

    Returns enhanced signal with ATR-based stops.
    """
    last = enriched.iloc[-1]
    prev = enriched.iloc[-2]

    decision: Decision = "FLAT"
    confidence = 0.0
    reason = "No clear signal."
    confirmations = []

    # Calculate EMA crossover
    ema_cross_up = prev["ema_fast"] <= prev["ema_slow"] and last["ema_fast"] > last["ema_slow"]
    ema_cross_down = prev["ema_fast"] >= prev["ema_slow"] and last["ema_fast"] < last["ema_slow"]

    # Check if already in trend (EMA alignment without fresh cross)
    ema_bullish = last["ema_fast"] > last["ema_slow"]
    ema_bearish = last["ema_fast"] < last["ema_slow"]

    # ADX trend strength check
    adx_value = last["adx"] if not pd.isna(last["adx"]) else 0
    is_trending = adx_value >= config.adx_threshold
    trend_strength = min(1.0, adx_value / 40)  # Normalize ADX to 0-1

    # Volume confirmation
    volume_ratio = last["volume_ratio"] if not pd.isna(last["volume_ratio"]) else 1.0
    has_volume = volume_ratio >= config.volume_threshold

    # MACD confirmation
    macd_bullish = last["macd_histogram"] > 0 and last["macd_histogram"] > prev["macd_histogram"]
    macd_bearish = last["macd_histogram"] < 0 and last["macd_histogram"] < prev["macd_histogram"]

    # RSI levels
    rsi_value = last["rsi"] if not pd.isna(last["rsi"]) else 50
    rsi_not_overbought = rsi_value < config.rsi_overbought
    rsi_not_oversold = rsi_value > config.rsi_oversold
    rsi_oversold = rsi_value <= config.rsi_oversold
    rsi_overbought = rsi_value >= config.rsi_overbought

    # RSI divergence detection (simplified)
    # Bullish divergence: price makes lower low but RSI makes higher low
    # Bearish divergence: price makes higher high but RSI makes lower high
    lookback = min(10, len(enriched) - 1)
    if lookback > 2:
        recent_prices = enriched["close"].iloc[-lookback:]
        recent_rsi = enriched["rsi"].iloc[-lookback:]

        price_lower_low = last["close"] < recent_prices.min()
        rsi_higher_low = rsi_value > recent_rsi.min()
        bullish_divergence = price_lower_low and rsi_higher_low and rsi_oversold

        price_higher_high = last["close"] > recent_prices.max()
        rsi_lower_high = rsi_value < recent_rsi.max()
        bearish_divergence = price_higher_high and rsi_lower_high and rsi_overbought
    else:
        bullish_divergence = False
        bearish_divergence = False

    # ============ SIGNAL GENERATION LOGIC ============

    # LONG Signal Conditions
    if ema_cross_up or (ema_bullish and bullish_divergence):
        base_confidence = 0.5 if ema_cross_up else 0.4
        confirmations.append("EMA crossover bullish" if ema_cross_up else "Bullish divergence")

        # Add confirmations
        if rsi_not_overbought:
            base_confidence += 0.1
            confirmations.append(f"RSI={rsi_value:.1f} not overbought")

        if config.use_adx_filter:
            if is_trending:
                base_confidence += 0.15 * trend_strength
                confirmations.append(f"ADX={adx_value:.1f} confirms trend")
            else:
                base_confidence -= 0.1
                confirmations.append(f"ADX={adx_value:.1f} weak trend")

        if config.use_volume_confirmation:
            if has_volume:
                base_confidence += 0.1
                confirmations.append(f"Volume {volume_ratio:.1f}x MA")
            else:
                base_confidence -= 0.05

        if config.use_macd_confirmation:
            if macd_bullish:
                base_confidence += 0.1
                confirmations.append("MACD histogram rising")
            elif last["macd_histogram"] < 0:
                base_confidence -= 0.1

        # +DI > -DI confirmation
        if last["adx_pos"] > last["adx_neg"]:
            base_confidence += 0.05
            confirmations.append("+DI > -DI")

        confidence = max(0, min(1.0, base_confidence))

        if confidence >= config.min_confidence_threshold:
            decision = "LONG"
            reason = " | ".join(confirmations)
        else:
            reason = f"LONG signal rejected (confidence {confidence:.0%} < {config.min_confidence_threshold:.0%})"

    # SHORT Signal Conditions
    elif ema_cross_down or (ema_bearish and bearish_divergence):
        base_confidence = 0.5 if ema_cross_down else 0.4
        confirmations.append("EMA crossover bearish" if ema_cross_down else "Bearish divergence")

        if rsi_not_oversold:
            base_confidence += 0.1
            confirmations.append(f"RSI={rsi_value:.1f} not oversold")

        if config.use_adx_filter:
            if is_trending:
                base_confidence += 0.15 * trend_strength
                confirmations.append(f"ADX={adx_value:.1f} confirms trend")
            else:
                base_confidence -= 0.1
                confirmations.append(f"ADX={adx_value:.1f} weak trend")

        if config.use_volume_confirmation:
            if has_volume:
                base_confidence += 0.1
                confirmations.append(f"Volume {volume_ratio:.1f}x MA")
            else:
                base_confidence -= 0.05

        if config.use_macd_confirmation:
            if macd_bearish:
                base_confidence += 0.1
                confirmations.append("MACD histogram falling")
            elif last["macd_histogram"] > 0:
                base_confidence -= 0.1

        # -DI > +DI confirmation
        if last["adx_neg"] > last["adx_pos"]:
            base_confidence += 0.05
            confirmations.append("-DI > +DI")

        confidence = max(0, min(1.0, base_confidence))

        if confidence >= config.min_confidence_threshold:
            decision = "SHORT"
            reason = " | ".join(confirmations)
        else:
            reason = f"SHORT signal rejected (confidence {confidence:.0%} < {config.min_confidence_threshold:.0%})"

    # RSI Extreme Signals (lower confidence, mean reversion)
    elif rsi_oversold and not ema_bearish:
        confirmations.append(f"RSI={rsi_value:.1f} oversold")
        base_confidence = 0.35

        if bullish_divergence:
            base_confidence += 0.2
            confirmations.append("Bullish divergence detected")

        if has_volume:
            base_confidence += 0.05

        confidence = max(0, min(1.0, base_confidence))

        if confidence >= config.min_confidence_threshold:
            decision = "LONG"
            reason = " | ".join(confirmations)

    elif rsi_overbought and not ema_bullish:
        confirmations.append(f"RSI={rsi_value:.1f} overbought")
        base_confidence = 0.35

        if bearish_divergence:
            base_confidence += 0.2
            confirmations.append("Bearish divergence detected")

        if has_volume:
            base_confidence += 0.05

        confidence = max(0, min(1.0, base_confidence))

        if confidence >= config.min_confidence_threshold:
            decision = "SHORT"
            reason = " | ".join(confirmations)

    # Calculate ATR-based stops
    atr_value = last["atr"] if not pd.isna(last["atr"]) else last["close"] * 0.02
    current_price = float(last["close"])

    if config.use_atr_stops:
        stop_distance = atr_value * config.atr_stop_multiplier
        tp_distance = atr_value * config.atr_tp_multiplier
    else:
        stop_distance = current_price * config.stop_loss_pct
        tp_distance = current_price * config.take_profit_pct

    if decision == "LONG":
        stop_loss = current_price - stop_distance
        take_profit = current_price + tp_distance
    elif decision == "SHORT":
        stop_loss = current_price + stop_distance
        take_profit = current_price - tp_distance
    else:
        stop_loss = None
        take_profit = None

    return {
        "decision": decision,
        "confidence": round(confidence, 4),
        "ema_fast": float(last["ema_fast"]),
        "ema_slow": float(last["ema_slow"]),
        "ema_spread": float(last["ema_spread"]) if not pd.isna(last["ema_spread"]) else 0,
        "rsi": float(rsi_value),
        "adx": float(adx_value),
        "adx_pos": float(last["adx_pos"]) if not pd.isna(last["adx_pos"]) else 0,
        "adx_neg": float(last["adx_neg"]) if not pd.isna(last["adx_neg"]) else 0,
        "macd_histogram": float(last["macd_histogram"])
        if not pd.isna(last["macd_histogram"])
        else 0,
        "atr": float(atr_value),
        "volume_ratio": float(volume_ratio),
        "close": float(current_price),
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "reason": reason,
    }


def calculate_position_size(
    balance: float,
    risk_pct: float,
    price: float,
    stop_loss_pct: float,
    atr: Optional[float] = None,
    atr_multiplier: float = 2.0,
) -> float:
    """
    Calculate position size based on risk management.

    Uses ATR-based stop distance if ATR is provided, otherwise uses percentage-based stop.

    Args:
        balance: Account balance
        risk_pct: Risk percentage per trade (e.g., 1.0 for 1%)
        price: Current asset price
        stop_loss_pct: Percentage-based stop loss (fallback)
        atr: Average True Range value (optional, for ATR-based sizing)
        atr_multiplier: Multiplier for ATR to determine stop distance

    Returns:
        Position size in base currency units
    """
    risk_amount = balance * (risk_pct / 100)

    # Calculate stop distance
    if atr is not None and atr > 0:
        stop_distance = atr * atr_multiplier
    else:
        stop_distance = price * stop_loss_pct

    if stop_distance <= 0:
        return 0.0

    # Position size = Risk Amount / Stop Distance
    size = risk_amount / stop_distance

    return max(size, 0.0)


def calculate_kelly_position_size(
    balance: float,
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    max_position_pct: float = 0.25,
    kelly_fraction: float = 0.5,  # Half-Kelly for safety
) -> float:
    """
    Calculate position size using Kelly Criterion.

    Kelly formula: f* = (p * b - q) / b
    where:
        p = probability of winning
        q = probability of losing (1 - p)
        b = win/loss ratio (avg_win / avg_loss)

    Args:
        balance: Account balance
        win_rate: Historical win rate (0-1)
        avg_win: Average winning trade amount
        avg_loss: Average losing trade amount
        max_position_pct: Maximum position size as % of balance
        kelly_fraction: Fraction of Kelly to use (0.5 = half Kelly)

    Returns:
        Position size in base currency
    """
    if win_rate <= 0 or win_rate >= 1 or avg_loss <= 0:
        return balance * 0.01  # Default 1% if invalid inputs

    p = win_rate
    q = 1 - p
    b = avg_win / avg_loss

    kelly = (p * b - q) / b

    # Apply Kelly fraction (half-Kelly is common for safety)
    kelly *= kelly_fraction

    # Clamp to valid range
    kelly = max(0, min(max_position_pct, kelly))

    return balance * kelly
