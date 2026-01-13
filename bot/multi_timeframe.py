"""
Multi-Timeframe Analysis Module.

Provides analysis across multiple timeframes (1h, 4h, 1d)
with signal aggregation and trend alignment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Trend(Enum):
    """Trend direction."""
    STRONG_UP = "strong_up"
    UP = "up"
    NEUTRAL = "neutral"
    DOWN = "down"
    STRONG_DOWN = "strong_down"


class Signal(Enum):
    """Trading signal."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class TimeframeAnalysis:
    """Analysis results for a single timeframe."""
    timeframe: str
    trend: Trend
    signal: Signal
    strength: float  # 0-1
    rsi: float
    macd_signal: float  # Positive = bullish, negative = bearish
    ma_alignment: float  # Positive = above MAs, negative = below
    volume_trend: float  # Volume relative to average
    support_level: float
    resistance_level: float
    atr: float
    volatility: float


@dataclass
class MultiTimeframeResult:
    """Combined multi-timeframe analysis result."""
    symbol: str
    timestamp: datetime
    analyses: Dict[str, TimeframeAnalysis]
    combined_signal: Signal
    combined_strength: float
    trend_alignment: float  # How aligned trends are across timeframes
    confidence: float
    recommendation: str
    risk_level: str


class MultiTimeframeAnalyzer:
    """
    Analyzes price action across multiple timeframes.

    Combines signals from different timeframes to provide
    more robust trading signals.
    """

    TIMEFRAMES = ["1h", "4h", "1d"]

    # Timeframe weights for signal combination
    TIMEFRAME_WEIGHTS = {
        "1h": 0.25,
        "4h": 0.35,
        "1d": 0.40,
    }

    def __init__(self):
        self._data: Dict[str, pd.DataFrame] = {}

    def load_data(
        self,
        symbol: str,
        data_1h: pd.DataFrame,
        data_4h: Optional[pd.DataFrame] = None,
        data_1d: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Load OHLCV data for multiple timeframes.

        Args:
            symbol: Trading symbol
            data_1h: 1-hour OHLCV data
            data_4h: 4-hour OHLCV data (resampled from 1h if not provided)
            data_1d: Daily OHLCV data (resampled from 1h if not provided)
        """
        self._data["1h"] = data_1h.copy()

        # Resample if higher timeframes not provided
        if data_4h is not None:
            self._data["4h"] = data_4h.copy()
        else:
            self._data["4h"] = self._resample(data_1h, "4h")

        if data_1d is not None:
            self._data["1d"] = data_1d.copy()
        else:
            self._data["1d"] = self._resample(data_1h, "1d")

    def _resample(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample data to higher timeframe."""
        tf_map = {"4h": "4H", "1d": "1D", "1w": "1W"}

        if timeframe not in tf_map:
            return data

        resampled = data.resample(tf_map[timeframe]).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

        return resampled

    def analyze(self, symbol: str = "BTC/USDT") -> MultiTimeframeResult:
        """
        Perform multi-timeframe analysis.

        Args:
            symbol: Trading symbol

        Returns:
            MultiTimeframeResult with combined analysis
        """
        analyses = {}

        for tf in self.TIMEFRAMES:
            if tf in self._data and len(self._data[tf]) > 0:
                analyses[tf] = self._analyze_timeframe(self._data[tf], tf)

        if not analyses:
            raise ValueError("No data available for analysis")

        # Combine signals
        combined_signal, combined_strength = self._combine_signals(analyses)

        # Calculate trend alignment
        trend_alignment = self._calculate_trend_alignment(analyses)

        # Calculate confidence
        confidence = self._calculate_confidence(analyses, trend_alignment)

        # Generate recommendation
        recommendation = self._generate_recommendation(combined_signal, combined_strength, trend_alignment)

        # Assess risk level
        risk_level = self._assess_risk(analyses, trend_alignment)

        return MultiTimeframeResult(
            symbol=symbol,
            timestamp=datetime.now(),
            analyses=analyses,
            combined_signal=combined_signal,
            combined_strength=combined_strength,
            trend_alignment=trend_alignment,
            confidence=confidence,
            recommendation=recommendation,
            risk_level=risk_level,
        )

    def _analyze_timeframe(self, data: pd.DataFrame, timeframe: str) -> TimeframeAnalysis:
        """Analyze a single timeframe."""
        close = data["close"].values
        high = data["high"].values
        low = data["low"].values
        volume = data["volume"].values

        # Calculate indicators
        rsi = self._calculate_rsi(close)
        macd, macd_signal_line, macd_hist = self._calculate_macd(close)
        sma_20 = self._calculate_sma(close, 20)
        sma_50 = self._calculate_sma(close, 50)
        sma_200 = self._calculate_sma(close, 200)
        atr = self._calculate_atr(high, low, close)
        volatility = self._calculate_volatility(close)

        # Determine trend
        trend = self._determine_trend(close, sma_20, sma_50, sma_200)

        # MA alignment
        ma_alignment = 0
        if len(close) > 0:
            current_price = close[-1]
            if current_price > sma_20:
                ma_alignment += 0.33
            if current_price > sma_50:
                ma_alignment += 0.33
            if current_price > sma_200:
                ma_alignment += 0.34

            if current_price < sma_20:
                ma_alignment -= 0.33
            if current_price < sma_50:
                ma_alignment -= 0.33
            if current_price < sma_200:
                ma_alignment -= 0.34

        # Volume trend
        volume_trend = 1.0
        if len(volume) > 20:
            avg_volume = np.mean(volume[-20:])
            if avg_volume > 0:
                volume_trend = volume[-1] / avg_volume

        # Support and resistance
        support, resistance = self._calculate_support_resistance(high, low, close)

        # Determine signal
        signal, strength = self._determine_signal(
            rsi=rsi,
            macd_hist=macd_hist,
            ma_alignment=ma_alignment,
            trend=trend,
        )

        return TimeframeAnalysis(
            timeframe=timeframe,
            trend=trend,
            signal=signal,
            strength=strength,
            rsi=round(rsi, 1),
            macd_signal=round(macd_hist, 4) if not np.isnan(macd_hist) else 0,
            ma_alignment=round(ma_alignment, 2),
            volume_trend=round(volume_trend, 2),
            support_level=round(support, 2),
            resistance_level=round(resistance, 2),
            atr=round(atr, 4),
            volatility=round(volatility, 2),
        )

    def _calculate_rsi(self, close: np.ndarray, period: int = 14) -> float:
        """Calculate RSI."""
        if len(close) < period + 1:
            return 50.0

        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return float(rsi)

    def _calculate_macd(
        self, close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Tuple[float, float, float]:
        """Calculate MACD."""
        if len(close) < slow + signal:
            return 0.0, 0.0, 0.0

        exp1 = pd.Series(close).ewm(span=fast, adjust=False).mean()
        exp2 = pd.Series(close).ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line

        return float(macd.iloc[-1]), float(signal_line.iloc[-1]), float(histogram.iloc[-1])

    def _calculate_sma(self, data: np.ndarray, period: int) -> float:
        """Calculate SMA."""
        if len(data) < period:
            return float(data[-1]) if len(data) > 0 else 0

        return float(np.mean(data[-period:]))

    def _calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        """Calculate ATR."""
        if len(close) < period + 1:
            return 0.0

        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )

        atr = np.mean(tr[-period:])
        return float(atr)

    def _calculate_volatility(self, close: np.ndarray, period: int = 20) -> float:
        """Calculate volatility as standard deviation of returns."""
        if len(close) < period + 1:
            return 0.0

        returns = np.diff(np.log(close[-period - 1:]))
        return float(np.std(returns) * np.sqrt(252) * 100)  # Annualized

    def _calculate_support_resistance(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> Tuple[float, float]:
        """Calculate basic support and resistance levels."""
        if len(close) < 20:
            return float(close[-1]), float(close[-1])

        recent_high = high[-20:]
        recent_low = low[-20:]

        resistance = float(np.percentile(recent_high, 90))
        support = float(np.percentile(recent_low, 10))

        return support, resistance

    def _determine_trend(
        self, close: np.ndarray, sma_20: float, sma_50: float, sma_200: float
    ) -> Trend:
        """Determine overall trend."""
        if len(close) == 0:
            return Trend.NEUTRAL

        current = close[-1]

        # Count bullish/bearish signals
        bullish = 0
        bearish = 0

        if current > sma_20:
            bullish += 1
        else:
            bearish += 1

        if current > sma_50:
            bullish += 1
        else:
            bearish += 1

        if current > sma_200:
            bullish += 1
        else:
            bearish += 1

        if sma_20 > sma_50:
            bullish += 1
        else:
            bearish += 1

        if sma_50 > sma_200:
            bullish += 1
        else:
            bearish += 1

        # Determine trend strength
        if bullish >= 5:
            return Trend.STRONG_UP
        elif bullish >= 4:
            return Trend.UP
        elif bearish >= 5:
            return Trend.STRONG_DOWN
        elif bearish >= 4:
            return Trend.DOWN
        else:
            return Trend.NEUTRAL

    def _determine_signal(
        self,
        rsi: float,
        macd_hist: float,
        ma_alignment: float,
        trend: Trend,
    ) -> Tuple[Signal, float]:
        """Determine trading signal and strength."""
        score = 0

        # RSI component
        if rsi < 30:
            score += 2  # Oversold - bullish
        elif rsi < 40:
            score += 1
        elif rsi > 70:
            score -= 2  # Overbought - bearish
        elif rsi > 60:
            score -= 1

        # MACD component
        if macd_hist > 0:
            score += 1 if macd_hist < 0.01 else 2
        else:
            score -= 1 if macd_hist > -0.01 else 2

        # MA alignment component
        score += int(ma_alignment * 2)

        # Trend component
        trend_scores = {
            Trend.STRONG_UP: 2,
            Trend.UP: 1,
            Trend.NEUTRAL: 0,
            Trend.DOWN: -1,
            Trend.STRONG_DOWN: -2,
        }
        score += trend_scores.get(trend, 0)

        # Normalize strength
        max_score = 8
        strength = min(1.0, max(0.0, abs(score) / max_score))

        # Determine signal
        if score >= 6:
            return Signal.STRONG_BUY, strength
        elif score >= 3:
            return Signal.BUY, strength
        elif score <= -6:
            return Signal.STRONG_SELL, strength
        elif score <= -3:
            return Signal.SELL, strength
        else:
            return Signal.NEUTRAL, strength

    def _combine_signals(
        self, analyses: Dict[str, TimeframeAnalysis]
    ) -> Tuple[Signal, float]:
        """Combine signals from multiple timeframes."""
        signal_values = {
            Signal.STRONG_BUY: 2,
            Signal.BUY: 1,
            Signal.NEUTRAL: 0,
            Signal.SELL: -1,
            Signal.STRONG_SELL: -2,
        }

        weighted_score = 0
        total_weight = 0

        for tf, analysis in analyses.items():
            weight = self.TIMEFRAME_WEIGHTS.get(tf, 0.33)
            signal_value = signal_values.get(analysis.signal, 0)
            weighted_score += signal_value * weight * analysis.strength
            total_weight += weight

        if total_weight > 0:
            weighted_score /= total_weight

        # Determine combined signal
        if weighted_score >= 1.5:
            signal = Signal.STRONG_BUY
        elif weighted_score >= 0.5:
            signal = Signal.BUY
        elif weighted_score <= -1.5:
            signal = Signal.STRONG_SELL
        elif weighted_score <= -0.5:
            signal = Signal.SELL
        else:
            signal = Signal.NEUTRAL

        strength = min(1.0, abs(weighted_score) / 2)

        return signal, strength

    def _calculate_trend_alignment(self, analyses: Dict[str, TimeframeAnalysis]) -> float:
        """Calculate how aligned trends are across timeframes."""
        if len(analyses) < 2:
            return 1.0

        trend_values = {
            Trend.STRONG_UP: 2,
            Trend.UP: 1,
            Trend.NEUTRAL: 0,
            Trend.DOWN: -1,
            Trend.STRONG_DOWN: -2,
        }

        trends = [trend_values.get(a.trend, 0) for a in analyses.values()]

        # Check if all same direction
        if all(t > 0 for t in trends) or all(t < 0 for t in trends):
            return 1.0
        elif all(t == 0 for t in trends):
            return 0.5
        else:
            # Calculate alignment score
            variance = np.var(trends)
            alignment = 1.0 - (variance / 8)  # Max variance is 8
            return max(0.0, alignment)

    def _calculate_confidence(
        self, analyses: Dict[str, TimeframeAnalysis], trend_alignment: float
    ) -> float:
        """Calculate overall confidence in the signal."""
        # Base confidence from signal strengths
        avg_strength = np.mean([a.strength for a in analyses.values()])

        # Adjust for trend alignment
        confidence = avg_strength * (0.5 + 0.5 * trend_alignment)

        # Adjust for volatility (higher volatility = lower confidence)
        avg_volatility = np.mean([a.volatility for a in analyses.values()])
        if avg_volatility > 50:
            confidence *= 0.8
        elif avg_volatility > 100:
            confidence *= 0.6

        return round(min(1.0, confidence), 2)

    def _generate_recommendation(
        self, signal: Signal, strength: float, alignment: float
    ) -> str:
        """Generate human-readable recommendation."""
        signal_text = {
            Signal.STRONG_BUY: "Strong Buy",
            Signal.BUY: "Buy",
            Signal.NEUTRAL: "Hold/Wait",
            Signal.SELL: "Sell",
            Signal.STRONG_SELL: "Strong Sell",
        }

        strength_text = "high" if strength > 0.7 else "moderate" if strength > 0.4 else "low"
        alignment_text = "well-aligned" if alignment > 0.7 else "partially aligned" if alignment > 0.4 else "conflicting"

        base = signal_text.get(signal, "Hold")

        return f"{base} with {strength_text} strength. Timeframes are {alignment_text}."

    def _assess_risk(
        self, analyses: Dict[str, TimeframeAnalysis], alignment: float
    ) -> str:
        """Assess risk level based on analysis."""
        avg_volatility = np.mean([a.volatility for a in analyses.values()])

        # Risk factors
        risk_score = 0

        if avg_volatility > 60:
            risk_score += 2
        elif avg_volatility > 40:
            risk_score += 1

        if alignment < 0.4:
            risk_score += 2
        elif alignment < 0.7:
            risk_score += 1

        if risk_score >= 3:
            return "HIGH"
        elif risk_score >= 2:
            return "MEDIUM"
        else:
            return "LOW"

    def to_api_response(self, result: MultiTimeframeResult) -> Dict[str, Any]:
        """Convert result to API response format."""
        return {
            "symbol": result.symbol,
            "timestamp": result.timestamp.isoformat(),
            "combined": {
                "signal": result.combined_signal.value,
                "strength": result.combined_strength,
                "trend_alignment": result.trend_alignment,
                "confidence": result.confidence,
                "recommendation": result.recommendation,
                "risk_level": result.risk_level,
            },
            "timeframes": {
                tf: {
                    "trend": analysis.trend.value,
                    "signal": analysis.signal.value,
                    "strength": analysis.strength,
                    "rsi": analysis.rsi,
                    "macd_signal": analysis.macd_signal,
                    "ma_alignment": analysis.ma_alignment,
                    "volume_trend": analysis.volume_trend,
                    "support": analysis.support_level,
                    "resistance": analysis.resistance_level,
                    "atr": analysis.atr,
                    "volatility": analysis.volatility,
                }
                for tf, analysis in result.analyses.items()
            },
        }

    def get_htf_trend_filter(
        self,
        signal_direction: str,
        htf_timeframe: str = "4h",
    ) -> Tuple[bool, float, str]:
        """
        Check if a signal aligns with higher timeframe trend.

        This is a key profitability improvement - trading with the HTF trend
        significantly increases win rate.

        Args:
            signal_direction: "LONG" or "SHORT"
            htf_timeframe: Higher timeframe to check (default "4h")

        Returns:
            Tuple of (should_take_signal, confidence_adjustment, reason)
        """
        if htf_timeframe not in self._data or len(self._data[htf_timeframe]) == 0:
            return True, 0.0, "No HTF data available"

        analysis = self._analyze_timeframe(self._data[htf_timeframe], htf_timeframe)

        # Map trends to directional bias
        bullish_trends = [Trend.STRONG_UP, Trend.UP]
        bearish_trends = [Trend.STRONG_DOWN, Trend.DOWN]

        if signal_direction == "LONG":
            if analysis.trend in bullish_trends:
                # Signal aligned with HTF trend
                boost = 0.15 if analysis.trend == Trend.STRONG_UP else 0.10
                return True, boost, f"LONG aligned with HTF {analysis.trend.value}"
            elif analysis.trend in bearish_trends:
                # Signal against HTF trend - reduce confidence or reject
                return False, -0.20, f"LONG rejected: HTF in {analysis.trend.value}"
            else:
                return True, 0.0, "HTF trend neutral"

        elif signal_direction == "SHORT":
            if analysis.trend in bearish_trends:
                boost = 0.15 if analysis.trend == Trend.STRONG_DOWN else 0.10
                return True, boost, f"SHORT aligned with HTF {analysis.trend.value}"
            elif analysis.trend in bullish_trends:
                return False, -0.20, f"SHORT rejected: HTF in {analysis.trend.value}"
            else:
                return True, 0.0, "HTF trend neutral"

        return True, 0.0, "Unknown signal direction"

    def filter_signal_with_htf(
        self,
        signal: Dict[str, Any],
        htf_timeframe: str = "4h",
        strict_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        Apply HTF trend filter to a trading signal.

        Args:
            signal: Signal dict with 'decision' and 'confidence' keys
            htf_timeframe: Higher timeframe for trend check
            strict_mode: If True, reject all counter-trend signals

        Returns:
            Modified signal with adjusted confidence and HTF info
        """
        decision = signal.get("decision", "FLAT")

        if decision == "FLAT":
            signal["htf_filter"] = {"applied": False, "reason": "No signal to filter"}
            return signal

        should_take, confidence_adj, reason = self.get_htf_trend_filter(decision, htf_timeframe)

        if not should_take:
            if strict_mode:
                # Reject the signal entirely
                signal["decision"] = "FLAT"
                signal["confidence"] = 0.0
                signal["htf_filter"] = {
                    "applied": True,
                    "rejected": True,
                    "reason": reason,
                }
            else:
                # Significantly reduce confidence
                original_conf = signal.get("confidence", 0.5)
                signal["confidence"] = max(0, original_conf + confidence_adj)
                signal["htf_filter"] = {
                    "applied": True,
                    "rejected": False,
                    "confidence_adjustment": confidence_adj,
                    "reason": reason,
                }
        else:
            # Boost confidence for aligned signals
            original_conf = signal.get("confidence", 0.5)
            signal["confidence"] = min(1.0, original_conf + confidence_adj)
            signal["htf_filter"] = {
                "applied": True,
                "rejected": False,
                "confidence_adjustment": confidence_adj,
                "reason": reason,
            }

        return signal


# Timeframe conversion utilities
TIMEFRAME_RESAMPLE_MAP = {
    "1m": "1T",
    "5m": "5T",
    "15m": "15T",
    "30m": "30T",
    "1h": "1H",
    "4h": "4H",
    "1d": "1D",
    "1w": "1W",
}

HTF_RECOMMENDATIONS = {
    "1m": "5m",
    "5m": "15m",
    "15m": "1h",
    "30m": "4h",
    "1h": "4h",
    "4h": "1d",
    "1d": "1w",
}


def get_recommended_htf(ltf: str) -> str:
    """Get recommended higher timeframe for trend confirmation."""
    return HTF_RECOMMENDATIONS.get(ltf.lower(), "4h")


def resample_to_htf(data: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """
    Resample OHLCV data to a higher timeframe.

    Args:
        data: OHLCV DataFrame with datetime index
        target_tf: Target timeframe (e.g., "4h", "1d")

    Returns:
        Resampled DataFrame
    """
    resample_str = TIMEFRAME_RESAMPLE_MAP.get(target_tf.lower(), "4H")

    resampled = data.resample(resample_str).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()

    return resampled


def confirm_signal_mtf(
    signal: str,
    ohlcv_1h: pd.DataFrame,
    min_agreement: float = 0.5,
) -> Dict[str, Any]:
    """
    Confirm a trading signal using multi-timeframe analysis.

    This is a convenience function that takes a signal (LONG, SHORT, FLAT)
    and confirms it against higher timeframe trends.

    Args:
        signal: The signal to confirm ("LONG", "SHORT", "FLAT")
        ohlcv_1h: 1-hour OHLCV data
        min_agreement: Minimum agreement threshold (0-1)

    Returns:
        Dictionary with confirmation result:
        - confirmed: True if signal is confirmed
        - confidence: Adjusted confidence (0-1)
        - alignment: Trend alignment score
        - recommendation: Action recommendation
        - details: Per-timeframe analysis
    """
    analyzer = MultiTimeframeAnalyzer()
    analyzer.load_data("symbol", ohlcv_1h)

    try:
        result = analyzer.analyze()
    except Exception as e:
        logger.warning(f"MTF analysis failed: {e}")
        return {
            "confirmed": True,  # Default to trusting original signal
            "confidence": 0.5,
            "alignment": 0.0,
            "recommendation": "PROCEED_WITH_CAUTION",
            "details": {},
        }

    # Map signals to direction
    signal_direction = {
        "LONG": 1,
        "SHORT": -1,
        "FLAT": 0,
    }

    combined_direction = {
        Signal.STRONG_BUY: 1,
        Signal.BUY: 1,
        Signal.NEUTRAL: 0,
        Signal.SELL: -1,
        Signal.STRONG_SELL: -1,
    }

    # Get direction of input signal
    input_dir = signal_direction.get(signal.upper(), 0)

    # Get direction from MTF analysis
    mtf_dir = combined_direction.get(result.combined_signal, 0)

    # Check alignment
    alignment = result.trend_alignment

    # Determine if confirmed
    if input_dir == 0:
        # FLAT signal - confirm if MTF also neutral or trend weak
        confirmed = alignment < 0.6
        confidence = 0.7 if confirmed else 0.3
    elif input_dir == mtf_dir:
        # Signals align
        confirmed = True
        confidence = min(1.0, result.confidence * (1 + alignment * 0.3))
    elif mtf_dir == 0:
        # MTF is neutral - partial confirmation
        confirmed = alignment < 0.5  # Only if not strong opposing trend
        confidence = result.confidence * 0.7
    else:
        # Signals conflict
        confirmed = False
        confidence = result.confidence * 0.3

    # Determine recommendation
    if confirmed and confidence > 0.6:
        recommendation = "PROCEED"
    elif confirmed:
        recommendation = "PROCEED_REDUCED_SIZE"
    elif confidence > 0.4:
        recommendation = "WAIT_FOR_CONFIRMATION"
    else:
        recommendation = "AVOID_TRADE"

    return {
        "confirmed": confirmed,
        "confidence": round(confidence, 4),
        "alignment": round(alignment, 4),
        "mtf_signal": result.combined_signal.value,
        "mtf_strength": round(result.combined_strength, 4),
        "recommendation": recommendation,
        "risk_level": result.risk_level,
        "details": {
            tf: {
                "trend": analysis.trend.value,
                "signal": analysis.signal.value,
                "rsi": analysis.rsi,
            }
            for tf, analysis in result.analyses.items()
        },
    }
