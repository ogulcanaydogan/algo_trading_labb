"""
Multi-Timeframe Analysis for Trading

Combines signals from multiple timeframes:
1. 1-hour: Short-term momentum and entries
2. 4-hour: Medium-term trend confirmation
3. Daily: Long-term trend direction
4. Weekly: Major support/resistance levels

Higher timeframes filter lower timeframe signals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TimeframeSignal:
    """Signal from a single timeframe."""
    timeframe: str
    trend: str  # 'bullish', 'bearish', 'neutral'
    strength: float  # 0 to 1
    momentum: float  # -1 to 1
    support: float
    resistance: float
    rsi: float
    macd_signal: str  # 'bullish', 'bearish', 'neutral'
    volume_trend: str  # 'increasing', 'decreasing', 'flat'


@dataclass
class MultiTimeframeSignal:
    """Combined signal from all timeframes."""
    timestamp: datetime
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    alignment: float  # How aligned are timeframes (0-1)
    primary_trend: str  # From highest timeframe
    entry_quality: str  # 'excellent', 'good', 'fair', 'poor'
    timeframe_signals: Dict[str, TimeframeSignal]
    reason: str


class TimeframeAnalyzer:
    """
    Analyzes a single timeframe and generates signals.
    """

    def __init__(self, timeframe: str):
        self.timeframe = timeframe
        self.minutes = self._parse_timeframe(timeframe)

    def _parse_timeframe(self, tf: str) -> int:
        """Parse timeframe string to minutes."""
        mapping = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '1d': 1440, '1w': 10080
        }
        return mapping.get(tf.lower(), 60)

    def analyze(self, ohlcv: pd.DataFrame) -> TimeframeSignal:
        """
        Analyze OHLCV data and generate signal.

        Args:
            ohlcv: DataFrame with columns [open, high, low, close, volume]

        Returns:
            TimeframeSignal
        """
        if len(ohlcv) < 50:
            return self._empty_signal()

        close = ohlcv['close']
        high = ohlcv['high']
        low = ohlcv['low']
        volume = ohlcv['volume'] if 'volume' in ohlcv.columns else pd.Series([0] * len(ohlcv))

        # Calculate indicators
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()

        # Trend
        current_price = close.iloc[-1]
        sma_20_current = sma_20.iloc[-1]
        sma_50_current = sma_50.iloc[-1]

        if current_price > sma_20_current > sma_50_current:
            trend = 'bullish'
            trend_strength = min(1.0, (current_price - sma_50_current) / sma_50_current * 10)
        elif current_price < sma_20_current < sma_50_current:
            trend = 'bearish'
            trend_strength = min(1.0, (sma_50_current - current_price) / sma_50_current * 10)
        else:
            trend = 'neutral'
            trend_strength = 0.3

        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        rsi_current = rsi.iloc[-1]

        # MACD
        macd = ema_12 - ema_26
        signal_line = macd.ewm(span=9).mean()
        macd_current = macd.iloc[-1]
        signal_current = signal_line.iloc[-1]

        if macd_current > signal_current and macd_current > 0:
            macd_signal = 'bullish'
        elif macd_current < signal_current and macd_current < 0:
            macd_signal = 'bearish'
        else:
            macd_signal = 'neutral'

        # Momentum
        momentum = (current_price - close.iloc[-10]) / close.iloc[-10]

        # Support/Resistance
        recent_high = high.iloc[-20:].max()
        recent_low = low.iloc[-20:].min()

        # Volume trend
        vol_sma = volume.rolling(20).mean()
        if vol_sma.iloc[-1] > vol_sma.iloc[-5] * 1.1:
            volume_trend = 'increasing'
        elif vol_sma.iloc[-1] < vol_sma.iloc[-5] * 0.9:
            volume_trend = 'decreasing'
        else:
            volume_trend = 'flat'

        return TimeframeSignal(
            timeframe=self.timeframe,
            trend=trend,
            strength=trend_strength,
            momentum=np.clip(momentum, -1, 1),
            support=recent_low,
            resistance=recent_high,
            rsi=rsi_current,
            macd_signal=macd_signal,
            volume_trend=volume_trend
        )

    def _empty_signal(self) -> TimeframeSignal:
        """Return empty signal when not enough data."""
        return TimeframeSignal(
            timeframe=self.timeframe,
            trend='neutral',
            strength=0,
            momentum=0,
            support=0,
            resistance=0,
            rsi=50,
            macd_signal='neutral',
            volume_trend='flat'
        )


class MultiTimeframeAnalyzer:
    """
    Combines analysis from multiple timeframes.

    Higher timeframes have more weight and act as filters.
    """

    def __init__(
        self,
        timeframes: List[str] = None,
        weights: Dict[str, float] = None
    ):
        if timeframes is None:
            timeframes = ['1h', '4h', '1d']

        self.timeframes = timeframes
        self.analyzers = {tf: TimeframeAnalyzer(tf) for tf in timeframes}

        # Default weights (higher timeframes = higher weight)
        self.weights = weights or {
            '1h': 0.25,
            '4h': 0.35,
            '1d': 0.40
        }

        # Normalize weights
        total_weight = sum(self.weights.get(tf, 0.33) for tf in timeframes)
        self.weights = {tf: self.weights.get(tf, 0.33) / total_weight for tf in timeframes}

    def resample_data(
        self,
        data_1h: pd.DataFrame,
        target_timeframe: str
    ) -> pd.DataFrame:
        """
        Resample 1h data to target timeframe.

        Args:
            data_1h: 1-hour OHLCV data
            target_timeframe: Target timeframe ('4h', '1d', etc.)

        Returns:
            Resampled DataFrame
        """
        if target_timeframe == '1h':
            return data_1h

        # Map timeframe to pandas offset
        offset_map = {
            '4h': '4h',
            '1d': '1D',
            '1w': '1W'
        }

        offset = offset_map.get(target_timeframe, '1D')

        resampled = data_1h.resample(offset).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        return resampled

    def analyze(
        self,
        symbol: str,
        data_1h: pd.DataFrame
    ) -> MultiTimeframeSignal:
        """
        Analyze symbol across all timeframes.

        Args:
            symbol: Trading symbol
            data_1h: 1-hour OHLCV data (needs enough history for all timeframes)

        Returns:
            MultiTimeframeSignal
        """
        signals = {}

        for tf in self.timeframes:
            data_tf = self.resample_data(data_1h, tf)
            signals[tf] = self.analyzers[tf].analyze(data_tf)

        # Combine signals
        return self._combine_signals(symbol, signals)

    def _combine_signals(
        self,
        symbol: str,
        signals: Dict[str, TimeframeSignal]
    ) -> MultiTimeframeSignal:
        """Combine signals from all timeframes."""

        # Calculate weighted trend score
        trend_scores = []
        for tf, signal in signals.items():
            weight = self.weights.get(tf, 0.33)
            if signal.trend == 'bullish':
                score = signal.strength
            elif signal.trend == 'bearish':
                score = -signal.strength
            else:
                score = 0
            trend_scores.append((score, weight))

        weighted_trend = sum(s * w for s, w in trend_scores)

        # Check alignment
        trends = [s.trend for s in signals.values()]
        if len(set(trends)) == 1 and trends[0] != 'neutral':
            alignment = 1.0
        elif all(t in ['bullish', 'neutral'] for t in trends) or all(t in ['bearish', 'neutral'] for t in trends):
            alignment = 0.7
        else:
            alignment = 0.3

        # Determine action
        if weighted_trend > 0.3 and alignment > 0.5:
            action = 'BUY'
            confidence = min(0.9, 0.5 + weighted_trend * alignment)
        elif weighted_trend < -0.3 and alignment > 0.5:
            action = 'SELL'
            confidence = min(0.9, 0.5 + abs(weighted_trend) * alignment)
        else:
            action = 'HOLD'
            confidence = 0.5

        # Entry quality based on RSI and momentum alignment
        entry_quality = self._assess_entry_quality(signals, action)

        # Primary trend from highest timeframe
        highest_tf = self.timeframes[-1]
        primary_trend = signals[highest_tf].trend

        # Generate reason
        reasons = []
        for tf, signal in signals.items():
            reasons.append(f"{tf}: {signal.trend} ({signal.strength:.0%})")
        reason = " | ".join(reasons)

        return MultiTimeframeSignal(
            timestamp=datetime.now(),
            symbol=symbol,
            action=action,
            confidence=confidence,
            alignment=alignment,
            primary_trend=primary_trend,
            entry_quality=entry_quality,
            timeframe_signals=signals,
            reason=reason
        )

    def _assess_entry_quality(
        self,
        signals: Dict[str, TimeframeSignal],
        action: str
    ) -> str:
        """
        Assess entry quality based on indicator alignment.

        Returns: 'excellent', 'good', 'fair', 'poor'
        """
        quality_score = 0

        for tf, signal in signals.items():
            # RSI confirmation
            if action == 'BUY' and signal.rsi < 40:
                quality_score += 1  # Oversold = good buy
            elif action == 'SELL' and signal.rsi > 60:
                quality_score += 1  # Overbought = good sell
            elif action == 'HOLD':
                if 40 < signal.rsi < 60:
                    quality_score += 0.5

            # MACD confirmation
            if action == 'BUY' and signal.macd_signal == 'bullish':
                quality_score += 1
            elif action == 'SELL' and signal.macd_signal == 'bearish':
                quality_score += 1

            # Volume confirmation
            if signal.volume_trend == 'increasing' and action != 'HOLD':
                quality_score += 0.5

        max_score = len(signals) * 2.5

        if quality_score >= max_score * 0.8:
            return 'excellent'
        elif quality_score >= max_score * 0.6:
            return 'good'
        elif quality_score >= max_score * 0.4:
            return 'fair'
        else:
            return 'poor'

    def get_features(
        self,
        data_1h: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Get multi-timeframe features for ML model.

        Returns dict of normalized features.
        """
        features = {}

        for tf in self.timeframes:
            data_tf = self.resample_data(data_1h, tf)
            signal = self.analyzers[tf].analyze(data_tf)

            prefix = f'mtf_{tf}'

            # Trend features
            features[f'{prefix}_trend'] = 1 if signal.trend == 'bullish' else (-1 if signal.trend == 'bearish' else 0)
            features[f'{prefix}_strength'] = signal.strength
            features[f'{prefix}_momentum'] = signal.momentum

            # Indicator features
            features[f'{prefix}_rsi'] = signal.rsi / 100
            features[f'{prefix}_rsi_oversold'] = 1 if signal.rsi < 30 else 0
            features[f'{prefix}_rsi_overbought'] = 1 if signal.rsi > 70 else 0

            features[f'{prefix}_macd'] = 1 if signal.macd_signal == 'bullish' else (-1 if signal.macd_signal == 'bearish' else 0)

            # Volume
            features[f'{prefix}_volume'] = 1 if signal.volume_trend == 'increasing' else (-1 if signal.volume_trend == 'decreasing' else 0)

        return features


class TimeframeCascadeFilter:
    """
    Uses higher timeframes to filter lower timeframe signals.

    Only take 1h signals that align with 4h and daily trends.
    """

    def __init__(self):
        self.analyzer = MultiTimeframeAnalyzer(timeframes=['1h', '4h', '1d'])

    def filter_signal(
        self,
        signal_1h: Dict[str, Any],
        data_1h: pd.DataFrame
    ) -> Tuple[bool, str]:
        """
        Filter 1h signal using higher timeframes.

        Args:
            signal_1h: Signal from 1h model/strategy
            data_1h: 1h OHLCV data

        Returns:
            (should_take_signal, reason)
        """
        # Get multi-timeframe analysis
        mtf = self.analyzer.analyze("", data_1h)

        action_1h = signal_1h.get('action', 'HOLD')

        # If higher timeframes are neutral, allow any signal
        if mtf.primary_trend == 'neutral':
            return True, "Higher timeframes neutral, signal allowed"

        # Check alignment
        if action_1h == 'BUY' and mtf.primary_trend == 'bullish':
            return True, f"BUY aligned with {mtf.primary_trend} daily trend"
        elif action_1h == 'SELL' and mtf.primary_trend == 'bearish':
            return True, f"SELL aligned with {mtf.primary_trend} daily trend"
        elif action_1h == 'HOLD':
            return True, "HOLD signal, no filter needed"
        else:
            # Counter-trend trade
            if mtf.alignment < 0.5:
                return True, "Low alignment, counter-trend signal allowed with caution"
            else:
                return False, f"Signal rejected: {action_1h} against {mtf.primary_trend} trend"

    def enhance_signal(
        self,
        signal_1h: Dict[str, Any],
        data_1h: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Enhance 1h signal with multi-timeframe context.

        Adjusts confidence and adds context.
        """
        mtf = self.analyzer.analyze("", data_1h)

        enhanced = signal_1h.copy()

        # Adjust confidence based on alignment
        original_confidence = signal_1h.get('confidence', 0.5)

        if mtf.alignment > 0.8:
            # Boost confidence when aligned
            enhanced['confidence'] = min(0.95, original_confidence * 1.2)
            enhanced['mtf_boost'] = 'aligned'
        elif mtf.alignment < 0.4:
            # Reduce confidence when not aligned
            enhanced['confidence'] = original_confidence * 0.7
            enhanced['mtf_boost'] = 'conflicting'
        else:
            enhanced['mtf_boost'] = 'neutral'

        # Add context
        enhanced['primary_trend'] = mtf.primary_trend
        enhanced['entry_quality'] = mtf.entry_quality
        enhanced['timeframe_alignment'] = mtf.alignment
        enhanced['mtf_reason'] = mtf.reason

        return enhanced


# Global analyzer
_mtf_analyzer: Optional[MultiTimeframeAnalyzer] = None


def get_mtf_analyzer() -> MultiTimeframeAnalyzer:
    """Get or create multi-timeframe analyzer."""
    global _mtf_analyzer
    if _mtf_analyzer is None:
        _mtf_analyzer = MultiTimeframeAnalyzer()
    return _mtf_analyzer


def get_mtf_signal(
    symbol: str,
    data_1h: pd.DataFrame
) -> MultiTimeframeSignal:
    """
    Convenience function to get multi-timeframe signal.

    Args:
        symbol: Trading symbol
        data_1h: 1-hour OHLCV data

    Returns:
        MultiTimeframeSignal
    """
    analyzer = get_mtf_analyzer()
    return analyzer.analyze(symbol, data_1h)
