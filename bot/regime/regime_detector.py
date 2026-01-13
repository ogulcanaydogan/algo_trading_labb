"""
Regime Detection Module.

Analyzes market data and classifies current market regime:
- BULL: Strong upward trend
- BEAR: Strong downward trend
- CRASH: Rapid decline with high volatility
- SIDEWAYS: Range-bound, no clear trend
- HIGH_VOL: Elevated volatility regardless of direction

The regime informs strategy selection and risk parameters.
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


class MarketRegime(Enum):
    """Market regime classifications."""

    BULL = "bull"
    BEAR = "bear"
    CRASH = "crash"
    SIDEWAYS = "sideways"
    HIGH_VOL = "high_vol"
    UNKNOWN = "unknown"

    @property
    def is_trending(self) -> bool:
        return self in (MarketRegime.BULL, MarketRegime.BEAR)

    @property
    def is_risk_off(self) -> bool:
        return self in (MarketRegime.CRASH, MarketRegime.HIGH_VOL)


@dataclass
class RegimeIndicators:
    """Computed indicators used for regime detection."""

    # Trend indicators
    trend_direction: float = 0.0  # +1 bullish, -1 bearish
    trend_strength: float = 0.0  # ADX-like, 0-100
    ma_slope: float = 0.0  # Slope of moving average
    price_vs_ma: float = 0.0  # % above/below MA

    # Volatility indicators
    atr: float = 0.0  # Average True Range
    atr_pct: float = 0.0  # ATR as % of price
    realized_vol: float = 0.0  # Realized volatility
    vol_percentile: float = 0.0  # Current vol vs historical
    vol_spike_ratio: float = 0.0  # Current vs normal vol

    # Drawdown indicators
    drawdown_pct: float = 0.0  # Current drawdown from peak
    short_term_return: float = 0.0  # Recent return

    # Range indicators
    bb_bandwidth: float = 0.0  # Bollinger bandwidth
    range_bound_score: float = 0.0  # How range-bound (0-1)

    def to_dict(self) -> Dict[str, float]:
        return {
            "trend_direction": self.trend_direction,
            "trend_strength": self.trend_strength,
            "ma_slope": self.ma_slope,
            "price_vs_ma": self.price_vs_ma,
            "atr": self.atr,
            "atr_pct": self.atr_pct,
            "realized_vol": self.realized_vol,
            "vol_percentile": self.vol_percentile,
            "vol_spike_ratio": self.vol_spike_ratio,
            "drawdown_pct": self.drawdown_pct,
            "short_term_return": self.short_term_return,
            "bb_bandwidth": self.bb_bandwidth,
            "range_bound_score": self.range_bound_score,
        }


@dataclass
class RegimeState:
    """Current regime state with metadata."""

    regime: MarketRegime
    confidence: float  # 0.0 to 1.0
    indicators: RegimeIndicators
    timestamp: datetime = field(default_factory=datetime.now)
    symbol: str = ""
    timeframe: str = "1h"

    # Previous regime for transition detection
    previous_regime: Optional[MarketRegime] = None
    regime_duration_bars: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.regime.value,
            "confidence": round(self.confidence, 4),
            "indicators": self.indicators.to_dict(),
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "previous_regime": self.previous_regime.value if self.previous_regime else None,
            "regime_duration_bars": self.regime_duration_bars,
        }

    @property
    def is_transition(self) -> bool:
        return self.previous_regime is not None and self.previous_regime != self.regime


@dataclass
class RegimeConfig:
    """Configuration for regime detection."""

    # Lookback
    lookback_bars: int = 100

    # Trend parameters
    ma_fast: int = 20
    ma_slow: int = 50
    adx_period: int = 14
    adx_threshold: float = 25.0
    adx_exit_threshold: float = 20.0  # Hysteresis: lower threshold to exit trending

    # Volatility parameters
    atr_period: int = 14
    vol_lookback: int = 20
    high_vol_percentile: float = 90.0
    vol_spike_multiplier: float = 2.0

    # Crash detection
    crash_drawdown_threshold: float = 0.10  # 10%
    crash_return_threshold: float = -0.05  # -5%
    crash_window: int = 24  # bars

    # Sideways detection
    bb_period: int = 20
    bb_std: float = 2.0
    bb_bandwidth_threshold: float = 0.04

    # Confidence thresholds
    min_confidence_threshold: float = 0.5

    # Anti-oscillation settings
    min_regime_duration: int = 3  # Minimum bars before regime can change
    regime_change_confidence_boost: float = 0.1  # Extra confidence needed to change regime

    @classmethod
    def from_dict(cls, config: Dict) -> "RegimeConfig":
        return cls(**{k: v for k, v in config.items() if hasattr(cls, k)})


class RegimeDetector:
    """
    Detects market regime from OHLCV data.

    Uses rules-based classification with the following priority:
    1. CRASH (highest priority - safety first)
    2. HIGH_VOL
    3. BULL / BEAR (trending)
    4. SIDEWAYS (default if no clear trend)
    """

    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
        self._current_state: Optional[RegimeState] = None
        self._regime_history: List[RegimeState] = []
        self._vol_history: List[float] = []

    def detect(
        self,
        df: pd.DataFrame,
        symbol: str = "",
        timeframe: str = "1h",
    ) -> RegimeState:
        """
        Detect current market regime from OHLCV data.

        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            symbol: Trading symbol
            timeframe: Candle timeframe

        Returns:
            RegimeState with regime classification and indicators
        """
        if len(df) < self.config.lookback_bars:
            logger.warning(
                f"Insufficient data: {len(df)} bars, need {self.config.lookback_bars}"
            )
            return RegimeState(
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                indicators=RegimeIndicators(),
                symbol=symbol,
                timeframe=timeframe,
            )

        # Ensure column names are lowercase
        df = df.copy()
        df.columns = df.columns.str.lower()

        # Compute all indicators
        indicators = self._compute_indicators(df)

        # Classify regime
        raw_regime, confidence = self._classify_regime(indicators)

        # Apply minimum regime duration (anti-oscillation)
        previous_regime = self._current_state.regime if self._current_state else None
        current_duration = self._current_state.regime_duration_bars if self._current_state else 0

        # Only allow regime change if:
        # 1. No previous regime (first detection)
        # 2. Current duration exceeds minimum
        # 3. New regime is CRASH (safety override - always allow crash detection)
        if previous_regime and previous_regime != raw_regime:
            if raw_regime != MarketRegime.CRASH and current_duration < self.config.min_regime_duration:
                # Block regime change, keep current regime
                regime = previous_regime
                logger.debug(
                    f"Regime change blocked: {previous_regime.value} -> {raw_regime.value} "
                    f"(duration {current_duration} < {self.config.min_regime_duration})"
                )
            else:
                regime = raw_regime
        else:
            regime = raw_regime

        # Create state
        duration = (
            self._current_state.regime_duration_bars + 1
            if self._current_state and self._current_state.regime == regime
            else 1
        )

        state = RegimeState(
            regime=regime,
            confidence=confidence,
            indicators=indicators,
            symbol=symbol,
            timeframe=timeframe,
            previous_regime=previous_regime,
            regime_duration_bars=duration,
        )

        # Update history
        self._current_state = state
        self._regime_history.append(state)
        if len(self._regime_history) > 1000:
            self._regime_history = self._regime_history[-500:]

        # Log regime transitions
        if state.is_transition:
            logger.info(
                f"Regime transition: {previous_regime.value} -> {regime.value} "
                f"(confidence: {confidence:.2%})"
            )

        return state

    def _compute_indicators(self, df: pd.DataFrame) -> RegimeIndicators:
        """Compute all technical indicators for regime detection."""

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        indicators = RegimeIndicators()

        # --- Trend Indicators ---

        # Moving averages
        ma_fast = self._sma(close, self.config.ma_fast)
        ma_slow = self._sma(close, self.config.ma_slow)

        # Trend direction: +1 if fast MA > slow MA, -1 otherwise
        if ma_fast[-1] > ma_slow[-1]:
            indicators.trend_direction = 1.0
        elif ma_fast[-1] < ma_slow[-1]:
            indicators.trend_direction = -1.0

        # Price vs slow MA
        indicators.price_vs_ma = (close[-1] - ma_slow[-1]) / ma_slow[-1]

        # MA slope (normalized)
        ma_slope = (ma_slow[-1] - ma_slow[-10]) / ma_slow[-10] if len(ma_slow) > 10 else 0
        indicators.ma_slope = ma_slope

        # ADX-like trend strength
        indicators.trend_strength = self._compute_adx(high, low, close, self.config.adx_period)

        # --- Volatility Indicators ---

        # ATR
        atr = self._compute_atr(high, low, close, self.config.atr_period)
        indicators.atr = atr[-1] if len(atr) > 0 else 0
        indicators.atr_pct = indicators.atr / close[-1] if close[-1] > 0 else 0

        # Realized volatility (annualized)
        returns = np.diff(np.log(close))
        if len(returns) >= self.config.vol_lookback:
            indicators.realized_vol = np.std(returns[-self.config.vol_lookback:]) * np.sqrt(252 * 24)

        # Volatility percentile (vs history)
        self._vol_history.append(indicators.realized_vol)
        if len(self._vol_history) > 500:
            self._vol_history = self._vol_history[-500:]

        if len(self._vol_history) > 20:
            indicators.vol_percentile = (
                np.sum(np.array(self._vol_history) <= indicators.realized_vol)
                / len(self._vol_history) * 100
            )

        # Volatility spike ratio
        if len(self._vol_history) > 20:
            normal_vol = np.median(self._vol_history[-100:]) if len(self._vol_history) > 100 else np.median(self._vol_history)
            indicators.vol_spike_ratio = indicators.realized_vol / normal_vol if normal_vol > 0 else 1.0

        # --- Drawdown Indicators ---

        # Rolling peak and drawdown
        rolling_max = pd.Series(close).rolling(self.config.lookback_bars).max().values
        indicators.drawdown_pct = (close[-1] - rolling_max[-1]) / rolling_max[-1] if rolling_max[-1] > 0 else 0

        # Short-term return
        window = min(self.config.crash_window, len(close) - 1)
        if window > 0:
            indicators.short_term_return = (close[-1] - close[-window-1]) / close[-window-1]

        # --- Range Indicators ---

        # Bollinger bandwidth
        bb_ma = self._sma(close, self.config.bb_period)
        bb_std = self._rolling_std(close, self.config.bb_period)
        if bb_ma[-1] > 0 and bb_std[-1] > 0:
            upper = bb_ma[-1] + self.config.bb_std * bb_std[-1]
            lower = bb_ma[-1] - self.config.bb_std * bb_std[-1]
            indicators.bb_bandwidth = (upper - lower) / bb_ma[-1]

        # Range-bound score (inverse of bandwidth)
        indicators.range_bound_score = max(0, 1 - indicators.bb_bandwidth / 0.1)

        return indicators

    def _classify_regime(self, ind: RegimeIndicators) -> Tuple[MarketRegime, float]:
        """
        Classify regime based on indicators.

        Priority order (safety first):
        1. CRASH
        2. HIGH_VOL
        3. BULL / BEAR
        4. SIDEWAYS

        Uses hysteresis to prevent oscillation at threshold boundaries.
        """

        confidences = {
            MarketRegime.CRASH: 0.0,
            MarketRegime.HIGH_VOL: 0.0,
            MarketRegime.BULL: 0.0,
            MarketRegime.BEAR: 0.0,
            MarketRegime.SIDEWAYS: 0.0,
        }

        # Get current regime for hysteresis
        current_regime = self._current_state.regime if self._current_state else None

        # --- Check for CRASH (highest priority) ---
        crash_score = 0.0
        if ind.drawdown_pct < -self.config.crash_drawdown_threshold:
            crash_score += 0.4
        if ind.short_term_return < self.config.crash_return_threshold:
            crash_score += 0.3
        if ind.vol_spike_ratio > self.config.vol_spike_multiplier:
            crash_score += 0.3

        if crash_score >= 0.6:
            confidences[MarketRegime.CRASH] = min(1.0, crash_score)
            return MarketRegime.CRASH, confidences[MarketRegime.CRASH]

        # --- Check for HIGH_VOL ---
        if ind.vol_percentile >= self.config.high_vol_percentile:
            vol_confidence = 0.5 + 0.5 * (ind.vol_percentile - 90) / 10
            confidences[MarketRegime.HIGH_VOL] = min(1.0, vol_confidence)

            # HIGH_VOL takes precedence if very high
            if confidences[MarketRegime.HIGH_VOL] > 0.7:
                return MarketRegime.HIGH_VOL, confidences[MarketRegime.HIGH_VOL]

        # --- Check for trending (BULL / BEAR) with hysteresis ---
        # Use different thresholds for entering vs staying in trending regime
        if current_regime in (MarketRegime.BULL, MarketRegime.BEAR):
            # Already trending: use lower exit threshold (stay in trend longer)
            trend_threshold = self.config.adx_exit_threshold
        else:
            # Not trending: use higher entry threshold (harder to enter trend)
            trend_threshold = self.config.adx_threshold

        is_trending = ind.trend_strength >= trend_threshold

        if is_trending:
            trend_confidence = 0.5 + 0.5 * min(1.0, ind.trend_strength / 50)

            if ind.trend_direction > 0 and ind.price_vs_ma > 0:
                confidences[MarketRegime.BULL] = trend_confidence
            elif ind.trend_direction < 0 and ind.price_vs_ma < 0:
                confidences[MarketRegime.BEAR] = trend_confidence

        # --- Check for SIDEWAYS ---
        if ind.bb_bandwidth < self.config.bb_bandwidth_threshold:
            sideways_confidence = 0.5 + 0.5 * ind.range_bound_score
            confidences[MarketRegime.SIDEWAYS] = sideways_confidence
        elif not is_trending:
            # Not trending and not tight range = weak sideways
            confidences[MarketRegime.SIDEWAYS] = 0.4

        # --- Select highest confidence regime with bias toward current ---
        best_regime = max(confidences, key=confidences.get)
        best_confidence = confidences[best_regime]

        # Apply regime change penalty (favor staying in current regime)
        if current_regime and current_regime != best_regime:
            current_confidence = confidences.get(current_regime, 0)
            confidence_boost = self.config.regime_change_confidence_boost

            # Stay in current regime if new regime doesn't have enough advantage
            if best_confidence < current_confidence + confidence_boost:
                return current_regime, current_confidence

        # Default to SIDEWAYS with low confidence if nothing else fits
        if best_confidence < self.config.min_confidence_threshold:
            return MarketRegime.SIDEWAYS, 0.5

        return best_regime, best_confidence

    @staticmethod
    def _sma(data: np.ndarray, period: int) -> np.ndarray:
        """Simple moving average."""
        if len(data) < period:
            return np.full(len(data), np.nan)
        return pd.Series(data).rolling(period).mean().values

    @staticmethod
    def _rolling_std(data: np.ndarray, period: int) -> np.ndarray:
        """Rolling standard deviation."""
        if len(data) < period:
            return np.full(len(data), np.nan)
        return pd.Series(data).rolling(period).std().values

    @staticmethod
    def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """Compute Average True Range."""
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))

        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First bar has no previous close

        return pd.Series(tr).rolling(period).mean().values

    @staticmethod
    def _compute_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> float:
        """Compute ADX (Average Directional Index) - simplified."""
        if len(close) < period * 2:
            return 0.0

        # True Range
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1))
            )
        )

        # +DM and -DM
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Smooth with EMA
        atr = pd.Series(tr).ewm(span=period).mean().values
        plus_di = 100 * pd.Series(plus_dm).ewm(span=period).mean().values / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(span=period).mean().values / atr

        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).ewm(span=period).mean().values

        return float(adx[-1]) if not np.isnan(adx[-1]) else 0.0

    @property
    def current_regime(self) -> Optional[MarketRegime]:
        """Get current detected regime."""
        return self._current_state.regime if self._current_state else None

    @property
    def current_state(self) -> Optional[RegimeState]:
        """Get full current state."""
        return self._current_state

    def get_regime_stats(self) -> Dict[str, Any]:
        """Get statistics about regime history."""
        if not self._regime_history:
            return {}

        regime_counts = {}
        for state in self._regime_history:
            r = state.regime.value
            regime_counts[r] = regime_counts.get(r, 0) + 1

        total = len(self._regime_history)

        return {
            "total_observations": total,
            "regime_distribution": {
                k: round(v / total * 100, 1) for k, v in regime_counts.items()
            },
            "current_regime": self.current_regime.value if self.current_regime else None,
            "current_duration_bars": self._current_state.regime_duration_bars if self._current_state else 0,
        }
