"""
Anomaly Detection - Flag unusual market conditions.

Detects anomalies in price, volume, and trading patterns to
warn of unusual market conditions and potential regime changes.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Deque, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class AnomalyType:
    """Types of anomalies."""

    PRICE_SPIKE = "price_spike"
    VOLUME_SURGE = "volume_surge"
    VOLATILITY_EXPLOSION = "volatility_explosion"
    SPREAD_WIDENING = "spread_widening"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    MOMENTUM_REVERSAL = "momentum_reversal"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    FLASH_CRASH = "flash_crash"


@dataclass
class Anomaly:
    """Detected anomaly."""

    anomaly_id: str
    anomaly_type: str
    symbol: str
    severity: Literal["low", "medium", "high", "critical"]
    score: float  # 0 to 1, how anomalous
    current_value: float
    expected_value: float
    deviation: float  # Standard deviations from mean
    description: str
    recommendation: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "anomaly_id": self.anomaly_id,
            "anomaly_type": self.anomaly_type,
            "symbol": self.symbol,
            "severity": self.severity,
            "score": round(self.score, 4),
            "current_value": round(self.current_value, 6),
            "expected_value": round(self.expected_value, 6),
            "deviation": round(self.deviation, 2),
            "description": self.description,
            "recommendation": self.recommendation,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class MarketState:
    """Current market state assessment."""

    symbol: str
    is_normal: bool
    anomaly_count: int
    risk_level: Literal["normal", "elevated", "high", "extreme"]
    active_anomalies: List[Anomaly]
    volatility_percentile: float
    volume_percentile: float
    spread_percentile: float
    market_stress_index: float  # 0 to 1
    recommendation: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "is_normal": self.is_normal,
            "anomaly_count": self.anomaly_count,
            "risk_level": self.risk_level,
            "active_anomalies": [a.to_dict() for a in self.active_anomalies],
            "volatility_percentile": round(self.volatility_percentile, 2),
            "volume_percentile": round(self.volume_percentile, 2),
            "spread_percentile": round(self.spread_percentile, 2),
            "market_stress_index": round(self.market_stress_index, 4),
            "recommendation": self.recommendation,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AnomalyConfig:
    """Anomaly detection configuration."""

    # Detection thresholds (in standard deviations)
    price_spike_threshold: float = 3.0
    volume_surge_threshold: float = 2.5
    volatility_threshold: float = 2.5
    spread_threshold: float = 3.0

    # Lookback windows
    short_window: int = 20
    long_window: int = 100

    # Severity thresholds
    low_severity_threshold: float = 2.0
    medium_severity_threshold: float = 3.0
    high_severity_threshold: float = 4.0
    critical_severity_threshold: float = 5.0

    # Flash crash detection
    flash_crash_pct: float = 0.05  # 5% drop
    flash_crash_minutes: int = 5

    # History
    history_size: int = 500


class AnomalyDetector:
    """
    Detect anomalies in market data.

    Methods:
    1. Statistical (z-score based)
    2. Isolation Forest
    3. Moving average deviation
    4. Percentile ranking
    """

    def __init__(self, config: Optional[AnomalyConfig] = None):
        self.config = config or AnomalyConfig()
        self._price_history: Dict[str, Deque[float]] = {}
        self._volume_history: Dict[str, Deque[float]] = {}
        self._volatility_history: Dict[str, Deque[float]] = {}
        self._spread_history: Dict[str, Deque[float]] = {}
        self._return_history: Dict[str, Deque[float]] = {}
        self._anomaly_count = 0
        self._active_anomalies: Dict[str, List[Anomaly]] = {}

    def update(
        self,
        symbol: str,
        price: float,
        volume: float,
        high: Optional[float] = None,
        low: Optional[float] = None,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
    ):
        """
        Update detector with new market data.

        Args:
            symbol: Trading symbol
            price: Current price
            volume: Current volume
            high: Period high
            low: Period low
            bid: Best bid
            ask: Best ask
        """
        # Initialize histories
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=self.config.history_size)
            self._volume_history[symbol] = deque(maxlen=self.config.history_size)
            self._volatility_history[symbol] = deque(maxlen=self.config.history_size)
            self._spread_history[symbol] = deque(maxlen=self.config.history_size)
            self._return_history[symbol] = deque(maxlen=self.config.history_size)
            self._active_anomalies[symbol] = []

        # Calculate return
        if self._price_history[symbol]:
            last_price = self._price_history[symbol][-1]
            ret = (price - last_price) / last_price if last_price > 0 else 0
            self._return_history[symbol].append(ret)

        # Update histories
        self._price_history[symbol].append(price)
        self._volume_history[symbol].append(volume)

        # Calculate volatility (if we have high/low)
        if high and low and price > 0:
            volatility = (high - low) / price
            self._volatility_history[symbol].append(volatility)

        # Calculate spread (if we have bid/ask)
        if bid and ask and bid > 0:
            spread = (ask - bid) / bid
            self._spread_history[symbol].append(spread)

    def detect_anomalies(self, symbol: str) -> List[Anomaly]:
        """
        Detect all anomalies for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            List of detected anomalies
        """
        anomalies = []

        if symbol not in self._price_history:
            return anomalies

        # Price spike detection
        price_anomaly = self._detect_price_spike(symbol)
        if price_anomaly:
            anomalies.append(price_anomaly)

        # Volume surge detection
        volume_anomaly = self._detect_volume_surge(symbol)
        if volume_anomaly:
            anomalies.append(volume_anomaly)

        # Volatility explosion
        volatility_anomaly = self._detect_volatility_explosion(symbol)
        if volatility_anomaly:
            anomalies.append(volatility_anomaly)

        # Spread widening
        spread_anomaly = self._detect_spread_widening(symbol)
        if spread_anomaly:
            anomalies.append(spread_anomaly)

        # Flash crash detection
        flash_crash = self._detect_flash_crash(symbol)
        if flash_crash:
            anomalies.append(flash_crash)

        # Update active anomalies
        self._active_anomalies[symbol] = anomalies

        return anomalies

    def _detect_price_spike(self, symbol: str) -> Optional[Anomaly]:
        """Detect unusual price movements."""
        returns = list(self._return_history.get(symbol, []))

        if len(returns) < self.config.short_window:
            return None

        current_return = returns[-1]
        mean_return = np.mean(returns[:-1])
        std_return = np.std(returns[:-1])

        if std_return == 0:
            return None

        z_score = abs(current_return - mean_return) / std_return

        if z_score > self.config.price_spike_threshold:
            severity = self._get_severity(z_score)
            direction = "up" if current_return > 0 else "down"

            self._anomaly_count += 1
            return Anomaly(
                anomaly_id=f"anomaly_{self._anomaly_count}",
                anomaly_type=AnomalyType.PRICE_SPIKE,
                symbol=symbol,
                severity=severity,
                score=min(1.0, z_score / 5),
                current_value=current_return,
                expected_value=mean_return,
                deviation=z_score,
                description=f"Unusual price spike {direction}: {current_return * 100:.2f}% ({z_score:.1f} std devs)",
                recommendation="Consider reducing position size or tightening stops",
            )

        return None

    def _detect_volume_surge(self, symbol: str) -> Optional[Anomaly]:
        """Detect unusual volume."""
        volumes = list(self._volume_history.get(symbol, []))

        if len(volumes) < self.config.short_window:
            return None

        current_volume = volumes[-1]
        mean_volume = np.mean(volumes[:-1])
        std_volume = np.std(volumes[:-1])

        if std_volume == 0 or mean_volume == 0:
            return None

        z_score = (current_volume - mean_volume) / std_volume

        if z_score > self.config.volume_surge_threshold:
            severity = self._get_severity(z_score)
            ratio = current_volume / mean_volume

            self._anomaly_count += 1
            return Anomaly(
                anomaly_id=f"anomaly_{self._anomaly_count}",
                anomaly_type=AnomalyType.VOLUME_SURGE,
                symbol=symbol,
                severity=severity,
                score=min(1.0, z_score / 5),
                current_value=current_volume,
                expected_value=mean_volume,
                deviation=z_score,
                description=f"Volume surge: {ratio:.1f}x average ({z_score:.1f} std devs)",
                recommendation="High activity - monitor for breakout or breakdown",
            )

        return None

    def _detect_volatility_explosion(self, symbol: str) -> Optional[Anomaly]:
        """Detect unusual volatility."""
        volatilities = list(self._volatility_history.get(symbol, []))

        if len(volatilities) < self.config.short_window:
            return None

        current_vol = volatilities[-1]
        mean_vol = np.mean(volatilities[:-1])
        std_vol = np.std(volatilities[:-1])

        if std_vol == 0:
            return None

        z_score = (current_vol - mean_vol) / std_vol

        if z_score > self.config.volatility_threshold:
            severity = self._get_severity(z_score)

            self._anomaly_count += 1
            return Anomaly(
                anomaly_id=f"anomaly_{self._anomaly_count}",
                anomaly_type=AnomalyType.VOLATILITY_EXPLOSION,
                symbol=symbol,
                severity=severity,
                score=min(1.0, z_score / 5),
                current_value=current_vol,
                expected_value=mean_vol,
                deviation=z_score,
                description=f"Volatility explosion: {current_vol * 100:.2f}% range ({z_score:.1f} std devs)",
                recommendation="Reduce position size, widen stops, or exit",
            )

        return None

    def _detect_spread_widening(self, symbol: str) -> Optional[Anomaly]:
        """Detect unusual bid-ask spread."""
        spreads = list(self._spread_history.get(symbol, []))

        if len(spreads) < self.config.short_window:
            return None

        current_spread = spreads[-1]
        mean_spread = np.mean(spreads[:-1])
        std_spread = np.std(spreads[:-1])

        if std_spread == 0:
            return None

        z_score = (current_spread - mean_spread) / std_spread

        if z_score > self.config.spread_threshold:
            severity = self._get_severity(z_score)

            self._anomaly_count += 1
            return Anomaly(
                anomaly_id=f"anomaly_{self._anomaly_count}",
                anomaly_type=AnomalyType.SPREAD_WIDENING,
                symbol=symbol,
                severity=severity,
                score=min(1.0, z_score / 5),
                current_value=current_spread,
                expected_value=mean_spread,
                deviation=z_score,
                description=f"Spread widening: {current_spread * 10000:.1f} bps ({z_score:.1f} std devs)",
                recommendation="Liquidity deteriorating - use limit orders, avoid large trades",
            )

        return None

    def _detect_flash_crash(self, symbol: str) -> Optional[Anomaly]:
        """Detect flash crash patterns."""
        prices = list(self._price_history.get(symbol, []))

        if len(prices) < self.config.flash_crash_minutes:
            return None

        recent_prices = prices[-self.config.flash_crash_minutes :]
        max_price = max(recent_prices)
        min_price = min(recent_prices)
        current_price = prices[-1]

        # Check for rapid decline
        if max_price > 0:
            decline = (max_price - min_price) / max_price

            if decline > self.config.flash_crash_pct:
                self._anomaly_count += 1
                return Anomaly(
                    anomaly_id=f"anomaly_{self._anomaly_count}",
                    anomaly_type=AnomalyType.FLASH_CRASH,
                    symbol=symbol,
                    severity="critical",
                    score=min(1.0, decline / 0.1),
                    current_value=current_price,
                    expected_value=max_price,
                    deviation=decline / self.config.flash_crash_pct,
                    description=f"Flash crash detected: {decline * 100:.1f}% drop in {self.config.flash_crash_minutes} periods",
                    recommendation="HALT TRADING - Wait for stability before re-entering",
                )

        return None

    def _get_severity(self, z_score: float) -> str:
        """Get severity level from z-score."""
        if z_score >= self.config.critical_severity_threshold:
            return "critical"
        elif z_score >= self.config.high_severity_threshold:
            return "high"
        elif z_score >= self.config.medium_severity_threshold:
            return "medium"
        return "low"

    def get_market_state(self, symbol: str) -> MarketState:
        """
        Get comprehensive market state assessment.

        Args:
            symbol: Trading symbol

        Returns:
            MarketState with full assessment
        """
        anomalies = self.detect_anomalies(symbol)

        # Calculate percentiles
        vol_percentile = self._get_percentile(self._volatility_history.get(symbol, deque()))
        volume_percentile = self._get_percentile(self._volume_history.get(symbol, deque()))
        spread_percentile = self._get_percentile(self._spread_history.get(symbol, deque()))

        # Calculate stress index
        stress_index = vol_percentile * 0.4 + spread_percentile * 0.3 + (len(anomalies) / 5) * 0.3
        stress_index = min(1.0, stress_index)

        # Determine risk level
        if stress_index > 0.8 or any(a.severity == "critical" for a in anomalies):
            risk_level = "extreme"
            recommendation = "HALT TRADING - Extreme market conditions"
        elif stress_index > 0.6 or any(a.severity == "high" for a in anomalies):
            risk_level = "high"
            recommendation = "Reduce exposure significantly, use tight stops"
        elif stress_index > 0.4 or any(a.severity == "medium" for a in anomalies):
            risk_level = "elevated"
            recommendation = "Trade with caution, reduce position sizes"
        else:
            risk_level = "normal"
            recommendation = "Normal trading conditions"

        is_normal = risk_level == "normal" and len(anomalies) == 0

        return MarketState(
            symbol=symbol,
            is_normal=is_normal,
            anomaly_count=len(anomalies),
            risk_level=risk_level,
            active_anomalies=anomalies,
            volatility_percentile=vol_percentile,
            volume_percentile=volume_percentile,
            spread_percentile=spread_percentile,
            market_stress_index=stress_index,
            recommendation=recommendation,
        )

    def _get_percentile(self, data: Deque[float]) -> float:
        """Get percentile rank of latest value."""
        if len(data) < 2:
            return 0.5

        values = list(data)
        current = values[-1]
        historical = values[:-1]

        percentile = sum(1 for v in historical if v < current) / len(historical)
        return percentile

    def get_all_market_states(self) -> Dict[str, MarketState]:
        """Get market states for all tracked symbols."""
        return {symbol: self.get_market_state(symbol) for symbol in self._price_history.keys()}

    def clear_history(self, symbol: Optional[str] = None):
        """Clear detection history."""
        if symbol:
            self._price_history.pop(symbol, None)
            self._volume_history.pop(symbol, None)
            self._volatility_history.pop(symbol, None)
            self._spread_history.pop(symbol, None)
            self._return_history.pop(symbol, None)
            self._active_anomalies.pop(symbol, None)
        else:
            self._price_history.clear()
            self._volume_history.clear()
            self._volatility_history.clear()
            self._spread_history.clear()
            self._return_history.clear()
            self._active_anomalies.clear()


def create_anomaly_detector(config: Optional[AnomalyConfig] = None) -> AnomalyDetector:
    """Factory function to create anomaly detector."""
    return AnomalyDetector(config=config)
