"""
Smart Shorting System.

Intelligent short selling with:
- Bear market detection
- Short squeeze risk assessment
- Optimal short entry timing
- Dynamic stop placement
- Hedge integration
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ShortSignal:
    """Short selling signal."""
    symbol: str
    action: str  # SHORT, COVER, HOLD
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    squeeze_risk: float  # 0-1, higher = more dangerous
    reasoning: str
    regime: str


@dataclass
class ShortingConfig:
    """Configuration for smart shorting."""
    enabled: bool = True
    min_confidence: float = 0.65
    max_squeeze_risk: float = 0.5  # Don't short if squeeze risk > 50%
    max_short_exposure_pct: float = 0.30  # Max 30% portfolio in shorts
    min_borrow_availability: float = 0.7  # Need 70%+ borrow availability

    # Entry conditions
    min_downtrend_strength: float = -0.2
    max_rsi_for_short: float = 65  # Only short when RSI < 65
    min_volume_decline_ratio: float = 0.8  # Volume declining on rallies

    # Stop/Target
    default_stop_pct: float = 0.03  # 3% stop loss
    default_target_pct: float = 0.08  # 8% take profit


class SmartShortingSystem:
    """
    Manages short selling decisions.

    Features:
    - Regime-aware shorting
    - Squeeze risk monitoring
    - Dynamic position sizing
    - Hedge coordination
    """

    def __init__(self, config: Optional[ShortingConfig] = None):
        self.config = config or ShortingConfig()

        # Active shorts tracking
        self._active_shorts: Dict[str, Dict] = {}
        self._short_history: List[Dict] = []

        # Squeeze risk tracking
        self._squeeze_warnings: Dict[str, datetime] = {}

        logger.info("SmartShortingSystem initialized")

    def evaluate_short_opportunity(
        self,
        symbol: str,
        current_price: float,
        regime: str,
        trend_strength: float,
        rsi: float,
        volume_ratio: float,
        returns_24h: float,
        fear_greed: float,
        borrow_rate: float = 0.0,
    ) -> ShortSignal:
        """
        Evaluate if a short opportunity exists.

        Args:
            symbol: Trading symbol
            current_price: Current price
            regime: Current market regime
            trend_strength: Trend strength (-1 to 1)
            rsi: RSI value
            volume_ratio: Volume vs average
            returns_24h: 24h return
            fear_greed: Fear & Greed index (0-100)
            borrow_rate: Borrow rate for shorting

        Returns:
            ShortSignal with recommendation
        """
        if not self.config.enabled:
            return ShortSignal(
                symbol=symbol,
                action="HOLD",
                confidence=0.0,
                entry_price=current_price,
                stop_loss=0,
                take_profit=0,
                squeeze_risk=0,
                reasoning="Shorting disabled",
                regime=regime,
            )

        reasons = []
        confidence = 0.5

        # 1. Regime Check
        bearish_regimes = ["BEAR", "STRONG_BEAR", "CRASH", "HIGH_VOL"]
        if regime in bearish_regimes:
            confidence += 0.15
            reasons.append(f"Bearish regime: {regime}")
        elif regime in ["BULL", "STRONG_BULL"]:
            confidence -= 0.2
            reasons.append(f"Bullish regime (risky short): {regime}")

        # 2. Trend Strength
        if trend_strength < self.config.min_downtrend_strength:
            confidence += 0.15
            reasons.append(f"Downtrend: {trend_strength:.2f}")
        elif trend_strength > 0.2:
            confidence -= 0.15
            reasons.append(f"Uptrend (risky short): {trend_strength:.2f}")

        # 3. RSI Check (short on overbought bounces in downtrend)
        if regime in bearish_regimes and rsi > 50:
            # Bounce in downtrend - good short entry
            confidence += 0.1
            reasons.append(f"Overbought bounce: RSI={rsi:.1f}")
        elif rsi > self.config.max_rsi_for_short:
            confidence -= 0.1
            reasons.append(f"RSI too high: {rsi:.1f}")
        elif rsi < 30:
            # Don't short oversold
            confidence -= 0.2
            reasons.append(f"Already oversold: RSI={rsi:.1f}")

        # 4. Volume Analysis
        if volume_ratio < self.config.min_volume_decline_ratio and trend_strength > -0.1:
            # Volume declining on rallies - bearish
            confidence += 0.1
            reasons.append(f"Weak rally volume: {volume_ratio:.2f}x")

        # 5. Recent Performance
        if returns_24h > 0.05:
            # Big bounce - could be reversal or short trap
            confidence -= 0.1
            reasons.append(f"Large bounce: +{returns_24h*100:.1f}%")
        elif returns_24h < -0.1:
            # Already dropped a lot - might bounce
            confidence -= 0.1
            reasons.append(f"Already dropped: {returns_24h*100:.1f}%")

        # 6. Sentiment
        if fear_greed < 25:
            # Extreme fear - might be bottom
            confidence -= 0.15
            reasons.append(f"Extreme fear: {fear_greed:.0f}")
        elif fear_greed > 60:
            # Complacency - good for shorts
            confidence += 0.1
            reasons.append(f"Complacent sentiment: {fear_greed:.0f}")

        # 7. Calculate Squeeze Risk
        squeeze_risk = self._calculate_squeeze_risk(
            symbol, rsi, volume_ratio, returns_24h, regime
        )

        if squeeze_risk > self.config.max_squeeze_risk:
            confidence -= 0.2
            reasons.append(f"HIGH SQUEEZE RISK: {squeeze_risk*100:.0f}%")

        # 8. Borrow Rate
        if borrow_rate > 0.05:  # 5% annual
            confidence -= 0.1
            reasons.append(f"High borrow rate: {borrow_rate*100:.1f}%")

        # Decision
        action = "HOLD"
        if confidence >= self.config.min_confidence:
            action = "SHORT"
        elif symbol in self._active_shorts:
            # Check if should cover
            if self._should_cover(symbol, current_price, regime, rsi):
                action = "COVER"
                confidence = 0.7
                reasons.append("Cover signal triggered")

        # Calculate levels
        stop_loss = current_price * (1 + self.config.default_stop_pct)
        take_profit = current_price * (1 - self.config.default_target_pct)

        # Adjust based on volatility/regime
        if regime == "HIGH_VOL":
            stop_loss = current_price * (1 + self.config.default_stop_pct * 1.5)
        elif regime == "CRASH":
            take_profit = current_price * (1 - self.config.default_target_pct * 1.5)

        return ShortSignal(
            symbol=symbol,
            action=action,
            confidence=min(0.95, max(0.0, confidence)),
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            squeeze_risk=squeeze_risk,
            reasoning=" | ".join(reasons),
            regime=regime,
        )

    def _calculate_squeeze_risk(
        self,
        symbol: str,
        rsi: float,
        volume_ratio: float,
        returns_24h: float,
        regime: str,
    ) -> float:
        """
        Calculate short squeeze risk (0-1).

        Higher value = more dangerous to short.
        """
        risk = 0.2  # Base risk

        # RSI factor (oversold = higher squeeze risk)
        if rsi < 30:
            risk += 0.3
        elif rsi < 40:
            risk += 0.15

        # Volume spike (could signal squeeze)
        if volume_ratio > 2.0:
            risk += 0.2
        elif volume_ratio > 1.5:
            risk += 0.1

        # Recent bounce (squeeze starting?)
        if returns_24h > 0.05:
            risk += 0.2
        elif returns_24h > 0.02:
            risk += 0.1

        # Regime factor
        if regime in ["BULL", "STRONG_BULL", "RECOVERY"]:
            risk += 0.15

        # Check recent squeeze warnings
        if symbol in self._squeeze_warnings:
            time_since_warning = datetime.now() - self._squeeze_warnings[symbol]
            if time_since_warning < timedelta(hours=4):
                risk += 0.2

        return min(1.0, risk)

    def _should_cover(
        self,
        symbol: str,
        current_price: float,
        regime: str,
        rsi: float,
    ) -> bool:
        """Check if should cover existing short."""
        if symbol not in self._active_shorts:
            return False

        short = self._active_shorts[symbol]
        entry_price = short["entry_price"]
        pnl_pct = (entry_price - current_price) / entry_price

        # Take profit
        if pnl_pct >= self.config.default_target_pct:
            return True

        # Stop loss
        if pnl_pct <= -self.config.default_stop_pct:
            return True

        # Regime flip to bullish
        if regime in ["BULL", "STRONG_BULL"] and pnl_pct > 0:
            return True

        # RSI oversold (potential bounce)
        if rsi < 25 and pnl_pct > 0.02:
            return True

        return False

    def register_short(
        self,
        symbol: str,
        entry_price: float,
        quantity: float,
        stop_loss: float,
        take_profit: float,
    ):
        """Register a new short position."""
        self._active_shorts[symbol] = {
            "entry_price": entry_price,
            "quantity": quantity,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "entry_time": datetime.now(),
        }
        logger.info(f"Registered short: {symbol} @ {entry_price}")

    def close_short(self, symbol: str, exit_price: float, pnl: float):
        """Close a short position."""
        if symbol in self._active_shorts:
            short = self._active_shorts.pop(symbol)
            self._short_history.append({
                "symbol": symbol,
                "entry_price": short["entry_price"],
                "exit_price": exit_price,
                "pnl": pnl,
                "hold_time": (datetime.now() - short["entry_time"]).total_seconds() / 3600,
            })
            logger.info(f"Closed short: {symbol} @ {exit_price}, PnL: ${pnl:.2f}")

    def flag_squeeze_warning(self, symbol: str):
        """Flag a potential squeeze warning."""
        self._squeeze_warnings[symbol] = datetime.now()
        logger.warning(f"Squeeze warning flagged for {symbol}")

    def get_active_shorts(self) -> Dict[str, Dict]:
        """Get all active short positions."""
        return self._active_shorts.copy()

    def get_stats(self) -> Dict:
        """Get shorting statistics."""
        if not self._short_history:
            return {"total_shorts": 0}

        wins = [s for s in self._short_history if s["pnl"] > 0]
        return {
            "total_shorts": len(self._short_history),
            "wins": len(wins),
            "win_rate": len(wins) / len(self._short_history),
            "total_pnl": sum(s["pnl"] for s in self._short_history),
            "avg_pnl": sum(s["pnl"] for s in self._short_history) / len(self._short_history),
            "avg_hold_hours": sum(s["hold_time"] for s in self._short_history) / len(self._short_history),
            "active_shorts": len(self._active_shorts),
        }


# Singleton
_shorting_system: Optional[SmartShortingSystem] = None


def get_smart_shorting_system() -> SmartShortingSystem:
    """Get or create the SmartShortingSystem singleton."""
    global _shorting_system
    if _shorting_system is None:
        _shorting_system = SmartShortingSystem()
    return _shorting_system
