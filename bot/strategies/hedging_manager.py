"""
Hedging Manager.

Manages portfolio hedging strategies:
- Delta hedging during uncertainty
- Correlation hedging (long BTC → hedge with ETH short)
- Pre-event hedging (reduce before FOMC)
- Dynamic hedge ratios
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class HedgePosition:
    """A hedge position."""
    symbol: str
    side: str  # LONG or SHORT
    quantity: float
    entry_price: float
    hedging_symbol: str  # Symbol being hedged
    hedge_ratio: float  # e.g., 0.5 = 50% hedge
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class HedgeRecommendation:
    """Recommendation for hedging."""
    action: str  # OPEN_HEDGE, CLOSE_HEDGE, ADJUST_HEDGE, NONE
    symbol: str
    side: str
    quantity: float
    reason: str
    urgency: int  # 1-10
    hedge_ratio: float


class HedgingManager:
    """
    Manages hedging strategies for the portfolio.

    Strategies:
    1. Correlation Hedge: Short correlated assets when long
    2. Delta Hedge: Neutralize directional exposure
    3. Event Hedge: Reduce exposure before high-impact events
    4. Drawdown Hedge: Add hedges when drawdown increases
    """

    # Asset correlations (simplified)
    CORRELATIONS = {
        ("BTC/USDT", "ETH/USDT"): 0.85,
        ("BTC/USDT", "SOL/USDT"): 0.75,
        ("ETH/USDT", "SOL/USDT"): 0.80,
        ("BTC/USDT", "XAU/USD"): 0.15,
        ("BTC/USDT", "SPX500"): 0.40,
    }

    def __init__(
        self,
        min_hedge_ratio: float = 0.2,
        max_hedge_ratio: float = 0.8,
        correlation_threshold: float = 0.6,
    ):
        self.min_hedge_ratio = min_hedge_ratio
        self.max_hedge_ratio = max_hedge_ratio
        self.correlation_threshold = correlation_threshold

        # Active hedges
        self._hedges: Dict[str, HedgePosition] = {}

        # Upcoming events (symbol -> event_time)
        self._upcoming_events: Dict[str, datetime] = {}

        logger.info("HedgingManager initialized")

    def evaluate_hedge_need(
        self,
        positions: Dict[str, Dict],  # symbol -> {side, value, entry_price}
        regime: str,
        volatility: float,
        daily_pnl: float,
        daily_target: float,
        upcoming_event: Optional[Tuple[str, datetime]] = None,
    ) -> List[HedgeRecommendation]:
        """
        Evaluate if hedging is needed.

        Args:
            positions: Current positions
            regime: Market regime
            volatility: Current volatility
            daily_pnl: Today's P&L
            daily_target: Daily target ($300)
            upcoming_event: Tuple of (event_name, event_time)

        Returns:
            List of hedge recommendations
        """
        recommendations = []

        if not positions:
            return recommendations

        # Calculate total exposure
        total_long = sum(
            p["value"] for p in positions.values() if p.get("side") == "long"
        )
        total_short = sum(
            abs(p["value"]) for p in positions.values() if p.get("side") == "short"
        )
        net_exposure = total_long - total_short

        # 1. Regime-based hedging
        if regime in ["HIGH_VOL", "CRASH", "BEAR"]:
            if net_exposure > 0 and not self._has_sufficient_hedge(positions):
                hedge_ratio = 0.4 if regime == "CRASH" else 0.3
                rec = self._create_correlation_hedge_recommendation(
                    positions, hedge_ratio, f"Regime hedge for {regime}"
                )
                if rec:
                    recommendations.append(rec)

        # 2. Volatility-based hedging
        if volatility > 0.04 and net_exposure > 0:  # High volatility
            existing_hedge_ratio = self._get_current_hedge_ratio(positions)
            if existing_hedge_ratio < 0.3:
                rec = HedgeRecommendation(
                    action="OPEN_HEDGE",
                    symbol=self._get_best_hedge_symbol(positions),
                    side="SHORT",
                    quantity=0,  # Will be calculated
                    reason=f"High volatility ({volatility*100:.1f}%) hedge",
                    urgency=7,
                    hedge_ratio=0.35,
                )
                recommendations.append(rec)

        # 3. Drawdown protection
        if daily_pnl < -daily_target * 0.5:  # Lost half daily target
            if net_exposure > 0:
                rec = HedgeRecommendation(
                    action="OPEN_HEDGE",
                    symbol=self._get_best_hedge_symbol(positions),
                    side="SHORT",
                    quantity=0,
                    reason=f"Drawdown protection (PnL: ${daily_pnl:.2f})",
                    urgency=8,
                    hedge_ratio=0.5,
                )
                recommendations.append(rec)

        # 4. Event-based hedging
        if upcoming_event:
            event_name, event_time = upcoming_event
            hours_until = (event_time - datetime.now()).total_seconds() / 3600

            if 0 < hours_until < 24:  # Within 24 hours
                rec = HedgeRecommendation(
                    action="OPEN_HEDGE",
                    symbol=self._get_best_hedge_symbol(positions),
                    side="SHORT",
                    quantity=0,
                    reason=f"Pre-event hedge: {event_name} in {hours_until:.1f}h",
                    urgency=6,
                    hedge_ratio=0.25,
                )
                recommendations.append(rec)

        # 5. Close unnecessary hedges
        if regime in ["BULL", "STRONG_BULL"] and volatility < 0.02:
            for hedge_id, hedge in self._hedges.items():
                rec = HedgeRecommendation(
                    action="CLOSE_HEDGE",
                    symbol=hedge.symbol,
                    side="COVER" if hedge.side == "SHORT" else "SELL",
                    quantity=hedge.quantity,
                    reason="Bullish regime, low vol - hedge no longer needed",
                    urgency=4,
                    hedge_ratio=0,
                )
                recommendations.append(rec)

        return recommendations

    def _create_correlation_hedge_recommendation(
        self,
        positions: Dict[str, Dict],
        target_ratio: float,
        reason: str,
    ) -> Optional[HedgeRecommendation]:
        """Create hedge recommendation based on correlation."""
        # Find the largest long position
        largest_long = None
        largest_value = 0

        for symbol, pos in positions.items():
            if pos.get("side") == "long" and pos.get("value", 0) > largest_value:
                largest_value = pos["value"]
                largest_long = symbol

        if not largest_long:
            return None

        # Find best hedge symbol
        hedge_symbol = self._get_best_hedge_symbol({largest_long: positions[largest_long]})
        if not hedge_symbol:
            return None

        return HedgeRecommendation(
            action="OPEN_HEDGE",
            symbol=hedge_symbol,
            side="SHORT",
            quantity=0,  # Calculate based on hedge_ratio and position value
            reason=reason,
            urgency=6,
            hedge_ratio=target_ratio,
        )

    def _get_best_hedge_symbol(self, positions: Dict[str, Dict]) -> str:
        """Find the best symbol to use as a hedge."""
        # For crypto, ETH is a good hedge for BTC
        # For stocks, could use inverse ETFs

        long_symbols = [s for s, p in positions.items() if p.get("side") == "long"]

        if "BTC/USDT" in long_symbols:
            return "ETH/USDT"  # ETH as hedge for BTC
        elif "ETH/USDT" in long_symbols:
            return "BTC/USDT"

        # Default to BTC as hedge
        return "BTC/USDT"

    def _has_sufficient_hedge(self, positions: Dict[str, Dict]) -> bool:
        """Check if portfolio has sufficient hedging."""
        current_ratio = self._get_current_hedge_ratio(positions)
        return current_ratio >= self.min_hedge_ratio

    def _get_current_hedge_ratio(self, positions: Dict[str, Dict]) -> float:
        """Calculate current hedge ratio."""
        total_long = sum(
            p["value"] for p in positions.values() if p.get("side") == "long"
        )
        total_short = sum(
            abs(p["value"]) for p in positions.values() if p.get("side") == "short"
        )

        if total_long == 0:
            return 0.0

        return total_short / total_long

    def calculate_hedge_size(
        self,
        position_value: float,
        hedge_ratio: float,
        hedge_price: float,
    ) -> float:
        """
        Calculate hedge position size.

        Args:
            position_value: Value of position to hedge
            hedge_ratio: Desired hedge ratio (0-1)
            hedge_price: Current price of hedge instrument

        Returns:
            Quantity to short for hedge
        """
        hedge_value = position_value * hedge_ratio
        quantity = hedge_value / hedge_price
        return quantity

    def register_hedge(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        hedging_symbol: str,
        hedge_ratio: float,
    ):
        """Register a new hedge position."""
        hedge_id = f"{symbol}_{hedging_symbol}_{datetime.now().timestamp()}"
        self._hedges[hedge_id] = HedgePosition(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            hedging_symbol=hedging_symbol,
            hedge_ratio=hedge_ratio,
        )
        logger.info(f"Registered hedge: {side} {quantity} {symbol} for {hedging_symbol}")
        return hedge_id

    def close_hedge(self, hedge_id: str):
        """Close a hedge position."""
        if hedge_id in self._hedges:
            hedge = self._hedges.pop(hedge_id)
            logger.info(f"Closed hedge: {hedge.symbol}")
            return hedge
        return None

    def register_upcoming_event(self, event_name: str, event_time: datetime, symbol: str = "ALL"):
        """Register an upcoming market event."""
        self._upcoming_events[f"{event_name}_{symbol}"] = event_time
        logger.info(f"Registered event: {event_name} at {event_time}")

    def get_active_hedges(self) -> Dict[str, HedgePosition]:
        """Get all active hedge positions."""
        return self._hedges.copy()

    def get_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation between two symbols."""
        key = (symbol1, symbol2)
        if key in self.CORRELATIONS:
            return self.CORRELATIONS[key]

        key = (symbol2, symbol1)
        if key in self.CORRELATIONS:
            return self.CORRELATIONS[key]

        return 0.5  # Default moderate correlation

    def get_stats(self) -> Dict:
        """Get hedging statistics."""
        return {
            "active_hedges": len(self._hedges),
            "hedges": [
                {
                    "symbol": h.symbol,
                    "side": h.side,
                    "hedging": h.hedging_symbol,
                    "ratio": h.hedge_ratio,
                }
                for h in self._hedges.values()
            ],
            "upcoming_events": len(self._upcoming_events),
        }


# Singleton
_hedging_manager: Optional[HedgingManager] = None


def get_hedging_manager() -> HedgingManager:
    """Get or create the HedgingManager singleton."""
    global _hedging_manager
    if _hedging_manager is None:
        _hedging_manager = HedgingManager()
    return _hedging_manager
