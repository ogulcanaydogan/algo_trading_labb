"""
Market Impact and Slippage Model.

Features:
- Volume-based slippage estimation
- Order book impact modeling
- Historical slippage tracking
- Adaptive slippage adjustment
- Execution quality analysis
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


class MarketCondition(Enum):
    """Market liquidity conditions."""

    VERY_LIQUID = "very_liquid"  # Tight spreads, deep book
    LIQUID = "liquid"  # Normal conditions
    MODERATE = "moderate"  # Wider spreads
    ILLIQUID = "illiquid"  # Poor liquidity
    STRESSED = "stressed"  # Extreme conditions


class OrderType(Enum):
    """Order types for slippage calculation."""

    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"


@dataclass
class SlippageConfig:
    """Configuration for slippage model."""

    # Base slippage parameters
    base_slippage_bps: float = 5.0  # 5 basis points base slippage
    min_slippage_bps: float = 1.0  # Minimum 1 bp
    max_slippage_bps: float = 50.0  # Maximum 50 bp

    # Volume impact parameters
    volume_impact_coefficient: float = 0.1  # Impact per % of ADV
    max_volume_pct_adv: float = 5.0  # Max 5% of ADV per trade

    # Volatility adjustment
    volatility_multiplier: float = 2.0  # Slippage increases 2x with vol

    # Order type multipliers
    order_type_multipliers: Dict[OrderType, float] = field(default_factory=dict)

    # Market condition multipliers
    condition_multipliers: Dict[MarketCondition, float] = field(default_factory=dict)

    # Time-of-day factors
    use_time_adjustments: bool = True
    opening_multiplier: float = 1.5  # Higher slippage at open
    closing_multiplier: float = 1.3  # Higher at close
    overnight_multiplier: float = 2.0  # Higher for overnight gaps

    # Tracking
    track_execution_quality: bool = True
    execution_history_days: int = 30

    def __post_init__(self):
        if not self.order_type_multipliers:
            self.order_type_multipliers = {
                OrderType.MARKET: 1.0,
                OrderType.LIMIT: 0.3,  # Lower slippage for limits
                OrderType.STOP_MARKET: 1.5,  # Higher for stop markets
                OrderType.STOP_LIMIT: 0.8,
            }

        if not self.condition_multipliers:
            self.condition_multipliers = {
                MarketCondition.VERY_LIQUID: 0.5,
                MarketCondition.LIQUID: 1.0,
                MarketCondition.MODERATE: 1.5,
                MarketCondition.ILLIQUID: 2.5,
                MarketCondition.STRESSED: 4.0,
            }


@dataclass
class SlippageEstimate:
    """Estimated slippage for a trade."""

    expected_slippage_bps: float  # Expected basis points
    expected_slippage_pct: float  # Expected percentage
    expected_slippage_price: float  # Expected price impact
    price_after_slippage: float  # Adjusted entry/exit price
    confidence: float  # Estimation confidence (0-1)
    components: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "expected_slippage_bps": round(self.expected_slippage_bps, 2),
            "expected_slippage_pct": round(self.expected_slippage_pct, 4),
            "expected_slippage_price": round(self.expected_slippage_price, 6),
            "price_after_slippage": round(self.price_after_slippage, 6),
            "confidence": round(self.confidence, 2),
            "components": {k: round(v, 4) for k, v in self.components.items()},
            "warnings": self.warnings,
        }


@dataclass
class ExecutionRecord:
    """Record of actual trade execution."""

    timestamp: datetime
    symbol: str
    side: str  # "buy" or "sell"
    order_type: OrderType
    expected_price: float  # Price at signal
    executed_price: float  # Actual fill price
    size: float
    volume_at_time: float  # Market volume
    spread_at_time: float  # Bid-ask spread
    volatility: float  # Recent volatility
    actual_slippage_bps: float  # Actual slippage
    estimated_slippage_bps: float  # What we predicted
    estimation_error: float  # Prediction error

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type.value,
            "expected_price": self.expected_price,
            "executed_price": self.executed_price,
            "size": self.size,
            "actual_slippage_bps": round(self.actual_slippage_bps, 2),
            "estimated_slippage_bps": round(self.estimated_slippage_bps, 2),
            "estimation_error": round(self.estimation_error, 2),
        }


class SlippageModel:
    """
    Market impact and slippage estimation model.

    Estimates expected slippage based on:
    - Trade size relative to ADV
    - Current volatility
    - Market conditions (liquidity)
    - Order type
    - Time of day
    - Historical execution data
    """

    def __init__(
        self,
        config: Optional[SlippageConfig] = None,
        data_dir: str = "data/slippage",
    ):
        self.config = config or SlippageConfig()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Symbol-specific ADV (Average Daily Volume)
        self._adv: Dict[str, float] = {}

        # Historical execution records
        self._execution_history: Dict[str, List[ExecutionRecord]] = {}

        # Adaptive parameters (learned from execution)
        self._adaptive_base: Dict[str, float] = {}
        self._adaptive_vol_mult: Dict[str, float] = {}

        # Load state
        self._load_state()

    def estimate_slippage(
        self,
        symbol: str,
        price: float,
        size: float,
        side: str,
        order_type: OrderType = OrderType.MARKET,
        volatility: Optional[float] = None,
        current_volume: Optional[float] = None,
        spread_bps: Optional[float] = None,
        market_condition: Optional[MarketCondition] = None,
        hour_of_day: Optional[int] = None,
    ) -> SlippageEstimate:
        """
        Estimate expected slippage for a trade.

        Args:
            symbol: Trading symbol
            price: Current price
            size: Trade size (in quote currency)
            side: "buy" or "sell"
            order_type: Type of order
            volatility: Current volatility (annualized, optional)
            current_volume: Current market volume (optional)
            spread_bps: Current bid-ask spread in basis points (optional)
            market_condition: Current market condition (optional)
            hour_of_day: Current hour (0-23) for time adjustments

        Returns:
            SlippageEstimate with expected slippage
        """
        components = {}
        warnings = []

        # Get base slippage (may be adaptive if we have history)
        base_slippage = self._get_adaptive_base(symbol)
        components["base"] = base_slippage

        # === Volume Impact ===
        volume_impact = 0.0
        adv = self._adv.get(symbol, 0)

        if adv > 0 and current_volume is not None:
            # Estimate proportion of daily volume
            volume_pct = (size / price) / adv * 100

            if volume_pct > self.config.max_volume_pct_adv:
                warnings.append(
                    f"Trade size ({volume_pct:.1f}%) exceeds {self.config.max_volume_pct_adv}% of ADV"
                )
                volume_pct = min(volume_pct, self.config.max_volume_pct_adv * 2)

            # Square root model for market impact
            volume_impact = (
                self.config.volume_impact_coefficient * np.sqrt(volume_pct) * 100
            )  # in bps
            components["volume_impact"] = volume_impact
        elif size / price > 10000:  # Large trade without ADV data
            warnings.append("No ADV data - using conservative volume impact estimate")
            volume_impact = 5.0
            components["volume_impact"] = volume_impact

        # === Volatility Adjustment ===
        vol_multiplier = 1.0
        if volatility is not None:
            # Normalize to a reference volatility (20% annualized)
            vol_ratio = volatility / 20.0
            vol_multiplier = 1.0 + (vol_ratio - 1.0) * (self._get_adaptive_vol_mult(symbol) - 1.0)
            vol_multiplier = max(0.5, min(3.0, vol_multiplier))
            components["volatility_mult"] = vol_multiplier

        # === Spread Component ===
        spread_component = 0.0
        if spread_bps is not None:
            spread_component = spread_bps * 0.5  # Half spread for one-way trade
            components["spread"] = spread_component

        # === Order Type Adjustment ===
        order_mult = self.config.order_type_multipliers.get(order_type, 1.0)
        components["order_type_mult"] = order_mult

        # === Market Condition Adjustment ===
        condition_mult = 1.0
        if market_condition is not None:
            condition_mult = self.config.condition_multipliers.get(market_condition, 1.0)
            components["condition_mult"] = condition_mult

        # === Time of Day Adjustment ===
        time_mult = 1.0
        if self.config.use_time_adjustments and hour_of_day is not None:
            if hour_of_day in [9, 10]:  # Market open (assuming US market)
                time_mult = self.config.opening_multiplier
            elif hour_of_day in [15, 16]:  # Market close
                time_mult = self.config.closing_multiplier
            elif hour_of_day < 9 or hour_of_day > 16:
                time_mult = self.config.overnight_multiplier
            components["time_mult"] = time_mult

        # === Calculate Total Slippage ===
        base_with_spread = base_slippage + spread_component + volume_impact
        total_slippage_bps = (
            base_with_spread * vol_multiplier * order_mult * condition_mult * time_mult
        )

        # Apply limits
        total_slippage_bps = max(
            self.config.min_slippage_bps, min(self.config.max_slippage_bps, total_slippage_bps)
        )

        # Calculate price impact
        slippage_pct = total_slippage_bps / 10000
        slippage_price = price * slippage_pct

        # Adjust price based on side
        if side.lower() == "buy":
            price_after = price + slippage_price
        else:
            price_after = price - slippage_price

        # Estimate confidence based on data availability
        confidence = 0.5  # Base confidence
        if symbol in self._execution_history and len(self._execution_history[symbol]) > 10:
            confidence += 0.3
        if adv > 0:
            confidence += 0.1
        if volatility is not None:
            confidence += 0.1
        confidence = min(1.0, confidence)

        return SlippageEstimate(
            expected_slippage_bps=total_slippage_bps,
            expected_slippage_pct=slippage_pct,
            expected_slippage_price=slippage_price,
            price_after_slippage=price_after,
            confidence=confidence,
            components=components,
            warnings=warnings,
        )

    def record_execution(
        self,
        symbol: str,
        side: str,
        order_type: OrderType,
        expected_price: float,
        executed_price: float,
        size: float,
        volume_at_time: float = 0.0,
        spread_at_time: float = 0.0,
        volatility: float = 0.0,
        estimated_slippage: Optional[SlippageEstimate] = None,
    ):
        """
        Record actual execution for model improvement.

        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            order_type: Type of order used
            expected_price: Price at signal time
            executed_price: Actual fill price
            size: Trade size
            volume_at_time: Market volume at execution
            spread_at_time: Spread at execution
            volatility: Volatility at execution
            estimated_slippage: The estimate we made (if any)
        """
        # Calculate actual slippage
        if side.lower() == "buy":
            actual_slippage = (executed_price - expected_price) / expected_price
        else:
            actual_slippage = (expected_price - executed_price) / expected_price

        actual_slippage_bps = actual_slippage * 10000

        # Get estimated slippage
        estimated_bps = estimated_slippage.expected_slippage_bps if estimated_slippage else 0
        estimation_error = actual_slippage_bps - estimated_bps

        record = ExecutionRecord(
            timestamp=datetime.now(),
            symbol=symbol,
            side=side,
            order_type=order_type,
            expected_price=expected_price,
            executed_price=executed_price,
            size=size,
            volume_at_time=volume_at_time,
            spread_at_time=spread_at_time,
            volatility=volatility,
            actual_slippage_bps=actual_slippage_bps,
            estimated_slippage_bps=estimated_bps,
            estimation_error=estimation_error,
        )

        if symbol not in self._execution_history:
            self._execution_history[symbol] = []
        self._execution_history[symbol].append(record)

        # Trim history
        cutoff = datetime.now() - timedelta(days=self.config.execution_history_days)
        self._execution_history[symbol] = [
            r for r in self._execution_history[symbol] if r.timestamp > cutoff
        ]

        # Update adaptive parameters
        self._update_adaptive_parameters(symbol)
        self._save_state()

    def _get_adaptive_base(self, symbol: str) -> float:
        """Get adaptive base slippage for symbol."""
        return self._adaptive_base.get(symbol, self.config.base_slippage_bps)

    def _get_adaptive_vol_mult(self, symbol: str) -> float:
        """Get adaptive volatility multiplier for symbol."""
        return self._adaptive_vol_mult.get(symbol, self.config.volatility_multiplier)

    def _update_adaptive_parameters(self, symbol: str):
        """Update adaptive parameters from execution history."""
        if symbol not in self._execution_history:
            return

        history = self._execution_history[symbol]
        if len(history) < 10:
            return

        # Calculate median actual slippage
        recent = history[-50:]
        actual_slippages = [r.actual_slippage_bps for r in recent]
        median_slippage = np.median(actual_slippages)

        # Update base slippage (exponential moving average)
        alpha = 0.1
        current_base = self._adaptive_base.get(symbol, self.config.base_slippage_bps)
        self._adaptive_base[symbol] = alpha * median_slippage + (1 - alpha) * current_base

        # Update volatility multiplier based on correlation
        vol_slippage_pairs = [
            (r.volatility, r.actual_slippage_bps) for r in recent if r.volatility > 0
        ]

        if len(vol_slippage_pairs) >= 10:
            vols = np.array([p[0] for p in vol_slippage_pairs])
            slips = np.array([p[1] for p in vol_slippage_pairs])

            # Simple linear fit for vol impact
            if np.std(vols) > 0:
                correlation = np.corrcoef(vols, slips)[0, 1]
                if not np.isnan(correlation) and correlation > 0.2:
                    # Volatility has positive impact
                    slope = np.polyfit(vols, slips, 1)[0]
                    vol_mult = 1.0 + slope * 0.01  # Scale appropriately
                    vol_mult = max(1.0, min(4.0, vol_mult))
                    self._adaptive_vol_mult[symbol] = vol_mult

    def update_adv(self, symbol: str, adv: float):
        """Update Average Daily Volume for a symbol."""
        self._adv[symbol] = adv

    def get_execution_quality(self, symbol: Optional[str] = None) -> Dict:
        """
        Get execution quality metrics.

        Args:
            symbol: Specific symbol (or None for all)

        Returns:
            Execution quality metrics
        """
        if symbol:
            history = self._execution_history.get(symbol, [])
            return self._calculate_quality_metrics(history, symbol)

        # All symbols
        all_metrics = {}
        for sym, hist in self._execution_history.items():
            all_metrics[sym] = self._calculate_quality_metrics(hist, sym)

        # Aggregate
        if all_metrics:
            all_records = []
            for hist in self._execution_history.values():
                all_records.extend(hist)
            all_metrics["_aggregate"] = self._calculate_quality_metrics(all_records, "all")

        return all_metrics

    def _calculate_quality_metrics(
        self,
        history: List[ExecutionRecord],
        symbol: str,
    ) -> Dict:
        """Calculate quality metrics from execution history."""
        if not history:
            return {"symbol": symbol, "executions": 0}

        recent = history[-100:]

        actual_slippages = [r.actual_slippage_bps for r in recent]
        estimation_errors = [r.estimation_error for r in recent]

        # Positive slippage = worse execution
        avg_slippage = np.mean(actual_slippages)
        median_slippage = np.median(actual_slippages)
        max_slippage = np.max(actual_slippages)
        std_slippage = np.std(actual_slippages)

        # Estimation accuracy
        mae = np.mean(np.abs(estimation_errors))  # Mean Absolute Error
        bias = np.mean(estimation_errors)  # Systematic over/under estimation

        # Execution improvement (are we getting better?)
        if len(recent) >= 20:
            first_half = np.mean([r.actual_slippage_bps for r in recent[: len(recent) // 2]])
            second_half = np.mean([r.actual_slippage_bps for r in recent[len(recent) // 2 :]])
            improvement = first_half - second_half
        else:
            improvement = 0

        return {
            "symbol": symbol,
            "executions": len(recent),
            "avg_slippage_bps": round(avg_slippage, 2),
            "median_slippage_bps": round(median_slippage, 2),
            "max_slippage_bps": round(max_slippage, 2),
            "std_slippage_bps": round(std_slippage, 2),
            "estimation_mae_bps": round(mae, 2),
            "estimation_bias_bps": round(bias, 2),
            "execution_improvement_bps": round(improvement, 2),
            "adaptive_base_bps": round(
                self._adaptive_base.get(symbol, self.config.base_slippage_bps), 2
            ),
        }

    def estimate_trade_cost(
        self,
        symbol: str,
        price: float,
        size: float,
        side: str,
        commission_rate: float = 0.001,  # 0.1% commission
        **kwargs,
    ) -> Dict:
        """
        Estimate total trade cost including slippage and commissions.

        Args:
            symbol: Trading symbol
            price: Current price
            size: Trade size
            side: "buy" or "sell"
            commission_rate: Commission rate (e.g., 0.001 for 0.1%)
            **kwargs: Additional args for slippage estimation

        Returns:
            Dict with cost breakdown
        """
        slippage = self.estimate_slippage(symbol, price, size, side, **kwargs)

        trade_value = size
        commission = trade_value * commission_rate
        slippage_cost = trade_value * slippage.expected_slippage_pct

        total_cost = commission + slippage_cost
        total_cost_pct = total_cost / trade_value * 100

        return {
            "trade_value": round(trade_value, 2),
            "commission": round(commission, 4),
            "commission_pct": round(commission_rate * 100, 4),
            "slippage_cost": round(slippage_cost, 4),
            "slippage_pct": round(slippage.expected_slippage_pct * 100, 4),
            "total_cost": round(total_cost, 4),
            "total_cost_pct": round(total_cost_pct, 4),
            "effective_entry": round(slippage.price_after_slippage, 6),
            "slippage_details": slippage.to_dict(),
        }

    def _save_state(self):
        """Save model state to disk."""
        state_file = self.data_dir / "slippage_state.json"

        # Serialize execution history
        history_data = {}
        for symbol, records in self._execution_history.items():
            history_data[symbol] = [r.to_dict() for r in records[-100:]]

        state = {
            "adv": self._adv,
            "adaptive_base": self._adaptive_base,
            "adaptive_vol_mult": self._adaptive_vol_mult,
            "execution_history": history_data,
            "updated_at": datetime.now().isoformat(),
        }

        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        """Load model state from disk."""
        state_file = self.data_dir / "slippage_state.json"
        if not state_file.exists():
            return

        try:
            with open(state_file, "r") as f:
                state = json.load(f)

            self._adv = state.get("adv", {})
            self._adaptive_base = state.get("adaptive_base", {})
            self._adaptive_vol_mult = state.get("adaptive_vol_mult", {})

            # Execution history is not fully reloaded (would need full deserialization)
            # Just use counts for now

        except (json.JSONDecodeError, KeyError):
            pass


def detect_market_condition(
    spread_bps: float,
    volume_ratio: float,  # current volume / average volume
    volatility_ratio: float,  # current vol / average vol
) -> MarketCondition:
    """
    Detect current market condition based on metrics.

    Args:
        spread_bps: Current bid-ask spread in basis points
        volume_ratio: Current volume / average volume
        volatility_ratio: Current volatility / average volatility

    Returns:
        Market condition classification
    """
    # Stress indicators
    stress_score = 0

    # Spread analysis
    if spread_bps < 5:
        stress_score -= 1  # Very tight spread
    elif spread_bps < 15:
        pass  # Normal
    elif spread_bps < 30:
        stress_score += 1  # Wider spread
    else:
        stress_score += 2  # Very wide

    # Volume analysis
    if volume_ratio > 2.0:
        # High volume can be good (more liquidity) or bad (panic)
        if volatility_ratio > 1.5:
            stress_score += 1  # High vol + high volume = stressed
    elif volume_ratio < 0.5:
        stress_score += 1  # Low volume = lower liquidity

    # Volatility analysis
    if volatility_ratio < 0.7:
        stress_score -= 1  # Low vol = calm markets
    elif volatility_ratio > 1.5:
        stress_score += 1
    elif volatility_ratio > 2.5:
        stress_score += 2  # Very high vol

    # Map score to condition
    if stress_score <= -2:
        return MarketCondition.VERY_LIQUID
    elif stress_score <= 0:
        return MarketCondition.LIQUID
    elif stress_score <= 1:
        return MarketCondition.MODERATE
    elif stress_score <= 2:
        return MarketCondition.ILLIQUID
    else:
        return MarketCondition.STRESSED
