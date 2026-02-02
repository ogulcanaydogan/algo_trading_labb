"""
Global Turnover Governor.

Reduces trading friction and improves net-of-cost edge by enforcing:
- Minimum decision interval per symbol
- Maximum decisions per day per symbol
- Minimum expected value multiple of costs

CRITICAL: This is a pre-filter only. TradeGate/RiskBudget/CapitalPreservation
retain final execution authority. No execution authority granted to this module.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SymbolOverrideConfig:
    """Per-symbol override configuration for turnover governor."""

    min_interval_minutes: Optional[float] = None
    max_decisions_per_day: Optional[int] = None
    min_ev_cost_multiple: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_interval_minutes": self.min_interval_minutes,
            "max_decisions_per_day": self.max_decisions_per_day,
            "min_ev_cost_multiple": self.min_ev_cost_multiple,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SymbolOverrideConfig":
        return cls(
            min_interval_minutes=data.get("min_interval_minutes"),
            max_decisions_per_day=data.get("max_decisions_per_day"),
            min_ev_cost_multiple=data.get("min_ev_cost_multiple"),
        )


# Default stricter BTC overrides (more volatile, higher risk)
DEFAULT_BTC_OVERRIDE = SymbolOverrideConfig(
    min_interval_minutes=30.0,  # 30 min vs 15 min default
    max_decisions_per_day=6,  # 6 vs 10 default
    min_ev_cost_multiple=2.5,  # 2.5x vs 2.0x default
)


@dataclass
class TurnoverGovernorConfig:
    """Configuration for the turnover governor."""

    # Time-based controls (defaults for symbols without overrides)
    min_decision_interval_minutes: float = 15.0  # Minimum minutes between decisions
    max_decisions_per_day: int = 10  # Maximum decisions per symbol per day

    # Cost-based controls
    min_expected_value_multiple: float = 2.0  # EV must be >= k * costs
    default_fee_bps: float = 10.0  # Default fee in basis points
    default_slippage_bps: float = 5.0  # Default slippage in basis points

    # Per-symbol overrides (symbol pattern -> override config)
    # Patterns: "BTC/USDT", "BTC_USDT", "BTC*" (prefix match), etc.
    symbol_overrides: Dict[str, SymbolOverrideConfig] = field(default_factory=dict)

    # Reporting
    state_path: Path = field(default_factory=lambda: Path("data/turnover_governor_state.json"))

    # Enable/disable
    enabled: bool = True

    def __post_init__(self):
        """Apply default BTC overrides if not already specified."""
        # Auto-apply stricter BTC rules unless explicitly overridden
        btc_patterns = ["BTC/USDT", "BTC_USDT", "BTC/USD", "BTCUSDT"]
        for pattern in btc_patterns:
            if pattern not in self.symbol_overrides:
                self.symbol_overrides[pattern] = DEFAULT_BTC_OVERRIDE


@dataclass
class SymbolTurnoverState:
    """Tracking state for a single symbol."""

    symbol: str
    last_decision_time: Optional[datetime] = None
    decisions_today: int = 0
    decisions_today_date: str = ""
    blocked_today: int = 0
    blocked_reasons: Dict[str, int] = field(default_factory=dict)
    estimated_cost_drag: float = 0.0  # Estimated costs avoided by blocking

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "last_decision_time": self.last_decision_time.isoformat() if self.last_decision_time else None,
            "decisions_today": self.decisions_today,
            "decisions_today_date": self.decisions_today_date,
            "blocked_today": self.blocked_today,
            "blocked_reasons": self.blocked_reasons,
            "estimated_cost_drag": self.estimated_cost_drag,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SymbolTurnoverState":
        return cls(
            symbol=data.get("symbol", ""),
            last_decision_time=datetime.fromisoformat(data["last_decision_time"])
            if data.get("last_decision_time")
            else None,
            decisions_today=data.get("decisions_today", 0),
            decisions_today_date=data.get("decisions_today_date", ""),
            blocked_today=data.get("blocked_today", 0),
            blocked_reasons=data.get("blocked_reasons", {}),
            estimated_cost_drag=data.get("estimated_cost_drag", 0.0),
        )


@dataclass
class EffectiveSymbolConfig:
    """Resolved effective configuration for a symbol (after applying overrides)."""

    symbol: str
    min_interval_minutes: float
    max_decisions_per_day: int
    min_ev_cost_multiple: float
    has_override: bool = False
    override_source: str = ""  # Which pattern matched, if any

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "min_interval_minutes": self.min_interval_minutes,
            "max_decisions_per_day": self.max_decisions_per_day,
            "min_ev_cost_multiple": self.min_ev_cost_multiple,
            "has_override": self.has_override,
            "override_source": self.override_source,
        }


@dataclass
class TurnoverDecision:
    """Result of turnover governor evaluation."""

    allowed: bool
    reason: str = ""
    blocked_by: str = ""  # "interval", "daily_limit", "ev_cost", or ""
    estimated_cost: float = 0.0
    expected_value: float = 0.0
    cost_multiple: float = 0.0
    effective_config: Optional[EffectiveSymbolConfig] = None


class TurnoverGovernor:
    """
    Global Turnover Governor for all symbols.

    Reduces excessive trading that erodes edge through fees and slippage.
    This is a PRE-FILTER only - TradeGate/RiskBudget/CapitalPreservation
    retain final authority over execution decisions.
    """

    def __init__(self, config: Optional[TurnoverGovernorConfig] = None):
        self.config = config or TurnoverGovernorConfig()
        self._symbol_states: Dict[str, SymbolTurnoverState] = {}
        self._daily_stats: Dict[str, Any] = {
            "date": "",
            "total_decisions_allowed": 0,
            "total_decisions_blocked": 0,
            "blocked_by_interval": 0,
            "blocked_by_daily_limit": 0,
            "blocked_by_ev_cost": 0,
            "total_cost_drag_avoided": 0.0,
        }
        self._load_state()
        logger.info(
            f"TurnoverGovernor initialized: interval={self.config.min_decision_interval_minutes}min, "
            f"max_daily={self.config.max_decisions_per_day}, ev_multiple={self.config.min_expected_value_multiple}x"
        )

    def _load_state(self) -> None:
        """Load persisted state."""
        if self.config.state_path.exists():
            try:
                with open(self.config.state_path) as f:
                    data = json.load(f)
                for symbol_data in data.get("symbols", []):
                    state = SymbolTurnoverState.from_dict(symbol_data)
                    self._symbol_states[state.symbol] = state
                self._daily_stats = data.get("daily_stats", self._daily_stats)
                logger.debug(f"Loaded turnover state for {len(self._symbol_states)} symbols")
            except Exception as e:
                logger.warning(f"Failed to load turnover state: {e}")

    def _save_state(self) -> None:
        """Persist current state."""
        try:
            self.config.state_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "symbols": [s.to_dict() for s in self._symbol_states.values()],
                "daily_stats": self._daily_stats,
                "timestamp": datetime.now().isoformat(),
            }
            with open(self.config.state_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save turnover state: {e}")

    def _get_symbol_state(self, symbol: str) -> SymbolTurnoverState:
        """Get or create state for a symbol."""
        if symbol not in self._symbol_states:
            self._symbol_states[symbol] = SymbolTurnoverState(symbol=symbol)
        return self._symbol_states[symbol]

    def _get_effective_config(self, symbol: str) -> EffectiveSymbolConfig:
        """
        Get the effective configuration for a symbol after applying overrides.

        Override matching priority:
        1. Exact match (e.g., "BTC/USDT")
        2. Normalized match (e.g., "BTC_USDT" matches "BTC/USDT")
        3. Prefix match (e.g., "BTC*" matches "BTC/USDT")
        4. Default config
        """
        # Normalize symbol for matching
        symbol_normalized = symbol.upper().replace("/", "_").replace("-", "_")

        override: Optional[SymbolOverrideConfig] = None
        override_source = ""

        # Check for exact match first
        if symbol in self.config.symbol_overrides:
            override = self.config.symbol_overrides[symbol]
            override_source = symbol
        else:
            # Check normalized patterns
            for pattern, config in self.config.symbol_overrides.items():
                pattern_normalized = pattern.upper().replace("/", "_").replace("-", "_")

                # Exact normalized match
                if pattern_normalized == symbol_normalized:
                    override = config
                    override_source = pattern
                    break

                # Prefix match (e.g., "BTC*" matches "BTC_USDT")
                if pattern.endswith("*"):
                    prefix = pattern[:-1].upper().replace("/", "_").replace("-", "_")
                    if symbol_normalized.startswith(prefix):
                        override = config
                        override_source = pattern
                        break

        # Build effective config
        if override:
            return EffectiveSymbolConfig(
                symbol=symbol,
                min_interval_minutes=override.min_interval_minutes
                if override.min_interval_minutes is not None
                else self.config.min_decision_interval_minutes,
                max_decisions_per_day=override.max_decisions_per_day
                if override.max_decisions_per_day is not None
                else self.config.max_decisions_per_day,
                min_ev_cost_multiple=override.min_ev_cost_multiple
                if override.min_ev_cost_multiple is not None
                else self.config.min_expected_value_multiple,
                has_override=True,
                override_source=override_source,
            )
        else:
            return EffectiveSymbolConfig(
                symbol=symbol,
                min_interval_minutes=self.config.min_decision_interval_minutes,
                max_decisions_per_day=self.config.max_decisions_per_day,
                min_ev_cost_multiple=self.config.min_expected_value_multiple,
                has_override=False,
                override_source="",
            )

    def _reset_daily_if_needed(self, symbol_state: SymbolTurnoverState) -> None:
        """Reset daily counters if date changed."""
        today = datetime.now().strftime("%Y-%m-%d")

        if symbol_state.decisions_today_date != today:
            symbol_state.decisions_today = 0
            symbol_state.blocked_today = 0
            symbol_state.blocked_reasons = {}
            symbol_state.estimated_cost_drag = 0.0
            symbol_state.decisions_today_date = today

        if self._daily_stats.get("date") != today:
            self._daily_stats = {
                "date": today,
                "total_decisions_allowed": 0,
                "total_decisions_blocked": 0,
                "blocked_by_interval": 0,
                "blocked_by_daily_limit": 0,
                "blocked_by_ev_cost": 0,
                "total_cost_drag_avoided": 0.0,
            }

    def _estimate_trade_cost(
        self,
        symbol: str,
        position_size_usd: float,
        fee_bps: Optional[float] = None,
        slippage_bps: Optional[float] = None,
    ) -> float:
        """
        Estimate total cost for a trade (fees + slippage).

        Args:
            symbol: Trading symbol
            position_size_usd: Position size in USD
            fee_bps: Fee in basis points (defaults to config)
            slippage_bps: Slippage in basis points (defaults to config)

        Returns:
            Estimated total cost in USD
        """
        fee = fee_bps if fee_bps is not None else self.config.default_fee_bps
        slippage = slippage_bps if slippage_bps is not None else self.config.default_slippage_bps

        # Total cost = position_size * (fee_bps + slippage_bps) / 10000
        # For round-trip, multiply by 2
        total_bps = (fee + slippage) * 2  # Round-trip
        total_cost = position_size_usd * total_bps / 10000

        return total_cost

    def evaluate_decision(
        self,
        symbol: str,
        action: str,
        expected_pnl: float = 0.0,
        position_size_usd: float = 1000.0,
        confidence: float = 0.5,
        fee_bps: Optional[float] = None,
        slippage_bps: Optional[float] = None,
        is_rl_advisory: bool = False,
    ) -> TurnoverDecision:
        """
        Evaluate whether a trading decision should be allowed.

        Args:
            symbol: Trading symbol
            action: Action type ("buy", "sell", "short", "hold")
            expected_pnl: Expected P&L from the trade
            position_size_usd: Position size in USD
            confidence: Signal confidence (0-1)
            fee_bps: Override fee in basis points
            slippage_bps: Override slippage in basis points
            is_rl_advisory: Whether this is an RL advisory (vs baseline strategy)

        Returns:
            TurnoverDecision with allowed status and reason
        """
        if not self.config.enabled:
            return TurnoverDecision(allowed=True, reason="Governor disabled")

        # Hold actions always allowed (no cost)
        if action.lower() in ("hold", "flat", "none", ""):
            return TurnoverDecision(allowed=True, reason="Hold action - no cost")

        # Get effective config for this symbol (may have overrides)
        effective_config = self._get_effective_config(symbol)

        symbol_state = self._get_symbol_state(symbol)
        self._reset_daily_if_needed(symbol_state)
        now = datetime.now()

        # Calculate estimated cost
        estimated_cost = self._estimate_trade_cost(
            symbol, position_size_usd, fee_bps, slippage_bps
        )

        # Calculate expected value (confidence-weighted expected PnL)
        # If no expected_pnl provided, estimate from confidence
        if expected_pnl == 0.0:
            # Rough estimate: position_size * confidence * typical_move
            typical_move_pct = 0.5  # 0.5% typical move
            expected_pnl = position_size_usd * confidence * typical_move_pct / 100

        expected_value = expected_pnl * confidence
        cost_multiple = expected_value / estimated_cost if estimated_cost > 0 else float("inf")

        # Check 1: Minimum decision interval (using effective config)
        if symbol_state.last_decision_time:
            elapsed = (now - symbol_state.last_decision_time).total_seconds() / 60
            if elapsed < effective_config.min_interval_minutes:
                remaining = effective_config.min_interval_minutes - elapsed
                self._record_blocked(symbol_state, "interval", estimated_cost)
                source = "RL advisory" if is_rl_advisory else "Strategy"
                return TurnoverDecision(
                    allowed=False,
                    reason=f"{source} blocked: {remaining:.1f}min until next decision allowed (limit: {effective_config.min_interval_minutes}min)",
                    effective_config=effective_config,
                    blocked_by="interval",
                    estimated_cost=estimated_cost,
                    expected_value=expected_value,
                    cost_multiple=cost_multiple,
                )

        # Check 2: Maximum decisions per day (using effective config)
        if symbol_state.decisions_today >= effective_config.max_decisions_per_day:
            self._record_blocked(symbol_state, "daily_limit", estimated_cost)
            source = "RL advisory" if is_rl_advisory else "Strategy"
            return TurnoverDecision(
                allowed=False,
                reason=f"{source} blocked: Daily limit reached ({effective_config.max_decisions_per_day})",
                blocked_by="daily_limit",
                estimated_cost=estimated_cost,
                expected_value=expected_value,
                cost_multiple=cost_multiple,
                effective_config=effective_config,
            )

        # Check 3: Expected value must exceed cost multiple (using effective config)
        if cost_multiple < effective_config.min_ev_cost_multiple:
            self._record_blocked(symbol_state, "ev_cost", estimated_cost)
            source = "RL advisory" if is_rl_advisory else "Strategy"
            return TurnoverDecision(
                allowed=False,
                reason=f"{source} blocked: EV/cost ratio {cost_multiple:.2f}x < {effective_config.min_ev_cost_multiple}x required",
                blocked_by="ev_cost",
                estimated_cost=estimated_cost,
                expected_value=expected_value,
                cost_multiple=cost_multiple,
                effective_config=effective_config,
            )

        # Decision allowed
        return TurnoverDecision(
            allowed=True,
            reason="Passed turnover checks",
            estimated_cost=estimated_cost,
            expected_value=expected_value,
            cost_multiple=cost_multiple,
            effective_config=effective_config,
        )

    def record_decision_taken(self, symbol: str) -> None:
        """
        Record that a decision was actually taken (after all gates passed).

        Call this only after TradeGate/RiskBudget/CapitalPreservation approve.
        """
        symbol_state = self._get_symbol_state(symbol)
        self._reset_daily_if_needed(symbol_state)
        effective_config = self._get_effective_config(symbol)

        symbol_state.last_decision_time = datetime.now()
        symbol_state.decisions_today += 1
        self._daily_stats["total_decisions_allowed"] += 1

        self._save_state()
        logger.debug(
            f"Turnover: Decision recorded for {symbol} "
            f"(today: {symbol_state.decisions_today}/{effective_config.max_decisions_per_day})"
        )

    def _record_blocked(
        self, symbol_state: SymbolTurnoverState, reason: str, estimated_cost: float
    ) -> None:
        """Record a blocked decision."""
        symbol_state.blocked_today += 1
        symbol_state.blocked_reasons[reason] = symbol_state.blocked_reasons.get(reason, 0) + 1
        symbol_state.estimated_cost_drag += estimated_cost

        self._daily_stats["total_decisions_blocked"] += 1
        self._daily_stats[f"blocked_by_{reason}"] = (
            self._daily_stats.get(f"blocked_by_{reason}", 0) + 1
        )
        self._daily_stats["total_cost_drag_avoided"] += estimated_cost

        self._save_state()
        logger.debug(
            f"Turnover: Decision blocked for {symbol_state.symbol} "
            f"(reason: {reason}, cost avoided: ${estimated_cost:.2f})"
        )

    def get_symbol_status(self, symbol: str) -> Dict[str, Any]:
        """Get current status for a symbol including effective config."""
        symbol_state = self._get_symbol_state(symbol)
        self._reset_daily_if_needed(symbol_state)
        effective_config = self._get_effective_config(symbol)

        time_until_next = 0.0
        if symbol_state.last_decision_time:
            elapsed = (datetime.now() - symbol_state.last_decision_time).total_seconds() / 60
            remaining = effective_config.min_interval_minutes - elapsed
            time_until_next = max(0, remaining)

        return {
            "symbol": symbol,
            "decisions_today": symbol_state.decisions_today,
            "max_decisions_per_day": effective_config.max_decisions_per_day,
            "decisions_remaining": max(0, effective_config.max_decisions_per_day - symbol_state.decisions_today),
            "time_until_next_allowed_minutes": time_until_next,
            "blocked_today": symbol_state.blocked_today,
            "blocked_reasons": symbol_state.blocked_reasons,
            "estimated_cost_drag_avoided": symbol_state.estimated_cost_drag,
            "effective_config": effective_config.to_dict(),
        }

    def get_daily_stats(self) -> Dict[str, Any]:
        """Get daily statistics across all symbols."""
        today = datetime.now().strftime("%Y-%m-%d")
        if self._daily_stats.get("date") != today:
            # Reset for new day
            self._daily_stats = {
                "date": today,
                "total_decisions_allowed": 0,
                "total_decisions_blocked": 0,
                "blocked_by_interval": 0,
                "blocked_by_daily_limit": 0,
                "blocked_by_ev_cost": 0,
                "total_cost_drag_avoided": 0.0,
            }

        return {
            **self._daily_stats,
            "symbols_tracked": len(self._symbol_states),
            "config": {
                "min_decision_interval_minutes": self.config.min_decision_interval_minutes,
                "max_decisions_per_day": self.config.max_decisions_per_day,
                "min_expected_value_multiple": self.config.min_expected_value_multiple,
            },
        }

    def get_weekly_stats(self) -> Dict[str, Any]:
        """
        Get weekly statistics (aggregated from state).

        Note: For full weekly tracking, integrate with weekly report generator.
        """
        total_blocked = sum(s.blocked_today for s in self._symbol_states.values())
        total_cost_avoided = sum(s.estimated_cost_drag for s in self._symbol_states.values())

        return {
            "total_decisions_blocked": total_blocked,
            "total_cost_drag_avoided": total_cost_avoided,
            "by_symbol": {
                symbol: {
                    "blocked": state.blocked_today,
                    "cost_avoided": state.estimated_cost_drag,
                    "reasons": state.blocked_reasons,
                    "effective_config": self._get_effective_config(symbol).to_dict(),
                }
                for symbol, state in self._symbol_states.items()
            },
        }

    def get_all_symbol_configs(self) -> Dict[str, Any]:
        """
        Get effective configuration for all tracked symbols.

        Returns configuration for:
        - All symbols with activity (from state)
        - Default config for symbols without overrides
        - Override patterns and their configs
        """
        configs = {}

        # Include all symbols that have been tracked
        for symbol in self._symbol_states:
            configs[symbol] = self._get_effective_config(symbol).to_dict()

        # Include override patterns
        override_patterns = {}
        for pattern, override in self.config.symbol_overrides.items():
            override_patterns[pattern] = override.to_dict()

        return {
            "default_config": {
                "min_interval_minutes": self.config.min_decision_interval_minutes,
                "max_decisions_per_day": self.config.max_decisions_per_day,
                "min_ev_cost_multiple": self.config.min_expected_value_multiple,
            },
            "override_patterns": override_patterns,
            "symbols": configs,
        }

    def reset_daily_stats(self) -> None:
        """Reset all daily statistics (call at start of new trading day)."""
        today = datetime.now().strftime("%Y-%m-%d")
        for state in self._symbol_states.values():
            state.decisions_today = 0
            state.blocked_today = 0
            state.blocked_reasons = {}
            state.estimated_cost_drag = 0.0
            state.decisions_today_date = today

        self._daily_stats = {
            "date": today,
            "total_decisions_allowed": 0,
            "total_decisions_blocked": 0,
            "blocked_by_interval": 0,
            "blocked_by_daily_limit": 0,
            "blocked_by_ev_cost": 0,
            "total_cost_drag_avoided": 0.0,
        }
        self._save_state()
        logger.info("Turnover governor daily stats reset")


# Singleton instance
_turnover_governor: Optional[TurnoverGovernor] = None


def get_turnover_governor(config: Optional[TurnoverGovernorConfig] = None) -> TurnoverGovernor:
    """Get or create the singleton turnover governor."""
    global _turnover_governor
    if _turnover_governor is None:
        _turnover_governor = TurnoverGovernor(config)
    return _turnover_governor


def reset_turnover_governor() -> None:
    """Reset the singleton (for testing)."""
    global _turnover_governor
    _turnover_governor = None
