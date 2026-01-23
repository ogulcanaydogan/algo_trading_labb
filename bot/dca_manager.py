"""
Dollar Cost Averaging (DCA) Module.

Implements DCA strategy for averaging down on drawdowns
and scaling into positions.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DCAConfig:
    """Configuration for DCA strategy."""

    enabled: bool = True

    # DCA levels (drawdown percentage -> additional position size multiplier)
    # Example: at 5% drawdown, add 0.5x of original position
    dca_levels: List[Dict[str, float]] = field(
        default_factory=lambda: [
            {"drawdown_pct": 0.03, "size_multiplier": 0.3},  # -3% -> add 30%
            {"drawdown_pct": 0.05, "size_multiplier": 0.4},  # -5% -> add 40%
            {"drawdown_pct": 0.08, "size_multiplier": 0.5},  # -8% -> add 50%
            {"drawdown_pct": 0.12, "size_multiplier": 0.6},  # -12% -> add 60%
        ]
    )

    # Maximum DCA orders per position
    max_dca_orders: int = 4

    # Minimum time between DCA orders (seconds)
    min_dca_interval: int = 3600  # 1 hour

    # Maximum total position size after DCA (multiple of original)
    max_position_multiplier: float = 3.0

    # Cooldown after taking profit before allowing new DCA
    profit_cooldown_seconds: int = 7200  # 2 hours


@dataclass
class DCAState:
    """State for a single position's DCA."""

    symbol: str
    original_entry_price: float
    original_quantity: float
    average_entry_price: float
    total_quantity: float
    total_cost: float
    dca_orders_count: int = 0
    dca_history: List[Dict[str, Any]] = field(default_factory=list)
    last_dca_time: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def current_size_multiplier(self) -> float:
        """Current position size as multiple of original."""
        if self.original_quantity == 0:
            return 0
        return self.total_quantity / self.original_quantity

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "original_entry_price": self.original_entry_price,
            "original_quantity": self.original_quantity,
            "average_entry_price": self.average_entry_price,
            "total_quantity": self.total_quantity,
            "total_cost": self.total_cost,
            "dca_orders_count": self.dca_orders_count,
            "dca_history": self.dca_history,
            "last_dca_time": self.last_dca_time.isoformat() if self.last_dca_time else None,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DCAState":
        return cls(
            symbol=data["symbol"],
            original_entry_price=data["original_entry_price"],
            original_quantity=data["original_quantity"],
            average_entry_price=data["average_entry_price"],
            total_quantity=data["total_quantity"],
            total_cost=data["total_cost"],
            dca_orders_count=data.get("dca_orders_count", 0),
            dca_history=data.get("dca_history", []),
            last_dca_time=datetime.fromisoformat(data["last_dca_time"])
            if data.get("last_dca_time")
            else None,
            created_at=datetime.fromisoformat(data["created_at"]),
        )


class DCAManager:
    """
    Manages Dollar Cost Averaging for positions.

    Features:
    - Automatic DCA at predefined drawdown levels
    - Position size scaling based on drawdown
    - Maximum position limits
    - Cooldown periods
    """

    def __init__(self, config: Optional[DCAConfig] = None):
        self.config = config or DCAConfig()
        self.positions: Dict[str, DCAState] = {}

    def add_position(
        self,
        symbol: str,
        entry_price: float,
        quantity: float,
    ) -> DCAState:
        """
        Add a new position for DCA tracking.

        Args:
            symbol: Trading symbol
            entry_price: Entry price
            quantity: Position quantity

        Returns:
            DCAState for the position
        """
        cost = entry_price * quantity
        state = DCAState(
            symbol=symbol,
            original_entry_price=entry_price,
            original_quantity=quantity,
            average_entry_price=entry_price,
            total_quantity=quantity,
            total_cost=cost,
        )

        self.positions[symbol] = state
        logger.info(f"DCA tracking added: {symbol} qty={quantity:.6f} @ ${entry_price:.2f}")

        return state

    def check_dca_opportunity(
        self,
        symbol: str,
        current_price: float,
        available_capital: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Check if a DCA opportunity exists.

        Args:
            symbol: Trading symbol
            current_price: Current market price
            available_capital: Available capital for DCA

        Returns:
            DCA order details if opportunity exists, None otherwise
        """
        if not self.config.enabled:
            return None

        if symbol not in self.positions:
            return None

        state = self.positions[symbol]

        # Check max DCA orders
        if state.dca_orders_count >= self.config.max_dca_orders:
            return None

        # Check max position multiplier
        if state.current_size_multiplier >= self.config.max_position_multiplier:
            return None

        # Check minimum interval
        if state.last_dca_time:
            elapsed = (datetime.now() - state.last_dca_time).total_seconds()
            if elapsed < self.config.min_dca_interval:
                return None

        # Calculate drawdown
        drawdown_pct = (state.average_entry_price - current_price) / state.average_entry_price

        # Find applicable DCA level
        applicable_level = None
        for level in self.config.dca_levels:
            # Check if we've already DCA'd at this level
            already_used = any(
                h.get("level_drawdown") == level["drawdown_pct"] for h in state.dca_history
            )
            if already_used:
                continue

            if drawdown_pct >= level["drawdown_pct"]:
                applicable_level = level

        if not applicable_level:
            return None

        # Calculate DCA order size
        dca_quantity = state.original_quantity * applicable_level["size_multiplier"]
        dca_cost = dca_quantity * current_price

        # Check if we have enough capital
        if dca_cost > available_capital:
            # Scale down to available capital
            dca_quantity = available_capital / current_price
            dca_cost = available_capital

        if dca_quantity <= 0:
            return None

        # Calculate new average
        new_total_cost = state.total_cost + dca_cost
        new_total_quantity = state.total_quantity + dca_quantity
        new_average = new_total_cost / new_total_quantity

        return {
            "symbol": symbol,
            "action": "DCA_BUY",
            "current_price": current_price,
            "dca_quantity": dca_quantity,
            "dca_cost": dca_cost,
            "drawdown_pct": drawdown_pct,
            "level_drawdown": applicable_level["drawdown_pct"],
            "size_multiplier": applicable_level["size_multiplier"],
            "current_average": state.average_entry_price,
            "new_average": new_average,
            "improvement_pct": (state.average_entry_price - new_average)
            / state.average_entry_price,
            "dca_order_number": state.dca_orders_count + 1,
        }

    def execute_dca(
        self,
        symbol: str,
        quantity: float,
        price: float,
        level_drawdown: float,
    ) -> DCAState:
        """
        Record a DCA order execution.

        Args:
            symbol: Trading symbol
            quantity: DCA quantity
            price: Execution price
            level_drawdown: The drawdown level this DCA was triggered at

        Returns:
            Updated DCAState
        """
        if symbol not in self.positions:
            raise ValueError(f"No DCA tracking for {symbol}")

        state = self.positions[symbol]

        # Update state
        cost = quantity * price
        state.total_cost += cost
        state.total_quantity += quantity
        state.average_entry_price = state.total_cost / state.total_quantity
        state.dca_orders_count += 1
        state.last_dca_time = datetime.now()

        # Record in history
        state.dca_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "quantity": quantity,
                "price": price,
                "cost": cost,
                "level_drawdown": level_drawdown,
                "new_average": state.average_entry_price,
                "order_number": state.dca_orders_count,
            }
        )

        logger.info(
            f"DCA executed: {symbol} +{quantity:.6f} @ ${price:.2f}, "
            f"new avg=${state.average_entry_price:.2f}, "
            f"total={state.total_quantity:.6f}"
        )

        return state

    def remove_position(self, symbol: str) -> bool:
        """Remove DCA tracking for a position."""
        if symbol in self.positions:
            del self.positions[symbol]
            logger.info(f"DCA tracking removed: {symbol}")
            return True
        return False

    def get_average_entry(self, symbol: str) -> Optional[float]:
        """Get average entry price for a symbol."""
        if symbol in self.positions:
            return self.positions[symbol].average_entry_price
        return None

    def get_position_state(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get DCA state for a symbol."""
        if symbol in self.positions:
            return self.positions[symbol].to_dict()
        return None

    def get_all_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get all DCA position states."""
        return {symbol: state.to_dict() for symbol, state in self.positions.items()}

    def load_state(self, state_dict: Dict[str, Dict[str, Any]]) -> None:
        """Load DCA states from dict."""
        self.positions = {symbol: DCAState.from_dict(data) for symbol, data in state_dict.items()}
        logger.info(f"Loaded {len(self.positions)} DCA position states")

    def get_summary(self, symbol: str, current_price: float) -> Optional[Dict[str, Any]]:
        """Get summary for a DCA position."""
        if symbol not in self.positions:
            return None

        state = self.positions[symbol]
        unrealized_pnl = (current_price - state.average_entry_price) * state.total_quantity
        unrealized_pnl_pct = (current_price - state.average_entry_price) / state.average_entry_price

        return {
            "symbol": symbol,
            "original_entry": state.original_entry_price,
            "average_entry": state.average_entry_price,
            "current_price": current_price,
            "total_quantity": state.total_quantity,
            "total_cost": state.total_cost,
            "current_value": current_price * state.total_quantity,
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pnl_pct": unrealized_pnl_pct,
            "dca_orders": state.dca_orders_count,
            "position_multiplier": state.current_size_multiplier,
            "average_improvement": (state.original_entry_price - state.average_entry_price)
            / state.original_entry_price
            if state.dca_orders_count > 0
            else 0,
        }


def create_dca_manager(
    enabled: bool = True,
    max_dca_orders: int = 4,
    max_position_multiplier: float = 3.0,
) -> DCAManager:
    """Factory function to create DCA manager."""
    config = DCAConfig(
        enabled=enabled,
        max_dca_orders=max_dca_orders,
        max_position_multiplier=max_position_multiplier,
    )
    return DCAManager(config)
