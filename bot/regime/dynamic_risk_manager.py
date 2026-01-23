"""
Dynamic Risk Manager.

Features:
- ATR-based dynamic stop-loss
- Drawdown-based position reduction
- Daily/weekly loss limits
- Correlation monitoring
- Regime-aware risk adjustments
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .regime_detector import MarketRegime

logger = logging.getLogger(__name__)


class RiskAction(Enum):
    """Risk management actions."""

    NORMAL = "normal"
    REDUCE_SIZE = "reduce_size"
    PAUSE_TRADING = "pause_trading"
    CLOSE_ALL = "close_all"
    HEDGE = "hedge"


@dataclass
class DynamicRiskConfig:
    """Configuration for dynamic risk management."""

    # ATR-based stop-loss
    atr_period: int = 14
    atr_stop_multiplier: float = 2.0  # Stop at 2x ATR
    atr_take_profit_multiplier: float = 3.0  # TP at 3x ATR (1.5:1 R:R)

    # Regime-specific ATR multipliers
    regime_atr_multipliers: Dict[str, float] = field(
        default_factory=lambda: {
            "bull": 2.0,
            "bear": 1.5,  # Tighter stops in bear
            "sideways": 1.5,
            "high_vol": 2.5,  # Wider stops in high vol
            "crash": 1.0,  # Very tight in crash
        }
    )

    # Drawdown protection
    drawdown_reduce_threshold: float = 0.05  # Reduce size at 5% DD
    drawdown_pause_threshold: float = 0.10  # Pause at 10% DD
    drawdown_close_threshold: float = 0.15  # Close all at 15% DD
    drawdown_recovery_threshold: float = 0.03  # Resume when DD < 3%

    # Daily/weekly limits
    daily_loss_limit: float = 0.02  # 2% max daily loss
    weekly_loss_limit: float = 0.05  # 5% max weekly loss
    daily_trade_limit: int = 20  # Max trades per day

    # Position sizing based on drawdown
    size_reduction_schedule: Dict[float, float] = field(
        default_factory=lambda: {
            0.03: 0.75,  # At 3% DD, reduce to 75% size
            0.05: 0.50,  # At 5% DD, reduce to 50% size
            0.08: 0.25,  # At 8% DD, reduce to 25% size
            0.10: 0.10,  # At 10% DD, reduce to 10% size
        }
    )

    # Correlation limits
    max_correlation: float = 0.7  # Reduce if assets too correlated
    correlation_lookback: int = 30  # Days for correlation calc


@dataclass
class RiskState:
    """Current risk state."""

    current_drawdown: float = 0.0
    peak_equity: float = 0.0
    current_equity: float = 0.0
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    daily_trades: int = 0
    action: RiskAction = RiskAction.NORMAL
    position_size_multiplier: float = 1.0
    reason: str = ""
    last_update: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "current_drawdown": self.current_drawdown,
            "current_drawdown_pct": f"{self.current_drawdown * 100:.2f}%",
            "peak_equity": self.peak_equity,
            "current_equity": self.current_equity,
            "daily_pnl": self.daily_pnl,
            "weekly_pnl": self.weekly_pnl,
            "daily_trades": self.daily_trades,
            "action": self.action.value,
            "position_size_multiplier": self.position_size_multiplier,
            "reason": self.reason,
            "last_update": self.last_update.isoformat(),
        }


@dataclass
class StopLossResult:
    """Result of stop-loss calculation."""

    stop_loss_price: float
    take_profit_price: float
    atr_value: float
    risk_per_share: float
    reward_per_share: float
    risk_reward_ratio: float

    def to_dict(self) -> Dict:
        return {
            "stop_loss": self.stop_loss_price,
            "take_profit": self.take_profit_price,
            "atr": self.atr_value,
            "risk_per_share": self.risk_per_share,
            "reward_per_share": self.reward_per_share,
            "risk_reward_ratio": self.risk_reward_ratio,
        }


class DynamicRiskManager:
    """
    Dynamic risk manager with ATR-based stops and drawdown protection.
    """

    def __init__(self, config: Optional[DynamicRiskConfig] = None):
        self.config = config or DynamicRiskConfig()

        # State tracking
        self._state = RiskState()
        self._daily_trades: List[Dict] = []
        self._weekly_trades: List[Dict] = []
        self._equity_history: List[Tuple[datetime, float]] = []
        self._pnl_history: List[Dict] = []

        # Trading state
        self._is_paused = False
        self._pause_reason = ""
        self._pause_until: Optional[datetime] = None

    def calculate_atr_stops(
        self,
        df: pd.DataFrame,
        entry_price: float,
        side: str,  # "long" or "short"
        regime: MarketRegime = MarketRegime.SIDEWAYS,
    ) -> StopLossResult:
        """
        Calculate ATR-based stop-loss and take-profit levels.

        Args:
            df: OHLCV DataFrame
            entry_price: Entry price
            side: "long" or "short"
            regime: Current market regime

        Returns:
            StopLossResult with calculated levels
        """
        # Calculate ATR
        atr = self._calculate_atr(df)

        # Get regime-specific multiplier
        regime_mult = self.config.regime_atr_multipliers.get(
            regime.value, self.config.atr_stop_multiplier
        )

        # Calculate stop distance
        stop_distance = atr * regime_mult
        tp_distance = atr * self.config.atr_take_profit_multiplier

        if side == "long":
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + tp_distance
            risk_per_share = entry_price - stop_loss
            reward_per_share = take_profit - entry_price
        else:  # short
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - tp_distance
            risk_per_share = stop_loss - entry_price
            reward_per_share = entry_price - take_profit

        rr_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 0

        return StopLossResult(
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
            atr_value=atr,
            risk_per_share=risk_per_share,
            reward_per_share=reward_per_share,
            risk_reward_ratio=rr_ratio,
        )

    def _calculate_atr(self, df: pd.DataFrame) -> float:
        """Calculate Average True Range."""
        if len(df) < self.config.atr_period + 1:
            # Fallback to simple range
            return df["High"].iloc[-1] - df["Low"].iloc[-1]

        high = df["High"].values
        low = df["Low"].values
        close = df["Close"].values

        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])

        tr = np.maximum(tr1, np.maximum(tr2, tr3))

        # EMA of TR
        atr = pd.Series(tr).ewm(span=self.config.atr_period, adjust=False).mean().iloc[-1]

        return float(atr)

    def update_equity(self, equity: float) -> RiskState:
        """
        Update equity and calculate drawdown-based risk actions.

        Args:
            equity: Current portfolio equity

        Returns:
            Updated RiskState with recommended action
        """
        now = datetime.now()

        # Update peak
        if equity > self._state.peak_equity:
            self._state.peak_equity = equity

        # Calculate drawdown
        if self._state.peak_equity > 0:
            self._state.current_drawdown = (
                self._state.peak_equity - equity
            ) / self._state.peak_equity
        else:
            self._state.current_drawdown = 0.0

        self._state.current_equity = equity
        self._state.last_update = now

        # Record history
        self._equity_history.append((now, equity))
        # Keep last 30 days
        cutoff = now - timedelta(days=30)
        self._equity_history = [(t, e) for t, e in self._equity_history if t > cutoff]

        # Determine action based on drawdown
        self._update_risk_action()

        return self._state

    def _update_risk_action(self):
        """Update risk action based on current state."""
        dd = self._state.current_drawdown

        # Check drawdown thresholds
        if dd >= self.config.drawdown_close_threshold:
            self._state.action = RiskAction.CLOSE_ALL
            self._state.position_size_multiplier = 0.0
            self._state.reason = (
                f"Drawdown {dd:.1%} >= {self.config.drawdown_close_threshold:.1%} close threshold"
            )
            self._is_paused = True
            self._pause_reason = self._state.reason

        elif dd >= self.config.drawdown_pause_threshold:
            self._state.action = RiskAction.PAUSE_TRADING
            self._state.position_size_multiplier = 0.0
            self._state.reason = (
                f"Drawdown {dd:.1%} >= {self.config.drawdown_pause_threshold:.1%} pause threshold"
            )
            self._is_paused = True
            self._pause_reason = self._state.reason

        elif dd >= self.config.drawdown_reduce_threshold:
            self._state.action = RiskAction.REDUCE_SIZE
            # Calculate size multiplier from schedule
            mult = 1.0
            for threshold, size in sorted(self.config.size_reduction_schedule.items()):
                if dd >= threshold:
                    mult = size
            self._state.position_size_multiplier = mult
            self._state.reason = f"Drawdown {dd:.1%} - reducing size to {mult:.0%}"

        else:
            # Check if we can resume from pause
            if self._is_paused and dd < self.config.drawdown_recovery_threshold:
                self._is_paused = False
                self._pause_reason = ""
                logger.info(f"Resuming trading - drawdown recovered to {dd:.1%}")

            if not self._is_paused:
                self._state.action = RiskAction.NORMAL
                self._state.position_size_multiplier = 1.0
                self._state.reason = "Normal operations"

        # Check daily/weekly limits
        self._check_periodic_limits()

    def _check_periodic_limits(self):
        """Check daily and weekly loss limits."""
        now = datetime.now()

        # Calculate daily P&L
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        daily_pnl = sum(
            t.get("pnl", 0) for t in self._daily_trades if t.get("timestamp", now) >= today_start
        )
        self._state.daily_pnl = daily_pnl

        # Check daily limit
        if self._state.peak_equity > 0:
            daily_loss_pct = abs(min(0, daily_pnl)) / self._state.peak_equity
            if daily_loss_pct >= self.config.daily_loss_limit:
                self._state.action = RiskAction.PAUSE_TRADING
                self._state.position_size_multiplier = 0.0
                self._state.reason = f"Daily loss limit reached: {daily_loss_pct:.1%}"
                self._is_paused = True
                self._pause_until = today_start + timedelta(days=1)

        # Calculate weekly P&L
        week_start = now - timedelta(days=now.weekday())
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        weekly_pnl = sum(
            t.get("pnl", 0) for t in self._weekly_trades if t.get("timestamp", now) >= week_start
        )
        self._state.weekly_pnl = weekly_pnl

        # Check weekly limit
        if self._state.peak_equity > 0:
            weekly_loss_pct = abs(min(0, weekly_pnl)) / self._state.peak_equity
            if weekly_loss_pct >= self.config.weekly_loss_limit:
                self._state.action = RiskAction.PAUSE_TRADING
                self._state.position_size_multiplier = 0.0
                self._state.reason = f"Weekly loss limit reached: {weekly_loss_pct:.1%}"
                self._is_paused = True

        # Check daily trade count
        daily_count = len([t for t in self._daily_trades if t.get("timestamp", now) >= today_start])
        self._state.daily_trades = daily_count
        if daily_count >= self.config.daily_trade_limit:
            self._state.action = RiskAction.PAUSE_TRADING
            self._state.reason = f"Daily trade limit reached: {daily_count}"

    def record_trade(self, trade: Dict):
        """Record a trade for limit tracking."""
        trade["timestamp"] = trade.get("timestamp", datetime.now())
        self._daily_trades.append(trade)
        self._weekly_trades.append(trade)
        self._pnl_history.append(trade)

        # Clean old trades
        now = datetime.now()
        cutoff_daily = now - timedelta(days=1)
        cutoff_weekly = now - timedelta(days=7)

        self._daily_trades = [t for t in self._daily_trades if t["timestamp"] > cutoff_daily]
        self._weekly_trades = [t for t in self._weekly_trades if t["timestamp"] > cutoff_weekly]

    def can_trade(self) -> Tuple[bool, str]:
        """Check if trading is allowed."""
        if self._is_paused:
            # Check if pause has expired
            if self._pause_until and datetime.now() > self._pause_until:
                self._is_paused = False
                self._pause_until = None
                self._pause_reason = ""
                return True, "Trading resumed after pause period"

            return False, self._pause_reason

        if self._state.action in (RiskAction.PAUSE_TRADING, RiskAction.CLOSE_ALL):
            return False, self._state.reason

        return True, "Trading allowed"

    def get_position_size_multiplier(self) -> float:
        """Get current position size multiplier based on risk state."""
        return self._state.position_size_multiplier

    def calculate_position_size(
        self,
        account_equity: float,
        risk_per_trade: float,  # e.g., 0.01 for 1%
        entry_price: float,
        stop_loss_price: float,
    ) -> float:
        """
        Calculate position size based on risk parameters.

        Args:
            account_equity: Total account equity
            risk_per_trade: Risk per trade as decimal (0.01 = 1%)
            entry_price: Entry price
            stop_loss_price: Stop-loss price

        Returns:
            Position size (quantity)
        """
        # Apply drawdown-based multiplier
        effective_risk = risk_per_trade * self._state.position_size_multiplier

        # Dollar risk
        dollar_risk = account_equity * effective_risk

        # Risk per unit
        risk_per_unit = abs(entry_price - stop_loss_price)

        if risk_per_unit <= 0:
            return 0.0

        # Position size
        position_size = dollar_risk / risk_per_unit

        return position_size

    def get_state(self) -> RiskState:
        """Get current risk state."""
        return self._state

    def get_status(self) -> Dict:
        """Get full risk manager status."""
        return {
            "state": self._state.to_dict(),
            "is_paused": self._is_paused,
            "pause_reason": self._pause_reason,
            "pause_until": self._pause_until.isoformat() if self._pause_until else None,
            "daily_trades_count": len(self._daily_trades),
            "weekly_trades_count": len(self._weekly_trades),
            "config": {
                "daily_loss_limit": f"{self.config.daily_loss_limit:.1%}",
                "weekly_loss_limit": f"{self.config.weekly_loss_limit:.1%}",
                "drawdown_pause_threshold": f"{self.config.drawdown_pause_threshold:.1%}",
                "daily_trade_limit": self.config.daily_trade_limit,
            },
        }

    def reset_daily(self):
        """Reset daily counters (call at start of trading day)."""
        self._daily_trades.clear()
        self._state.daily_pnl = 0.0
        self._state.daily_trades = 0

        # Check if we can resume from daily pause
        if self._pause_until and datetime.now() > self._pause_until:
            self._is_paused = False
            self._pause_until = None
            self._pause_reason = ""
            logger.info("Daily reset - trading resumed")

    def reset_weekly(self):
        """Reset weekly counters (call at start of trading week)."""
        self._weekly_trades.clear()
        self._state.weekly_pnl = 0.0
