"""
Production Monitor.

Real-time monitoring dashboard for live trading:
- Daily P&L tracking
- Win rate (rolling 20 trades)
- Sharpe ratio (rolling 7 days)
- System health (latency, errors)
- Risk metrics (exposure, VaR)
- Alerts and notifications
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from collections import deque
from enum import Enum
import statistics
import json
import os

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MetricStatus(Enum):
    """Health status for metrics."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class Alert:
    """System alert."""
    level: AlertLevel
    category: str
    message: str
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False


@dataclass
class TradeRecord:
    """Record of a trade for monitoring."""
    trade_id: str
    symbol: str
    side: str
    pnl: float
    pnl_pct: float
    entry_time: datetime
    exit_time: datetime
    leverage: float
    is_short: bool


@dataclass
class SystemHealth:
    """Overall system health status."""
    status: MetricStatus
    api_latency_ms: float
    error_rate: float
    last_heartbeat: datetime
    active_processes: int
    memory_usage_mb: float
    cpu_usage_pct: float


@dataclass
class RiskMetrics:
    """Risk metrics snapshot."""
    total_exposure: float
    net_exposure: float
    exposure_pct: float
    position_count: int
    max_position_pct: float
    correlation_risk: float
    var_95: float  # 95% Value at Risk
    var_99: float  # 99% Value at Risk


class ProductionMonitor:
    """
    Comprehensive production monitoring system.

    Features:
    - Real-time P&L tracking
    - Performance metrics (win rate, Sharpe)
    - System health monitoring
    - Risk metrics
    - Alerting system
    - Historical data storage
    """

    # Alert thresholds
    ALERT_THRESHOLDS = {
        "daily_loss_pct": {
            AlertLevel.WARNING: -0.01,  # 1% loss
            AlertLevel.CRITICAL: -0.02,  # 2% loss
            AlertLevel.EMERGENCY: -0.03,  # 3% loss
        },
        "win_rate": {
            AlertLevel.WARNING: 0.45,
            AlertLevel.CRITICAL: 0.40,
        },
        "sharpe_ratio": {
            AlertLevel.WARNING: 0.3,
            AlertLevel.CRITICAL: 0.0,
        },
        "api_latency_ms": {
            AlertLevel.WARNING: 1000,
            AlertLevel.CRITICAL: 3000,
        },
        "error_rate": {
            AlertLevel.WARNING: 0.05,
            AlertLevel.CRITICAL: 0.10,
        },
        "exposure_pct": {
            AlertLevel.WARNING: 0.80,
            AlertLevel.CRITICAL: 0.95,
        },
    }

    def __init__(
        self,
        total_capital: float = 30000.0,
        daily_target: float = 300.0,
        alert_callback: Optional[Callable[[Alert], None]] = None,
        state_file: Optional[str] = None,
    ):
        """
        Initialize production monitor.

        Args:
            total_capital: Total trading capital
            daily_target: Daily P&L target
            alert_callback: Callback for alerts
            state_file: Path to persist state
        """
        self.total_capital = total_capital
        self.daily_target = daily_target
        self.alert_callback = alert_callback
        self.state_file = state_file

        # Trade tracking
        self._trades: deque = deque(maxlen=1000)
        self._daily_pnl: Dict[str, float] = {}  # date -> pnl

        # Metrics tracking
        self._hourly_snapshots: deque = deque(maxlen=168)  # 1 week
        self._daily_snapshots: deque = deque(maxlen=365)  # 1 year

        # Alerts
        self._alerts: List[Alert] = []
        self._alert_cooldown: Dict[str, datetime] = {}

        # System health
        self._health = SystemHealth(
            status=MetricStatus.UNKNOWN,
            api_latency_ms=0,
            error_rate=0,
            last_heartbeat=datetime.now(),
            active_processes=0,
            memory_usage_mb=0,
            cpu_usage_pct=0,
        )

        # Risk metrics
        self._risk = RiskMetrics(
            total_exposure=0,
            net_exposure=0,
            exposure_pct=0,
            position_count=0,
            max_position_pct=0,
            correlation_risk=0,
            var_95=0,
            var_99=0,
        )

        # Session tracking
        self._session_start = datetime.now()
        self._session_trades = 0
        self._session_pnl = 0.0

        logger.info("ProductionMonitor initialized")

    def record_trade(self, trade: TradeRecord):
        """Record a completed trade."""
        self._trades.append(trade)

        # Update session stats
        self._session_trades += 1
        self._session_pnl += trade.pnl

        # Update daily P&L
        date_key = trade.exit_time.strftime("%Y-%m-%d")
        self._daily_pnl[date_key] = self._daily_pnl.get(date_key, 0) + trade.pnl

        # Check for alerts
        self._check_trade_alerts(trade)

        # Check daily alerts
        self._check_daily_alerts()

        logger.debug(f"Recorded trade: {trade.symbol} PnL=${trade.pnl:.2f}")

    def update_system_health(
        self,
        api_latency_ms: float,
        error_rate: float,
        active_processes: int = 1,
        memory_usage_mb: float = 0,
        cpu_usage_pct: float = 0,
    ):
        """Update system health metrics."""
        self._health.api_latency_ms = api_latency_ms
        self._health.error_rate = error_rate
        self._health.last_heartbeat = datetime.now()
        self._health.active_processes = active_processes
        self._health.memory_usage_mb = memory_usage_mb
        self._health.cpu_usage_pct = cpu_usage_pct

        # Determine status
        if error_rate > 0.10 or api_latency_ms > 3000:
            self._health.status = MetricStatus.CRITICAL
        elif error_rate > 0.05 or api_latency_ms > 1000:
            self._health.status = MetricStatus.DEGRADED
        else:
            self._health.status = MetricStatus.HEALTHY

        # Check for health alerts
        self._check_health_alerts()

    def update_risk_metrics(
        self,
        positions: Dict[str, Dict],
        correlations: Optional[Dict] = None,
    ):
        """Update risk metrics based on current positions."""
        if not positions:
            self._risk = RiskMetrics(
                total_exposure=0,
                net_exposure=0,
                exposure_pct=0,
                position_count=0,
                max_position_pct=0,
                correlation_risk=0,
                var_95=0,
                var_99=0,
            )
            return

        # Calculate exposures
        long_exposure = sum(
            p.get("value", 0) for p in positions.values()
            if p.get("side") == "long"
        )
        short_exposure = sum(
            abs(p.get("value", 0)) for p in positions.values()
            if p.get("side") == "short"
        )

        total_exposure = long_exposure + short_exposure
        net_exposure = long_exposure - short_exposure

        # Max position
        max_position = max(
            (abs(p.get("value", 0)) for p in positions.values()),
            default=0
        )
        max_position_pct = max_position / self.total_capital if self.total_capital > 0 else 0

        # Correlation risk (simplified)
        correlation_risk = 0.5 if len(positions) > 3 else 0.2

        # VaR calculation (simplified - assumes 2% daily volatility)
        daily_vol = 0.02
        self._risk = RiskMetrics(
            total_exposure=total_exposure,
            net_exposure=net_exposure,
            exposure_pct=total_exposure / self.total_capital if self.total_capital > 0 else 0,
            position_count=len(positions),
            max_position_pct=max_position_pct,
            correlation_risk=correlation_risk,
            var_95=net_exposure * daily_vol * 1.65,  # 95% confidence
            var_99=net_exposure * daily_vol * 2.33,  # 99% confidence
        )

        # Check risk alerts
        self._check_risk_alerts()

    def _check_trade_alerts(self, trade: TradeRecord):
        """Check for trade-related alerts."""
        # Large loss alert
        if trade.pnl < -self.total_capital * 0.01:  # >1% loss
            self._create_alert(
                AlertLevel.WARNING,
                "trade",
                f"Large loss on {trade.symbol}: ${trade.pnl:.2f}",
                "trade_loss",
                trade.pnl,
            )

    def _check_daily_alerts(self):
        """Check for daily performance alerts."""
        today = datetime.now().strftime("%Y-%m-%d")
        daily_pnl = self._daily_pnl.get(today, 0)
        daily_pnl_pct = daily_pnl / self.total_capital if self.total_capital > 0 else 0

        thresholds = self.ALERT_THRESHOLDS["daily_loss_pct"]

        for level in [AlertLevel.EMERGENCY, AlertLevel.CRITICAL, AlertLevel.WARNING]:
            if daily_pnl_pct <= thresholds.get(level, -1):
                self._create_alert(
                    level,
                    "daily_pnl",
                    f"Daily loss: ${daily_pnl:.2f} ({daily_pnl_pct:.1%})",
                    "daily_loss_pct",
                    daily_pnl_pct,
                    thresholds[level],
                )
                break

    def _check_health_alerts(self):
        """Check for system health alerts."""
        # Latency alerts
        thresholds = self.ALERT_THRESHOLDS["api_latency_ms"]
        for level in [AlertLevel.CRITICAL, AlertLevel.WARNING]:
            if self._health.api_latency_ms >= thresholds.get(level, float('inf')):
                self._create_alert(
                    level,
                    "health",
                    f"High API latency: {self._health.api_latency_ms:.0f}ms",
                    "api_latency_ms",
                    self._health.api_latency_ms,
                    thresholds[level],
                )
                break

        # Error rate alerts
        thresholds = self.ALERT_THRESHOLDS["error_rate"]
        for level in [AlertLevel.CRITICAL, AlertLevel.WARNING]:
            if self._health.error_rate >= thresholds.get(level, float('inf')):
                self._create_alert(
                    level,
                    "health",
                    f"High error rate: {self._health.error_rate:.1%}",
                    "error_rate",
                    self._health.error_rate,
                    thresholds[level],
                )
                break

    def _check_risk_alerts(self):
        """Check for risk-related alerts."""
        thresholds = self.ALERT_THRESHOLDS["exposure_pct"]
        for level in [AlertLevel.CRITICAL, AlertLevel.WARNING]:
            if self._risk.exposure_pct >= thresholds.get(level, float('inf')):
                self._create_alert(
                    level,
                    "risk",
                    f"High exposure: {self._risk.exposure_pct:.1%}",
                    "exposure_pct",
                    self._risk.exposure_pct,
                    thresholds[level],
                )
                break

    def _create_alert(
        self,
        level: AlertLevel,
        category: str,
        message: str,
        metric_name: Optional[str] = None,
        metric_value: Optional[float] = None,
        threshold: Optional[float] = None,
    ):
        """Create and process an alert."""
        # Check cooldown (same alert type within 5 minutes)
        cooldown_key = f"{category}_{metric_name}_{level.value}"
        if cooldown_key in self._alert_cooldown:
            if datetime.now() - self._alert_cooldown[cooldown_key] < timedelta(minutes=5):
                return

        self._alert_cooldown[cooldown_key] = datetime.now()

        alert = Alert(
            level=level,
            category=category,
            message=message,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=threshold,
        )

        self._alerts.append(alert)

        # Trigger callback
        if self.alert_callback:
            try:
                self.alert_callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

        # Log based on level
        if level == AlertLevel.EMERGENCY:
            logger.critical(f"EMERGENCY ALERT: {message}")
        elif level == AlertLevel.CRITICAL:
            logger.error(f"CRITICAL ALERT: {message}")
        elif level == AlertLevel.WARNING:
            logger.warning(f"WARNING: {message}")
        else:
            logger.info(f"INFO: {message}")

    def get_performance_metrics(self) -> Dict:
        """Get performance metrics."""
        recent_trades = list(self._trades)[-20:]

        # Win rate
        if recent_trades:
            wins = sum(1 for t in recent_trades if t.pnl > 0)
            win_rate = wins / len(recent_trades)
        else:
            win_rate = 0.0

        # Daily returns for Sharpe
        daily_returns = []
        for date, pnl in sorted(self._daily_pnl.items())[-20:]:
            daily_returns.append(pnl / self.total_capital)

        # Sharpe ratio
        if len(daily_returns) >= 2:
            mean_return = statistics.mean(daily_returns)
            std_return = statistics.stdev(daily_returns) if len(daily_returns) > 1 else 0.001
            sharpe = (mean_return / max(0.001, std_return)) * (252 ** 0.5)
        else:
            sharpe = 0.0

        # Today's P&L
        today = datetime.now().strftime("%Y-%m-%d")
        today_pnl = self._daily_pnl.get(today, 0)

        # Progress to daily target
        progress = today_pnl / self.daily_target if self.daily_target > 0 else 0

        return {
            "win_rate_20": win_rate,
            "sharpe_7d": sharpe,
            "today_pnl": today_pnl,
            "today_pnl_pct": today_pnl / self.total_capital if self.total_capital > 0 else 0,
            "daily_target_progress": progress,
            "session_trades": self._session_trades,
            "session_pnl": self._session_pnl,
            "total_trades_recorded": len(self._trades),
        }

    def get_dashboard_data(self) -> Dict:
        """Get all data for dashboard display."""
        performance = self.get_performance_metrics()

        return {
            "timestamp": datetime.now().isoformat(),
            "capital": self.total_capital,
            "daily_target": self.daily_target,
            "performance": performance,
            "health": {
                "status": self._health.status.value,
                "api_latency_ms": self._health.api_latency_ms,
                "error_rate": self._health.error_rate,
                "last_heartbeat": self._health.last_heartbeat.isoformat(),
                "memory_mb": self._health.memory_usage_mb,
                "cpu_pct": self._health.cpu_usage_pct,
            },
            "risk": {
                "total_exposure": self._risk.total_exposure,
                "net_exposure": self._risk.net_exposure,
                "exposure_pct": self._risk.exposure_pct,
                "position_count": self._risk.position_count,
                "var_95": self._risk.var_95,
                "var_99": self._risk.var_99,
            },
            "alerts": {
                "total": len(self._alerts),
                "unacknowledged": sum(1 for a in self._alerts if not a.acknowledged),
                "recent": [
                    {
                        "level": a.level.value,
                        "category": a.category,
                        "message": a.message,
                        "timestamp": a.timestamp.isoformat(),
                    }
                    for a in self._alerts[-10:]
                ],
            },
            "session": {
                "start": self._session_start.isoformat(),
                "duration_hours": (datetime.now() - self._session_start).total_seconds() / 3600,
                "trades": self._session_trades,
                "pnl": self._session_pnl,
            },
        }

    def get_daily_summary(self, date: Optional[str] = None) -> Dict:
        """Get summary for a specific day."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        pnl = self._daily_pnl.get(date, 0)

        # Get trades for the day
        day_trades = [
            t for t in self._trades
            if t.exit_time.strftime("%Y-%m-%d") == date
        ]

        wins = [t for t in day_trades if t.pnl > 0]
        losses = [t for t in day_trades if t.pnl < 0]

        return {
            "date": date,
            "pnl": pnl,
            "pnl_pct": pnl / self.total_capital if self.total_capital > 0 else 0,
            "target_met": pnl >= self.daily_target,
            "trades": len(day_trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(day_trades) if day_trades else 0,
            "avg_win": statistics.mean(t.pnl for t in wins) if wins else 0,
            "avg_loss": statistics.mean(t.pnl for t in losses) if losses else 0,
        }

    def get_alerts(
        self,
        level: Optional[AlertLevel] = None,
        acknowledged: Optional[bool] = None,
        limit: int = 50,
    ) -> List[Alert]:
        """Get alerts with optional filtering."""
        alerts = self._alerts

        if level is not None:
            alerts = [a for a in alerts if a.level == level]

        if acknowledged is not None:
            alerts = [a for a in alerts if a.acknowledged == acknowledged]

        return alerts[-limit:]

    def acknowledge_alert(self, index: int):
        """Acknowledge an alert by index."""
        if 0 <= index < len(self._alerts):
            self._alerts[index].acknowledged = True

    def acknowledge_all_alerts(self):
        """Acknowledge all alerts."""
        for alert in self._alerts:
            alert.acknowledged = True

    def export_report(self, days: int = 7) -> Dict:
        """Export performance report for last N days."""
        cutoff = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff.strftime("%Y-%m-%d")

        # Daily P&Ls
        daily_data = {
            date: pnl for date, pnl in self._daily_pnl.items()
            if date >= cutoff_str
        }

        # Trades in period
        period_trades = [
            t for t in self._trades
            if t.exit_time >= cutoff
        ]

        # Calculate metrics
        total_pnl = sum(daily_data.values())
        wins = sum(1 for t in period_trades if t.pnl > 0)
        win_rate = wins / len(period_trades) if period_trades else 0

        return {
            "period_start": cutoff_str,
            "period_end": datetime.now().strftime("%Y-%m-%d"),
            "days": days,
            "daily_pnl": daily_data,
            "total_pnl": total_pnl,
            "avg_daily_pnl": total_pnl / len(daily_data) if daily_data else 0,
            "total_trades": len(period_trades),
            "win_rate": win_rate,
            "targets_met": sum(1 for pnl in daily_data.values() if pnl >= self.daily_target),
            "positive_days": sum(1 for pnl in daily_data.values() if pnl > 0),
            "negative_days": sum(1 for pnl in daily_data.values() if pnl < 0),
            "best_day": max(daily_data.values()) if daily_data else 0,
            "worst_day": min(daily_data.values()) if daily_data else 0,
        }

    def save_state(self):
        """Save monitor state to file."""
        if not self.state_file:
            return

        try:
            state = {
                "daily_pnl": self._daily_pnl,
                "session_start": self._session_start.isoformat(),
                "session_trades": self._session_trades,
                "session_pnl": self._session_pnl,
                "alerts": [
                    {
                        "level": a.level.value,
                        "category": a.category,
                        "message": a.message,
                        "timestamp": a.timestamp.isoformat(),
                        "acknowledged": a.acknowledged,
                    }
                    for a in self._alerts[-100:]
                ],
            }

            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def load_state(self):
        """Load monitor state from file."""
        if not self.state_file or not os.path.exists(self.state_file):
            return

        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)

            self._daily_pnl = state.get("daily_pnl", {})

            # Restore alerts
            for alert_data in state.get("alerts", []):
                self._alerts.append(Alert(
                    level=AlertLevel(alert_data["level"]),
                    category=alert_data["category"],
                    message=alert_data["message"],
                    timestamp=datetime.fromisoformat(alert_data["timestamp"]),
                    acknowledged=alert_data["acknowledged"],
                ))

            logger.info("Loaded monitor state")

        except Exception as e:
            logger.error(f"Failed to load state: {e}")


# Singleton
_production_monitor: Optional[ProductionMonitor] = None


def get_production_monitor(
    total_capital: float = 30000.0,
    daily_target: float = 300.0,
) -> ProductionMonitor:
    """Get or create the ProductionMonitor singleton."""
    global _production_monitor
    if _production_monitor is None:
        _production_monitor = ProductionMonitor(
            total_capital=total_capital,
            daily_target=daily_target,
        )
    return _production_monitor
