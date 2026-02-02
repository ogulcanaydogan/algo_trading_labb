"""
Trade Qualification Gate.

Production-grade pre-trade validation that computes a gate score and
approves/rejects trades based on multiple risk factors:
- Data freshness validation
- Regime confidence thresholds
- Model uncertainty bounds
- Event risk assessment
- Slippage risk estimation
- Correlation exposure limits
- Drawdown state checks

Implements "no-trade zones" when confidence is insufficient.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class GateDecision(Enum):
    """Gate decision outcomes."""
    APPROVED = "approved"
    REJECTED = "rejected"
    DEFERRED = "deferred"  # Try again later


class RejectionReason(Enum):
    """Reasons for trade rejection."""
    DATA_STALE = "data_stale"
    REGIME_UNCERTAINTY = "regime_uncertainty"
    MODEL_DISAGREEMENT = "model_disagreement"
    EVENT_RISK = "event_risk"
    SLIPPAGE_RISK = "slippage_risk"
    CORRELATION_OVERLOAD = "correlation_overload"
    DRAWDOWN_PROTECTION = "drawdown_protection"
    CONFIDENCE_TOO_LOW = "confidence_too_low"
    NO_TRADE_ZONE = "no_trade_zone"
    CIRCUIT_BREAKER = "circuit_breaker"
    COOLDOWN_ACTIVE = "cooldown_active"


@dataclass
class GateScore:
    """Composite gate score with component breakdown."""
    total_score: float  # 0-100, higher = safer to trade
    data_freshness_score: float  # 0-100
    regime_confidence_score: float  # 0-100
    model_agreement_score: float  # 0-100
    event_risk_score: float  # 0-100 (inverted: high = low risk)
    slippage_risk_score: float  # 0-100 (inverted)
    correlation_score: float  # 0-100 (inverted)
    drawdown_score: float  # 0-100

    weights: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        return {
            "total": self.total_score,
            "data_freshness": self.data_freshness_score,
            "regime_confidence": self.regime_confidence_score,
            "model_agreement": self.model_agreement_score,
            "event_risk": self.event_risk_score,
            "slippage_risk": self.slippage_risk_score,
            "correlation": self.correlation_score,
            "drawdown": self.drawdown_score,
        }


@dataclass
class GateResult:
    """Result of trade gate evaluation."""
    decision: GateDecision
    score: GateScore
    rejection_reasons: List[RejectionReason] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommended_size_multiplier: float = 1.0  # Reduce size if borderline
    retry_after_seconds: int = 0  # For deferred decisions
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_approved(self) -> bool:
        return self.decision == GateDecision.APPROVED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.value,
            "score": self.score.to_dict(),
            "rejection_reasons": [r.value for r in self.rejection_reasons],
            "warnings": self.warnings,
            "recommended_size_multiplier": self.recommended_size_multiplier,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class TradeRequest:
    """Trade request to be evaluated by the gate."""
    symbol: str
    side: str  # BUY/SELL or LONG/SHORT
    quantity: float
    price: float
    order_type: str  # market, limit
    leverage: float = 1.0

    # Signal context
    signal_confidence: float = 0.0
    signal_source: str = ""

    # Market context
    current_regime: str = "unknown"
    regime_confidence: float = 0.0
    volatility: float = 0.0
    spread_bps: float = 0.0  # Spread in basis points
    volume_24h: float = 0.0

    # Model predictions
    model_predictions: Dict[str, float] = field(default_factory=dict)
    model_confidences: Dict[str, float] = field(default_factory=dict)

    # News/events
    upcoming_events_hours: float = 999.0  # Hours until next major event
    news_urgency: int = 0  # 1-10

    # Portfolio context
    correlation_with_portfolio: float = 0.0
    current_drawdown_pct: float = 0.0
    daily_loss_pct: float = 0.0


@dataclass
class GateConfig:
    """Configuration for trade gate thresholds."""

    # Minimum scores for approval (0-100)
    min_total_score: float = 60.0
    min_data_freshness: float = 70.0
    min_regime_confidence: float = 50.0
    min_model_agreement: float = 55.0

    # Risk thresholds
    max_slippage_risk_score: float = 40.0  # Below this = reject
    max_correlation_exposure: float = 0.85
    max_drawdown_for_trading: float = 0.08  # 8%
    max_daily_loss_for_trading: float = 0.02  # 2%

    # Data staleness
    max_data_age_seconds: float = 30.0

    # Event risk
    min_hours_before_event: float = 2.0  # No trading within 2h of major event

    # Model agreement
    max_model_disagreement: float = 0.3  # Max spread between model predictions

    # Cooldown
    cooldown_after_loss_seconds: int = 60
    cooldown_after_consecutive_losses: int = 300  # 5 minutes
    consecutive_loss_threshold: int = 3

    # Score weights
    weights: Dict[str, float] = field(default_factory=lambda: {
        "data_freshness": 0.15,
        "regime_confidence": 0.20,
        "model_agreement": 0.20,
        "event_risk": 0.10,
        "slippage_risk": 0.15,
        "correlation": 0.10,
        "drawdown": 0.10,
    })

    # Size reduction thresholds
    size_reduction_score_70: float = 0.8  # 80% size if score 60-70
    size_reduction_score_65: float = 0.6  # 60% size if score 55-65


class TradeGate:
    """
    Production-grade trade qualification gate.

    Evaluates trade requests against multiple risk factors and
    computes a composite gate score to approve/reject trades.

    Features:
    - Multi-factor risk assessment
    - No-trade zone detection
    - Position size recommendations
    - Cooldown management
    - Audit trail
    """

    def __init__(self, config: Optional[GateConfig] = None):
        self.config = config or GateConfig()
        self._lock = RLock()

        # State tracking
        self._last_trade_time: Optional[datetime] = None
        self._last_trade_was_loss: bool = False
        self._consecutive_losses: int = 0
        self._in_cooldown_until: Optional[datetime] = None

        # Data freshness tracking
        self._last_data_timestamps: Dict[str, datetime] = {}

        # Circuit breaker state
        self._circuit_breaker_open: bool = False
        self._circuit_breaker_reason: Optional[str] = None

        # Audit trail
        self._recent_evaluations: List[GateResult] = []

        logger.info("TradeGate initialized")

    def evaluate(self, request: TradeRequest) -> GateResult:
        """
        Evaluate a trade request against all gate criteria.

        Args:
            request: TradeRequest with full context

        Returns:
            GateResult with decision, score, and recommendations
        """
        with self._lock:
            rejection_reasons = []
            warnings = []

            # Check circuit breaker first
            if self._circuit_breaker_open:
                return GateResult(
                    decision=GateDecision.REJECTED,
                    score=self._zero_score(),
                    rejection_reasons=[RejectionReason.CIRCUIT_BREAKER],
                    warnings=[f"Circuit breaker: {self._circuit_breaker_reason}"],
                )

            # Check cooldown
            if self._in_cooldown_until and datetime.now() < self._in_cooldown_until:
                remaining = (self._in_cooldown_until - datetime.now()).total_seconds()
                return GateResult(
                    decision=GateDecision.DEFERRED,
                    score=self._zero_score(),
                    rejection_reasons=[RejectionReason.COOLDOWN_ACTIVE],
                    retry_after_seconds=int(remaining),
                    warnings=[f"Cooldown active for {remaining:.0f}s"],
                )

            # Calculate component scores
            data_score = self._score_data_freshness(request)
            regime_score = self._score_regime_confidence(request)
            model_score = self._score_model_agreement(request)
            event_score = self._score_event_risk(request)
            slippage_score = self._score_slippage_risk(request)
            correlation_score = self._score_correlation(request)
            drawdown_score = self._score_drawdown(request)

            # Calculate weighted total
            weights = self.config.weights
            total_score = (
                data_score * weights["data_freshness"] +
                regime_score * weights["regime_confidence"] +
                model_score * weights["model_agreement"] +
                event_score * weights["event_risk"] +
                slippage_score * weights["slippage_risk"] +
                correlation_score * weights["correlation"] +
                drawdown_score * weights["drawdown"]
            )

            score = GateScore(
                total_score=total_score,
                data_freshness_score=data_score,
                regime_confidence_score=regime_score,
                model_agreement_score=model_score,
                event_risk_score=event_score,
                slippage_risk_score=slippage_score,
                correlation_score=correlation_score,
                drawdown_score=drawdown_score,
                weights=weights,
            )

            # Check hard rejections
            if data_score < self.config.min_data_freshness:
                rejection_reasons.append(RejectionReason.DATA_STALE)

            if regime_score < self.config.min_regime_confidence:
                rejection_reasons.append(RejectionReason.REGIME_UNCERTAINTY)

            if model_score < self.config.min_model_agreement:
                rejection_reasons.append(RejectionReason.MODEL_DISAGREEMENT)

            if event_score < 30:  # Very close to event
                rejection_reasons.append(RejectionReason.EVENT_RISK)

            if slippage_score < self.config.max_slippage_risk_score:
                rejection_reasons.append(RejectionReason.SLIPPAGE_RISK)

            if correlation_score < 30:
                rejection_reasons.append(RejectionReason.CORRELATION_OVERLOAD)

            if drawdown_score < 30:
                rejection_reasons.append(RejectionReason.DRAWDOWN_PROTECTION)

            if request.signal_confidence < 0.5:
                rejection_reasons.append(RejectionReason.CONFIDENCE_TOO_LOW)

            # Determine decision
            if rejection_reasons:
                decision = GateDecision.REJECTED
            elif total_score < self.config.min_total_score:
                decision = GateDecision.REJECTED
                rejection_reasons.append(RejectionReason.NO_TRADE_ZONE)
            else:
                decision = GateDecision.APPROVED

            # Calculate size multiplier for borderline approvals
            size_multiplier = 1.0
            if decision == GateDecision.APPROVED:
                if total_score < 65:
                    size_multiplier = self.config.size_reduction_score_65
                    warnings.append(f"Borderline score ({total_score:.0f}) - reducing size to {size_multiplier*100:.0f}%")
                elif total_score < 70:
                    size_multiplier = self.config.size_reduction_score_70
                    warnings.append(f"Moderate score ({total_score:.0f}) - reducing size to {size_multiplier*100:.0f}%")

            result = GateResult(
                decision=decision,
                score=score,
                rejection_reasons=rejection_reasons,
                warnings=warnings,
                recommended_size_multiplier=size_multiplier,
            )

            # Store for audit
            self._recent_evaluations.append(result)
            if len(self._recent_evaluations) > 100:
                self._recent_evaluations = self._recent_evaluations[-100:]

            self._log_evaluation(request, result)

            return result

    def _score_data_freshness(self, request: TradeRequest) -> float:
        """Score data freshness (0-100)."""
        # Check if we have recent data
        symbol = request.symbol
        if symbol in self._last_data_timestamps:
            age = (datetime.now() - self._last_data_timestamps[symbol]).total_seconds()
            if age > self.config.max_data_age_seconds * 2:
                return 0.0
            elif age > self.config.max_data_age_seconds:
                return 50.0 * (1 - (age - self.config.max_data_age_seconds) / self.config.max_data_age_seconds)
            else:
                return 100.0 * (1 - age / self.config.max_data_age_seconds * 0.3)

        # If no timestamp tracked, assume recent but warn
        return 70.0

    def _score_regime_confidence(self, request: TradeRequest) -> float:
        """Score regime detection confidence (0-100)."""
        # Base score from regime confidence
        base_score = request.regime_confidence * 100

        # Penalty for unknown regime
        if request.current_regime == "unknown":
            base_score *= 0.5

        # Bonus for trending regimes (easier to trade)
        trending_regimes = ["BULL", "STRONG_BULL", "BEAR", "STRONG_BEAR"]
        if request.current_regime in trending_regimes:
            base_score = min(100, base_score * 1.1)

        return base_score

    def _score_model_agreement(self, request: TradeRequest) -> float:
        """Score model agreement/disagreement (0-100)."""
        predictions = request.model_predictions
        confidences = request.model_confidences

        if not predictions:
            return 50.0  # Neutral if no predictions

        if len(predictions) < 2:
            # Single model - use its confidence
            return list(confidences.values())[0] * 100 if confidences else 60.0

        # Calculate disagreement
        pred_values = list(predictions.values())
        pred_min = min(pred_values)
        pred_max = max(pred_values)
        disagreement = pred_max - pred_min

        # Score inversely proportional to disagreement
        if disagreement > self.config.max_model_disagreement:
            return max(0, 50 - (disagreement - self.config.max_model_disagreement) * 100)

        return 100 * (1 - disagreement / self.config.max_model_disagreement * 0.5)

    def _score_event_risk(self, request: TradeRequest) -> float:
        """Score event risk (0-100, high = low risk)."""
        hours = request.upcoming_events_hours
        urgency = request.news_urgency

        # Event proximity risk
        if hours < self.config.min_hours_before_event:
            return 0.0
        elif hours < self.config.min_hours_before_event * 2:
            return 30.0
        elif hours < self.config.min_hours_before_event * 6:
            return 60.0

        event_score = 100.0

        # News urgency penalty
        if urgency >= 8:
            event_score = 20.0
        elif urgency >= 6:
            event_score = 50.0
        elif urgency >= 4:
            event_score = 70.0

        return event_score

    def _score_slippage_risk(self, request: TradeRequest) -> float:
        """Score slippage risk (0-100, high = low risk)."""
        spread_bps = request.spread_bps
        volatility = request.volatility
        order_type = request.order_type

        # Spread component (wider spread = higher slippage risk)
        if spread_bps > 50:  # 0.5% spread
            spread_score = 20.0
        elif spread_bps > 20:
            spread_score = 50.0
        elif spread_bps > 10:
            spread_score = 70.0
        else:
            spread_score = 90.0

        # Volatility component
        if volatility > 0.05:  # >5% volatility
            vol_score = 30.0
        elif volatility > 0.03:
            vol_score = 60.0
        else:
            vol_score = 90.0

        # Order type adjustment (limit orders have less slippage risk)
        type_multiplier = 1.0 if order_type == "limit" else 0.8

        return (spread_score * 0.5 + vol_score * 0.5) * type_multiplier

    def _score_correlation(self, request: TradeRequest) -> float:
        """Score correlation with existing portfolio (0-100, high = low correlation)."""
        correlation = abs(request.correlation_with_portfolio)

        if correlation > self.config.max_correlation_exposure:
            return 0.0
        elif correlation > 0.7:
            return 30.0
        elif correlation > 0.5:
            return 60.0
        else:
            return 100.0 * (1 - correlation)

    def _score_drawdown(self, request: TradeRequest) -> float:
        """Score based on current drawdown state (0-100)."""
        drawdown = request.current_drawdown_pct
        daily_loss = request.daily_loss_pct

        # Hard block if limits exceeded
        if drawdown > self.config.max_drawdown_for_trading:
            return 0.0
        if daily_loss > self.config.max_daily_loss_for_trading:
            return 10.0  # Very low but not zero

        # Score based on how much room is left
        drawdown_ratio = drawdown / self.config.max_drawdown_for_trading
        daily_ratio = daily_loss / self.config.max_daily_loss_for_trading

        return 100 * (1 - max(drawdown_ratio, daily_ratio))

    def _zero_score(self) -> GateScore:
        """Return a zero score."""
        return GateScore(
            total_score=0,
            data_freshness_score=0,
            regime_confidence_score=0,
            model_agreement_score=0,
            event_risk_score=0,
            slippage_risk_score=0,
            correlation_score=0,
            drawdown_score=0,
        )

    def _log_evaluation(self, request: TradeRequest, result: GateResult):
        """Log the evaluation for debugging."""
        if result.is_approved:
            logger.debug(
                f"TradeGate APPROVED: {request.symbol} {request.side} "
                f"score={result.score.total_score:.1f} size_mult={result.recommended_size_multiplier:.2f}"
            )
        else:
            logger.info(
                f"TradeGate REJECTED: {request.symbol} {request.side} "
                f"score={result.score.total_score:.1f} reasons={[r.value for r in result.rejection_reasons]}"
            )

    def update_data_timestamp(self, symbol: str, timestamp: Optional[datetime] = None):
        """Update the last data timestamp for a symbol."""
        with self._lock:
            self._last_data_timestamps[symbol] = timestamp or datetime.now()

    def record_trade_outcome(self, was_loss: bool):
        """Record trade outcome for cooldown management."""
        with self._lock:
            self._last_trade_time = datetime.now()
            self._last_trade_was_loss = was_loss

            if was_loss:
                self._consecutive_losses += 1

                # Apply cooldown
                if self._consecutive_losses >= self.config.consecutive_loss_threshold:
                    self._in_cooldown_until = datetime.now() + timedelta(
                        seconds=self.config.cooldown_after_consecutive_losses
                    )
                    logger.warning(
                        f"TradeGate entering cooldown for {self.config.cooldown_after_consecutive_losses}s "
                        f"after {self._consecutive_losses} consecutive losses"
                    )
                else:
                    self._in_cooldown_until = datetime.now() + timedelta(
                        seconds=self.config.cooldown_after_loss_seconds
                    )
            else:
                self._consecutive_losses = 0
                self._in_cooldown_until = None

    def trip_circuit_breaker(self, reason: str):
        """Trip the circuit breaker to block all trades."""
        with self._lock:
            self._circuit_breaker_open = True
            self._circuit_breaker_reason = reason
            logger.critical(f"TradeGate circuit breaker TRIPPED: {reason}")

    def reset_circuit_breaker(self):
        """Reset the circuit breaker."""
        with self._lock:
            self._circuit_breaker_open = False
            self._circuit_breaker_reason = None
            logger.info("TradeGate circuit breaker RESET")

    def get_status(self) -> Dict[str, Any]:
        """Get current gate status."""
        with self._lock:
            return {
                "circuit_breaker_open": self._circuit_breaker_open,
                "circuit_breaker_reason": self._circuit_breaker_reason,
                "in_cooldown": self._in_cooldown_until is not None and datetime.now() < self._in_cooldown_until,
                "cooldown_until": self._in_cooldown_until.isoformat() if self._in_cooldown_until else None,
                "consecutive_losses": self._consecutive_losses,
                "last_trade_time": self._last_trade_time.isoformat() if self._last_trade_time else None,
                "recent_evaluations": len(self._recent_evaluations),
            }

    def get_recent_rejections(self, limit: int = 10) -> List[Dict]:
        """Get recent rejection reasons for analysis."""
        with self._lock:
            rejections = [
                r.to_dict() for r in self._recent_evaluations
                if r.decision == GateDecision.REJECTED
            ]
            return rejections[-limit:]


# Singleton instance
_trade_gate: Optional[TradeGate] = None


def get_trade_gate(config: Optional[GateConfig] = None) -> TradeGate:
    """Get or create the TradeGate singleton."""
    global _trade_gate
    if _trade_gate is None:
        _trade_gate = TradeGate(config=config)
    return _trade_gate


def reset_trade_gate() -> None:
    """Reset the singleton instance (for testing)."""
    global _trade_gate
    if _trade_gate is not None:
        _trade_gate._initialized = False
    _trade_gate = None
    TradeGate._instance = None
