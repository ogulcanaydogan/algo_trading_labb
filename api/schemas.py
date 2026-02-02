from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# =============================================================================
# Error Response Models
# =============================================================================


class ErrorDetail(BaseModel):
    """Standard error detail for API responses."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "detail": "Resource not found",
                "error_code": "NOT_FOUND",
                "timestamp": "2024-01-15T10:30:00Z",
            }
        }
    )

    detail: str = Field(..., description="Human-readable error message")
    error_code: Optional[str] = Field(None, description="Machine-readable error code")
    timestamp: Optional[datetime] = Field(None, description="Time when error occurred")


class ValidationErrorDetail(BaseModel):
    """Validation error response model."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "detail": [
                    {
                        "loc": ["body", "symbol"],
                        "msg": "field required",
                        "type": "value_error.missing",
                    }
                ]
            }
        }
    )

    detail: List[Dict[str, str]] = Field(..., description="List of validation errors")


# =============================================================================
# Bot State Models
# =============================================================================


class BotStateResponse(BaseModel):
    """Current state of the trading bot including positions and AI analysis.

    This is the primary endpoint for monitoring bot activity in real-time.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "timestamp": "2024-01-15T10:30:00Z",
                "symbol": "BTC/USDT",
                "position": "LONG",
                "entry_price": 42500.0,
                "position_size": 0.5,
                "balance": 10000.0,
                "unrealized_pnl_pct": 2.5,
                "last_signal": "LONG",
                "last_signal_reason": "EMA crossover with strong momentum",
                "confidence": 0.75,
                "rsi": 55.2,
                "ema_fast": 42800.0,
                "ema_slow": 42600.0,
                "risk_per_trade_pct": 2.0,
                "ai_action": "LONG",
                "ai_confidence": 0.82,
                "macro_bias": 0.6,
                "macro_summary": "Bullish sentiment driven by institutional adoption",
            }
        }
    )

    timestamp: datetime
    symbol: str = Field(..., description="Trading pair symbol (e.g., BTC/USDT)")
    position: Literal["LONG", "SHORT", "FLAT"] = Field(
        ..., description="Current position direction"
    )
    entry_price: Optional[float] = Field(
        None, description="Last entry price for the open position"
    )
    position_size: float
    balance: float
    unrealized_pnl_pct: float
    last_signal: Optional[str] = None
    last_signal_reason: Optional[str] = Field(
        None, description="Narrative explaining why the last decision was taken."
    )
    confidence: Optional[float] = None
    technical_signal: Optional[str] = Field(
        None, description="Baseline technical signal before any AI adjustments."
    )
    technical_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    technical_reason: Optional[str] = None
    ai_override_active: bool = Field(
        False, description="Whether the AI layer overrode the technical signal."
    )
    rsi: Optional[float] = None
    ema_fast: Optional[float] = None
    ema_slow: Optional[float] = None
    risk_per_trade_pct: float
    ai_action: Optional[str] = Field(None, description="AI layer recommended action.")
    ai_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    ai_probability_long: Optional[float] = Field(None, ge=0.0, le=1.0)
    ai_probability_short: Optional[float] = Field(None, ge=0.0, le=1.0)
    ai_probability_flat: Optional[float] = Field(None, ge=0.0, le=1.0)
    ai_expected_move_pct: Optional[float] = None
    ai_summary: Optional[str] = None
    ai_features: Optional[Dict[str, float]] = None
    macro_bias: Optional[float] = None
    macro_confidence: Optional[float] = None
    macro_summary: Optional[str] = None
    macro_drivers: List[str] = Field(default_factory=list)
    macro_interest_rate_outlook: Optional[str] = None
    macro_political_risk: Optional[str] = None
    macro_events: List[Dict[str, Optional[str]]] = Field(default_factory=list)
    portfolio_playbook: Optional[PortfolioPlaybookResponse] = None


class PortfolioBotStatusResponse(BaseModel):
    """Slim status payload for each asset tracked inside the portfolio runner."""

    timestamp: datetime
    symbol: str
    position: Literal["LONG", "SHORT", "FLAT"]
    position_size: float
    balance: float
    initial_balance: float = Field(
        ..., description="Starting capital allocated to this portfolio bot."
    )
    entry_price: Optional[float] = None
    unrealized_pnl_pct: float
    last_signal: Optional[str] = None
    confidence: Optional[float] = None
    risk_per_trade_pct: float
    ai_action: Optional[str] = None
    ai_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    data_directory: str = Field(
        ..., description="Relative data directory that produced this status payload."
    )
    asset_type: Optional[str] = Field(
        None, description="Asset class label assigned in the portfolio config."
    )
    timeframe: Optional[str] = Field(None, description="Primary timeframe configured for the bot.")
    allocation_pct: Optional[float] = Field(
        None, description="Share of the total portfolio allocated to this asset."
    )
    paper_mode: Optional[bool] = Field(
        None, description="Indicates whether the asset runs in paper or live trading."
    )
    loop_interval_seconds: Optional[int] = Field(
        None, description="Configured loop cadence for the asset-specific runner."
    )
    stop_loss_pct: Optional[float] = Field(
        None, description="Stop-loss percentage configured for this bot."
    )
    take_profit_pct: Optional[float] = Field(
        None, description="Take-profit percentage configured for this bot."
    )
    is_paused: bool = Field(
        False, description="True when manual controls have paused execution for the bot."
    )
    pause_reason: Optional[str] = Field(
        None, description="Optional note explaining why the bot is paused."
    )
    pause_updated_at: Optional[datetime] = Field(
        None, description="Timestamp for the latest pause/resume toggle, when known."
    )
    is_placeholder: bool = Field(
        False,
        description="True when the entry is synthesized from configuration and no bot heartbeat has been observed yet.",
    )


class PlaybookHorizonResponse(BaseModel):
    label: str
    timeframe: str
    candles_tested: int
    initial_balance: float
    final_balance: float
    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    win_rate: float
    trades: int
    macro_bias: float
    macro_summary: Optional[str] = None


class AssetPlaybookResponse(BaseModel):
    symbol: str
    asset_class: str
    macro_bias: float
    macro_confidence: float
    macro_summary: Optional[str] = None
    macro_drivers: List[str] = Field(default_factory=list)
    macro_interest_rate_outlook: Optional[str] = None
    macro_political_risk: Optional[str] = None
    horizons: List[PlaybookHorizonResponse] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class PortfolioPlaybookResponse(BaseModel):
    generated_at: datetime
    starting_balance: float
    commodities: List[AssetPlaybookResponse] = Field(default_factory=list)
    equities: List[AssetPlaybookResponse] = Field(default_factory=list)
    highlights: List[str] = Field(default_factory=list)


class SignalResponse(BaseModel):
    """Trading signal generated by the strategy engine.

    Signals represent trading decisions with confidence scores and reasoning.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "timestamp": "2024-01-15T10:30:00Z",
                "symbol": "BTC/USDT",
                "decision": "LONG",
                "confidence": 0.75,
                "reason": "EMA crossover bullish, RSI not overbought",
                "ai_action": "LONG",
                "ai_confidence": 0.82,
                "ai_expected_move_pct": 2.5,
            }
        }
    )

    timestamp: datetime = Field(..., description="Time when signal was generated")
    symbol: str = Field(..., description="Trading pair symbol")
    decision: Literal["LONG", "SHORT", "FLAT"] = Field(
        ..., description="Recommended trading action"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Signal confidence (0-1)")
    reason: str = Field(..., description="Explanation for the trading decision")
    ai_action: Optional[str] = Field(None, description="AI layer recommendation")
    ai_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    ai_expected_move_pct: Optional[float] = Field(
        None, description="AI predicted price move percentage"
    )


class EquityPointResponse(BaseModel):
    timestamp: datetime
    value: float


class StrategyOverviewResponse(BaseModel):
    symbol: str
    timeframe: str
    ema_fast: int
    ema_slow: int
    rsi_period: int
    rsi_overbought: float
    rsi_oversold: float
    risk_per_trade_pct: float
    stop_loss_pct: float
    take_profit_pct: float
    decision_rules: List[str]
    risk_management_notes: List[str]


class AIPredictionResponse(BaseModel):
    timestamp: datetime
    symbol: str
    recommended_action: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    probability_long: Optional[float] = Field(None, ge=0.0, le=1.0)
    probability_short: Optional[float] = Field(None, ge=0.0, le=1.0)
    probability_flat: Optional[float] = Field(None, ge=0.0, le=1.0)
    expected_move_pct: Optional[float] = None
    summary: Optional[str] = None
    features: Optional[Dict[str, float]] = None
    macro_bias: Optional[float] = None
    macro_confidence: Optional[float] = None
    macro_summary: Optional[str] = None
    macro_drivers: List[str] = Field(default_factory=list)
    macro_interest_rate_outlook: Optional[str] = None
    macro_political_risk: Optional[str] = None


class AIQuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)


class AIAnswerResponse(BaseModel):
    question: str
    answer: str


class PortfolioControlUpdateRequest(BaseModel):
    symbol: str = Field(..., min_length=1)
    paused: bool
    reason: Optional[str] = Field(
        None,
        max_length=280,
        description="Optional note describing why the bot was paused.",
    )


class PortfolioControlStateResponse(BaseModel):
    symbol: str
    paused: bool
    reason: Optional[str] = None
    updated_at: Optional[datetime] = None


class MacroEventResponse(BaseModel):
    title: str
    category: Optional[str] = None
    sentiment: Optional[str] = None
    impact: Optional[str] = None
    actor: Optional[str] = None
    interest_rate_expectation: Optional[str] = None
    summary: Optional[str] = None
    timestamp: Optional[str] = None
    source: Optional[str] = None


class MacroInsightResponse(BaseModel):
    timestamp: datetime
    symbol: str
    bias_score: Optional[float] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    summary: Optional[str] = None
    drivers: List[str] = Field(default_factory=list)
    interest_rate_outlook: Optional[str] = None
    political_risk: Optional[str] = None
    events: List[MacroEventResponse] = Field(default_factory=list)


class HealthCheckResponse(BaseModel):
    """Response model for health check endpoint.

    Provides system health status including component-level checks.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00Z",
                "uptime_seconds": 3600.5,
                "bot_last_update": "2024-01-15T10:29:55Z",
                "bot_stale": False,
                "stale_threshold_seconds": 300,
                "components": {
                    "database": "healthy",
                    "redis": "healthy",
                    "ml_models": "healthy",
                    "exchange_api": "healthy",
                },
            }
        }
    )

    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        ..., description="Overall system health status"
    )
    timestamp: datetime = Field(..., description="Time of health check")
    uptime_seconds: float = Field(..., description="Seconds since API started")
    bot_last_update: Optional[datetime] = Field(
        None, description="Last time bot state was updated"
    )
    bot_stale: bool = Field(..., description="Whether bot data is stale")
    stale_threshold_seconds: Optional[int] = Field(
        None, description="Threshold for considering data stale"
    )
    components: Dict[str, str] = Field(
        ..., description="Health status of individual components"
    )


class ShadowHealthResponse(BaseModel):
    """Response model for shadow data collection health metrics.

    Provides PAPER_LIVE progress metrics for Phase 2C promotion tracking.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "paper_live_decisions_today": 15,
                "paper_live_decisions_7d": 85,
                "paper_live_days_streak": 5,
                "paper_live_weeks_counted": 3,
                "heartbeat_recent": 1,
                "latest_report_timestamp": "2026-01-29T10:30:00",
                "gate_1_progress": {"required": 12, "current": 3, "met": False},
                "overall_health": "HEALTHY",
            }
        }
    )

    paper_live_decisions_today: int = Field(
        ..., description="Number of PAPER_LIVE decisions collected today"
    )
    paper_live_decisions_7d: int = Field(
        ..., description="Number of PAPER_LIVE decisions in the last 7 days"
    )
    paper_live_days_streak: int = Field(
        ..., description="Consecutive days with PAPER_LIVE data collection"
    )
    paper_live_weeks_counted: int = Field(
        ..., description="Number of weeks with PAPER_LIVE data (Gate 1 progress)"
    )
    heartbeat_recent: int = Field(
        ..., description="1 if heartbeat is recent (< 2 hours), 0 otherwise"
    )
    latest_report_timestamp: Optional[str] = Field(
        None, description="Timestamp of the latest daily shadow health report"
    )
    gate_1_progress: Dict[str, Any] = Field(
        ..., description="Gate 1 (weeks collected) progress details"
    )
    overall_health: str = Field(
        ..., description="Overall shadow collection health status"
    )


class LiveHealthResponse(BaseModel):
    """Response model for live trading guardrails health status.

    Provides real-time status of micro-live rollout guardrails.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "live_mode_enabled": False,
                "kill_switch_active": False,
                "kill_switch_reason": None,
                "daily_trades": {"count": 1, "limit": 3, "remaining": 2},
                "capital": {"deployed_today": 150.0, "max_pct": 0.01},
                "position": {"max_pct": 0.02},
                "leverage": {"max": 1.0},
                "symbol_allowlist": ["ETH/USDT"],
                "last_trade": "2026-01-29T10:30:00Z",
                "overall_status": "SAFE",
            }
        }
    )

    live_mode_enabled: bool = Field(
        ..., description="Whether live trading mode is enabled"
    )
    kill_switch_active: bool = Field(
        ..., description="Whether the kill switch is currently active"
    )
    kill_switch_reason: Optional[str] = Field(
        None, description="Reason for kill switch activation if active"
    )
    daily_trades: Dict[str, Any] = Field(
        ..., description="Daily trade count and limits"
    )
    capital: Dict[str, Any] = Field(
        ..., description="Capital deployment status"
    )
    position: Dict[str, Any] = Field(
        ..., description="Position size limits"
    )
    leverage: Dict[str, Any] = Field(
        ..., description="Leverage limits"
    )
    symbol_allowlist: List[str] = Field(
        ..., description="Symbols allowed for live trading"
    )
    last_trade: Optional[str] = Field(
        None, description="Timestamp of last live trade"
    )
    overall_status: str = Field(
        ..., description="Overall status: SAFE, LIVE_ACTIVE, or BLOCKED"
    )


class ReadinessResponse(BaseModel):
    """Response model for system readiness check.

    Aggregates shadow, live guardrails, turnover, and capital preservation
    status into a unified readiness assessment.

    Includes live_rollout_* fields for April 1st deployment readiness.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "overall_readiness": "CONDITIONAL",
                "reasons": ["Shadow heartbeat not recent"],
                "recommended_next_actions": ["Verify shadow collector is running"],
                "live_rollout_readiness": "CONDITIONAL",
                "live_rollout_reasons": [
                    "PAPER_LIVE streak (10 days) below minimum (14 days)",
                    "PAPER_LIVE weeks counted (1) below minimum (2)",
                ],
                "live_rollout_next_actions": [
                    "Continue PAPER_LIVE trading for 4 more consecutive days",
                    "Accumulate 1 more week of PAPER_LIVE data",
                ],
                "components": {
                    "shadow": {
                        "paper_live_decisions_today": 5,
                        "paper_live_decisions_7d": 45,
                        "paper_live_days_streak": 10,
                        "paper_live_weeks_counted": 1,
                        "heartbeat_recent": 1,
                        "overall_health": "WARNING",
                    },
                    "live": {
                        "live_mode_enabled": False,
                        "kill_switch_active": False,
                        "daily_trades_remaining": 3,
                        "overall_status": "SAFE",
                    },
                    "turnover": {
                        "enabled": True,
                        "symbols_configured": 2,
                        "total_blocks_today": 5,
                        "total_decisions_today": 20,
                        "block_rate_pct": 25.0,
                    },
                    "capital_preservation": {
                        "current_level": "NORMAL",
                        "last_escalation": None,
                    },
                    "daily_reports": {
                        "latest_report_age_hours": 2.5,
                        "reports_last_24h": True,
                        "critical_alerts_14d": 0,
                    },
                    "execution_realism": {
                        "drift_detected": False,
                        "slippage_7d_avg": 0.0015,
                        "slippage_prior_7d_avg": 0.0012,
                    },
                },
            }
        }
    )

    overall_readiness: Literal["GO", "CONDITIONAL", "NO_GO"] = Field(
        ..., description="Overall readiness status"
    )
    reasons: List[str] = Field(
        ..., description="Reasons for the readiness determination"
    )
    recommended_next_actions: List[str] = Field(
        ..., description="Recommended actions based on current status"
    )
    live_rollout_readiness: Literal["GO", "CONDITIONAL", "NO_GO"] = Field(
        ..., description="Live rollout readiness for April 1st deployment"
    )
    live_rollout_reasons: List[str] = Field(
        ..., description="Reasons for the live rollout readiness determination"
    )
    live_rollout_next_actions: List[str] = Field(
        ..., description="Recommended actions for live rollout readiness"
    )
    components: Dict[str, Any] = Field(
        ..., description="Detailed status of each component"
    )
