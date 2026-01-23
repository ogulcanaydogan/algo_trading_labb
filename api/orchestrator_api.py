"""
Orchestrator REST API Endpoints

Phase 9: REST API endpoints for the unified trading orchestrator,
risk controls, strategy management, news features, and walk-forward validation.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Body
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2", tags=["Orchestrator"])


# ==============================================================================
# Pydantic Models
# ==============================================================================


class SystemStatusResponse(BaseModel):
    """System status response."""

    state: str = Field(..., description="Current system state")
    mode: str = Field(..., description="Trading mode (live, paper, shadow)")
    uptime_seconds: float = Field(..., description="System uptime in seconds")
    current_regime: str = Field(..., description="Current market regime")
    active_strategies: List[str] = Field(..., description="List of active strategies")
    disabled_strategies: List[str] = Field(..., description="List of disabled strategies")
    metrics: Dict[str, int] = Field(..., description="System metrics")
    components: Dict[str, bool] = Field(..., description="Component availability")


class HealthResponse(BaseModel):
    """Health check response."""

    state: str
    uptime_seconds: float
    risk_guardian_active: bool
    execution_engine_connected: bool
    data_feed_active: bool
    websocket_clients: int
    decisions_today: int
    trades_today: int
    errors_today: int
    current_drawdown: float
    daily_pnl: float
    open_positions: int
    warnings: List[str]


class KillSwitchRequest(BaseModel):
    """Kill switch request."""

    action: str = Field(..., description="Action: 'stop', 'resume', or 'close_all'")
    reason: str = Field("Manual", description="Reason for activation")


class KillSwitchResponse(BaseModel):
    """Kill switch response."""

    success: bool
    action: str
    message: str
    timestamp: str


class RiskLimitUpdateRequest(BaseModel):
    """Risk limit update request."""

    limit_type: str = Field(
        ..., description="Type of limit: max_drawdown, max_position, daily_loss, max_trades"
    )
    value: float = Field(..., description="New limit value")


class RiskLimitResponse(BaseModel):
    """Risk limit response."""

    success: bool
    limit_type: str
    old_value: Optional[float]
    new_value: float
    message: str


class RiskLimitsResponse(BaseModel):
    """Current risk limits response."""

    max_drawdown: float = Field(..., description="Maximum allowed drawdown (e.g., 0.10 for 10%)")
    max_position_size: float = Field(..., description="Maximum position size as % of portfolio")
    daily_loss_limit: float = Field(..., description="Maximum daily loss in currency")
    max_trades_per_day: int = Field(..., description="Maximum trades per day")
    max_leverage: float = Field(..., description="Maximum leverage allowed")
    max_correlated_exposure: float = Field(..., description="Maximum exposure to correlated assets")


class StrategyInfo(BaseModel):
    """Strategy information."""

    name: str
    enabled: bool
    suitable_regimes: List[str]
    description: Optional[str] = None
    parameters: Dict[str, Any] = {}
    performance: Optional[Dict[str, float]] = None


class StrategiesResponse(BaseModel):
    """List of strategies response."""

    strategies: List[StrategyInfo]
    total_count: int
    active_count: int


class StrategyToggleRequest(BaseModel):
    """Request to enable/disable a strategy."""

    enabled: bool = Field(..., description="Whether to enable or disable the strategy")


class StrategyToggleResponse(BaseModel):
    """Response for strategy toggle."""

    strategy_name: str
    enabled: bool
    message: str


class NewsFeatureResponse(BaseModel):
    """News features response."""

    timestamp: str
    overall_sentiment: float
    sentiment_momentum: float
    sentiment_dispersion: float
    high_impact_events: int
    medium_impact_events: int
    low_impact_events: int
    avg_surprise: float
    central_bank_sentiment: float
    economic_data_sentiment: float
    corporate_sentiment: float
    geopolitical_sentiment: float
    decay_weighted_sentiment: float
    recent_news_intensity: float
    hours_to_next_high_impact: Optional[float]
    next_event_type: Optional[str]
    news_velocity: float
    headline_risk_score: float


class AddNewsEventRequest(BaseModel):
    """Request to add a news event."""

    title: str = Field(..., description="Event/news title")
    timestamp: Optional[str] = Field(None, description="ISO timestamp")
    event_type: str = Field("other", description="Event type")
    actual: Optional[float] = Field(None, description="Actual value (for economic events)")
    expected: Optional[float] = Field(None, description="Expected value")
    previous: Optional[float] = Field(None, description="Previous value")
    source: str = Field("manual", description="Data source")


class AddNewsEventResponse(BaseModel):
    """Response for adding news event."""

    success: bool
    event_id: str
    message: str


class WalkForwardResultResponse(BaseModel):
    """Walk-forward validation result."""

    strategy_name: str
    validation_date: str
    result: str  # PASSED, FAILED, MARGINAL
    metrics: Dict[str, float]
    windows_passed: int
    windows_total: int
    stress_tests_passed: int
    stress_tests_total: int
    recommendation: str


class AllocationResponse(BaseModel):
    """Meta-allocator allocation response."""

    strategy_name: str
    weight: float
    capital_allocated: float
    reason: str


class MetaAllocatorStatusResponse(BaseModel):
    """Meta-allocator status response."""

    method: str
    total_capital: float
    allocations: List[AllocationResponse]
    last_rebalance: Optional[str]
    next_rebalance: Optional[str]


class DecisionResponse(BaseModel):
    """Trading decision response."""

    decision_id: str
    timestamp: str
    symbol: str
    action: str
    quantity: float
    price: float
    strategy_name: str
    risk_approved: bool
    risk_reason: str
    executed: bool
    execution_price: Optional[float]


class RegimeUpdateRequest(BaseModel):
    """Request to update market regime."""

    regime: str = Field(..., description="New regime name")
    confidence: float = Field(0.8, description="Confidence level 0-1")


class RegimeResponse(BaseModel):
    """Regime status response."""

    current_regime: str
    confidence: float
    last_change: Optional[str]
    regime_duration_hours: Optional[float]


# ==============================================================================
# Global orchestrator instance (injected at startup)
# ==============================================================================

_orchestrator = None
_news_extractor = None
_walk_forward_validator = None
_meta_allocator = None
_risk_guardian = None


def get_orchestrator():
    """Get the orchestrator instance."""
    if _orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    return _orchestrator


def get_news_extractor():
    """Get the news extractor instance."""
    return _news_extractor


def get_walk_forward_validator():
    """Get the walk-forward validator instance."""
    return _walk_forward_validator


def get_meta_allocator():
    """Get the meta-allocator instance."""
    return _meta_allocator


def get_risk_guardian():
    """Get the risk guardian instance."""
    return _risk_guardian


def init_orchestrator_api(
    orchestrator=None,
    news_extractor=None,
    walk_forward_validator=None,
    meta_allocator=None,
    risk_guardian=None,
):
    """Initialize the orchestrator API with component instances."""
    global _orchestrator, _news_extractor, _walk_forward_validator, _meta_allocator, _risk_guardian
    _orchestrator = orchestrator
    _news_extractor = news_extractor
    _walk_forward_validator = walk_forward_validator
    _meta_allocator = meta_allocator
    _risk_guardian = risk_guardian
    logger.info("Orchestrator API initialized with components")


# ==============================================================================
# System Status Endpoints
# ==============================================================================


@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status():
    """
    Get comprehensive system status.

    Returns current state, mode, active strategies, and component availability.
    """
    orchestrator = get_orchestrator()
    status = orchestrator.get_status()
    return SystemStatusResponse(**status)


@router.get("/health", response_model=HealthResponse)
async def get_system_health():
    """
    Get system health metrics.

    Returns component health, trading metrics, and risk indicators.
    """
    orchestrator = get_orchestrator()
    health = orchestrator.get_health()

    return HealthResponse(
        state=health.state.value,
        uptime_seconds=health.uptime_seconds,
        risk_guardian_active=health.risk_guardian_active,
        execution_engine_connected=health.execution_engine_connected,
        data_feed_active=health.data_feed_active,
        websocket_clients=health.websocket_clients,
        decisions_today=health.decisions_today,
        trades_today=health.trades_today,
        errors_today=health.errors_today,
        current_drawdown=health.current_drawdown,
        daily_pnl=health.daily_pnl,
        open_positions=health.open_positions,
        warnings=health.warnings,
    )


# ==============================================================================
# Kill Switch & Control Endpoints
# ==============================================================================


@router.post("/kill-switch", response_model=KillSwitchResponse)
async def activate_kill_switch(request: KillSwitchRequest):
    """
    Activate the emergency kill switch.

    Actions:
    - 'stop': Stop all trading immediately
    - 'resume': Resume trading
    - 'close_all': Stop trading and close all positions
    """
    orchestrator = get_orchestrator()

    if request.action not in ["stop", "resume", "close_all"]:
        raise HTTPException(status_code=400, detail="Invalid action")

    await orchestrator.kill_switch(action=request.action, reason=request.reason)

    return KillSwitchResponse(
        success=True,
        action=request.action,
        message=f"Kill switch {request.action} executed: {request.reason}",
        timestamp=datetime.now().isoformat(),
    )


@router.post("/pause")
async def pause_trading(reason: str = Query("Manual pause")):
    """Pause trading without closing positions."""
    orchestrator = get_orchestrator()
    await orchestrator.pause(reason=reason)
    return {"success": True, "message": f"Trading paused: {reason}"}


@router.post("/resume")
async def resume_trading():
    """Resume trading after pause."""
    orchestrator = get_orchestrator()
    await orchestrator.resume()
    return {"success": True, "message": "Trading resumed"}


# ==============================================================================
# Risk Control Endpoints
# ==============================================================================


@router.get("/risk/limits", response_model=RiskLimitsResponse)
async def get_risk_limits():
    """
    Get current risk limits.

    Returns all configurable risk parameters.
    """
    risk_guardian = get_risk_guardian()

    if risk_guardian and hasattr(risk_guardian, "get_limits"):
        limits = risk_guardian.get_limits()
        return RiskLimitsResponse(**limits)

    # Default limits if no guardian
    return RiskLimitsResponse(
        max_drawdown=0.10,
        max_position_size=0.25,
        daily_loss_limit=5000.0,
        max_trades_per_day=100,
        max_leverage=1.0,
        max_correlated_exposure=0.40,
    )


@router.put("/risk/limits", response_model=RiskLimitResponse)
async def update_risk_limit(request: RiskLimitUpdateRequest):
    """
    Update a risk limit.

    Limit types:
    - max_drawdown: Maximum allowed drawdown (e.g., 0.10 for 10%)
    - max_position: Maximum single position size (e.g., 0.25 for 25%)
    - daily_loss: Maximum daily loss in currency
    - max_trades: Maximum trades per day
    """
    orchestrator = get_orchestrator()

    valid_limits = ["max_drawdown", "max_position", "daily_loss", "max_trades"]
    if request.limit_type not in valid_limits:
        raise HTTPException(
            status_code=400, detail=f"Invalid limit type. Valid types: {valid_limits}"
        )

    success = await orchestrator.adjust_risk_limit(request.limit_type, request.value)

    return RiskLimitResponse(
        success=success,
        limit_type=request.limit_type,
        old_value=None,  # Would need to track old value
        new_value=request.value,
        message="Limit updated successfully" if success else "Failed to update limit",
    )


@router.get("/risk/metrics")
async def get_risk_metrics():
    """
    Get current risk metrics.

    Returns real-time risk indicators like drawdown, exposure, VaR.
    """
    risk_guardian = get_risk_guardian()

    if risk_guardian and hasattr(risk_guardian, "get_current_metrics"):
        return risk_guardian.get_current_metrics()

    return {
        "current_drawdown": 0.0,
        "max_drawdown_today": 0.0,
        "total_exposure": 0.0,
        "largest_position": 0.0,
        "correlated_exposure": 0.0,
        "daily_trades": 0,
        "daily_pnl": 0.0,
        "var_95": 0.0,
        "position_count": 0,
    }


# ==============================================================================
# Strategy Management Endpoints
# ==============================================================================


@router.get("/strategies", response_model=StrategiesResponse)
async def list_strategies():
    """
    List all available strategies.

    Returns strategy details, status, and performance metrics.
    """
    orchestrator = get_orchestrator()

    strategies = []
    for name, enabled in orchestrator.active_strategies.items():
        strategy_info = StrategyInfo(
            name=name,
            enabled=enabled,
            suitable_regimes=["trending_bullish", "trending_bearish"],  # Would come from strategy
            description=f"Strategy: {name}",
            parameters={},
            performance=None,
        )
        strategies.append(strategy_info)

    return StrategiesResponse(
        strategies=strategies,
        total_count=len(strategies),
        active_count=sum(1 for s in strategies if s.enabled),
    )


@router.get("/strategies/{strategy_name}", response_model=StrategyInfo)
async def get_strategy(strategy_name: str):
    """Get details for a specific strategy."""
    orchestrator = get_orchestrator()

    if strategy_name not in orchestrator.active_strategies:
        raise HTTPException(status_code=404, detail="Strategy not found")

    enabled = orchestrator.active_strategies[strategy_name]

    return StrategyInfo(
        name=strategy_name,
        enabled=enabled,
        suitable_regimes=["trending_bullish", "trending_bearish"],
        description=f"Strategy: {strategy_name}",
        parameters={},
        performance=None,
    )


@router.put("/strategies/{strategy_name}", response_model=StrategyToggleResponse)
async def toggle_strategy(strategy_name: str, request: StrategyToggleRequest):
    """
    Enable or disable a strategy.

    Use with caution - disabling active strategies affects live trading.
    """
    orchestrator = get_orchestrator()

    if strategy_name not in orchestrator.active_strategies:
        raise HTTPException(status_code=404, detail="Strategy not found")

    if request.enabled:
        success = orchestrator.enable_strategy(strategy_name)
    else:
        success = orchestrator.disable_strategy(strategy_name)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to toggle strategy")

    return StrategyToggleResponse(
        strategy_name=strategy_name,
        enabled=request.enabled,
        message=f"Strategy {'enabled' if request.enabled else 'disabled'}",
    )


# ==============================================================================
# News Features Endpoints
# ==============================================================================


@router.get("/news/features", response_model=NewsFeatureResponse)
async def get_news_features(regime: Optional[str] = Query(None, description="Filter by regime")):
    """
    Get current news/sentiment features.

    Returns aggregated features from news and economic events for ML models.
    """
    news_extractor = get_news_extractor()

    if not news_extractor:
        raise HTTPException(status_code=503, detail="News extractor not initialized")

    features = news_extractor.extract_features(regime=regime)

    return NewsFeatureResponse(
        timestamp=features.timestamp.isoformat(),
        overall_sentiment=features.overall_sentiment,
        sentiment_momentum=features.sentiment_momentum,
        sentiment_dispersion=features.sentiment_dispersion,
        high_impact_events=features.high_impact_events,
        medium_impact_events=features.medium_impact_events,
        low_impact_events=features.low_impact_events,
        avg_surprise=features.avg_surprise,
        central_bank_sentiment=features.central_bank_sentiment,
        economic_data_sentiment=features.economic_data_sentiment,
        corporate_sentiment=features.corporate_sentiment,
        geopolitical_sentiment=features.geopolitical_sentiment,
        decay_weighted_sentiment=features.decay_weighted_sentiment,
        recent_news_intensity=features.recent_news_intensity,
        hours_to_next_high_impact=features.hours_to_next_high_impact,
        next_event_type=features.next_event_type,
        news_velocity=features.news_velocity,
        headline_risk_score=features.headline_risk_score,
    )


@router.get("/news/features/vector")
async def get_news_feature_vector():
    """
    Get news features as a numeric vector for ML models.

    Returns feature names and values suitable for model input.
    """
    news_extractor = get_news_extractor()

    if not news_extractor:
        raise HTTPException(status_code=503, detail="News extractor not initialized")

    features = news_extractor.extract_features()

    return {
        "feature_names": features.feature_names(),
        "feature_values": features.to_vector(),
        "timestamp": features.timestamp.isoformat(),
    }


@router.post("/news/events", response_model=AddNewsEventResponse)
async def add_news_event(request: AddNewsEventRequest):
    """
    Add a news or economic event manually.

    Useful for ingesting data from external sources.
    """
    news_extractor = get_news_extractor()

    if not news_extractor:
        raise HTTPException(status_code=503, detail="News extractor not initialized")

    try:
        from bot.news_feature_extractor import EconomicCalendarParser, NewsParser

        timestamp = (
            datetime.fromisoformat(request.timestamp) if request.timestamp else datetime.now()
        )

        if request.actual is not None or request.expected is not None:
            # Economic event
            parser = EconomicCalendarParser()
            event = parser.parse_event(
                title=request.title,
                timestamp=timestamp,
                actual=request.actual,
                expected=request.expected,
                previous=request.previous,
                source=request.source,
            )
            news_extractor.add_economic_event(event)
            event_id = event.event_id
        else:
            # News item
            parser = NewsParser()
            news = parser.parse_headline(
                headline=request.title, timestamp=timestamp, source=request.source
            )
            news_extractor.add_news_item(news)
            event_id = news.news_id

        return AddNewsEventResponse(
            success=True, event_id=event_id, message="Event added successfully"
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/news/upcoming")
async def get_upcoming_events():
    """Get upcoming scheduled economic events."""
    news_extractor = get_news_extractor()

    if not news_extractor:
        raise HTTPException(status_code=503, detail="News extractor not initialized")

    events = []
    for event in news_extractor.upcoming_events[:20]:
        events.append(
            {
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "timestamp": event.timestamp.isoformat(),
                "title": event.title,
                "impact": event.impact.value,
                "expected": event.expected,
                "previous": event.previous,
            }
        )

    return {"upcoming_events": events}


# ==============================================================================
# Walk-Forward Validation Endpoints
# ==============================================================================


@router.get("/validation/results")
async def get_validation_results(
    strategy_name: Optional[str] = Query(None, description="Filter by strategy"),
):
    """
    Get walk-forward validation results.

    Returns validation status and metrics for strategies.
    """
    validator = get_walk_forward_validator()

    if not validator:
        raise HTTPException(status_code=503, detail="Walk-forward validator not initialized")

    # Get cached results
    results = []
    if hasattr(validator, "get_cached_results"):
        cached = validator.get_cached_results()
        for name, report in cached.items():
            if strategy_name and name != strategy_name:
                continue
            results.append(
                {
                    "strategy_name": name,
                    "validation_date": report.validation_date.isoformat()
                    if hasattr(report, "validation_date")
                    else None,
                    "result": report.result.value if hasattr(report, "result") else "unknown",
                    "metrics": report.metrics if hasattr(report, "metrics") else {},
                    "windows_passed": getattr(report, "windows_passed", 0),
                    "windows_total": getattr(report, "windows_total", 0),
                    "stress_tests_passed": getattr(report, "stress_tests_passed", 0),
                    "stress_tests_total": getattr(report, "stress_tests_total", 0),
                    "recommendation": getattr(report, "recommendation", ""),
                }
            )

    return {"validation_results": results}


@router.post("/validation/run/{strategy_name}")
async def run_validation(
    strategy_name: str, force: bool = Query(False, description="Force re-validation even if cached")
):
    """
    Run walk-forward validation for a strategy.

    This is a long-running operation. Consider running asynchronously.
    """
    validator = get_walk_forward_validator()
    orchestrator = get_orchestrator()

    if not validator:
        raise HTTPException(status_code=503, detail="Walk-forward validator not initialized")

    if strategy_name not in orchestrator.active_strategies:
        raise HTTPException(status_code=404, detail="Strategy not found")

    # Would trigger validation asynchronously in production
    return {
        "message": f"Validation requested for {strategy_name}",
        "status": "queued",
        "force": force,
    }


# ==============================================================================
# Meta-Allocator Endpoints
# ==============================================================================


@router.get("/allocator/status", response_model=MetaAllocatorStatusResponse)
async def get_allocator_status():
    """
    Get meta-allocator status and current allocations.

    Shows how capital is distributed across strategies.
    """
    allocator = get_meta_allocator()

    if not allocator:
        raise HTTPException(status_code=503, detail="Meta-allocator not initialized")

    # Get current allocations
    allocations = []
    if hasattr(allocator, "current_allocations"):
        for name, weight in allocator.current_allocations.items():
            allocations.append(
                AllocationResponse(
                    strategy_name=name,
                    weight=weight,
                    capital_allocated=weight * 100000,  # Example total capital
                    reason="Current allocation",
                )
            )

    return MetaAllocatorStatusResponse(
        method=getattr(allocator, "method", "equal_weight"),
        total_capital=100000,  # Would come from portfolio
        allocations=allocations,
        last_rebalance=None,
        next_rebalance=None,
    )


@router.post("/allocator/rebalance")
async def trigger_rebalance():
    """
    Trigger a portfolio rebalance.

    Recalculates optimal allocations based on current performance.
    """
    allocator = get_meta_allocator()

    if not allocator:
        raise HTTPException(status_code=503, detail="Meta-allocator not initialized")

    # Would trigger rebalance
    return {"message": "Rebalance requested", "status": "queued"}


# ==============================================================================
# Regime Endpoints
# ==============================================================================


@router.get("/regime", response_model=RegimeResponse)
async def get_current_regime():
    """Get current market regime."""
    orchestrator = get_orchestrator()

    return RegimeResponse(
        current_regime=orchestrator.current_regime,
        confidence=0.8,  # Would come from regime detector
        last_change=None,
        regime_duration_hours=None,
    )


@router.put("/regime", response_model=RegimeResponse)
async def update_regime(request: RegimeUpdateRequest):
    """
    Manually update market regime.

    Use with caution - normally detected automatically.
    """
    orchestrator = get_orchestrator()

    await orchestrator.update_regime(request.regime, request.confidence)

    return RegimeResponse(
        current_regime=request.regime,
        confidence=request.confidence,
        last_change=datetime.now().isoformat(),
        regime_duration_hours=0,
    )


# ==============================================================================
# Decisions & Trades Endpoints
# ==============================================================================


@router.get("/decisions/recent")
async def get_recent_decisions(
    limit: int = Query(50, ge=1, le=500, description="Maximum decisions to return"),
    executed_only: bool = Query(False, description="Only show executed decisions"),
):
    """
    Get recent trading decisions.

    Includes both approved and rejected decisions for audit.
    """
    orchestrator = get_orchestrator()

    decisions = orchestrator.recent_decisions[-limit:]

    if executed_only:
        decisions = [d for d in decisions if d.executed]

    return {
        "decisions": [
            {
                "decision_id": d.decision_id,
                "timestamp": d.timestamp.isoformat(),
                "symbol": d.symbol,
                "action": d.action,
                "quantity": d.quantity,
                "price": d.price,
                "strategy_name": d.strategy_name,
                "risk_approved": d.risk_approved,
                "risk_reason": d.risk_reason,
                "executed": d.executed,
                "execution_price": d.execution_price,
            }
            for d in decisions
        ],
        "total": len(decisions),
    }


@router.get("/metrics")
async def get_system_metrics():
    """Get system performance metrics."""
    orchestrator = get_orchestrator()

    return {"metrics": orchestrator.metrics, "timestamp": datetime.now().isoformat()}
