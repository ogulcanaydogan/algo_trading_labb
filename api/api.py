from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from bot.ai import FeatureSnapshot, PredictionSnapshot, QuestionAnsweringEngine
from bot.macro import MacroInsight
from bot.state import StateStore, create_state_store
from bot.strategy import StrategyConfig
from bot.market_data import sanitize_symbol_for_fs
from bot.config_loader import load_overrides, merge_config

from .schemas import (
    AIAnswerResponse,
    AIQuestionRequest,
    AIPredictionResponse,
    BotStateResponse,
    EquityPointResponse,
    MacroEventResponse,
    MacroInsightResponse,
    SignalResponse,
    StrategyOverviewResponse,
)

STATE_DIR = Path(os.getenv("DATA_DIR", "./data"))
DASHBOARD_TEMPLATE = Path(__file__).with_name("dashboard.html")

app = FastAPI(
    title="Algo Trading Lab Status API",
    version="0.1.0",
    description="Status endpoints for the Algo Trading Lab bot.",
)

_state_store: Optional[StateStore] = None


def get_store() -> StateStore:
    global _state_store
    if _state_store is None:
        _state_store = create_state_store(STATE_DIR)
    return _state_store


@app.get("/status", response_model=BotStateResponse)
def read_status(store: StateStore = Depends(get_store)) -> BotStateResponse:
    store.load()
    payload = store.get_state_dict()
    if not payload:
        raise HTTPException(status_code=404, detail="State not found.")
    return BotStateResponse(**payload)


@app.get("/signals", response_model=List[SignalResponse])
def read_signals(
    limit: int = Query(default=50, ge=1, le=500),
    store: StateStore = Depends(get_store),
) -> List[SignalResponse]:
    store.load()
    signals = store.get_signals(limit)
    return [SignalResponse(**item) for item in signals]


@app.get("/equity", response_model=List[EquityPointResponse])
def read_equity(store: StateStore = Depends(get_store)) -> List[EquityPointResponse]:
    store.load()
    curve = store.get_equity_curve()
    return [EquityPointResponse(**point) for point in curve]


@app.get("/strategy", response_model=StrategyOverviewResponse)
def read_strategy_overview(symbol: Optional[str] = Query(default=None)) -> StrategyOverviewResponse:
    """Return the active strategy settings.

    If a symbol is provided, the API will try to load the per-asset overrides from
    DATA_DIR/portfolio/<safe_symbol>/strategy_config.json and fall back to the
    global DATA_DIR/strategy_config.json when not found.
    """
    config = StrategyConfig.from_env()

    # Try per-asset strategy first when query symbol is provided
    if symbol:
        safe = sanitize_symbol_for_fs(symbol)
        per_asset_path = STATE_DIR / "portfolio" / safe / "strategy_config.json"
        overrides = load_overrides(per_asset_path)
        if overrides:
            merged = merge_config(StrategyConfig(symbol=symbol, timeframe=config.timeframe), overrides)
            config = merged
        else:
            # fall back to global overrides
            overrides = load_overrides(STATE_DIR / "strategy_config.json")
            if overrides:
                config = merge_config(config, overrides)
    else:
        overrides = load_overrides(STATE_DIR / "strategy_config.json")
        if overrides:
            config = merge_config(config, overrides)
    decision_rules = [
        "Go LONG when the fast EMA crosses above the slow EMA and RSI stays below the overbought threshold.",
        "Go SHORT when the fast EMA crosses below the slow EMA and RSI stays above the oversold threshold.",
        "Fallback LONG when RSI dips below the oversold threshold even without a crossover.",
        "Fallback SHORT when RSI rises above the overbought threshold even without a crossover.",
    ]
    risk_notes = [
        "Risk per trade is capped by the configured percentage of the current balance.",
        "Stop-loss and take-profit levels are derived from the last entry price and the configured percentages.",
        "Position sizing scales inversely with stop distance to maintain consistent risk exposure.",
    ]

    return StrategyOverviewResponse(
        symbol=config.symbol,
        timeframe=config.timeframe,
        ema_fast=config.ema_fast,
        ema_slow=config.ema_slow,
        rsi_period=config.rsi_period,
        rsi_overbought=config.rsi_overbought,
        rsi_oversold=config.rsi_oversold,
        risk_per_trade_pct=config.risk_per_trade_pct,
        stop_loss_pct=config.stop_loss_pct,
        take_profit_pct=config.take_profit_pct,
        decision_rules=decision_rules,
        risk_management_notes=risk_notes,
    )


@app.get("/portfolio/strategies", response_model=List[StrategyOverviewResponse])
def list_portfolio_strategies() -> List[StrategyOverviewResponse]:
    """Enumerate per-asset strategies found under DATA_DIR/portfolio/*/strategy_config.json."""
    results: List[StrategyOverviewResponse] = []
    portfolio_dir = STATE_DIR / "portfolio"
    if not portfolio_dir.exists():
        return results
    for item in sorted(portfolio_dir.iterdir()):
        strategy_file = item / "strategy_config.json"
        if not strategy_file.exists():
            continue
        overrides = load_overrides(strategy_file)
        if not overrides:
            continue
        symbol = str(overrides.get("symbol") or item.name)
        timeframe = str(overrides.get("timeframe") or StrategyConfig.from_env().timeframe)
        cfg = merge_config(StrategyConfig(symbol=symbol, timeframe=timeframe), overrides)
        results.append(
            StrategyOverviewResponse(
                symbol=cfg.symbol,
                timeframe=cfg.timeframe,
                ema_fast=cfg.ema_fast,
                ema_slow=cfg.ema_slow,
                rsi_period=cfg.rsi_period,
                rsi_overbought=cfg.rsi_overbought,
                rsi_oversold=cfg.rsi_oversold,
                risk_per_trade_pct=cfg.risk_per_trade_pct,
                stop_loss_pct=cfg.stop_loss_pct,
                take_profit_pct=cfg.take_profit_pct,
                decision_rules=[
                    "Go LONG when the fast EMA crosses above the slow EMA and RSI stays below the overbought threshold.",
                    "Go SHORT when the fast EMA crosses below the slow EMA and RSI stays above the oversold threshold.",
                    "Fallback LONG when RSI dips below the oversold threshold even without a crossover.",
                    "Fallback SHORT when RSI rises above the overbought threshold even without a crossover.",
                ],
                risk_management_notes=[
                    "Risk per trade is capped by the configured percentage of the current balance.",
                    "Stop-loss and take-profit levels are derived from the last entry price and the configured percentages.",
                    "Position sizing scales inversely with stop distance to maintain consistent risk exposure.",
                ],
            )
        )
    return results


@app.get("/ai/prediction", response_model=AIPredictionResponse)
def read_ai_prediction(store: StateStore = Depends(get_store)) -> AIPredictionResponse:
    store.load()
    payload = store.get_state_dict()
    if not payload:
        raise HTTPException(status_code=404, detail="State not found.")

    features = payload.get("ai_features") or None

    return AIPredictionResponse(
        timestamp=payload["timestamp"],
        symbol=payload.get("symbol", ""),
        recommended_action=payload.get("ai_action"),
        confidence=payload.get("ai_confidence"),
        probability_long=payload.get("ai_probability_long"),
        probability_short=payload.get("ai_probability_short"),
        probability_flat=payload.get("ai_probability_flat"),
        expected_move_pct=payload.get("ai_expected_move_pct"),
        summary=payload.get("ai_summary"),
        features=features,
        macro_bias=payload.get("macro_bias"),
        macro_confidence=payload.get("macro_confidence"),
        macro_summary=payload.get("macro_summary"),
        macro_drivers=payload.get("macro_drivers") or [],
        macro_interest_rate_outlook=payload.get("macro_interest_rate_outlook"),
        macro_political_risk=payload.get("macro_political_risk"),
    )


@app.post("/ai/question", response_model=AIAnswerResponse)
def ask_ai(
    request: AIQuestionRequest,
    store: StateStore = Depends(get_store),
) -> AIAnswerResponse:
    store.load()
    config = StrategyConfig.from_env()
    engine = QuestionAnsweringEngine(config)

    state = store.state
    ai_features = getattr(state, "ai_features", {}) or {}
    ai_snapshot: Optional[PredictionSnapshot]
    if state.ai_action:
        ai_snapshot = PredictionSnapshot(
            recommended_action=state.ai_action,
            confidence=state.ai_confidence or 0.0,
            probability_long=state.ai_probability_long or 0.0,
            probability_short=state.ai_probability_short or 0.0,
            probability_flat=state.ai_probability_flat or 0.0,
            expected_move_pct=state.ai_expected_move_pct or 0.0,
            summary=state.ai_summary or "",
            features=FeatureSnapshot(
                ema_gap_pct=ai_features.get("ema_gap_pct", 0.0),
                momentum_pct=ai_features.get("momentum_pct", 0.0),
                rsi_distance_from_mid=ai_features.get("rsi_distance_from_mid", 0.0),
                volatility_pct=ai_features.get("volatility_pct", 0.0),
            ),
            macro_bias=state.macro_bias or 0.0,
            macro_confidence=state.macro_confidence or 0.0,
            macro_summary=state.macro_summary or "",
            macro_drivers=list(state.macro_drivers or []),
            macro_interest_rate_outlook=state.macro_interest_rate_outlook,
            macro_political_risk=state.macro_political_risk,
        )
    else:
        ai_snapshot = None

    macro_insight: Optional[MacroInsight] = None
    if state.macro_summary or state.macro_drivers or state.macro_bias is not None:
        macro_insight = MacroInsight(
            symbol=state.symbol,
            bias_score=state.macro_bias or 0.0,
            confidence=state.macro_confidence or 0.0,
            summary=state.macro_summary or "",
            drivers=list(state.macro_drivers or []),
            interest_rate_outlook=state.macro_interest_rate_outlook,
            political_risk=state.macro_political_risk,
            events=list(state.macro_events or []),
        )

    answer = engine.answer(
        request.question,
        state=state,
        ai_snapshot=ai_snapshot,
        macro_insight=macro_insight,
    )
    return AIAnswerResponse(question=request.question, answer=answer)


@app.get("/macro/insights", response_model=MacroInsightResponse)
def read_macro_insights(store: StateStore = Depends(get_store)) -> MacroInsightResponse:
    store.load()
    payload = store.get_state_dict()
    if not payload:
        raise HTTPException(status_code=404, detail="State not found.")

    events_payload: List[MacroEventResponse] = []
    for item in payload.get("macro_events") or []:
        if isinstance(item, dict) and item.get("title"):
            events_payload.append(MacroEventResponse(**item))

    return MacroInsightResponse(
        timestamp=payload["timestamp"],
        symbol=payload.get("symbol", ""),
        bias_score=payload.get("macro_bias"),
        confidence=payload.get("macro_confidence"),
        summary=payload.get("macro_summary"),
        drivers=payload.get("macro_drivers") or [],
        interest_rate_outlook=payload.get("macro_interest_rate_outlook"),
        political_risk=payload.get("macro_political_risk"),
        events=events_payload,
    )


def load_dashboard_template() -> str:
    try:
        return DASHBOARD_TEMPLATE.read_text(encoding="utf-8")
    except FileNotFoundError:
        return """<!DOCTYPE html><html><body><h1>Dashboard template missing.</h1></body></html>"""


@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
@app.get("/dashboard/preview", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    """Serve a lightweight HTML dashboard for quick monitoring."""
    return HTMLResponse(content=load_dashboard_template())

