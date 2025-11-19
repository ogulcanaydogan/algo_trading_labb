from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from bot.ai import FeatureSnapshot, PredictionSnapshot, QuestionAnsweringEngine
from bot.control import load_bot_control, update_bot_control
from bot.macro import MacroInsight
from bot.state import StateStore, create_state_store, load_bot_state_from_path
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
    PortfolioBotStatusResponse,
    PortfolioControlStateResponse,
    PortfolioControlUpdateRequest,
    PortfolioPlaybookResponse,
    SignalResponse,
    StrategyOverviewResponse,
)

STATE_DIR = Path(os.getenv("DATA_DIR", "./data"))
DASHBOARD_TEMPLATE = Path(__file__).with_name("dashboard.html")
PORTFOLIO_CONFIG_PATH = Path(
    os.getenv("PORTFOLIO_CONFIG_PATH", str(STATE_DIR / "portfolio.json"))
).expanduser()

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


def _load_portfolio_config_payload() -> Dict[str, Any]:
    """Best-effort loader for the portfolio.json definition file."""

    if not PORTFOLIO_CONFIG_PATH.exists():
        return {}
    try:
        with PORTFOLIO_CONFIG_PATH.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


def _build_asset_index(config_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    asset_map: Dict[str, Dict[str, Any]] = {}
    for asset in config_payload.get("assets", []):
        symbol_value = str(asset.get("symbol") or "").strip()
        if not symbol_value:
            continue
        asset_map[symbol_value.upper()] = asset
    return asset_map


def _resolve_asset_data_dir(
    symbol_key: str,
    asset_map: Dict[str, Dict[str, Any]],
    portfolio_dir: Path,
) -> Path:
    asset = asset_map.get(symbol_key)
    if asset and asset.get("data_dir"):
        return Path(asset["data_dir"]).expanduser()
    symbol_value = asset.get("symbol") if asset else symbol_key
    safe = sanitize_symbol_for_fs(str(symbol_value))
    return portfolio_dir / safe


def _derive_asset_metadata(
    asset: Optional[Dict[str, Any]],
    *,
    default_timeframe: Optional[str],
    default_loop_interval: Optional[int],
    default_paper_mode: Optional[bool],
    default_stop_loss_pct: Optional[float],
    default_take_profit_pct: Optional[float],
) -> Dict[str, Any]:
    """Return a dict of optional metadata for dashboard consumption."""

    if not asset:
        return {}

    allocation_value = asset.get("allocation_pct")
    loop_value = asset.get("loop_interval_seconds")
    if loop_value is not None:
        try:
            loop_value = int(loop_value)
        except (TypeError, ValueError):
            loop_value = None
    elif default_loop_interval is not None:
        loop_value = default_loop_interval

    paper_mode_value = asset.get("paper_mode")
    if paper_mode_value is None:
        paper_mode_value = default_paper_mode
    if paper_mode_value is not None:
        paper_mode_value = bool(paper_mode_value)

    timeframe_value = asset.get("timeframe") or default_timeframe
    stop_loss_value = asset.get("stop_loss_pct")
    if stop_loss_value is None:
        stop_loss_value = default_stop_loss_pct
    take_profit_value = asset.get("take_profit_pct")
    if take_profit_value is None:
        take_profit_value = default_take_profit_pct

    return {
        "asset_type": asset.get("asset_type"),
        "timeframe": timeframe_value,
        "allocation_pct": float(allocation_value)
        if allocation_value is not None
        else None,
        "paper_mode": paper_mode_value,
        "loop_interval_seconds": loop_value,
        "stop_loss_pct": float(stop_loss_value)
        if stop_loss_value is not None
        else None,
        "take_profit_pct": float(take_profit_value)
        if take_profit_value is not None
        else None,
    }


def load_portfolio_states(
    symbol: Optional[str] = None,
) -> List[PortfolioBotStatusResponse]:
    """Load per-asset states from DATA_DIR/portfolio and config fallbacks."""

    portfolio_dir = STATE_DIR / "portfolio"
    symbol_filter = symbol.upper() if symbol else None

    config_payload = _load_portfolio_config_payload()
    asset_map = _build_asset_index(config_payload)
    default_risk_pct = config_payload.get("default_risk_per_trade_pct")
    default_risk_pct = float(default_risk_pct) if default_risk_pct is not None else None
    default_timeframe = config_payload.get("default_timeframe")
    default_loop_interval = config_payload.get("default_loop_interval_seconds")
    default_loop_interval = (
        int(default_loop_interval) if default_loop_interval is not None else None
    )
    default_paper_mode = config_payload.get("default_paper_mode")
    default_stop_loss_pct = config_payload.get("default_stop_loss_pct")
    default_stop_loss_pct = (
        float(default_stop_loss_pct) if default_stop_loss_pct is not None else None
    )
    default_take_profit_pct = config_payload.get("default_take_profit_pct")
    default_take_profit_pct = (
        float(default_take_profit_pct) if default_take_profit_pct is not None else None
    )
    portfolio_capital = float(config_payload.get("portfolio_capital", 0.0))

    results: Dict[str, PortfolioBotStatusResponse] = {}

    if portfolio_dir.exists():
        for item in sorted(portfolio_dir.iterdir()):
            if not item.is_dir():
                continue
            state_file = item / "state.json"
            state = load_bot_state_from_path(state_file)
            if state is None:
                continue
            symbol_value = state.symbol or ""
            symbol_key = symbol_value.upper()
            if symbol_filter and symbol_key != symbol_filter:
                continue
            try:
                relative_dir = item.relative_to(STATE_DIR)
            except ValueError:
                relative_dir = item

            metadata = _derive_asset_metadata(
                asset_map.get(symbol_key),
                default_timeframe=default_timeframe,
                default_loop_interval=default_loop_interval,
                default_paper_mode=default_paper_mode,
                default_stop_loss_pct=default_stop_loss_pct,
                default_take_profit_pct=default_take_profit_pct,
            )
            control = load_bot_control(item)
            results[symbol_key] = PortfolioBotStatusResponse(
                timestamp=state.timestamp,
                symbol=state.symbol,
                position=state.position,
                position_size=state.position_size,
                balance=state.balance,
                initial_balance=state.initial_balance or state.balance,
                entry_price=state.entry_price,
                unrealized_pnl_pct=state.unrealized_pnl_pct,
                last_signal=state.last_signal,
                confidence=state.confidence,
                risk_per_trade_pct=state.risk_per_trade_pct,
                ai_action=state.ai_action,
                ai_confidence=state.ai_confidence,
                data_directory=str(relative_dir),
                is_paused=control.paused,
                pause_reason=control.reason,
                pause_updated_at=control.updated_at,
                is_placeholder=False,
                **metadata,
            )

    for symbol_key, asset in asset_map.items():
        if symbol_filter and symbol_key != symbol_filter:
            continue
        if symbol_key in results:
            continue

        data_dir_path = _resolve_asset_data_dir(
            symbol_key,
            asset_map,
            portfolio_dir,
        )
        try:
            relative_dir = data_dir_path.relative_to(STATE_DIR)
        except ValueError:
            relative_dir = data_dir_path

        allocation_pct = asset.get("allocation_pct")
        starting_balance = asset.get("starting_balance")
        if starting_balance is None and allocation_pct is not None and portfolio_capital:
            starting_balance = (portfolio_capital * float(allocation_pct)) / 100.0
        balance_value = float(starting_balance) if starting_balance is not None else 0.0

        risk_value = asset.get("risk_per_trade_pct")
        if risk_value is None:
            risk_value = default_risk_pct
        risk_value = float(risk_value) if risk_value is not None else 0.0

        metadata = _derive_asset_metadata(
            asset,
            default_timeframe=default_timeframe,
            default_loop_interval=default_loop_interval,
            default_paper_mode=default_paper_mode,
            default_stop_loss_pct=default_stop_loss_pct,
            default_take_profit_pct=default_take_profit_pct,
        )
        control = load_bot_control(data_dir_path)

        results[symbol_key] = PortfolioBotStatusResponse(
            timestamp=datetime.now(timezone.utc),
            symbol=asset["symbol"],
            position="FLAT",
            position_size=0.0,
            balance=balance_value,
            initial_balance=balance_value,
            entry_price=None,
            unrealized_pnl_pct=0.0,
            last_signal=None,
            confidence=None,
            risk_per_trade_pct=risk_value,
            ai_action=None,
            ai_confidence=None,
            data_directory=str(relative_dir),
            is_paused=control.paused,
            pause_reason=control.reason,
            pause_updated_at=control.updated_at,
            is_placeholder=True,
            **metadata,
        )

    results_list = list(results.values())
    results_list.sort(key=lambda entry: (entry.symbol or ""))
    return results_list


@app.get("/portfolio/controls", response_model=List[PortfolioControlStateResponse])
def read_portfolio_controls(
    symbol: Optional[str] = Query(default=None),
) -> List[PortfolioControlStateResponse]:
    """Expose the current manual pause states for configured bots."""

    statuses = load_portfolio_states(symbol)
    return [
        PortfolioControlStateResponse(
            symbol=entry.symbol,
            paused=entry.is_paused,
            reason=entry.pause_reason,
            updated_at=entry.pause_updated_at,
        )
        for entry in statuses
    ]


@app.post("/portfolio/controls", response_model=PortfolioControlStateResponse)
def update_portfolio_control(
    payload: PortfolioControlUpdateRequest,
) -> PortfolioControlStateResponse:
    """Toggle manual pause/resume for a specific configured bot."""

    symbol_value = payload.symbol.strip()
    if not symbol_value:
        raise HTTPException(status_code=400, detail="Symbol is required.")
    symbol_key = symbol_value.upper()

    config_payload = _load_portfolio_config_payload()
    asset_map = _build_asset_index(config_payload)
    asset = asset_map.get(symbol_key)
    if asset is None:
        raise HTTPException(
            status_code=404,
            detail="Symbol is not part of the configured portfolio.",
        )

    portfolio_dir = STATE_DIR / "portfolio"
    data_dir = _resolve_asset_data_dir(symbol_key, asset_map, portfolio_dir)
    reason_value = payload.reason.strip() if payload.reason else None
    control = update_bot_control(
        data_dir,
        paused=payload.paused,
        reason=reason_value,
    )

    resolved_symbol = asset.get("symbol") or symbol_key
    return PortfolioControlStateResponse(
        symbol=resolved_symbol,
        paused=control.paused,
        reason=control.reason,
        updated_at=control.updated_at,
    )


@app.get("/status", response_model=BotStateResponse)
def read_status(store: StateStore = Depends(get_store)) -> BotStateResponse:
    store.load()
    payload = store.get_state_dict()
    if not payload:
        raise HTTPException(status_code=404, detail="State not found.")
    return BotStateResponse.model_validate(payload)


@app.get("/signals", response_model=List[SignalResponse])
def read_signals(
    limit: int = Query(default=50, ge=1, le=500),
    store: StateStore = Depends(get_store),
) -> List[SignalResponse]:
    store.load()
    signals = store.get_signals(limit)
    return [SignalResponse.model_validate(item) for item in signals]


@app.get("/equity", response_model=List[EquityPointResponse])
def read_equity(store: StateStore = Depends(get_store)) -> List[EquityPointResponse]:
    store.load()
    curve = store.get_equity_curve()
    return [EquityPointResponse.model_validate(point) for point in curve]


@app.get("/strategy", response_model=StrategyOverviewResponse)
def read_strategy_overview(
    symbol: Optional[str] = Query(default=None),
) -> StrategyOverviewResponse:
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
            merged = merge_config(
                StrategyConfig(symbol=symbol, timeframe=config.timeframe), overrides
            )
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
        timeframe = str(
            overrides.get("timeframe") or StrategyConfig.from_env().timeframe
        )
        cfg = merge_config(
            StrategyConfig(symbol=symbol, timeframe=timeframe), overrides
        )
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
                    (
                        "Go LONG when the fast EMA crosses above the slow EMA and "
                        "RSI stays below the overbought threshold."
                    ),
                    (
                        "Go SHORT when the fast EMA crosses below the slow EMA and "
                        "RSI stays above the oversold threshold."
                    ),
                    "Fallback LONG when RSI dips below the oversold threshold even without a crossover.",
                    "Fallback SHORT when RSI rises above the overbought threshold even without a crossover.",
                ],
                risk_management_notes=[
                    "Risk per trade is capped by the configured percentage of the current balance.",
                    (
                        "Stop-loss and take-profit levels are derived from the last entry price "
                        "and the configured percentages."
                    ),
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
            events_payload.append(MacroEventResponse.model_validate(item))

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


@app.get("/portfolio/playbook", response_model=PortfolioPlaybookResponse)
def read_portfolio_playbook(
    store: StateStore = Depends(get_store),
) -> PortfolioPlaybookResponse:
    store.load()
    payload = store.get_state_dict()
    if not payload:
        raise HTTPException(status_code=404, detail="State not found.")

    playbook = payload.get("portfolio_playbook")
    if not playbook:
        raise HTTPException(status_code=404, detail="Portfolio playbook unavailable.")

    return PortfolioPlaybookResponse.model_validate(playbook)


@app.get("/portfolio/status", response_model=List[PortfolioBotStatusResponse])
def read_portfolio_status(
    symbol: Optional[str] = Query(default=None),
) -> List[PortfolioBotStatusResponse]:
    """Expose each running asset's status for the dashboard."""

    statuses = load_portfolio_states(symbol)
    if symbol and not statuses:
        raise HTTPException(status_code=404, detail=f"No state found for symbol {symbol}.")
    return statuses


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
