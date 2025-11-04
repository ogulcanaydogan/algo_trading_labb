from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class BotStateResponse(BaseModel):
    timestamp: datetime
    symbol: str
    position: Literal["LONG", "SHORT", "FLAT"]
    entry_price: Optional[float] = Field(None, description="Last entry price for the open position.")
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
    timestamp: datetime
    symbol: str
    decision: Literal["LONG", "SHORT", "FLAT"]
    confidence: float
    reason: str
    ai_action: Optional[str] = None
    ai_confidence: Optional[float] = None
    ai_expected_move_pct: Optional[float] = None


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
