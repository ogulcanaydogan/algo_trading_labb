from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

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
    confidence: Optional[float] = None
    rsi: Optional[float] = None
    ema_fast: Optional[float] = None
    ema_slow: Optional[float] = None
    risk_per_trade_pct: float


class SignalResponse(BaseModel):
    timestamp: datetime
    symbol: str
    decision: Literal["LONG", "SHORT", "FLAT"]
    confidence: float
    reason: str


class EquityPointResponse(BaseModel):
    timestamp: datetime
    value: float

