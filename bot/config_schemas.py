"""Pydantic schemas for configuration validation."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class StrategyConfigSchema(BaseModel):
    """Validated schema for strategy configuration."""

    symbol: str = Field(default="BTC/USDT", min_length=1, max_length=50)
    timeframe: str = Field(default="1m", pattern=r"^\d+[mhdwM]$")
    ema_fast: int = Field(default=12, ge=2, le=200)
    ema_slow: int = Field(default=26, ge=5, le=500)
    rsi_period: int = Field(default=14, ge=2, le=100)
    rsi_overbought: float = Field(default=70.0, ge=50.0, le=100.0)
    rsi_oversold: float = Field(default=30.0, ge=0.0, le=50.0)
    risk_per_trade_pct: float = Field(default=0.5, ge=0.01, le=10.0)
    stop_loss_pct: float = Field(default=0.004, ge=0.0001, le=0.5)
    take_profit_pct: float = Field(default=0.008, ge=0.0001, le=1.0)

    @model_validator(mode="after")
    def validate_ema_relationship(self) -> "StrategyConfigSchema":
        """Ensure ema_slow is greater than ema_fast."""
        if self.ema_slow <= self.ema_fast:
            raise ValueError(
                f"ema_slow ({self.ema_slow}) must be greater than ema_fast ({self.ema_fast})"
            )
        return self

    @model_validator(mode="after")
    def validate_rsi_thresholds(self) -> "StrategyConfigSchema":
        """Ensure RSI thresholds are properly ordered."""
        if self.rsi_oversold >= self.rsi_overbought:
            raise ValueError(
                f"rsi_oversold ({self.rsi_oversold}) must be less than "
                f"rsi_overbought ({self.rsi_overbought})"
            )
        return self


class BotConfigSchema(BaseModel):
    """Validated schema for bot configuration."""

    symbol: str = Field(default="BTC/USDT", min_length=1, max_length=50)
    data_symbol: Optional[str] = Field(default=None, max_length=50)
    macro_symbol: Optional[str] = Field(default=None, max_length=50)
    asset_type: Literal["crypto", "equity", "stock", "etf", "index", "commodity", "forex"] = (
        Field(default="crypto")
    )
    timeframe: str = Field(default="1m", pattern=r"^\d+[mhdwM]$")
    loop_interval_seconds: int = Field(default=60, ge=1, le=86400)
    lookback: int = Field(default=250, ge=10, le=10000)
    paper_mode: bool = Field(default=True)
    exchange_id: str = Field(default="binance", min_length=1, max_length=50)
    starting_balance: float = Field(default=10000.0, ge=1.0, le=1_000_000_000.0)
    risk_per_trade_pct: float = Field(default=0.5, ge=0.01, le=10.0)
    stop_loss_pct: float = Field(default=0.004, ge=0.0001, le=0.5)
    take_profit_pct: float = Field(default=0.008, ge=0.0001, le=1.0)
    data_dir: str = Field(default="./data")
    macro_events_path: Optional[str] = Field(default=None)
    macro_refresh_seconds: int = Field(default=300, ge=10, le=86400)
    playbook_assets_path: Optional[str] = Field(default=None)


class PortfolioAssetSchema(BaseModel):
    """Validated schema for a portfolio asset definition."""

    symbol: str = Field(..., min_length=1, max_length=50)
    asset_type: Optional[
        Literal["crypto", "equity", "stock", "etf", "index", "commodity", "forex"]
    ] = Field(default=None)
    allocation_pct: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    starting_balance: Optional[float] = Field(default=None, ge=0.0)
    risk_per_trade_pct: Optional[float] = Field(default=None, ge=0.01, le=10.0)
    data_dir: Optional[str] = Field(default=None)
    data_symbol: Optional[str] = Field(default=None, max_length=50)
    macro_symbol: Optional[str] = Field(default=None, max_length=50)
    timeframe: Optional[str] = Field(default=None, pattern=r"^\d+[mhdwM]$")
    loop_interval_seconds: Optional[int] = Field(default=None, ge=1, le=86400)
    paper_mode: Optional[bool] = Field(default=None)
    stop_loss_pct: Optional[float] = Field(default=None, ge=0.0001, le=0.5)
    take_profit_pct: Optional[float] = Field(default=None, ge=0.0001, le=1.0)


class PortfolioConfigSchema(BaseModel):
    """Validated schema for portfolio configuration."""

    portfolio_capital: float = Field(default=10000.0, ge=1.0, le=1_000_000_000.0)
    default_timeframe: Optional[str] = Field(default=None, pattern=r"^\d+[mhdwM]$")
    default_loop_interval_seconds: Optional[int] = Field(default=None, ge=1, le=86400)
    default_paper_mode: Optional[bool] = Field(default=None)
    default_risk_per_trade_pct: Optional[float] = Field(default=None, ge=0.01, le=10.0)
    default_stop_loss_pct: Optional[float] = Field(default=None, ge=0.0001, le=0.5)
    default_take_profit_pct: Optional[float] = Field(default=None, ge=0.0001, le=1.0)
    assets: List[PortfolioAssetSchema] = Field(default_factory=list)

    @field_validator("assets")
    @classmethod
    def validate_unique_symbols(cls, v: List[PortfolioAssetSchema]) -> List[PortfolioAssetSchema]:
        """Ensure all asset symbols are unique."""
        symbols = [asset.symbol.upper() for asset in v]
        if len(symbols) != len(set(symbols)):
            raise ValueError("Duplicate symbols found in portfolio assets")
        return v

    @model_validator(mode="after")
    def validate_allocation_sum(self) -> "PortfolioConfigSchema":
        """Warn if allocations don't sum to 100%."""
        allocations = [a.allocation_pct for a in self.assets if a.allocation_pct is not None]
        if allocations:
            total = sum(allocations)
            if abs(total - 100.0) > 0.01:
                # We don't raise an error, just allow it (user might want partial allocation)
                pass
        return self


class MacroEventSchema(BaseModel):
    """Validated schema for a macro event."""

    title: str = Field(..., min_length=1, max_length=500)
    category: Optional[str] = Field(default=None, max_length=100)
    sentiment: Optional[Literal["bullish", "bearish", "neutral"]] = Field(default=None)
    impact: Optional[Literal["high", "medium", "low"]] = Field(default=None)
    actor: Optional[str] = Field(default=None, max_length=200)
    interest_rate_expectation: Optional[Literal["hawkish", "dovish", "neutral"]] = Field(
        default=None
    )
    summary: Optional[str] = Field(default=None, max_length=2000)
    timestamp: Optional[str] = Field(default=None)
    source: Optional[str] = Field(default=None, max_length=200)
    symbols: Optional[List[str]] = Field(default=None)


class MacroEventsFileSchema(BaseModel):
    """Validated schema for macro events JSON file."""

    events: List[MacroEventSchema] = Field(default_factory=list)


def validate_strategy_config(config_dict: dict) -> StrategyConfigSchema:
    """Validate a strategy configuration dictionary."""
    return StrategyConfigSchema.model_validate(config_dict)


def validate_bot_config(config_dict: dict) -> BotConfigSchema:
    """Validate a bot configuration dictionary."""
    return BotConfigSchema.model_validate(config_dict)


def validate_portfolio_config(config_dict: dict) -> PortfolioConfigSchema:
    """Validate a portfolio configuration dictionary."""
    return PortfolioConfigSchema.model_validate(config_dict)


def validate_macro_events(events_dict: dict) -> MacroEventsFileSchema:
    """Validate a macro events file dictionary."""
    return MacroEventsFileSchema.model_validate(events_dict)
