from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from .exchange import PaperExchangeClient
from .macro import MacroInsight, MacroSentimentEngine
from .research import backtest_strategy, calculate_metrics
from .strategy import StrategyConfig, compute_indicators


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class HorizonConfig:
    label: str
    timeframe: str
    lookback: int
    ema_fast: int
    ema_slow: int
    rsi_overbought: float
    rsi_oversold: float


@dataclass
class HorizonSnapshot:
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

    def to_dict(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "timeframe": self.timeframe,
            "candles_tested": self.candles_tested,
            "initial_balance": self.initial_balance,
            "final_balance": self.final_balance,
            "total_return_pct": self.total_return_pct,
            "annualized_return_pct": self.annualized_return_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "win_rate": self.win_rate,
            "trades": self.trades,
            "macro_bias": self.macro_bias,
            "macro_summary": self.macro_summary,
        }


@dataclass
class AssetPlaybook:
    symbol: str
    asset_class: str
    macro_bias: float
    macro_confidence: float
    macro_summary: Optional[str]
    macro_drivers: Sequence[str]
    macro_interest_rate_outlook: Optional[str]
    macro_political_risk: Optional[str]
    horizons: List[HorizonSnapshot]
    notes: Sequence[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "symbol": self.symbol,
            "asset_class": self.asset_class,
            "macro_bias": self.macro_bias,
            "macro_confidence": self.macro_confidence,
            "macro_summary": self.macro_summary,
            "macro_drivers": list(self.macro_drivers),
            "macro_interest_rate_outlook": self.macro_interest_rate_outlook,
            "macro_political_risk": self.macro_political_risk,
            "horizons": [snapshot.to_dict() for snapshot in self.horizons],
            "notes": list(self.notes),
        }


@dataclass
class PortfolioPlaybook:
    generated_at: str
    starting_balance: float
    commodities: List[AssetPlaybook]
    equities: List[AssetPlaybook]
    highlights: Sequence[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "generated_at": self.generated_at,
            "starting_balance": self.starting_balance,
            "commodities": [asset.to_dict() for asset in self.commodities],
            "equities": [asset.to_dict() for asset in self.equities],
            "highlights": list(self.highlights),
        }


DEFAULT_COMMODITY_ASSETS: Sequence[str] = (
    "BTC/USDT",
    "ETH/USDT",
    "XAU/USD",
    "XAG/USD",
    "USOIL/USD",
)

DEFAULT_EQUITY_ASSETS: Sequence[str] = (
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOG",
    "TSLA",
    "NVDA",
)

DEFAULT_HORIZONS: Sequence[HorizonConfig] = (
    HorizonConfig(
        label="short", timeframe="1m", lookback=720, ema_fast=8, ema_slow=21, rsi_overbought=68.0, rsi_oversold=32.0
    ),
    HorizonConfig(
        label="medium", timeframe="15m", lookback=480, ema_fast=12, ema_slow=26, rsi_overbought=70.0, rsi_oversold=30.0
    ),
    HorizonConfig(
        label="long", timeframe="1h", lookback=360, ema_fast=20, ema_slow=50, rsi_overbought=72.0, rsi_oversold=28.0
    ),
)


def timeframe_to_minutes(timeframe: str) -> int:
    timeframe = timeframe.strip().lower()
    if timeframe.endswith("m"):
        return int(timeframe[:-1] or 1)
    if timeframe.endswith("h"):
        return int(timeframe[:-1] or 1) * 60
    if timeframe.endswith("d"):
        return int(timeframe[:-1] or 1) * 1440
    return 1


def build_strategy_config(symbol: str, horizon: HorizonConfig) -> StrategyConfig:
    return StrategyConfig(
        symbol=symbol,
        timeframe=horizon.timeframe,
        ema_fast=horizon.ema_fast,
        ema_slow=horizon.ema_slow,
        rsi_overbought=horizon.rsi_overbought,
        rsi_oversold=horizon.rsi_oversold,
    )


def _load_candles(symbol: str, horizon: HorizonConfig) -> pd.DataFrame:
    client = PaperExchangeClient(symbol=symbol, timeframe=horizon.timeframe)
    return client.fetch_ohlcv(limit=horizon.lookback)


def _annualised_return(total_return_pct: float, candles: int, timeframe: str) -> float:
    minutes = timeframe_to_minutes(timeframe)
    duration_days = (candles * minutes) / 1440 if candles else 0
    if duration_days <= 0:
        return 0.0
    growth = 1 + total_return_pct / 100
    periods_per_year = 365 / max(duration_days, 1e-6)
    annualised = (growth ** periods_per_year) - 1
    return float(round(annualised * 100, 4))


def _summarise_asset(horizons: Sequence[HorizonSnapshot], macro: Optional[MacroInsight]) -> List[str]:
    notes: List[str] = []
    if not horizons:
        return notes

    best = max(horizons, key=lambda snap: snap.total_return_pct)
    worst = min(horizons, key=lambda snap: snap.total_return_pct)
    notes.append(
        f"Best horizon: {best.label} ({best.total_return_pct:+.2f}% return)."
    )
    if best.label != worst.label:
        notes.append(
            f"Most pressured horizon: {worst.label} ({worst.total_return_pct:+.2f}% return)."
        )
    else:
        notes.append("Performance is even across horizons; monitor macro catalysts for differentiation.")

    if macro:
        bias_descriptor = "supportive" if macro.bias_score > 0 else "defensive" if macro.bias_score < 0 else "balanced"
        notes.append(
            f"Macro stance is {bias_descriptor} (bias {macro.bias_score:+.2f}, confidence {macro.confidence:.2f})."
        )
        if macro.interest_rate_outlook:
            notes.append(f"Rate outlook: {macro.interest_rate_outlook}")
        if macro.political_risk:
            notes.append(f"Political watch: {macro.political_risk}")
    return notes


def _build_asset_playbook(
    symbol: str,
    asset_class: str,
    horizons: Sequence[HorizonConfig],
    starting_balance: float,
    macro_engine: Optional[MacroSentimentEngine],
) -> AssetPlaybook:
    macro: Optional[MacroInsight] = None
    if macro_engine is not None:
        try:
            macro = macro_engine.assess(symbol)
        except Exception:
            macro = None

    snapshots: List[HorizonSnapshot] = []
    for horizon in horizons:
        try:
            candles = _load_candles(symbol, horizon)
            config = build_strategy_config(symbol, horizon)
            enriched = compute_indicators(candles, config)
        except Exception:
            continue

        equity_curve, trade_returns = backtest_strategy(enriched, config)
        total_return, sharpe, win_rate, _max_dd, trades = calculate_metrics(equity_curve, trade_returns)
        final_balance = starting_balance * (1 + total_return / 100)
        annualised = _annualised_return(total_return, len(equity_curve), horizon.timeframe)

        snapshots.append(
            HorizonSnapshot(
                label=horizon.label,
                timeframe=horizon.timeframe,
                candles_tested=len(enriched),
                initial_balance=round(starting_balance, 2),
                final_balance=round(final_balance, 2),
                total_return_pct=round(total_return, 4),
                annualized_return_pct=annualised,
                sharpe_ratio=round(sharpe, 4),
                win_rate=round(win_rate, 4),
                trades=trades,
                macro_bias=round(macro.bias_score, 4) if macro else 0.0,
                macro_summary=macro.summary if macro else None,
            )
        )

    notes = _summarise_asset(snapshots, macro)

    return AssetPlaybook(
        symbol=symbol,
        asset_class=asset_class,
        macro_bias=round(macro.bias_score, 4) if macro else 0.0,
        macro_confidence=round(macro.confidence, 4) if macro else 0.0,
        macro_summary=macro.summary if macro else None,
        macro_drivers=macro.drivers if macro else [],
        macro_interest_rate_outlook=macro.interest_rate_outlook if macro else None,
        macro_political_risk=macro.political_risk if macro else None,
        horizons=snapshots,
        notes=notes,
    )


def build_portfolio_playbook(
    *,
    starting_balance: float,
    macro_engine: Optional[MacroSentimentEngine] = None,
    commodity_assets: Sequence[str] = DEFAULT_COMMODITY_ASSETS,
    equity_assets: Sequence[str] = DEFAULT_EQUITY_ASSETS,
    horizons: Sequence[HorizonConfig] = DEFAULT_HORIZONS,
) -> PortfolioPlaybook:
    commodities: List[AssetPlaybook] = []
    equities: List[AssetPlaybook] = []

    for symbol in commodity_assets:
        commodities.append(
            _build_asset_playbook(
                symbol,
                asset_class="crypto" if symbol.endswith("/USDT") else "commodity",
                horizons=horizons,
                starting_balance=starting_balance,
                macro_engine=macro_engine,
            )
        )

    for symbol in equity_assets:
        equities.append(
            _build_asset_playbook(
                symbol,
                asset_class="equity",
                horizons=horizons,
                starting_balance=starting_balance,
                macro_engine=macro_engine,
            )
        )

    highlights: List[str] = []
    all_assets = commodities + equities
    if all_assets:
        best_asset = max(all_assets, key=lambda asset: max((h.total_return_pct for h in asset.horizons), default=-math.inf))
        worst_asset = min(all_assets, key=lambda asset: min((h.total_return_pct for h in asset.horizons), default=math.inf))
        if best_asset.horizons:
            top_horizon = max(best_asset.horizons, key=lambda h: h.total_return_pct)
            highlights.append(
                f"Top performer: {best_asset.symbol} on the {top_horizon.label} horizon ({top_horizon.total_return_pct:+.2f}%)."
            )
        if worst_asset.horizons:
            bottom_horizon = min(worst_asset.horizons, key=lambda h: h.total_return_pct)
            highlights.append(
                f"Lagging setup: {worst_asset.symbol} on the {bottom_horizon.label} horizon ({bottom_horizon.total_return_pct:+.2f}%)."
            )
        avg_macro = sum(asset.macro_bias for asset in all_assets) / len(all_assets)
        highlights.append(
            f"Average macro bias across tracked assets: {avg_macro:+.2f}."
        )

    return PortfolioPlaybook(
        generated_at=_utcnow_iso(),
        starting_balance=round(starting_balance, 2),
        commodities=commodities,
        equities=equities,
        highlights=highlights,
    )


def build_default_portfolio_playbook(
    *,
    starting_balance: float,
    macro_events_path: Optional[Path] = None,
    macro_refresh_seconds: int = 300,
) -> PortfolioPlaybook:
    macro_engine: Optional[MacroSentimentEngine] = None
    if macro_events_path is not None:
        macro_engine = MacroSentimentEngine(
            events_path=macro_events_path,
            refresh_interval=macro_refresh_seconds,
        )
    else:
        macro_engine = MacroSentimentEngine(refresh_interval=macro_refresh_seconds)

    return build_portfolio_playbook(
        starting_balance=starting_balance,
        macro_engine=macro_engine,
    )
