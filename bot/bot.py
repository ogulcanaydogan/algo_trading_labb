from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union, cast

import pandas as pd

from .ai import PredictionSnapshot, RuleBasedAIPredictor
from .control import load_bot_control
from .exchange import ExchangeClient, PaperExchangeClient
from .macro import MacroInsight, MacroSentimentEngine
from .playbook import (
    HorizonConfig,
    PortfolioPlaybook,
    PlaybookAssetDefinition,
    PlaybookAssetFile,
    build_portfolio_playbook,
    load_playbook_asset_file,
)
from .state import BotState, EquityPoint, PositionType, SignalEvent, StateStore, create_state_store
from .config_loader import load_overrides, merge_config
from .market_data import MarketDataError, YFinanceMarketDataClient
from .strategy import (
    StrategyConfig,
    calculate_position_size,
    calculate_kelly_position_size,
    compute_indicators,
    generate_signal,
)
from .position_sizer import PositionSizer, SizingMethod
from .multi_timeframe import MultiTimeframeAnalyzer, get_recommended_htf, resample_to_htf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("algo-trading-bot")


@dataclass
class BotConfig:
    symbol: str = "BTC/USDT"
    data_symbol: Optional[str] = None  # allows mapping to provider-specific tickers (e.g. GC=F)
    macro_symbol: Optional[str] = None  # optional override for macro engine lookups
    asset_type: str = "crypto"
    timeframe: str = "1m"
    loop_interval_seconds: int = 60
    lookback: int = 250
    paper_mode: bool = True
    exchange_id: str = "binance"
    starting_balance: float = 10_000.0
    risk_per_trade_pct: float = 0.5
    stop_loss_pct: float = 0.004
    take_profit_pct: float = 0.008
    data_dir: Path = Path("./data")
    macro_events_path: Optional[Path] = None
    macro_refresh_seconds: int = 300
    playbook_assets_path: Optional[Path] = None
    # Advanced position sizing
    use_kelly_sizing: bool = False
    sizing_method: str = "volatility_adjusted"  # fixed_fraction, kelly, half_kelly, volatility_adjusted, atr_based
    max_position_pct: float = 0.25  # Maximum 25% of portfolio per position
    # Multi-timeframe filtering
    use_htf_filter: bool = True
    htf_strict_mode: bool = False  # If True, reject counter-trend signals entirely

    @classmethod
    def from_env(cls) -> "BotConfig":
        return cls(
            symbol=os.getenv("SYMBOL", "BTC/USDT"),
            data_symbol=os.getenv("DATA_SYMBOL"),
            macro_symbol=os.getenv("MACRO_SYMBOL"),
            asset_type=os.getenv("ASSET_TYPE", "crypto"),
            timeframe=os.getenv("TIMEFRAME", "1m"),
            loop_interval_seconds=int(os.getenv("LOOP_INTERVAL_SECONDS", "60")),
            lookback=int(os.getenv("LOOKBACK", "250")),
            paper_mode=os.getenv("PAPER_MODE", "true").lower() == "true",
            exchange_id=os.getenv("EXCHANGE_ID", "binance"),
            starting_balance=float(os.getenv("STARTING_BALANCE", "10000")),
            risk_per_trade_pct=float(os.getenv("RISK_PER_TRADE_PCT", "0.5")),
            stop_loss_pct=float(os.getenv("STOP_LOSS_PCT", "0.004")),
            take_profit_pct=float(os.getenv("TAKE_PROFIT_PCT", "0.008")),
            data_dir=Path(os.getenv("DATA_DIR", "./data")),
            macro_events_path=(
                Path(os.getenv("MACRO_EVENTS_PATH", "")).expanduser()
                if os.getenv("MACRO_EVENTS_PATH")
                else None
            ),
            macro_refresh_seconds=int(
                os.getenv("MACRO_REFRESH_SECONDS", "300")
            ),
            playbook_assets_path=(
                Path(os.getenv("PLAYBOOK_ASSETS_PATH", "")).expanduser()
                if os.getenv("PLAYBOOK_ASSETS_PATH")
                else None
            ),
            use_kelly_sizing=os.getenv("USE_KELLY_SIZING", "false").lower() == "true",
            sizing_method=os.getenv("SIZING_METHOD", "volatility_adjusted"),
            max_position_pct=float(os.getenv("MAX_POSITION_PCT", "0.25")),
            use_htf_filter=os.getenv("USE_HTF_FILTER", "true").lower() == "true",
            htf_strict_mode=os.getenv("HTF_STRICT_MODE", "false").lower() == "true",
        )


@dataclass
class ExecutionSnapshot:
    decision: str
    confidence: float
    reason: str
    technical_decision: str
    technical_confidence: float
    technical_reason: str
    ai_override: bool


def build_strategy_config(config: BotConfig) -> StrategyConfig:
    """Build StrategyConfig from BotConfig and optional overrides file.

    If data/strategy_config.json exists (within DATA_DIR), overlay its fields.
    """
    base = StrategyConfig(
        symbol=config.symbol,
        timeframe=config.timeframe,
        risk_per_trade_pct=config.risk_per_trade_pct,
        stop_loss_pct=config.stop_loss_pct,
        take_profit_pct=config.take_profit_pct,
    )

    overrides_path = config.data_dir / "strategy_config.json"
    overrides = load_overrides(overrides_path)
    if overrides:
        return merge_config(base, overrides)
    return base


def create_market_client(
    config: BotConfig,
) -> PaperExchangeClient | ExchangeClient | YFinanceMarketDataClient:
    asset_type = (config.asset_type or "crypto").lower()

    if config.paper_mode:
        logger.info(
            "Running in paper mode with synthetic exchange data for %s (%s).",
            config.symbol,
            asset_type,
        )
        return PaperExchangeClient(
            symbol=config.symbol,
            timeframe=config.timeframe,
        )

    if asset_type in {"equity", "stock", "etf", "index", "commodity", "forex"}:
        try:
            return YFinanceMarketDataClient()
        except MarketDataError as exc:
            logger.warning("Falling back to paper market data: %s", exc)
            return PaperExchangeClient(symbol=config.symbol, timeframe=config.timeframe)

    # Check for Binance testnet configuration
    testnet_enabled = os.getenv("BINANCE_TESTNET_ENABLED", "false").lower() == "true"

    if testnet_enabled and config.exchange_id == "binance":
        api_key = os.getenv("BINANCE_TESTNET_API_KEY")
        api_secret = os.getenv("BINANCE_TESTNET_API_SECRET")
        logger.info("Using Binance Spot Testnet")
        try:
            return ExchangeClient(
                exchange_id=config.exchange_id,
                api_key=api_key,
                api_secret=api_secret,
                sandbox=False,
                testnet=True,
            )
        except RuntimeError as exc:
            logger.warning("Falling back to paper exchange: %s", exc)
            return PaperExchangeClient(symbol=config.symbol, timeframe=config.timeframe)

    # Regular exchange configuration
    api_key = os.getenv("EXCHANGE_API_KEY")
    api_secret = os.getenv("EXCHANGE_API_SECRET")
    sandbox = os.getenv("EXCHANGE_SANDBOX", "false").lower() == "true"

    try:
        return ExchangeClient(
            exchange_id=config.exchange_id,
            api_key=api_key,
            api_secret=api_secret,
            sandbox=sandbox,
            testnet=False,
        )
    except RuntimeError as exc:
        logger.warning("Falling back to paper exchange: %s", exc)
        return PaperExchangeClient(symbol=config.symbol, timeframe=config.timeframe)


def sync_state_with_config(store: StateStore, config: BotConfig) -> None:
    """Ensure the persisted state aligns with runtime configuration."""

    state = store.state
    updates: dict[str, object] = {}

    if state.symbol != config.symbol:
        updates["symbol"] = config.symbol
    if abs(state.initial_balance - config.starting_balance) > 1e-6:
        updates["initial_balance"] = config.starting_balance
    if abs(state.balance - config.starting_balance) > 1e-6:
        updates["balance"] = config.starting_balance
    if state.risk_per_trade_pct != config.risk_per_trade_pct:
        updates["risk_per_trade_pct"] = config.risk_per_trade_pct

    if updates:
        store.update_state(**updates)


def run_loop(config: BotConfig) -> None:
    market_client = create_market_client(config)
    store = create_state_store(config.data_dir)
    sync_state_with_config(store, config)
    # predictor will be re-created if strategy settings change
    predictor: RuleBasedAIPredictor | None = None
    macro_engine = MacroSentimentEngine(
        events_path=config.macro_events_path,
        refresh_interval=config.macro_refresh_seconds,
    )

    # Initialize advanced position sizer
    sizing_method_map = {
        "fixed_fraction": SizingMethod.FIXED_FRACTION,
        "kelly": SizingMethod.KELLY,
        "half_kelly": SizingMethod.HALF_KELLY,
        "volatility_adjusted": SizingMethod.VOLATILITY_ADJUSTED,
        "atr_based": SizingMethod.ATR_BASED,
        "confidence_scaled": SizingMethod.CONFIDENCE_SCALED,
    }
    position_sizer = PositionSizer(
        portfolio_value=config.starting_balance,
        max_position_size=config.max_position_pct,
        max_portfolio_risk=config.risk_per_trade_pct / 100,
    )

    # Initialize multi-timeframe analyzer
    mtf_analyzer = MultiTimeframeAnalyzer() if config.use_htf_filter else None

    playbook_assets: Sequence[PlaybookAssetDefinition] | None = None
    playbook_horizons: Sequence[HorizonConfig] | None = None
    if config.playbook_assets_path:
        try:
            asset_file: PlaybookAssetFile = load_playbook_asset_file(
                config.playbook_assets_path
            )
            playbook_assets = asset_file.assets
            playbook_horizons = asset_file.horizons
            logger.info(
                "Loaded %d custom playbook assets from %s",
                len(playbook_assets),
                config.playbook_assets_path,
            )
        except Exception as exc:
            logger.warning(
                "Failed to load playbook assets from %s: %s",
                config.playbook_assets_path,
                exc,
            )

    logger.info(
        "Starting trading loop | symbol=%s data_symbol=%s timeframe=%s asset_type=%s paper=%s",
        config.symbol,
        config.data_symbol or config.symbol,
        config.timeframe,
        config.asset_type,
        config.paper_mode,
    )

    while True:
        started = time.time()
        try:
            control_state = load_bot_control(config.data_dir)
            if control_state.paused:
                note = control_state.reason or "Manual safety pause active."
                if (
                    store.state.last_signal != "PAUSED"
                    or store.state.last_signal_reason != note
                ):
                    store.update_state(
                        last_signal="PAUSED",
                        last_signal_reason=note,
                        ai_action=None,
                        ai_confidence=None,
                    )
                    logger.info(
                        "Trading paused for %s: %s",
                        config.symbol,
                        note,
                    )
                time.sleep(max(5, config.loop_interval_seconds))
                continue
            # re-load strategy config every loop to allow hot updates
            strategy_config = build_strategy_config(config)
            if predictor is None or predictor.config != strategy_config:
                predictor = RuleBasedAIPredictor(strategy_config)
                logger.info(
                    (
                        "Strategy reloaded | ema_fast=%d ema_slow=%d rsi_period=%d "
                        "rsi_overbought=%.1f rsi_oversold=%.1f risk=%.3f%% "
                        "sl=%.3f%% tp=%.3f%%"
                    ),
                    strategy_config.ema_fast,
                    strategy_config.ema_slow,
                    strategy_config.rsi_period,
                    strategy_config.rsi_overbought,
                    strategy_config.rsi_oversold,
                    strategy_config.risk_per_trade_pct,
                    strategy_config.stop_loss_pct * 100,
                    strategy_config.take_profit_pct * 100,
                )
            market_symbol = config.data_symbol or config.symbol
            candles = fetch_candles(
                market_client,
                market_symbol,
                config.timeframe,
                config.lookback,
            )
            enriched = compute_indicators(candles, strategy_config)
            signal = generate_signal(enriched, strategy_config)

            # Apply multi-timeframe filter if enabled
            htf_info = None
            if mtf_analyzer and config.use_htf_filter and signal["decision"] != "FLAT":
                try:
                    # Load data into MTF analyzer
                    mtf_analyzer.load_data(config.symbol, candles)

                    # Get recommended HTF for this timeframe
                    htf = get_recommended_htf(config.timeframe)

                    # Apply HTF filter to signal
                    signal = mtf_analyzer.filter_signal_with_htf(
                        signal,
                        htf_timeframe=htf,
                        strict_mode=config.htf_strict_mode,
                    )
                    htf_info = signal.get("htf_filter", {})

                    if htf_info and htf_info.get("rejected"):
                        logger.info(
                            "Signal filtered by HTF: %s -> FLAT | reason=%s",
                            signal.get("decision"),
                            htf_info.get("reason"),
                        )
                except Exception as htf_error:
                    logger.debug("HTF filter skipped: %s", htf_error)

            # Update position sizer with current portfolio value
            position_sizer.update_portfolio_value(store.state.balance)

            # Calculate advanced position size if signal is not FLAT
            advanced_sizing = None
            if signal["decision"] != "FLAT":
                try:
                    sizing_method = sizing_method_map.get(
                        config.sizing_method,
                        SizingMethod.VOLATILITY_ADJUSTED
                    )
                    # Get volatility and ATR from signal
                    atr = signal.get("atr")
                    price = float(signal["close"])

                    # Estimate volatility from ATR
                    volatility = (atr / price * 100 * 16) if atr else None  # Annualized approx

                    advanced_sizing = position_sizer.calculate_size(
                        symbol=config.symbol,
                        method=sizing_method,
                        price=price,
                        confidence=float(signal.get("confidence", 0.5)),
                        volatility=volatility,
                        atr=atr,
                    )
                    logger.debug(
                        "Advanced sizing: method=%s size=%.4f reason=%s",
                        advanced_sizing.method.value,
                        advanced_sizing.position_size,
                        advanced_sizing.reasoning,
                    )
                except Exception as sizing_error:
                    logger.debug("Advanced sizing skipped: %s", sizing_error)
            macro_insight: MacroInsight | None
            try:
                target_macro_symbol = config.macro_symbol or config.symbol
                macro_insight = macro_engine.assess(target_macro_symbol)
            except Exception as macro_error:  # pragma: no cover - defensive
                logger.warning("Macro assessment failed: %s", macro_error)
                macro_insight = None
            ai_snapshot: PredictionSnapshot | None = None
            try:
                ai_snapshot = predictor.predict(enriched, macro_insight)
            except ValueError as exc:
                logger.debug("AI prediction skipped: %s", exc)
            try:
                playbook_kwargs: Dict[str, object] = {}
                if playbook_assets is not None:
                    playbook_kwargs["asset_definitions"] = playbook_assets
                if playbook_horizons is not None:
                    playbook_kwargs["horizons"] = playbook_horizons
                portfolio_playbook = build_portfolio_playbook(
                    starting_balance=config.starting_balance,
                    macro_engine=macro_engine,
                    **playbook_kwargs,
                )
            except Exception as playbook_error:
                logger.debug("Portfolio playbook generation skipped: %s", playbook_error)
                portfolio_playbook = None
            state = update_state(
                store,
                signal,
                config,
                ai_snapshot,
                macro_insight,
                portfolio_playbook,
                advanced_sizing,
            )
            record_metrics(store, signal, state, config, ai_snapshot)
            logger.info(
                "decision=%s price=%.2f rsi=%.2f ema_fast=%.2f ema_slow=%.2f ai=%s macro_bias=%.2f",
                signal["decision"],
                signal["close"],
                signal["rsi"],
                signal["ema_fast"],
                signal["ema_slow"],
                ai_snapshot.recommended_action if ai_snapshot else "n/a",
                macro_insight.bias_score if macro_insight else 0.0,
            )
        except Exception as exc:
            logger.exception("Error inside bot loop: %s", exc)

        elapsed = time.time() - started
        sleep_for = max(0, config.loop_interval_seconds - int(elapsed))
        time.sleep(sleep_for or 1)


def fetch_candles(
    client: PaperExchangeClient | ExchangeClient | YFinanceMarketDataClient,
    symbol: str,
    timeframe: str,
    limit: int,
) -> pd.DataFrame:
    if isinstance(client, PaperExchangeClient):
        return client.fetch_ohlcv(limit=limit)
    return client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)


def update_state(
    store: StateStore,
    signal: Dict[str, Union[float, str]],
    config: BotConfig,
    ai_snapshot: PredictionSnapshot | None,
    macro_insight: MacroInsight | None,
    portfolio_playbook: PortfolioPlaybook | None,
    advanced_sizing: Optional[object] = None,  # SizingResult from PositionSizer
) -> BotState:
    decision = cast(str, signal["decision"])
    price = float(signal["close"])  # ensure float
    state = store.state

    entry_price: Optional[float] = state.entry_price
    position: PositionType = state.position

    if decision == "FLAT":
        position = "FLAT"
        entry_price = None
    else:
        desired = cast(PositionType, decision)
        if position != desired:
            position = desired
            entry_price = price
        elif entry_price is None:
            entry_price = price

    # Use advanced sizing if available, otherwise fall back to basic calculation
    if advanced_sizing and hasattr(advanced_sizing, 'shares_or_units'):
        position_size = advanced_sizing.shares_or_units
    else:
        # Get ATR from signal for better position sizing
        atr = signal.get("atr")
        position_size = calculate_position_size(
            balance=state.balance,
            risk_pct=config.risk_per_trade_pct,
            price=price,
            stop_loss_pct=config.stop_loss_pct,
            atr=atr,
        )

    unrealized_pnl_pct = 0.0
    if entry_price and position != "FLAT":
        direction = 1 if position == "LONG" else -1
        unrealized_pnl_pct = (price - entry_price) / entry_price * 100 * direction

    ai_features: Dict[str, float] = {}
    if ai_snapshot:
        ai_features = {
            "ema_gap_pct": ai_snapshot.features.ema_gap_pct,
            "momentum_pct": ai_snapshot.features.momentum_pct,
            "rsi_distance_from_mid": ai_snapshot.features.rsi_distance_from_mid,
            "volatility_pct": ai_snapshot.features.volatility_pct,
        }

    macro_events: List[Dict[str, object]] = []
    macro_summary: Optional[str] = None
    macro_bias: Optional[float] = None
    macro_confidence: Optional[float] = None
    macro_interest: Optional[str] = None
    macro_political: Optional[str] = None
    macro_drivers: List[str] = []

    if macro_insight:
        macro_events = [dict(event) for event in getattr(macro_insight, "events", [])]
        macro_summary = getattr(macro_insight, "summary", None)
        macro_bias = getattr(macro_insight, "bias_score", None)
        macro_confidence = getattr(macro_insight, "confidence", None)
        macro_interest = getattr(macro_insight, "interest_rate_outlook", None)
        macro_political = getattr(macro_insight, "political_risk", None)
        macro_drivers = list(getattr(macro_insight, "drivers", []))

    playbook_payload = (
        portfolio_playbook.to_dict() if portfolio_playbook else store.state.portfolio_playbook
    )

    signal_confidence = float(signal.get("confidence", 0.0))
    signal_reason = cast(str, signal.get("reason", ""))

    ai_override = False
    if ai_snapshot and ai_snapshot.recommended_action:
        ai_override = ai_snapshot.recommended_action != decision

    updated = store.update_state(
        symbol=config.symbol,
        position=position,
        entry_price=entry_price,
        position_size=position_size,
        unrealized_pnl_pct=round(unrealized_pnl_pct, 4),
        last_signal=decision,
        last_signal_reason=signal_reason,
        confidence=round(signal_confidence, 4),
        technical_signal=decision,
        technical_confidence=round(signal_confidence, 4),
        technical_reason=signal_reason,
        ai_override_active=ai_override,
        rsi=float(signal.get("rsi", state.rsi or 0.0)),
        ema_fast=float(signal.get("ema_fast", state.ema_fast or 0.0)),
        ema_slow=float(signal.get("ema_slow", state.ema_slow or 0.0)),
        risk_per_trade_pct=config.risk_per_trade_pct,
        ai_action=ai_snapshot.recommended_action if ai_snapshot else None,
        ai_confidence=ai_snapshot.confidence if ai_snapshot else None,
        ai_probability_long=ai_snapshot.probability_long if ai_snapshot else None,
        ai_probability_short=ai_snapshot.probability_short if ai_snapshot else None,
        ai_probability_flat=ai_snapshot.probability_flat if ai_snapshot else None,
        ai_expected_move_pct=ai_snapshot.expected_move_pct if ai_snapshot else None,
        ai_summary=ai_snapshot.summary if ai_snapshot else None,
        ai_features=ai_features,
        macro_bias=macro_bias,
        macro_confidence=macro_confidence,
        macro_summary=macro_summary,
        macro_drivers=macro_drivers,
        macro_interest_rate_outlook=macro_interest,
        macro_political_risk=macro_political,
        macro_events=macro_events,
        portfolio_playbook=playbook_payload,
    )
    return updated


def record_metrics(
    store: StateStore,
    signal: Dict[str, Union[float, str]],
    state: BotState,
    config: BotConfig,
    ai_snapshot: PredictionSnapshot | None,
) -> None:
    timestamp = state.timestamp
    # Record a simplified signal event and equity snapshot
    store.record_signal(
        SignalEvent(
            timestamp=timestamp,
            symbol=config.symbol,
            decision=cast(PositionType, signal["decision"]),
            confidence=float(signal.get("confidence", 0.0)),
            reason=cast(str, signal.get("reason", "")),
            ai_action=ai_snapshot.recommended_action if ai_snapshot else None,
            ai_confidence=ai_snapshot.confidence if ai_snapshot else None,
            ai_expected_move_pct=ai_snapshot.expected_move_pct if ai_snapshot else None,
        )
    )

    equity_value = state.balance * (1 + (state.unrealized_pnl_pct or 0.0) / 100)
    store.record_equity(
        EquityPoint(
            timestamp=timestamp,
            value=round(equity_value, 2),
        )
    )


if __name__ == "__main__":
    cfg = BotConfig.from_env()
    run_loop(cfg)
