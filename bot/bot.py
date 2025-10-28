from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from .exchange import ExchangeClient, PaperExchangeClient
from .state import BotState, EquityPoint, SignalEvent, StateStore, create_state_store
from .strategy import (
    StrategyConfig,
    calculate_position_size,
    compute_indicators,
    generate_signal,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("algo-trading-bot")


@dataclass
class BotConfig:
    symbol: str = "BTC/USDT"
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

    @classmethod
    def from_env(cls) -> "BotConfig":
        return cls(
            symbol=os.getenv("SYMBOL", "BTC/USDT"),
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
        )


def build_strategy_config(config: BotConfig) -> StrategyConfig:
    return StrategyConfig(
        symbol=config.symbol,
        timeframe=config.timeframe,
        risk_per_trade_pct=config.risk_per_trade_pct,
        stop_loss_pct=config.stop_loss_pct,
        take_profit_pct=config.take_profit_pct,
    )


def create_exchange_client(config: BotConfig) -> PaperExchangeClient | ExchangeClient:
    if config.paper_mode:
        logger.info("Running in paper mode with synthetic exchange data.")
        return PaperExchangeClient(symbol=config.symbol, timeframe=config.timeframe)

    api_key = os.getenv("EXCHANGE_API_KEY")
    api_secret = os.getenv("EXCHANGE_API_SECRET")
    sandbox = os.getenv("EXCHANGE_SANDBOX", "false").lower() == "true"

    try:
        return ExchangeClient(
            exchange_id=config.exchange_id,
            api_key=api_key,
            api_secret=api_secret,
            sandbox=sandbox,
        )
    except RuntimeError as exc:
        logger.warning("Falling back to paper exchange: %s", exc)
        return PaperExchangeClient(symbol=config.symbol, timeframe=config.timeframe)


def run_loop(config: BotConfig) -> None:
    strategy_config = build_strategy_config(config)
    exchange = create_exchange_client(config)
    store = create_state_store(config.data_dir)

    logger.info(
        "Starting trading loop | symbol=%s timeframe=%s paper=%s",
        config.symbol,
        config.timeframe,
        config.paper_mode,
    )

    while True:
        started = time.time()
        try:
            candles = fetch_candles(exchange, config.symbol, config.timeframe, config.lookback)
            enriched = compute_indicators(candles, strategy_config)
            signal = generate_signal(enriched, strategy_config)
            state = update_state(store, signal, config)
            record_metrics(store, signal, state, config)
            logger.info(
                "decision=%s price=%.2f rsi=%.2f ema_fast=%.2f ema_slow=%.2f",
                signal["decision"],
                signal["close"],
                signal["rsi"],
                signal["ema_fast"],
                signal["ema_slow"],
            )
        except Exception as exc:
            logger.exception("Error inside bot loop: %s", exc)

        elapsed = time.time() - started
        sleep_for = max(0, config.loop_interval_seconds - int(elapsed))
        time.sleep(sleep_for or 1)


def fetch_candles(
    exchange: PaperExchangeClient | ExchangeClient,
    symbol: str,
    timeframe: str,
    limit: int,
) -> pd.DataFrame:
    if isinstance(exchange, PaperExchangeClient):
        return exchange.fetch_ohlcv(limit=limit)
    return exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)


def update_state(store: StateStore, signal: dict, config: BotConfig) -> BotState:
    decision = signal["decision"]
    price = signal["close"]
    state = store.state

    entry_price: Optional[float] = state.entry_price
    position = state.position

    if decision == "FLAT":
        position = "FLAT"
        entry_price = None
    else:
        if position != decision:
            entry_price = price
            position = decision
        elif entry_price is None:
            entry_price = price

    position_size = calculate_position_size(
        balance=state.balance,
        risk_pct=config.risk_per_trade_pct,
        price=price,
        stop_loss_pct=config.stop_loss_pct,
    )

    unrealized_pnl_pct = 0.0
    if entry_price and position != "FLAT":
        direction = 1 if position == "LONG" else -1
        unrealized_pnl_pct = (price - entry_price) / entry_price * 100 * direction

    updated = store.update_state(
        symbol=config.symbol,
        position=position,
        entry_price=entry_price,
        position_size=position_size,
        unrealized_pnl_pct=round(unrealized_pnl_pct, 4),
        last_signal=decision,
        confidence=signal["confidence"],
        rsi=signal["rsi"],
        ema_fast=signal["ema_fast"],
        ema_slow=signal["ema_slow"],
        risk_per_trade_pct=config.risk_per_trade_pct,
    )
    return updated


def record_metrics(store: StateStore, signal: dict, state: BotState, config: BotConfig) -> None:
    timestamp = state.timestamp
    store.record_signal(
        SignalEvent(
            timestamp=timestamp,
            symbol=config.symbol,
            decision=signal["decision"],
            confidence=signal["confidence"],
            reason=signal["reason"],
        )
    )

    equity_value = config.starting_balance * (1 + state.unrealized_pnl_pct / 100)
    store.record_equity(
        EquityPoint(
            timestamp=timestamp,
            value=round(equity_value, 2),
        )
    )


if __name__ == "__main__":
    cfg = BotConfig.from_env()
    run_loop(cfg)
