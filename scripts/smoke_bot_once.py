from __future__ import annotations

import logging
from pathlib import Path

from bot.ai import RuleBasedAIPredictor
from bot.bot import (
    BotConfig,
    build_strategy_config,
    create_market_client,
    fetch_candles,
    update_state,
    record_metrics,
)
from bot.strategy import compute_indicators, generate_signal
from bot.state import create_state_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smoke-bot-once")


def main() -> None:
    cfg = BotConfig.from_env()
    # force paper mode and a modest lookback for a quick run
    cfg.paper_mode = True
    cfg.lookback = 250
    cfg.data_dir = Path(cfg.data_dir or Path("./data"))

    market_client = create_market_client(cfg)
    store = create_state_store(cfg.data_dir)

    strategy_config = build_strategy_config(cfg)
    predictor = RuleBasedAIPredictor(strategy_config)

    market_symbol = cfg.data_symbol or cfg.symbol
    logger.info("Fetching candles for %s", market_symbol)
    candles = fetch_candles(market_client, market_symbol, cfg.timeframe, cfg.lookback)

    logger.info("Computing indicators")
    enriched = compute_indicators(candles, strategy_config)

    logger.info("Generating signal")
    signal = generate_signal(enriched, strategy_config)

    logger.info("Running AI predictor")
    ai_snapshot = predictor.predict(enriched)

    logger.info("Updating state and recording metrics")
    state = update_state(store, signal, cfg, ai_snapshot, None, None)
    record_metrics(store, signal, state, cfg, ai_snapshot)

    logger.info("Done. State snapshot: %s", store.get_state_dict())


if __name__ == "__main__":
    main()
