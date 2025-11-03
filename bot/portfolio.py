from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .bot import BotConfig, run_loop
from .market_data import sanitize_symbol_for_fs

logger = logging.getLogger(__name__)


@dataclass
class PortfolioAssetConfig:
    """Per-asset overrides for the multi-instrument portfolio runner."""

    symbol: str
    asset_type: str = "crypto"
    data_symbol: Optional[str] = None
    macro_symbol: Optional[str] = None
    timeframe: Optional[str] = None
    lookback: Optional[int] = None
    paper_mode: Optional[bool] = None
    exchange_id: Optional[str] = None
    starting_balance: Optional[float] = None
    allocation_pct: Optional[float] = None
    risk_per_trade_pct: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    loop_interval_seconds: Optional[int] = None
    macro_events_path: Optional[str] = None
    macro_refresh_seconds: Optional[int] = None
    data_dir: Optional[str] = None


@dataclass
class PortfolioConfig:
    """Structured configuration for a multi-asset deployment."""

    assets: List[PortfolioAssetConfig] = field(default_factory=list)
    portfolio_capital: float = 100_000.0
    default_timeframe: str = "1h"
    default_loop_interval_seconds: int = 60
    default_lookback: int = 500
    default_paper_mode: bool = True
    default_exchange_id: str = "binance"
    default_risk_per_trade_pct: float = 0.5
    default_stop_loss_pct: float = 0.01
    default_take_profit_pct: float = 0.02
    macro_events_path: Optional[Path] = None
    macro_refresh_seconds: int = 300
    data_dir: Path = Path("./data/portfolio")

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PortfolioConfig":
        assets_payload = payload.get("assets", [])
        assets = [
            PortfolioAssetConfig(**asset)  # type: ignore[arg-type]
            for asset in assets_payload
        ]
        macro_path_value = payload.get("macro_events_path")
        macro_path = Path(macro_path_value).expanduser() if macro_path_value else None
        data_dir_value = payload.get("data_dir", "./data/portfolio")
        return cls(
            assets=assets,
            portfolio_capital=float(payload.get("portfolio_capital", 100_000.0)),
            default_timeframe=payload.get("default_timeframe", "1h"),
            default_loop_interval_seconds=int(
                payload.get("default_loop_interval_seconds", 60)
            ),
            default_lookback=int(payload.get("default_lookback", 500)),
            default_paper_mode=bool(payload.get("default_paper_mode", True)),
            default_exchange_id=payload.get("default_exchange_id", "binance"),
            default_risk_per_trade_pct=float(
                payload.get("default_risk_per_trade_pct", 0.5)
            ),
            default_stop_loss_pct=float(payload.get("default_stop_loss_pct", 0.01)),
            default_take_profit_pct=float(
                payload.get("default_take_profit_pct", 0.02)
            ),
            macro_events_path=macro_path,
            macro_refresh_seconds=int(payload.get("macro_refresh_seconds", 300)),
            data_dir=Path(data_dir_value).expanduser(),
        )

    @classmethod
    def load(cls, path: Path) -> "PortfolioConfig":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_dict(payload)


class PortfolioRunner:
    """Launch and supervise multiple bot instances inside the same process."""

    def __init__(
        self,
        config: PortfolioConfig,
        sleep_interval_seconds: float = 1.0,
    ) -> None:
        if not config.assets:
            raise ValueError("Portfolio configuration must include at least one asset.")
        self.config = config
        self.sleep_interval_seconds = sleep_interval_seconds
        self._stop_event = threading.Event()
        self._threads: List[threading.Thread] = []
        self._allocations = self._resolve_allocations(config.assets)

    def start(self) -> None:
        logger.info(
            "Starting portfolio runner with %d assets (capital %.2f).",
            len(self.config.assets),
            self.config.portfolio_capital,
        )
        self.config.data_dir.mkdir(parents=True, exist_ok=True)

        for asset_cfg in self.config.assets:
            bot_config = self._build_bot_config(asset_cfg)
            name = f"bot-{sanitize_symbol_for_fs(asset_cfg.symbol)}"
            thread = threading.Thread(
                target=run_loop,
                args=(bot_config,),
                name=name,
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)
            logger.info(
                "Launched asset thread | symbol=%s data_dir=%s allocation=%.2f%%",
                bot_config.symbol,
                bot_config.data_dir,
                self._allocations[asset_cfg.symbol],
            )

        try:
            while not self._stop_event.is_set():
                time.sleep(self.sleep_interval_seconds)
        except KeyboardInterrupt:
            logger.info("Portfolio runner received shutdown signal.")
            self.stop()

    def stop(self) -> None:
        self._stop_event.set()
        logger.info("Portfolio runner stop requested. Threads set to daemon=True, exiting main loop.")

    def _build_bot_config(self, asset: PortfolioAssetConfig) -> BotConfig:
        data_dir = (
            Path(asset.data_dir).expanduser()
            if asset.data_dir
            else self.config.data_dir / sanitize_symbol_for_fs(asset.symbol)
        )
        data_dir.mkdir(parents=True, exist_ok=True)

        allocation_pct = self._allocations[asset.symbol]
        starting_balance = (
            float(asset.starting_balance)
            if asset.starting_balance is not None
            else (self.config.portfolio_capital * allocation_pct / 100.0)
        )

        macro_path = (
            Path(asset.macro_events_path).expanduser()
            if asset.macro_events_path
            else self.config.macro_events_path
        )

        bot_config = BotConfig(
            symbol=asset.symbol,
            data_symbol=asset.data_symbol,
            macro_symbol=asset.macro_symbol,
            asset_type=asset.asset_type or "crypto",
            timeframe=asset.timeframe or self.config.default_timeframe,
            loop_interval_seconds=asset.loop_interval_seconds
            if asset.loop_interval_seconds is not None
            else self.config.default_loop_interval_seconds,
            lookback=asset.lookback or self.config.default_lookback,
            paper_mode=self._resolve_paper_mode(asset.paper_mode),
            exchange_id=asset.exchange_id or self.config.default_exchange_id,
            starting_balance=starting_balance,
            risk_per_trade_pct=asset.risk_per_trade_pc
            if asset.risk_per_trade_pct is not None
            else self.config.default_risk_per_trade_pct,
            stop_loss_pct=asset.stop_loss_pc
            if asset.stop_loss_pct is not None
            else self.config.default_stop_loss_pct,
            take_profit_pct=asset.take_profit_pc
            if asset.take_profit_pct is not None
            else self.config.default_take_profit_pct,
            data_dir=data_dir,
            macro_events_path=macro_path,
            macro_refresh_seconds=asset.macro_refresh_seconds
            if asset.macro_refresh_seconds is not None
            else self.config.macro_refresh_seconds,
        )
        return bot_config

    def _resolve_allocations(
        self, assets: Iterable[PortfolioAssetConfig]
    ) -> Dict[str, float]:
        allocations: Dict[str, float] = {}
        explicit_total = 0.0
        empty_symbols: List[str] = []
        for asset in assets:
            if asset.allocation_pct is None:
                empty_symbols.append(asset.symbol)
            else:
                allocations[asset.symbol] = float(asset.allocation_pct)
                explicit_total += float(asset.allocation_pct)

        remaining = max(0.0, 100.0 - explicit_total)
        if explicit_total > 100.0:
            logger.warning(
                "Explicit allocation exceeds 100%% (%.2f). Values will be proportionally scaled.",
                explicit_total,
            )
            scale = 100.0 / explicit_total if explicit_total else 0.0
            allocations = {symbol: pct * scale for symbol, pct in allocations.items()}
            remaining = 0.0

        default_allocation = (remaining / len(empty_symbols)) if empty_symbols else 0.0
        for symbol in empty_symbols:
            allocations[symbol] = default_allocation

        if not allocations:
            raise ValueError("Failed to derive allocation percentages for portfolio assets.")

        return allocations

    def _resolve_paper_mode(self, asset_value: Optional[bool]) -> bool:
        if asset_value is None:
            return self.config.default_paper_mode
        return bool(asset_value)

