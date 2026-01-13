"""Shared pytest fixtures for all tests."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import pytest

from bot.state import BotState
from bot.strategy import StrategyConfig


@pytest.fixture
def sample_bot_state() -> BotState:
    """Create a sample BotState for testing."""
    return BotState(
        symbol="BTC/USDT",
        position="LONG",
        entry_price=50000.0,
        position_size=0.1,
        balance=10000.0,
        initial_balance=10000.0,
        unrealized_pnl_pct=2.5,
        last_signal="LONG",
        confidence=0.75,
        risk_per_trade_pct=0.5,
        rsi=45.0,
        ema_fast=50100.0,
        ema_slow=49900.0,
    )


@pytest.fixture
def sample_strategy_config() -> StrategyConfig:
    """Create a sample StrategyConfig for testing."""
    return StrategyConfig(
        symbol="BTC/USDT",
        timeframe="1m",
        ema_fast=12,
        ema_slow=26,
        rsi_period=14,
        rsi_overbought=70.0,
        rsi_oversold=30.0,
        risk_per_trade_pct=0.5,
        stop_loss_pct=0.004,
        take_profit_pct=0.008,
    )


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Create sample OHLCV DataFrame for testing."""
    np.random.seed(42)
    n_rows = 100

    base_price = 100.0
    returns = np.random.randn(n_rows) * 0.01
    prices = base_price * np.exp(np.cumsum(returns))

    data = pd.DataFrame({
        "open": prices * (1 + np.random.randn(n_rows) * 0.001),
        "high": prices * (1 + np.abs(np.random.randn(n_rows)) * 0.005),
        "low": prices * (1 - np.abs(np.random.randn(n_rows)) * 0.005),
        "close": prices,
        "volume": np.random.randint(1000, 10000, n_rows),
    })

    data.index = pd.date_range(start="2024-01-01", periods=n_rows, freq="1min", tz="UTC")
    return data


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory with required structure."""
    # Create state.json
    state = BotState(
        symbol="BTC/USDT",
        position="FLAT",
        balance=10000.0,
        initial_balance=10000.0,
        risk_per_trade_pct=0.5,
    )
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps(state.to_dict()), encoding="utf-8")

    # Create empty signals.json
    signals_file = tmp_path / "signals.json"
    signals_file.write_text("[]", encoding="utf-8")

    # Create empty equity.json
    equity_file = tmp_path / "equity.json"
    equity_file.write_text("[]", encoding="utf-8")

    return tmp_path


@pytest.fixture
def sample_portfolio_config() -> Dict[str, Any]:
    """Create sample portfolio configuration for testing."""
    return {
        "portfolio_capital": 50000.0,
        "default_timeframe": "1h",
        "default_loop_interval_seconds": 60,
        "default_paper_mode": True,
        "default_risk_per_trade_pct": 0.5,
        "default_stop_loss_pct": 0.015,
        "default_take_profit_pct": 0.03,
        "assets": [
            {
                "symbol": "BTC/USDT",
                "asset_type": "crypto",
                "allocation_pct": 40.0,
                "starting_balance": 20000.0,
            },
            {
                "symbol": "ETH/USDT",
                "asset_type": "crypto",
                "allocation_pct": 30.0,
                "starting_balance": 15000.0,
            },
            {
                "symbol": "XAU/USD",
                "asset_type": "commodity",
                "allocation_pct": 30.0,
                "starting_balance": 15000.0,
            },
        ],
    }


@pytest.fixture
def sample_macro_events() -> Dict[str, Any]:
    """Create sample macro events for testing."""
    return {
        "events": [
            {
                "title": "Federal Reserve Rate Decision",
                "category": "monetary_policy",
                "sentiment": "hawkish",
                "impact": "high",
                "actor": "Federal Reserve",
                "interest_rate_expectation": "hawkish",
                "summary": "Fed signals potential rate hikes",
            },
            {
                "title": "US Employment Report",
                "category": "economic_data",
                "sentiment": "bullish",
                "impact": "medium",
                "summary": "Strong job growth exceeds expectations",
            },
        ]
    }


# Markers for test categorization
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
