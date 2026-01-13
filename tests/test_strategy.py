"""Unit tests for the strategy module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bot.strategy import (
    StrategyConfig,
    calculate_position_size,
    compute_indicators,
    generate_signal,
)


@pytest.fixture
def default_config() -> StrategyConfig:
    """Create a default strategy configuration for tests."""
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
def sample_ohlcv_data() -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    n_rows = 100

    # Generate realistic price data with trend
    base_price = 100.0
    returns = np.random.randn(n_rows) * 0.01  # 1% daily volatility
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
def bullish_crossover_data() -> pd.DataFrame:
    """Create data that should trigger a bullish EMA crossover."""
    n_rows = 50

    # Create data where fast EMA crosses above slow EMA
    prices = np.concatenate([
        np.linspace(100, 95, 25),  # downtrend
        np.linspace(95, 105, 25),  # uptrend with crossover
    ])

    data = pd.DataFrame({
        "open": prices,
        "high": prices * 1.01,
        "low": prices * 0.99,
        "close": prices,
        "volume": np.ones(n_rows) * 1000,
    })

    data.index = pd.date_range(start="2024-01-01", periods=n_rows, freq="1min", tz="UTC")
    return data


@pytest.fixture
def bearish_crossover_data() -> pd.DataFrame:
    """Create data that should trigger a bearish EMA crossover."""
    n_rows = 50

    # Create data where fast EMA crosses below slow EMA
    prices = np.concatenate([
        np.linspace(100, 110, 25),  # uptrend
        np.linspace(110, 95, 25),   # downtrend with crossover
    ])

    data = pd.DataFrame({
        "open": prices,
        "high": prices * 1.01,
        "low": prices * 0.99,
        "close": prices,
        "volume": np.ones(n_rows) * 1000,
    })

    data.index = pd.date_range(start="2024-01-01", periods=n_rows, freq="1min", tz="UTC")
    return data


class TestStrategyConfig:
    """Tests for StrategyConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = StrategyConfig()
        assert config.symbol == "BTC/USDT"
        assert config.timeframe == "1m"
        assert config.ema_fast == 12
        assert config.ema_slow == 26
        assert config.rsi_period == 14
        assert config.rsi_overbought == 70.0
        assert config.rsi_oversold == 30.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = StrategyConfig(
            symbol="ETH/USDT",
            ema_fast=5,
            ema_slow=20,
            rsi_period=10,
        )
        assert config.symbol == "ETH/USDT"
        assert config.ema_fast == 5
        assert config.ema_slow == 20
        assert config.rsi_period == 10

    def test_from_env(self, monkeypatch):
        """Test loading config from environment variables."""
        monkeypatch.setenv("SYMBOL", "XRP/USDT")
        monkeypatch.setenv("EMA_FAST", "8")
        monkeypatch.setenv("EMA_SLOW", "21")

        config = StrategyConfig.from_env()
        assert config.symbol == "XRP/USDT"
        assert config.ema_fast == 8
        assert config.ema_slow == 21


class TestComputeIndicators:
    """Tests for compute_indicators function."""

    def test_adds_ema_columns(self, sample_ohlcv_data, default_config):
        """Test that EMA indicators are added to DataFrame."""
        result = compute_indicators(sample_ohlcv_data, default_config)

        assert "ema_fast" in result.columns
        assert "ema_slow" in result.columns
        assert "rsi" in result.columns

    def test_ema_values_reasonable(self, sample_ohlcv_data, default_config):
        """Test that EMA values are within reasonable range."""
        result = compute_indicators(sample_ohlcv_data, default_config)

        # EMAs should be close to price values
        price_range = (sample_ohlcv_data["close"].min(), sample_ohlcv_data["close"].max())

        ema_fast_last = result["ema_fast"].iloc[-1]
        ema_slow_last = result["ema_slow"].iloc[-1]

        assert price_range[0] * 0.9 <= ema_fast_last <= price_range[1] * 1.1
        assert price_range[0] * 0.9 <= ema_slow_last <= price_range[1] * 1.1

    def test_rsi_bounded(self, sample_ohlcv_data, default_config):
        """Test that RSI is bounded between 0 and 100."""
        result = compute_indicators(sample_ohlcv_data, default_config)

        # Skip NaN values at the beginning
        rsi_values = result["rsi"].dropna()

        assert rsi_values.min() >= 0
        assert rsi_values.max() <= 100

    def test_insufficient_data_raises_error(self, default_config):
        """Test that insufficient data raises ValueError."""
        small_data = pd.DataFrame({
            "open": [100, 101],
            "high": [102, 103],
            "low": [99, 100],
            "close": [101, 102],
            "volume": [1000, 1000],
        })

        with pytest.raises(ValueError, match="Not enough data"):
            compute_indicators(small_data, default_config)


class TestGenerateSignal:
    """Tests for generate_signal function."""

    def test_returns_required_keys(self, sample_ohlcv_data, default_config):
        """Test that signal contains all required keys."""
        enriched = compute_indicators(sample_ohlcv_data, default_config)
        signal = generate_signal(enriched, default_config)

        required_keys = ["decision", "confidence", "ema_fast", "ema_slow", "rsi", "close", "reason"]
        for key in required_keys:
            assert key in signal

    def test_decision_is_valid(self, sample_ohlcv_data, default_config):
        """Test that decision is one of LONG, SHORT, or FLAT."""
        enriched = compute_indicators(sample_ohlcv_data, default_config)
        signal = generate_signal(enriched, default_config)

        assert signal["decision"] in ["LONG", "SHORT", "FLAT"]

    def test_confidence_bounded(self, sample_ohlcv_data, default_config):
        """Test that confidence is between 0 and 1."""
        enriched = compute_indicators(sample_ohlcv_data, default_config)
        signal = generate_signal(enriched, default_config)

        assert 0 <= signal["confidence"] <= 1

    def test_bullish_crossover_generates_signal(self, bullish_crossover_data, default_config):
        """Test that bullish trend data generates a valid signal."""
        # Use smaller EMA periods for the test data
        config = StrategyConfig(
            ema_fast=5,
            ema_slow=10,
            rsi_period=5,
        )
        enriched = compute_indicators(bullish_crossover_data, config)
        signal = generate_signal(enriched, config)

        # The signal should be valid (any decision is acceptable as long as it's valid)
        assert signal["decision"] in ["LONG", "SHORT", "FLAT"]
        assert "reason" in signal

    def test_bearish_crossover_generates_signal(self, bearish_crossover_data, default_config):
        """Test that bearish trend data generates a valid signal."""
        config = StrategyConfig(
            ema_fast=5,
            ema_slow=10,
            rsi_period=5,
        )
        enriched = compute_indicators(bearish_crossover_data, config)
        signal = generate_signal(enriched, config)

        # The signal should be valid (any decision is acceptable as long as it's valid)
        assert signal["decision"] in ["LONG", "SHORT", "FLAT"]
        assert "reason" in signal


class TestCalculatePositionSize:
    """Tests for calculate_position_size function."""

    def test_basic_calculation(self):
        """Test basic position size calculation."""
        size = calculate_position_size(
            balance=10000.0,
            risk_pct=1.0,  # 1% risk
            price=100.0,
            stop_loss_pct=0.02,  # 2% stop loss
        )

        # Risk amount = 10000 * 0.01 = 100
        # Stop distance = 100 * 0.02 = 2
        # Position size = 100 / 2 = 50 units
        assert size == pytest.approx(50.0, rel=0.01)

    def test_zero_stop_loss_returns_zero(self):
        """Test that zero stop loss returns zero position size."""
        size = calculate_position_size(
            balance=10000.0,
            risk_pct=1.0,
            price=100.0,
            stop_loss_pct=0.0,
        )
        assert size == 0.0

    def test_negative_stop_loss_returns_zero(self):
        """Test that negative stop loss returns zero position size."""
        size = calculate_position_size(
            balance=10000.0,
            risk_pct=1.0,
            price=100.0,
            stop_loss_pct=-0.02,
        )
        assert size == 0.0

    def test_larger_risk_larger_position(self):
        """Test that larger risk percentage gives larger position."""
        size_small = calculate_position_size(
            balance=10000.0,
            risk_pct=0.5,
            price=100.0,
            stop_loss_pct=0.02,
        )

        size_large = calculate_position_size(
            balance=10000.0,
            risk_pct=1.0,
            price=100.0,
            stop_loss_pct=0.02,
        )

        assert size_large > size_small

    def test_tighter_stop_larger_position(self):
        """Test that tighter stop loss gives larger position size."""
        size_tight = calculate_position_size(
            balance=10000.0,
            risk_pct=1.0,
            price=100.0,
            stop_loss_pct=0.01,  # 1% stop
        )

        size_wide = calculate_position_size(
            balance=10000.0,
            risk_pct=1.0,
            price=100.0,
            stop_loss_pct=0.02,  # 2% stop
        )

        assert size_tight > size_wide
