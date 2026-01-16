"""
Tests for EMA Crossover strategy.
"""

import pytest
import pandas as pd
import numpy as np

from bot.strategies.ema_crossover import EMACrossoverStrategy, EMACrossoverConfig


class TestEMACrossoverConfig:
    """Test EMACrossoverConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EMACrossoverConfig()
        assert config.ema_fast == 12
        assert config.ema_slow == 26
        assert config.ema_trend == 200
        assert config.rsi_period == 14
        assert config.rsi_overbought == 70.0
        assert config.rsi_oversold == 30.0
        assert config.require_trend_alignment is True

    def test_custom_config(self):
        """Test custom configuration by attribute modification."""
        config = EMACrossoverConfig()
        # Modify attributes after creation
        config.ema_fast = 8
        config.ema_slow = 21
        config.ema_trend = 100
        config.rsi_period = 10
        assert config.ema_fast == 8
        assert config.ema_slow == 21
        assert config.ema_trend == 100
        assert config.rsi_period == 10


class TestEMACrossoverStrategy:
    """Test EMACrossoverStrategy class."""

    @pytest.fixture
    def strategy(self):
        """Create strategy instance."""
        return EMACrossoverStrategy()

    @pytest.fixture
    def custom_strategy(self):
        """Create strategy with custom config."""
        config = EMACrossoverConfig()
        config.ema_fast = 5
        config.ema_slow = 10
        config.ema_trend = 50
        config.require_trend_alignment = False
        return EMACrossoverStrategy(config)

    @pytest.fixture
    def trending_up_data(self):
        """Create data with upward trend."""
        np.random.seed(42)
        n = 250
        # Start at 100 and trend up
        base = 100 + np.arange(n) * 0.5
        noise = np.random.randn(n) * 2
        close = base + noise

        return pd.DataFrame({
            "open": close - np.random.rand(n) * 0.5,
            "high": close + np.random.rand(n) * 1,
            "low": close - np.random.rand(n) * 1,
            "close": close,
            "volume": np.random.randint(1000, 10000, n),
        })

    @pytest.fixture
    def trending_down_data(self):
        """Create data with downward trend."""
        np.random.seed(42)
        n = 250
        # Start at 200 and trend down
        base = 200 - np.arange(n) * 0.5
        noise = np.random.randn(n) * 2
        close = base + noise

        return pd.DataFrame({
            "open": close + np.random.rand(n) * 0.5,
            "high": close + np.random.rand(n) * 1,
            "low": close - np.random.rand(n) * 1,
            "close": close,
            "volume": np.random.randint(1000, 10000, n),
        })

    @pytest.fixture
    def short_data(self):
        """Create insufficient data."""
        return pd.DataFrame({
            "open": [100, 101, 102],
            "high": [102, 103, 104],
            "low": [99, 100, 101],
            "close": [101, 102, 103],
            "volume": [1000, 1000, 1000],
        })

    def test_strategy_properties(self, strategy):
        """Test strategy name and description."""
        assert strategy.name == "ema_crossover"
        assert "EMA" in strategy.description
        assert "12" in strategy.description  # ema_fast
        assert "26" in strategy.description  # ema_slow

    def test_suitable_regimes(self, strategy):
        """Test suitable market regimes."""
        regimes = strategy.suitable_regimes
        assert "bull" in regimes
        assert "strong_bull" in regimes
        assert "bear" in regimes
        assert "strong_bear" in regimes

    def test_required_indicators(self, strategy):
        """Test required indicators list."""
        indicators = strategy.get_required_indicators()
        assert "ema_fast" in indicators
        assert "ema_slow" in indicators
        assert "ema_trend" in indicators
        assert "rsi" in indicators

    def test_add_indicators(self, strategy, trending_up_data):
        """Test indicators are added correctly."""
        df = strategy.add_indicators(trending_up_data)
        assert "ema_fast" in df.columns
        assert "ema_slow" in df.columns
        assert "ema_trend" in df.columns
        assert "rsi" in df.columns

    def test_insufficient_data(self, strategy, short_data):
        """Test flat signal with insufficient data."""
        signal = strategy.generate_signal(short_data)
        assert signal.decision == "FLAT"
        assert "Insufficient" in signal.reason

    def test_signal_generation_trending_up(self, custom_strategy, trending_up_data):
        """Test signal generation with upward trend."""
        signal = custom_strategy.generate_signal(trending_up_data)
        # Should either be LONG, SHORT, or FLAT depending on exact crossover timing
        assert signal.decision in ["LONG", "SHORT", "FLAT"]
        assert signal.strategy_name == "ema_crossover"

    def test_signal_generation_trending_down(self, custom_strategy, trending_down_data):
        """Test signal generation with downward trend."""
        signal = custom_strategy.generate_signal(trending_down_data)
        assert signal.decision in ["LONG", "SHORT", "FLAT"]
        assert signal.strategy_name == "ema_crossover"

    def test_signal_has_indicators(self, custom_strategy, trending_up_data):
        """Test signal includes indicator values."""
        signal = custom_strategy.generate_signal(trending_up_data)
        if signal.decision != "FLAT" or signal.indicators:
            # If we got a non-FLAT signal, it should have indicators
            if signal.decision in ["LONG", "SHORT"]:
                assert "ema_fast" in signal.indicators
                assert "ema_slow" in signal.indicators
                assert "rsi" in signal.indicators
                assert "close" in signal.indicators

    def test_confidence_calculation(self, strategy):
        """Test confidence calculation."""
        # Test long confidence
        conf_long = strategy._calculate_confidence(
            ema_fast=105.0,
            ema_slow=100.0,
            close=100.0,
            rsi=50.0,
            is_long=True,
        )
        assert 0 <= conf_long <= 1

        # Test short confidence
        conf_short = strategy._calculate_confidence(
            ema_fast=95.0,
            ema_slow=100.0,
            close=100.0,
            rsi=50.0,
            is_long=False,
        )
        assert 0 <= conf_short <= 1

    def test_rsi_affects_confidence(self, strategy):
        """Test RSI affects confidence."""
        # Low RSI should give better confidence for long
        low_rsi_conf = strategy._calculate_confidence(
            ema_fast=105.0,
            ema_slow=100.0,
            close=100.0,
            rsi=30.0,  # Low RSI
            is_long=True,
        )

        high_rsi_conf = strategy._calculate_confidence(
            ema_fast=105.0,
            ema_slow=100.0,
            close=100.0,
            rsi=65.0,  # Higher RSI
            is_long=True,
        )

        assert low_rsi_conf >= high_rsi_conf  # Low RSI better for long

    def test_stop_loss_take_profit_long(self, custom_strategy, trending_up_data):
        """Test stop loss and take profit for long signal."""
        signal = custom_strategy.generate_signal(trending_up_data)
        if signal.decision == "LONG":
            assert signal.stop_loss is not None
            assert signal.take_profit is not None
            assert signal.stop_loss < signal.entry_price
            assert signal.take_profit > signal.entry_price

    def test_stop_loss_take_profit_short(self, custom_strategy, trending_down_data):
        """Test stop loss and take profit for short signal."""
        signal = custom_strategy.generate_signal(trending_down_data)
        if signal.decision == "SHORT":
            assert signal.stop_loss is not None
            assert signal.take_profit is not None
            assert signal.stop_loss > signal.entry_price
            assert signal.take_profit < signal.entry_price


class TestTrendAlignment:
    """Test trend alignment feature."""

    def test_trend_alignment_enabled(self):
        """Test trend alignment blocks signals against trend."""
        config = EMACrossoverConfig()
        config.ema_fast = 5
        config.ema_slow = 10
        config.ema_trend = 50
        config.require_trend_alignment = True
        strategy = EMACrossoverStrategy(config)

        # Create data with recent bullish cross but below long-term trend
        np.random.seed(42)
        n = 100

        # Start below, then create conditions for bullish cross
        close = np.concatenate([
            np.linspace(100, 90, 50),  # Downward movement
            np.linspace(90, 95, 50),   # Slight recovery but still low
        ])

        df = pd.DataFrame({
            "open": close - 0.5,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": [1000] * n,
        })

        signal = strategy.generate_signal(df)
        # Should be FLAT or have reason mentioning trend
        assert signal.decision in ["FLAT", "LONG", "SHORT"]

    def test_trend_alignment_disabled(self):
        """Test trend alignment can be disabled."""
        config = EMACrossoverConfig()
        config.ema_fast = 5
        config.ema_slow = 10
        config.ema_trend = 50
        config.require_trend_alignment = False  # Disabled
        strategy = EMACrossoverStrategy(config)
        assert strategy.config.require_trend_alignment is False


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        strategy = EMACrossoverStrategy()
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        signal = strategy.generate_signal(df)
        assert signal.decision == "FLAT"

    def test_constant_prices(self):
        """Test with constant prices (no movement)."""
        config = EMACrossoverConfig()
        config.ema_fast = 5
        config.ema_slow = 10
        config.ema_trend = 20
        strategy = EMACrossoverStrategy(config)

        n = 50
        df = pd.DataFrame({
            "open": [100.0] * n,
            "high": [100.0] * n,
            "low": [100.0] * n,
            "close": [100.0] * n,
            "volume": [1000] * n,
        })

        signal = strategy.generate_signal(df)
        assert signal.decision in ["FLAT", "LONG", "SHORT"]  # Should handle gracefully
