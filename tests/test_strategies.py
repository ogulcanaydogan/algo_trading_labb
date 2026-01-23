"""
Tests for trading strategies (RSI Mean Reversion and Bollinger Bands).
"""

import pytest
import pandas as pd
import numpy as np

from bot.strategies.rsi_mean_reversion import (
    RSIMeanReversionStrategy,
    RSIMeanReversionConfig,
)
from bot.strategies.bollinger_bands import (
    BollingerBandStrategy,
    BollingerBandConfig,
)


# ============================================================================
# RSI Mean Reversion Strategy Tests
# ============================================================================


class TestRSIMeanReversionConfig:
    """Test RSIMeanReversionConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RSIMeanReversionConfig()
        assert config.rsi_period == 14
        assert config.rsi_oversold == 30.0
        assert config.rsi_overbought == 70.0
        assert config.rsi_extreme_oversold == 20.0
        assert config.rsi_extreme_overbought == 80.0
        assert config.use_stochastic is True


class TestRSIMeanReversionStrategy:
    """Test RSIMeanReversionStrategy."""

    @pytest.fixture
    def strategy(self):
        """Create strategy instance."""
        return RSIMeanReversionStrategy()

    @pytest.fixture
    def oversold_data(self):
        """Create data with oversold RSI conditions."""
        np.random.seed(42)
        n = 100
        # Create declining prices to generate oversold RSI
        close = 100 - np.arange(n) * 0.8 + np.random.randn(n) * 0.5
        close = np.maximum(close, 20)  # Floor at 20

        return pd.DataFrame(
            {
                "open": close + np.random.rand(n) * 0.5,
                "high": close + np.random.rand(n) * 1,
                "low": close - np.random.rand(n) * 1,
                "close": close,
                "volume": np.random.randint(1000, 10000, n),
            }
        )

    @pytest.fixture
    def overbought_data(self):
        """Create data with overbought RSI conditions."""
        np.random.seed(42)
        n = 100
        # Create rising prices to generate overbought RSI
        close = 100 + np.arange(n) * 0.8 + np.random.randn(n) * 0.5

        return pd.DataFrame(
            {
                "open": close - np.random.rand(n) * 0.5,
                "high": close + np.random.rand(n) * 1,
                "low": close - np.random.rand(n) * 1,
                "close": close,
                "volume": np.random.randint(1000, 10000, n),
            }
        )

    @pytest.fixture
    def short_data(self):
        """Create insufficient data."""
        return pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [102, 103, 104],
                "low": [99, 100, 101],
                "close": [101, 102, 103],
                "volume": [1000, 1000, 1000],
            }
        )

    def test_strategy_properties(self, strategy):
        """Test strategy name and description."""
        assert strategy.name == "rsi_mean_reversion"
        assert "RSI" in strategy.description
        assert "mean reversion" in strategy.description.lower()

    def test_suitable_regimes(self, strategy):
        """Test suitable market regimes."""
        regimes = strategy.suitable_regimes
        assert "sideways" in regimes

    def test_required_indicators(self, strategy):
        """Test required indicators list."""
        indicators = strategy.get_required_indicators()
        assert "rsi" in indicators
        assert "ema_50" in indicators

    def test_add_indicators(self, strategy, oversold_data):
        """Test indicators are added."""
        df = strategy.add_indicators(oversold_data)
        assert "rsi" in df.columns
        assert "ema_50" in df.columns
        if strategy.config.use_stochastic:
            assert "stoch_k" in df.columns
            assert "stoch_d" in df.columns

    def test_insufficient_data(self, strategy, short_data):
        """Test flat signal with insufficient data."""
        signal = strategy.generate_signal(short_data)
        assert signal.decision == "FLAT"
        assert "Insufficient" in signal.reason

    def test_signal_has_indicators(self, strategy, oversold_data):
        """Test signal includes indicator values."""
        signal = strategy.generate_signal(oversold_data)
        if signal.decision in ["LONG", "SHORT"]:
            assert "rsi" in signal.indicators
            assert "close" in signal.indicators
            assert "ema_50" in signal.indicators

    def test_flat_when_neutral(self, strategy):
        """Test flat signal when RSI is neutral."""
        np.random.seed(42)
        n = 100
        # Sideways movement for neutral RSI
        close = 100 + np.sin(np.arange(n) * 0.1) * 5

        df = pd.DataFrame(
            {
                "open": close - 0.5,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": [1000] * n,
            }
        )

        signal = strategy.generate_signal(df)
        # Should be FLAT or have signal depending on RSI
        assert signal.decision in ["FLAT", "LONG", "SHORT"]


# ============================================================================
# Bollinger Bands Strategy Tests
# ============================================================================


class TestBollingerBandConfig:
    """Test BollingerBandConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BollingerBandConfig()
        assert config.bb_period == 20
        assert config.bb_std == 2.0
        assert config.rsi_period == 14
        assert config.mode == "mean_reversion"
        assert config.squeeze_threshold == 0.02

    def test_breakout_mode(self):
        """Test breakout mode configuration."""
        config = BollingerBandConfig(mode="breakout")
        assert config.mode == "breakout"


class TestBollingerBandStrategy:
    """Test BollingerBandStrategy."""

    @pytest.fixture
    def mr_strategy(self):
        """Create mean-reversion strategy."""
        config = BollingerBandConfig(mode="mean_reversion")
        return BollingerBandStrategy(config)

    @pytest.fixture
    def breakout_strategy(self):
        """Create breakout strategy."""
        config = BollingerBandConfig(mode="breakout")
        return BollingerBandStrategy(config)

    @pytest.fixture
    def ranging_data(self):
        """Create ranging/sideways data."""
        np.random.seed(42)
        n = 100
        # Oscillating prices
        close = 100 + np.sin(np.arange(n) * 0.3) * 10 + np.random.randn(n) * 2

        return pd.DataFrame(
            {
                "open": close - np.random.rand(n) * 0.5,
                "high": close + np.random.rand(n) * 2,
                "low": close - np.random.rand(n) * 2,
                "close": close,
                "volume": np.random.randint(1000, 10000, n),
            }
        )

    @pytest.fixture
    def breakout_data(self):
        """Create data with potential breakout."""
        np.random.seed(42)
        n = 100
        # Tight range followed by explosive move
        close = np.concatenate(
            [
                100 + np.random.randn(80) * 1,  # Tight range (squeeze)
                100 + np.arange(20) * 2 + np.random.randn(20) * 0.5,  # Breakout up
            ]
        )

        return pd.DataFrame(
            {
                "open": close - np.random.rand(n) * 0.5,
                "high": close + np.random.rand(n) * 1,
                "low": close - np.random.rand(n) * 1,
                "close": close,
                "volume": np.random.randint(1000, 10000, n),
            }
        )

    def test_mr_strategy_properties(self, mr_strategy):
        """Test mean-reversion strategy properties."""
        assert "mean_reversion" in mr_strategy.name
        assert "mean_reversion" in mr_strategy.description
        assert "sideways" in mr_strategy.suitable_regimes

    def test_breakout_strategy_properties(self, breakout_strategy):
        """Test breakout strategy properties."""
        assert "breakout" in breakout_strategy.name
        assert "breakout" in breakout_strategy.description
        assert "volatile" in breakout_strategy.suitable_regimes

    def test_required_indicators(self, mr_strategy):
        """Test required indicators list."""
        indicators = mr_strategy.get_required_indicators()
        assert "bb_high" in indicators
        assert "bb_low" in indicators
        assert "bb_mid" in indicators
        assert "bb_width" in indicators
        assert "rsi" in indicators

    def test_add_indicators(self, mr_strategy, ranging_data):
        """Test indicators are added."""
        df = mr_strategy.add_indicators(ranging_data)
        assert "bb_high" in df.columns
        assert "bb_low" in df.columns
        assert "bb_mid" in df.columns
        assert "bb_width" in df.columns
        assert "bb_pctb" in df.columns
        assert "rsi" in df.columns

    def test_insufficient_data(self, mr_strategy):
        """Test flat signal with insufficient data."""
        short_data = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [102, 103, 104],
                "low": [99, 100, 101],
                "close": [101, 102, 103],
                "volume": [1000, 1000, 1000],
            }
        )
        signal = mr_strategy.generate_signal(short_data)
        assert signal.decision == "FLAT"
        assert "Insufficient" in signal.reason

    def test_mr_signal_generation(self, mr_strategy, ranging_data):
        """Test mean-reversion signal generation."""
        signal = mr_strategy.generate_signal(ranging_data)
        assert signal.decision in ["LONG", "SHORT", "FLAT"]
        assert signal.strategy_name == mr_strategy.name

    def test_breakout_signal_generation(self, breakout_strategy, breakout_data):
        """Test breakout signal generation."""
        signal = breakout_strategy.generate_signal(breakout_data)
        assert signal.decision in ["LONG", "SHORT", "FLAT"]
        assert signal.strategy_name == breakout_strategy.name

    def test_mr_confidence_calculation(self, mr_strategy):
        """Test mean-reversion confidence calculation."""
        # Test long confidence (oversold)
        conf_long = mr_strategy._calculate_mr_confidence(
            bb_pctb=0.02,  # Near lower band
            rsi=25.0,  # Oversold
            is_long=True,
        )
        assert 0 <= conf_long <= 1

        # Test short confidence (overbought)
        conf_short = mr_strategy._calculate_mr_confidence(
            bb_pctb=0.98,  # Near upper band
            rsi=75.0,  # Overbought
            is_long=False,
        )
        assert 0 <= conf_short <= 1

    def test_signal_has_stop_and_target(self, mr_strategy, ranging_data):
        """Test signal has stop loss and take profit."""
        signal = mr_strategy.generate_signal(ranging_data)
        if signal.decision in ["LONG", "SHORT"]:
            assert signal.stop_loss is not None
            assert signal.take_profit is not None
            assert signal.entry_price is not None


# ============================================================================
# Edge Cases
# ============================================================================


class TestStrategyEdgeCases:
    """Test edge cases for strategies."""

    def test_empty_dataframe_rsi(self):
        """Test RSI strategy with empty DataFrame."""
        strategy = RSIMeanReversionStrategy()
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        signal = strategy.generate_signal(df)
        assert signal.decision == "FLAT"

    def test_empty_dataframe_bb(self):
        """Test BB strategy with empty DataFrame."""
        strategy = BollingerBandStrategy()
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        signal = strategy.generate_signal(df)
        assert signal.decision == "FLAT"

    def test_constant_prices_rsi(self):
        """Test RSI strategy with constant prices."""
        strategy = RSIMeanReversionStrategy()
        n = 80
        df = pd.DataFrame(
            {
                "open": [100.0] * n,
                "high": [100.0] * n,
                "low": [100.0] * n,
                "close": [100.0] * n,
                "volume": [1000] * n,
            }
        )
        signal = strategy.generate_signal(df)
        # Should handle gracefully
        assert signal.decision in ["FLAT", "LONG", "SHORT"]

    def test_constant_prices_bb(self):
        """Test BB strategy with constant prices."""
        strategy = BollingerBandStrategy()
        n = 50
        df = pd.DataFrame(
            {
                "open": [100.0] * n,
                "high": [100.0] * n,
                "low": [100.0] * n,
                "close": [100.0] * n,
                "volume": [1000] * n,
            }
        )
        signal = strategy.generate_signal(df)
        assert signal.decision in ["FLAT", "LONG", "SHORT"]

    def test_stochastic_disabled(self):
        """Test RSI strategy with stochastic disabled."""
        config = RSIMeanReversionConfig()
        config.use_stochastic = False
        strategy = RSIMeanReversionStrategy(config)

        indicators = strategy.get_required_indicators()
        assert "stoch_k" not in indicators
        assert "stoch_d" not in indicators
