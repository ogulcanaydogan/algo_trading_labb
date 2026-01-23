"""
Tests for strategy base classes.
"""

import pytest
import pandas as pd
from datetime import datetime

from bot.strategies.base import StrategySignal, StrategyConfig, BaseStrategy


class TestStrategySignal:
    """Test StrategySignal dataclass."""

    def test_signal_creation_long(self):
        """Test creating a LONG signal."""
        signal = StrategySignal(
            decision="LONG",
            confidence=0.85,
            reason="Strong bullish momentum",
            strategy_name="test_strategy",
        )
        assert signal.decision == "LONG"
        assert signal.confidence == 0.85
        assert signal.reason == "Strong bullish momentum"
        assert signal.strategy_name == "test_strategy"

    def test_signal_creation_short(self):
        """Test creating a SHORT signal."""
        signal = StrategySignal(
            decision="SHORT",
            confidence=0.75,
            reason="Bearish reversal",
            strategy_name="test_strategy",
        )
        assert signal.decision == "SHORT"
        assert signal.confidence == 0.75

    def test_signal_creation_flat(self):
        """Test creating a FLAT signal."""
        signal = StrategySignal(
            decision="FLAT",
            confidence=0.0,
            reason="No clear direction",
            strategy_name="test_strategy",
        )
        assert signal.decision == "FLAT"
        assert signal.confidence == 0.0

    def test_signal_with_prices(self):
        """Test signal with entry, stop, and take profit."""
        signal = StrategySignal(
            decision="LONG",
            confidence=0.8,
            reason="Test",
            strategy_name="test_strategy",
            entry_price=50000.0,
            stop_loss=49000.0,
            take_profit=52000.0,
        )
        assert signal.entry_price == 50000.0
        assert signal.stop_loss == 49000.0
        assert signal.take_profit == 52000.0

    def test_signal_with_indicators(self):
        """Test signal with indicator values."""
        indicators = {"rsi": 70.5, "macd": 0.02, "ema_20": 50500.0}
        signal = StrategySignal(
            decision="LONG",
            confidence=0.75,
            reason="RSI confirmation",
            strategy_name="test_strategy",
            indicators=indicators,
        )
        assert signal.indicators["rsi"] == 70.5
        assert signal.indicators["macd"] == 0.02

    def test_signal_to_dict(self):
        """Test conversion to dictionary."""
        signal = StrategySignal(
            decision="LONG",
            confidence=0.8,
            reason="Test signal",
            strategy_name="test_strategy",
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            indicators={"rsi": 65.0},
        )
        d = signal.to_dict()
        assert d["decision"] == "LONG"
        assert d["confidence"] == 0.8
        assert d["strategy_name"] == "test_strategy"
        assert d["entry_price"] == 100.0
        assert "timestamp" in d

    def test_signal_to_dict_rounds_values(self):
        """Test to_dict rounds values to 4 decimals."""
        signal = StrategySignal(
            decision="LONG",
            confidence=0.123456789,
            reason="Test",
            strategy_name="test",
            entry_price=100.123456789,
            indicators={"rsi": 65.123456789},
        )
        d = signal.to_dict()
        assert d["confidence"] == 0.1235
        assert d["entry_price"] == 100.1235
        assert d["indicators"]["rsi"] == 65.1235

    def test_signal_to_dict_handles_none_prices(self):
        """Test to_dict handles None values."""
        signal = StrategySignal(
            decision="FLAT",
            confidence=0.0,
            reason="No signal",
            strategy_name="test",
        )
        d = signal.to_dict()
        assert d["entry_price"] is None
        assert d["stop_loss"] is None
        assert d["take_profit"] is None


class TestStrategyConfig:
    """Test StrategyConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = StrategyConfig()
        assert config.symbol == "BTC/USDT"
        assert config.timeframe == "1h"
        assert config.risk_per_trade_pct == 0.5
        assert config.stop_loss_pct == 0.02
        assert config.take_profit_pct == 0.04
        assert config.min_confidence == 0.4

    def test_custom_config(self):
        """Test custom configuration values."""
        config = StrategyConfig(
            symbol="ETH/USDT",
            timeframe="4h",
            risk_per_trade_pct=1.0,
            stop_loss_pct=0.03,
            take_profit_pct=0.06,
            min_confidence=0.6,
        )
        assert config.symbol == "ETH/USDT"
        assert config.timeframe == "4h"
        assert config.risk_per_trade_pct == 1.0


class ConcreteStrategy(BaseStrategy):
    """Concrete implementation of BaseStrategy for testing."""

    @property
    def name(self) -> str:
        return "test_concrete_strategy"

    @property
    def description(self) -> str:
        return "A test strategy for unit testing"

    def generate_signal(self, ohlcv: pd.DataFrame) -> StrategySignal:
        """Generate a simple test signal."""
        if len(ohlcv) == 0:
            return self._flat_signal("No data")

        close = ohlcv["close"].iloc[-1]
        open_price = ohlcv["open"].iloc[-1]

        if close > open_price:
            return StrategySignal(
                decision="LONG",
                confidence=0.6,
                reason="Close > Open",
                strategy_name=self.name,
                entry_price=close,
                stop_loss=close * 0.98,
                take_profit=close * 1.02,
            )
        else:
            return StrategySignal(
                decision="SHORT",
                confidence=0.6,
                reason="Close < Open",
                strategy_name=self.name,
                entry_price=close,
                stop_loss=close * 1.02,
                take_profit=close * 0.98,
            )

    def get_required_indicators(self):
        return ["close", "open"]


class TestBaseStrategy:
    """Test BaseStrategy abstract class."""

    @pytest.fixture
    def strategy(self):
        """Create concrete strategy instance."""
        return ConcreteStrategy()

    @pytest.fixture
    def ohlcv(self):
        """Create sample OHLCV data."""
        data = {
            "open": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            "high": [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            "low": [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            "close": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            "volume": [1000] * 10,
        }
        return pd.DataFrame(data)

    def test_strategy_properties(self, strategy):
        """Test strategy name and description."""
        assert strategy.name == "test_concrete_strategy"
        assert strategy.description == "A test strategy for unit testing"

    def test_default_config(self, strategy):
        """Test default config is created."""
        assert strategy.config is not None
        assert isinstance(strategy.config, StrategyConfig)

    def test_custom_config(self):
        """Test strategy with custom config."""
        config = StrategyConfig(symbol="SOL/USDT", risk_per_trade_pct=2.0)
        strategy = ConcreteStrategy(config=config)
        assert strategy.config.symbol == "SOL/USDT"
        assert strategy.config.risk_per_trade_pct == 2.0

    def test_suitable_regimes_default(self, strategy):
        """Test default suitable regimes."""
        assert strategy.suitable_regimes == ["all"]

    def test_generate_signal_long(self, strategy, ohlcv):
        """Test generating a LONG signal."""
        signal = strategy.generate_signal(ohlcv)
        assert signal.decision == "LONG"  # Close > Open in our data
        assert signal.confidence > 0
        assert signal.strategy_name == strategy.name

    def test_generate_signal_short(self, strategy):
        """Test generating a SHORT signal."""
        data = {
            "open": [100, 101, 102, 103, 104, 105, 106, 107, 108, 110],  # Last open higher
            "high": [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            "low": [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            "close": [101, 102, 103, 104, 105, 106, 107, 108, 109, 108],  # Last close lower
            "volume": [1000] * 10,
        }
        ohlcv = pd.DataFrame(data)
        signal = strategy.generate_signal(ohlcv)
        assert signal.decision == "SHORT"  # Close < Open

    def test_generate_signal_flat_empty_data(self, strategy):
        """Test flat signal with empty data."""
        empty_ohlcv = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        signal = strategy.generate_signal(empty_ohlcv)
        assert signal.decision == "FLAT"
        assert signal.reason == "No data"

    def test_get_required_indicators(self, strategy):
        """Test getting required indicators."""
        indicators = strategy.get_required_indicators()
        assert isinstance(indicators, list)
        assert len(indicators) > 0

    def test_add_indicators_returns_copy(self, strategy, ohlcv):
        """Test add_indicators returns a copy."""
        result = strategy.add_indicators(ohlcv)
        assert result is not ohlcv  # Should be a copy

    def test_calculate_position_size(self, strategy):
        """Test position size calculation."""
        balance = 10000.0
        entry_price = 100.0
        stop_loss_price = 95.0  # 5% stop loss

        position_size = strategy.calculate_position_size(balance, entry_price, stop_loss_price)

        # Risk amount = 10000 * 0.005 = 50
        # Risk per unit = 100 - 95 = 5
        # Position size = 50 / 5 = 10
        expected_size = 10.0
        assert position_size == expected_size

    def test_calculate_position_size_zero_risk(self, strategy):
        """Test position size with zero risk per unit."""
        balance = 10000.0
        entry_price = 100.0
        stop_loss_price = 100.0  # Same as entry = zero risk

        position_size = strategy.calculate_position_size(balance, entry_price, stop_loss_price)
        assert position_size == 0.0

    def test_flat_signal_helper(self, strategy):
        """Test _flat_signal helper method."""
        signal = strategy._flat_signal("Custom reason")
        assert signal.decision == "FLAT"
        assert signal.confidence == 0.0
        assert signal.reason == "Custom reason"
        assert signal.strategy_name == strategy.name

    def test_flat_signal_default_reason(self, strategy):
        """Test _flat_signal with default reason."""
        signal = strategy._flat_signal()
        assert signal.reason == "No clear signal"
