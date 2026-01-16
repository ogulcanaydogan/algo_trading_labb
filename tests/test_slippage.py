"""
Tests for slippage module.
"""

import pytest
import tempfile
import shutil
from datetime import datetime
from pathlib import Path

from bot.slippage import (
    MarketCondition,
    OrderType,
    SlippageConfig,
    SlippageEstimate,
    ExecutionRecord,
    SlippageModel,
)


class TestMarketCondition:
    """Test MarketCondition enum."""

    def test_all_conditions_exist(self):
        """Test all market conditions exist."""
        assert MarketCondition.VERY_LIQUID.value == "very_liquid"
        assert MarketCondition.LIQUID.value == "liquid"
        assert MarketCondition.MODERATE.value == "moderate"
        assert MarketCondition.ILLIQUID.value == "illiquid"
        assert MarketCondition.STRESSED.value == "stressed"


class TestOrderType:
    """Test OrderType enum."""

    def test_all_order_types_exist(self):
        """Test all order types exist."""
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"
        assert OrderType.STOP_MARKET.value == "stop_market"
        assert OrderType.STOP_LIMIT.value == "stop_limit"


class TestSlippageConfig:
    """Test SlippageConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = SlippageConfig()
        assert config.base_slippage_bps == 5.0
        assert config.min_slippage_bps == 1.0
        assert config.max_slippage_bps == 50.0
        assert config.volume_impact_coefficient == 0.1
        assert config.volatility_multiplier == 2.0

    def test_default_order_type_multipliers(self):
        """Test default order type multipliers are set."""
        config = SlippageConfig()
        assert OrderType.MARKET in config.order_type_multipliers
        assert OrderType.LIMIT in config.order_type_multipliers
        assert config.order_type_multipliers[OrderType.MARKET] == 1.0
        assert config.order_type_multipliers[OrderType.LIMIT] == 0.3

    def test_default_condition_multipliers(self):
        """Test default condition multipliers are set."""
        config = SlippageConfig()
        assert MarketCondition.VERY_LIQUID in config.condition_multipliers
        assert config.condition_multipliers[MarketCondition.VERY_LIQUID] == 0.5
        assert config.condition_multipliers[MarketCondition.ILLIQUID] == 2.5

    def test_custom_config(self):
        """Test custom configuration."""
        config = SlippageConfig(
            base_slippage_bps=10.0,
            max_slippage_bps=100.0,
            volatility_multiplier=3.0,
        )
        assert config.base_slippage_bps == 10.0
        assert config.max_slippage_bps == 100.0
        assert config.volatility_multiplier == 3.0

    def test_time_adjustment_settings(self):
        """Test time adjustment settings."""
        config = SlippageConfig(
            use_time_adjustments=False,
            opening_multiplier=2.0,
        )
        assert config.use_time_adjustments is False
        assert config.opening_multiplier == 2.0


class TestSlippageEstimate:
    """Test SlippageEstimate dataclass."""

    def test_estimate_creation(self):
        """Test creating a slippage estimate."""
        estimate = SlippageEstimate(
            expected_slippage_bps=10.0,
            expected_slippage_pct=0.001,
            expected_slippage_price=50.0,
            price_after_slippage=50050.0,
            confidence=0.8,
        )
        assert estimate.expected_slippage_bps == 10.0
        assert estimate.expected_slippage_pct == 0.001
        assert estimate.confidence == 0.8
        assert estimate.components == {}
        assert estimate.warnings == []

    def test_estimate_with_components(self):
        """Test estimate with components."""
        estimate = SlippageEstimate(
            expected_slippage_bps=15.0,
            expected_slippage_pct=0.0015,
            expected_slippage_price=75.0,
            price_after_slippage=50075.0,
            confidence=0.7,
            components={"base": 5.0, "volume_impact": 10.0},
            warnings=["Large order size"],
        )
        assert estimate.components["base"] == 5.0
        assert len(estimate.warnings) == 1

    def test_to_dict(self):
        """Test conversion to dict."""
        estimate = SlippageEstimate(
            expected_slippage_bps=12.5,
            expected_slippage_pct=0.00125,
            expected_slippage_price=62.5,
            price_after_slippage=50062.5,
            confidence=0.85,
            components={"base": 5.0, "spread": 7.5},
        )
        d = estimate.to_dict()

        assert d["expected_slippage_bps"] == 12.5
        assert d["confidence"] == 0.85
        assert "base" in d["components"]


class TestExecutionRecord:
    """Test ExecutionRecord dataclass."""

    def test_record_creation(self):
        """Test creating an execution record."""
        now = datetime.now()
        record = ExecutionRecord(
            timestamp=now,
            symbol="BTC/USDT",
            side="buy",
            order_type=OrderType.MARKET,
            expected_price=50000.0,
            executed_price=50010.0,
            size=1000.0,
            volume_at_time=1000000.0,
            spread_at_time=5.0,
            volatility=30.0,
            actual_slippage_bps=2.0,
            estimated_slippage_bps=5.0,
            estimation_error=-3.0,
        )
        assert record.symbol == "BTC/USDT"
        assert record.expected_price == 50000.0
        assert record.executed_price == 50010.0
        assert record.estimation_error == -3.0

    def test_to_dict(self):
        """Test conversion to dict."""
        now = datetime.now()
        record = ExecutionRecord(
            timestamp=now,
            symbol="ETH/USDT",
            side="sell",
            order_type=OrderType.LIMIT,
            expected_price=3000.0,
            executed_price=2995.0,
            size=500.0,
            volume_at_time=500000.0,
            spread_at_time=3.0,
            volatility=25.0,
            actual_slippage_bps=-16.67,
            estimated_slippage_bps=5.0,
            estimation_error=-21.67,
        )
        d = record.to_dict()

        assert d["symbol"] == "ETH/USDT"
        assert d["side"] == "sell"
        assert d["order_type"] == "limit"
        assert d["expected_price"] == 3000.0


class TestSlippageModel:
    """Test SlippageModel class."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory."""
        temp_dir = tempfile.mkdtemp()
        yield f"{temp_dir}/slippage"
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def model(self, temp_data_dir):
        """Create slippage model."""
        return SlippageModel(data_dir=temp_data_dir)

    def test_model_creation(self, model):
        """Test model is created."""
        assert model is not None
        assert model.config is not None

    def test_model_with_custom_config(self, temp_data_dir):
        """Test model with custom config."""
        config = SlippageConfig(base_slippage_bps=10.0)
        model = SlippageModel(config=config, data_dir=temp_data_dir)

        assert model.config.base_slippage_bps == 10.0

    def test_estimate_slippage_basic(self, model):
        """Test basic slippage estimation."""
        estimate = model.estimate_slippage(
            symbol="BTC/USDT",
            price=50000.0,
            size=1000.0,
            side="buy",
        )

        assert estimate is not None
        assert estimate.expected_slippage_bps > 0
        assert estimate.price_after_slippage > 50000.0  # Buy adds slippage
        assert estimate.confidence > 0

    def test_estimate_slippage_sell(self, model):
        """Test slippage estimation for sell."""
        estimate = model.estimate_slippage(
            symbol="BTC/USDT",
            price=50000.0,
            size=1000.0,
            side="sell",
        )

        assert estimate.price_after_slippage < 50000.0  # Sell subtracts

    def test_estimate_slippage_with_order_type(self, model):
        """Test slippage with different order types."""
        market_estimate = model.estimate_slippage(
            symbol="BTC/USDT",
            price=50000.0,
            size=1000.0,
            side="buy",
            order_type=OrderType.MARKET,
        )

        limit_estimate = model.estimate_slippage(
            symbol="BTC/USDT",
            price=50000.0,
            size=1000.0,
            side="buy",
            order_type=OrderType.LIMIT,
        )

        # Limit orders should have lower slippage
        assert limit_estimate.expected_slippage_bps < market_estimate.expected_slippage_bps

    def test_estimate_slippage_with_market_condition(self, model):
        """Test slippage with market conditions."""
        liquid_estimate = model.estimate_slippage(
            symbol="BTC/USDT",
            price=50000.0,
            size=1000.0,
            side="buy",
            market_condition=MarketCondition.VERY_LIQUID,
        )

        illiquid_estimate = model.estimate_slippage(
            symbol="BTC/USDT",
            price=50000.0,
            size=1000.0,
            side="buy",
            market_condition=MarketCondition.ILLIQUID,
        )

        # Illiquid markets have higher slippage
        assert illiquid_estimate.expected_slippage_bps > liquid_estimate.expected_slippage_bps

    def test_estimate_slippage_with_volatility(self, model):
        """Test slippage with volatility."""
        low_vol_estimate = model.estimate_slippage(
            symbol="BTC/USDT",
            price=50000.0,
            size=1000.0,
            side="buy",
            volatility=10.0,  # Low volatility
        )

        high_vol_estimate = model.estimate_slippage(
            symbol="BTC/USDT",
            price=50000.0,
            size=1000.0,
            side="buy",
            volatility=50.0,  # High volatility
        )

        # Higher volatility = higher slippage
        assert high_vol_estimate.expected_slippage_bps >= low_vol_estimate.expected_slippage_bps

    def test_estimate_slippage_with_spread(self, model):
        """Test slippage with spread."""
        no_spread_estimate = model.estimate_slippage(
            symbol="BTC/USDT",
            price=50000.0,
            size=1000.0,
            side="buy",
        )

        spread_estimate = model.estimate_slippage(
            symbol="BTC/USDT",
            price=50000.0,
            size=1000.0,
            side="buy",
            spread_bps=10.0,
        )

        # Spread adds to slippage
        assert spread_estimate.expected_slippage_bps > no_spread_estimate.expected_slippage_bps

    def test_estimate_slippage_time_adjustments(self, model):
        """Test slippage with time of day adjustments."""
        # Normal hours
        normal_estimate = model.estimate_slippage(
            symbol="BTC/USDT",
            price=50000.0,
            size=1000.0,
            side="buy",
            hour_of_day=12,
        )

        # Market open
        open_estimate = model.estimate_slippage(
            symbol="BTC/USDT",
            price=50000.0,
            size=1000.0,
            side="buy",
            hour_of_day=9,
        )

        # Opening hours have higher slippage
        assert open_estimate.expected_slippage_bps > normal_estimate.expected_slippage_bps

    def test_estimate_slippage_respects_limits(self, temp_data_dir):
        """Test slippage respects min/max limits."""
        config = SlippageConfig(
            min_slippage_bps=2.0,
            max_slippage_bps=30.0,
        )
        model = SlippageModel(config=config, data_dir=temp_data_dir)

        estimate = model.estimate_slippage(
            symbol="BTC/USDT",
            price=50000.0,
            size=1000.0,
            side="buy",
        )

        assert estimate.expected_slippage_bps >= 2.0
        assert estimate.expected_slippage_bps <= 30.0

    def test_estimate_slippage_components_tracked(self, model):
        """Test that estimate components are tracked."""
        estimate = model.estimate_slippage(
            symbol="BTC/USDT",
            price=50000.0,
            size=1000.0,
            side="buy",
            volatility=30.0,
            market_condition=MarketCondition.MODERATE,
            hour_of_day=10,
        )

        assert "base" in estimate.components
        assert "volatility_mult" in estimate.components
        assert "condition_mult" in estimate.components


class TestSlippageModelRecordExecution:
    """Test execution recording functionality."""

    @pytest.fixture
    def temp_data_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield f"{temp_dir}/slippage"
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def model(self, temp_data_dir):
        return SlippageModel(data_dir=temp_data_dir)

    def test_record_execution_basic(self, model):
        """Test recording an execution."""
        model.record_execution(
            symbol="BTC/USDT",
            side="buy",
            order_type=OrderType.MARKET,
            expected_price=50000.0,
            executed_price=50010.0,
            size=1000.0,
        )

        assert "BTC/USDT" in model._execution_history
        assert len(model._execution_history["BTC/USDT"]) == 1

    def test_record_execution_calculates_slippage(self, model):
        """Test slippage is calculated correctly."""
        model.record_execution(
            symbol="BTC/USDT",
            side="buy",
            order_type=OrderType.MARKET,
            expected_price=50000.0,
            executed_price=50010.0,  # 2 bps slippage
            size=1000.0,
        )

        record = model._execution_history["BTC/USDT"][0]
        assert record.actual_slippage_bps == pytest.approx(2.0, rel=0.01)

    def test_record_execution_sell_slippage(self, model):
        """Test slippage calculation for sell."""
        model.record_execution(
            symbol="BTC/USDT",
            side="sell",
            order_type=OrderType.MARKET,
            expected_price=50000.0,
            executed_price=49990.0,  # 2 bps slippage (worse than expected)
            size=1000.0,
        )

        record = model._execution_history["BTC/USDT"][0]
        assert record.actual_slippage_bps == pytest.approx(2.0, rel=0.01)

    def test_record_execution_with_estimate(self, model):
        """Test recording with prior estimate."""
        estimate = model.estimate_slippage(
            symbol="BTC/USDT",
            price=50000.0,
            size=1000.0,
            side="buy",
        )

        model.record_execution(
            symbol="BTC/USDT",
            side="buy",
            order_type=OrderType.MARKET,
            expected_price=50000.0,
            executed_price=50010.0,
            size=1000.0,
            estimated_slippage=estimate,
        )

        record = model._execution_history["BTC/USDT"][0]
        assert record.estimated_slippage_bps == estimate.expected_slippage_bps

    def test_multiple_executions(self, model):
        """Test multiple executions are stored."""
        for i in range(5):
            model.record_execution(
                symbol="BTC/USDT",
                side="buy",
                order_type=OrderType.MARKET,
                expected_price=50000.0,
                executed_price=50000.0 + i * 10,
                size=1000.0,
            )

        assert len(model._execution_history["BTC/USDT"]) == 5

    def test_multiple_symbols(self, model):
        """Test executions for multiple symbols."""
        model.record_execution(
            symbol="BTC/USDT",
            side="buy",
            order_type=OrderType.MARKET,
            expected_price=50000.0,
            executed_price=50010.0,
            size=1000.0,
        )
        model.record_execution(
            symbol="ETH/USDT",
            side="buy",
            order_type=OrderType.MARKET,
            expected_price=3000.0,
            executed_price=3001.0,
            size=1000.0,
        )

        assert "BTC/USDT" in model._execution_history
        assert "ETH/USDT" in model._execution_history


class TestSlippageModelAdaptive:
    """Test adaptive parameter learning."""

    @pytest.fixture
    def temp_data_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield f"{temp_dir}/slippage"
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def model(self, temp_data_dir):
        return SlippageModel(data_dir=temp_data_dir)

    def test_adaptive_base_default(self, model):
        """Test adaptive base uses default without history."""
        base = model._get_adaptive_base("BTC/USDT")
        assert base == model.config.base_slippage_bps

    def test_adaptive_vol_mult_default(self, model):
        """Test adaptive vol multiplier uses default without history."""
        mult = model._get_adaptive_vol_mult("BTC/USDT")
        assert mult == model.config.volatility_multiplier


class TestSlippageModelPersistence:
    """Test state persistence."""

    @pytest.fixture
    def temp_data_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield f"{temp_dir}/slippage"
        shutil.rmtree(temp_dir)

    def test_data_dir_created(self, temp_data_dir):
        """Test data directory is created."""
        model = SlippageModel(data_dir=temp_data_dir)
        assert Path(temp_data_dir).exists()

    def test_state_persistence(self, temp_data_dir):
        """Test state is persisted and can be loaded."""
        # Create model and set some adaptive parameters
        model1 = SlippageModel(data_dir=temp_data_dir)
        model1._adv["BTC/USDT"] = 1000000.0
        model1._adaptive_base["BTC/USDT"] = 7.5
        model1._save_state()

        # Create new model - should load state
        model2 = SlippageModel(data_dir=temp_data_dir)

        # Adaptive parameters should be loaded
        assert "BTC/USDT" in model2._adv
        assert model2._adv["BTC/USDT"] == 1000000.0
        assert model2._adaptive_base.get("BTC/USDT") == 7.5
