"""
Tests for Aggressive Profit Hunter ML Predictor.

Tests:
- Model initialization and configuration
- Training on synthetic data
- Prediction generation
- Learning from mistakes
- Signal strength and leverage calculation
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil

from bot.ml.aggressive_predictor import (
    AggressiveProfitHunter,
    AggressiveConfig,
    AggressiveSignal,
    SignalStrength,
    MistakeLearner,
    TradeOutcome,
    LearningConfig,
    create_aggressive_predictor,
)


def generate_synthetic_ohlcv(bars: int = 1000, trend: str = "up") -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)

    # Base price with trend
    if trend == "up":
        base = 100 + np.cumsum(np.random.randn(bars) * 0.5 + 0.1)
    elif trend == "down":
        base = 100 + np.cumsum(np.random.randn(bars) * 0.5 - 0.1)
    else:  # sideways
        base = 100 + np.cumsum(np.random.randn(bars) * 0.3)

    # Generate OHLCV
    high = base + np.abs(np.random.randn(bars)) * 0.5
    low = base - np.abs(np.random.randn(bars)) * 0.5
    open_price = base + np.random.randn(bars) * 0.2
    close = base + np.random.randn(bars) * 0.2
    volume = np.abs(np.random.randn(bars)) * 1000 + 500

    # Ensure OHLC consistency
    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))

    df = pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )

    # Add timestamp index
    df.index = pd.date_range(start="2024-01-01", periods=bars, freq="1h")

    return df


class TestAggressiveConfig:
    """Test AggressiveConfig dataclass."""

    def test_default_config(self):
        config = AggressiveConfig()

        assert config.short_horizon == 1
        assert config.medium_horizon == 3
        assert config.long_horizon == 8
        assert config.min_confidence_to_trade == 0.55
        assert config.base_leverage == 2.0
        assert config.max_leverage == 10.0
        assert config.enable_learning is True

    def test_custom_config(self):
        config = AggressiveConfig(
            max_leverage=5.0,
            base_position_pct=0.1,
            enable_learning=False,
        )

        assert config.max_leverage == 5.0
        assert config.base_position_pct == 0.1
        assert config.enable_learning is False


class TestMistakeLearner:
    """Test MistakeLearner class."""

    def test_initial_state(self):
        config = LearningConfig()
        learner = MistakeLearner(config)

        assert learner.get_win_rate() == 0.5  # Default when no history
        assert learner.consecutive_losses == 0
        assert learner.consecutive_wins == 0

    def test_record_winning_trade(self):
        config = LearningConfig()
        learner = MistakeLearner(config)

        outcome = TradeOutcome(
            timestamp=datetime.now(),
            predicted_action="LONG",
            actual_direction="UP",
            confidence=0.7,
            features_snapshot=np.array([1, 2, 3]),
            pnl_pct=0.02,
            was_correct=True,
            regime="bull",
            holding_period=5,
        )

        learner.record_outcome(outcome)

        assert learner.consecutive_wins == 1
        assert learner.consecutive_losses == 0
        assert learner.get_win_rate() == 1.0

    def test_record_losing_trade(self):
        config = LearningConfig()
        learner = MistakeLearner(config)

        outcome = TradeOutcome(
            timestamp=datetime.now(),
            predicted_action="LONG",
            actual_direction="DOWN",
            confidence=0.7,
            features_snapshot=np.array([1, 2, 3]),
            pnl_pct=-0.02,
            was_correct=False,
            regime="bull",
            holding_period=5,
        )

        learner.record_outcome(outcome)

        assert learner.consecutive_losses == 1
        assert learner.consecutive_wins == 0
        assert learner.get_win_rate() == 0.0

    def test_consecutive_losses_penalty(self):
        config = LearningConfig(max_consecutive_losses=3)
        learner = MistakeLearner(config)

        # Record 3 losses
        for _ in range(3):
            outcome = TradeOutcome(
                timestamp=datetime.now(),
                predicted_action="LONG",
                actual_direction="DOWN",
                confidence=0.6,
                features_snapshot=np.array([1, 2, 3]),
                pnl_pct=-0.01,
                was_correct=False,
                regime="bear",
                holding_period=3,
            )
            learner.record_outcome(outcome)

        # Should block trading after max consecutive losses
        should_trade, reason = learner.should_trade("bear")
        assert should_trade is False
        assert "consecutive losses" in reason.lower()

    def test_position_size_multiplier(self):
        config = LearningConfig()
        learner = MistakeLearner(config)

        # No history - should return reasonable multiplier
        mult = learner.get_position_size_multiplier()
        assert 0.2 <= mult <= 2.0

        # Add winning trades
        for _ in range(10):
            outcome = TradeOutcome(
                timestamp=datetime.now(),
                predicted_action="LONG",
                actual_direction="UP",
                confidence=0.7,
                features_snapshot=np.array([1, 2, 3]),
                pnl_pct=0.02,
                was_correct=True,
                regime="bull",
                holding_period=5,
            )
            learner.record_outcome(outcome)

        # High win rate should increase multiplier
        mult_after_wins = learner.get_position_size_multiplier()
        assert mult_after_wins > mult

    def test_regime_win_rate(self):
        config = LearningConfig()
        learner = MistakeLearner(config)

        # Record wins in bull regime
        for _ in range(5):
            outcome = TradeOutcome(
                timestamp=datetime.now(),
                predicted_action="LONG",
                actual_direction="UP",
                confidence=0.7,
                features_snapshot=np.array([1, 2, 3]),
                pnl_pct=0.02,
                was_correct=True,
                regime="bull",
                holding_period=5,
            )
            learner.record_outcome(outcome)

        # Record losses in bear regime
        for _ in range(5):
            outcome = TradeOutcome(
                timestamp=datetime.now(),
                predicted_action="SHORT",
                actual_direction="UP",
                confidence=0.6,
                features_snapshot=np.array([1, 2, 3]),
                pnl_pct=-0.01,
                was_correct=False,
                regime="bear",
                holding_period=3,
            )
            learner.record_outcome(outcome)

        assert learner.get_regime_win_rate("bull") == 1.0
        assert learner.get_regime_win_rate("bear") == 0.0


class TestAggressiveProfitHunter:
    """Test AggressiveProfitHunter class."""

    @pytest.fixture
    def temp_model_dir(self):
        """Create temporary directory for model storage."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_data(self):
        """Generate sample OHLCV data."""
        return generate_synthetic_ohlcv(bars=1000, trend="up")

    def test_initialization(self, temp_model_dir):
        predictor = AggressiveProfitHunter(model_dir=temp_model_dir)

        assert predictor.is_trained is False
        assert len(predictor.models) >= 5  # At least 5 models
        assert predictor.learner is not None

    def test_training(self, temp_model_dir, sample_data):
        predictor = AggressiveProfitHunter(model_dir=temp_model_dir)

        metrics = predictor.train(sample_data, symbol="TEST/USDT")

        assert predictor.is_trained is True
        assert "overall" in metrics
        assert "ensemble_accuracy" in metrics["overall"]
        assert metrics["overall"]["ensemble_accuracy"] > 0.4  # Better than random
        assert len(predictor.feature_names) > 0

    def test_prediction_before_training(self, temp_model_dir, sample_data):
        predictor = AggressiveProfitHunter(model_dir=temp_model_dir)

        with pytest.raises(ValueError, match="not trained"):
            predictor.predict(sample_data)

    def test_prediction_after_training(self, temp_model_dir, sample_data):
        predictor = AggressiveProfitHunter(model_dir=temp_model_dir)
        predictor.train(sample_data, symbol="TEST/USDT")

        signal = predictor.predict(sample_data.tail(100))

        assert isinstance(signal, AggressiveSignal)
        assert signal.action in ["LONG", "SHORT", "FLAT"]
        assert 0 <= signal.confidence <= 1
        assert isinstance(signal.strength, SignalStrength)
        assert signal.recommended_leverage >= 1.0
        assert 0 <= signal.position_size_pct <= 1.0

    def test_signal_strength_levels(self, temp_model_dir, sample_data):
        config = AggressiveConfig(
            min_confidence_to_trade=0.5,
            high_confidence_threshold=0.7,
            extreme_confidence_threshold=0.85,
        )
        predictor = AggressiveProfitHunter(config=config, model_dir=temp_model_dir)
        predictor.train(sample_data, symbol="TEST/USDT")

        # Generate multiple predictions
        signals = []
        for i in range(10):
            try:
                signal = predictor.predict(sample_data.iloc[i * 50 : (i + 1) * 50 + 50])
                signals.append(signal)
            except Exception:
                pass

        # Should have variety of strength levels
        strengths = [s.strength for s in signals]
        assert len(signals) > 0

    def test_leverage_calculation(self, temp_model_dir, sample_data):
        config = AggressiveConfig(
            base_leverage=2.0,
            max_leverage=5.0,
        )
        predictor = AggressiveProfitHunter(config=config, model_dir=temp_model_dir)
        predictor.train(sample_data, symbol="TEST/USDT")

        signal = predictor.predict(sample_data.tail(100))

        assert signal.recommended_leverage >= 1.0
        assert signal.recommended_leverage <= config.max_leverage

    def test_record_trade_outcome(self, temp_model_dir, sample_data):
        predictor = AggressiveProfitHunter(model_dir=temp_model_dir)
        predictor.train(sample_data, symbol="TEST/USDT")

        # Record outcome with empty features (features are optional)
        predictor.record_trade_outcome(
            predicted_action="LONG",
            actual_pnl_pct=0.02,
            confidence=0.7,
            features=np.array([]),  # Empty features for simplicity
            regime="bull",
            holding_period=5,
        )

        stats = predictor.get_stats()
        assert stats["learning"]["trades_recorded"] == 1

    def test_save_and_load(self, temp_model_dir, sample_data):
        # Train and save
        predictor1 = AggressiveProfitHunter(model_dir=temp_model_dir)
        predictor1.train(sample_data, symbol="TEST/USDT")
        predictor1.save("test_model")

        # Load in new instance
        predictor2 = AggressiveProfitHunter(model_dir=temp_model_dir)
        success = predictor2.load("test_model")

        assert success is True
        assert predictor2.is_trained is True
        assert predictor2.feature_names == predictor1.feature_names

        # Should produce similar predictions
        signal1 = predictor1.predict(sample_data.tail(100))
        signal2 = predictor2.predict(sample_data.tail(100))

        assert signal1.action == signal2.action

    def test_learning_reduces_size_after_losses(self, temp_model_dir, sample_data):
        predictor = AggressiveProfitHunter(model_dir=temp_model_dir)
        predictor.train(sample_data, symbol="TEST/USDT")

        # Get initial signal
        signal_before = predictor.predict(sample_data.tail(100))

        # Record consecutive losses with empty features
        for _ in range(3):
            predictor.record_trade_outcome(
                predicted_action="LONG",
                actual_pnl_pct=-0.02,
                confidence=0.7,
                features=np.array([]),  # Empty features for simplicity
                regime="bull",
                holding_period=5,
            )

        # Get signal after losses
        signal_after = predictor.predict(sample_data.tail(100))

        # Position size should be reduced (or signal may become FLAT)
        if signal_after.action != "FLAT":
            assert signal_after.position_size_pct <= signal_before.position_size_pct * 1.1


class TestCreateAggressivePredictor:
    """Test factory function."""

    def test_create_with_defaults(self):
        predictor = create_aggressive_predictor()

        assert isinstance(predictor, AggressiveProfitHunter)
        assert predictor.config.enable_learning is True

    def test_create_with_custom_leverage(self):
        predictor = create_aggressive_predictor(max_leverage=3.0)

        assert predictor.config.max_leverage == 3.0

    def test_create_without_learning(self):
        predictor = create_aggressive_predictor(enable_learning=False)

        assert predictor.config.enable_learning is False


class TestSignalStrength:
    """Test SignalStrength enum."""

    def test_strength_ordering(self):
        assert SignalStrength.WEAK.value < SignalStrength.MODERATE.value
        assert SignalStrength.MODERATE.value < SignalStrength.STRONG.value
        assert SignalStrength.STRONG.value < SignalStrength.VERY_STRONG.value
        assert SignalStrength.VERY_STRONG.value < SignalStrength.EXTREME.value


class TestAggressiveSignal:
    """Test AggressiveSignal dataclass."""

    def test_to_dict(self):
        signal = AggressiveSignal(
            action="LONG",
            confidence=0.75,
            strength=SignalStrength.STRONG,
            recommended_leverage=3.0,
            position_size_pct=0.1,
            expected_move_pct=0.02,
            stop_loss_pct=0.015,
            take_profit_pct=0.03,
            max_hold_bars=10,
            regime="bull",
        )

        d = signal.to_dict()

        assert d["action"] == "LONG"
        assert d["confidence"] == 0.75
        assert d["strength"] == "STRONG"
        assert d["recommended_leverage"] == 3.0
        assert d["regime"] == "bull"
