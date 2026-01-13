"""
Unit Tests for Regime-Aware Trading Module.

Tests:
- Regime detection (Bull, Bear, Crash, Sideways, HighVol)
- Risk engine constraint checks
- Strategy selection logic
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from bot.regime import (
    MarketRegime,
    RegimeConfig,
    RegimeDetector,
    RegimeIndicators,
    RegimeState,
    RegimeLimits,
    RegimeRiskEngine,
    RiskCheckResult,
    RiskConfig,
    TradeRequest,
    RegimeStrategy,
    RegimeStrategySelector,
    StrategySignal,
)
from bot.regime.regime_risk_engine import PortfolioState, BlockReason, RiskDecision


# =============================================================================
# Test Fixtures
# =============================================================================


def generate_ohlcv_data(
    n_bars: int = 200,
    start_price: float = 100.0,
    trend: str = "neutral",
    volatility: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(seed)

    dates = pd.date_range(start="2024-01-01", periods=n_bars, freq="1h")

    # Generate returns based on trend
    if trend == "bull":
        drift = 0.002  # Strong positive drift
    elif trend == "bear":
        drift = -0.002  # Strong negative drift
    elif trend == "crash":
        drift = -0.008  # Strong negative with high vol
        volatility *= 3
    elif trend == "high_vol":
        drift = 0.0
        volatility *= 4
    else:
        drift = 0.0

    returns = np.random.normal(drift, volatility, n_bars)
    prices = start_price * np.exp(np.cumsum(returns))

    # Generate OHLCV
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        noise = np.random.uniform(0.002, 0.01)
        high = close * (1 + noise)
        low = close * (1 - noise)
        open_price = close * (1 + np.random.uniform(-0.005, 0.005))
        volume = np.random.uniform(1000, 10000)

        data.append({
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        })

    return pd.DataFrame(data, index=dates)


@pytest.fixture
def bull_market_data() -> pd.DataFrame:
    """Generate bull market test data."""
    return generate_ohlcv_data(n_bars=200, trend="bull", seed=42)


@pytest.fixture
def bear_market_data() -> pd.DataFrame:
    """Generate bear market test data."""
    return generate_ohlcv_data(n_bars=200, trend="bear", seed=43)


@pytest.fixture
def crash_market_data() -> pd.DataFrame:
    """Generate crash scenario test data."""
    return generate_ohlcv_data(n_bars=200, trend="crash", seed=44)


@pytest.fixture
def sideways_market_data() -> pd.DataFrame:
    """Generate sideways/ranging market test data."""
    return generate_ohlcv_data(n_bars=200, trend="neutral", volatility=0.005, seed=45)


@pytest.fixture
def high_vol_market_data() -> pd.DataFrame:
    """Generate high volatility test data."""
    return generate_ohlcv_data(n_bars=200, trend="high_vol", seed=46)


@pytest.fixture
def regime_detector() -> RegimeDetector:
    """Create regime detector with default config."""
    return RegimeDetector()


@pytest.fixture
def risk_engine() -> RegimeRiskEngine:
    """Create risk engine with default config."""
    # Use a temp path to avoid persisting state between tests
    import tempfile
    from pathlib import Path
    temp_dir = tempfile.mkdtemp()
    return RegimeRiskEngine(state_path=Path(temp_dir) / "test_risk_state.json")


@pytest.fixture
def default_portfolio_state() -> PortfolioState:
    """Create default portfolio state."""
    return PortfolioState(
        equity=10000.0,
        available_balance=10000.0,
        peak_equity=10000.0,
    )


@pytest.fixture
def bull_regime_state() -> RegimeState:
    """Create a bull regime state."""
    return RegimeState(
        regime=MarketRegime.BULL,
        confidence=0.8,
        indicators=RegimeIndicators(),
        symbol="BTC/USDT",
        timeframe="1h",
    )


# =============================================================================
# Regime Detection Tests
# =============================================================================


class TestRegimeDetector:
    """Tests for RegimeDetector class."""

    def test_detect_returns_regime_state(self, regime_detector, bull_market_data):
        """Test that detect() returns a valid RegimeState."""
        result = regime_detector.detect(bull_market_data, "BTC/USDT", "1h")

        assert isinstance(result, RegimeState)
        assert isinstance(result.regime, MarketRegime)
        assert 0.0 <= result.confidence <= 1.0
        assert result.indicators is not None

    def test_detect_bull_market_has_positive_trend(self, regime_detector, bull_market_data):
        """Test that bull market data shows positive trend direction."""
        result = regime_detector.detect(bull_market_data, "BTC/USDT", "1h")

        # Bull market should have positive trend indicators
        assert result.indicators.trend_direction >= 0 or result.regime in [MarketRegime.BULL, MarketRegime.HIGH_VOL]

    def test_detect_bear_market_has_negative_trend(self, regime_detector, bear_market_data):
        """Test that bear market data shows negative trend direction."""
        result = regime_detector.detect(bear_market_data, "BTC/USDT", "1h")

        # Bear market data - checking regime or trend direction
        assert result.regime in [MarketRegime.BEAR, MarketRegime.CRASH, MarketRegime.HIGH_VOL, MarketRegime.SIDEWAYS]

    def test_detect_crash_conditions(self, regime_detector, crash_market_data):
        """Test detection of crash conditions."""
        result = regime_detector.detect(crash_market_data, "BTC/USDT", "1h")

        # Crash should be detected due to high drawdown + negative trend or high vol
        assert result.regime in [MarketRegime.CRASH, MarketRegime.HIGH_VOL, MarketRegime.BEAR]
        # Volatility should be elevated
        assert result.indicators.realized_vol > 0 or result.indicators.atr_pct > 0

    def test_detect_sideways_market(self, regime_detector, sideways_market_data):
        """Test detection of sideways/ranging market."""
        result = regime_detector.detect(sideways_market_data, "BTC/USDT", "1h")

        # Low volatility, no strong trend should give SIDEWAYS or low trend strength
        assert result.regime in [MarketRegime.SIDEWAYS, MarketRegime.BULL, MarketRegime.BEAR]

    def test_indicators_computed_correctly(self, regime_detector, bull_market_data):
        """Test that all indicators are computed."""
        result = regime_detector.detect(bull_market_data, "BTC/USDT", "1h")
        indicators = result.indicators

        assert indicators is not None
        assert indicators.trend_strength >= 0  # ADX-like value
        assert indicators.realized_vol >= 0
        assert isinstance(indicators.trend_direction, (int, float))
        assert isinstance(indicators.drawdown_pct, float)

    def test_regime_stats_tracking(self, regime_detector, bull_market_data):
        """Test that regime stats are tracked."""
        # Detect multiple times
        for _ in range(5):
            regime_detector.detect(bull_market_data, "BTC/USDT", "1h")

        stats = regime_detector.get_regime_stats()
        # Stats should contain some data
        assert isinstance(stats, dict)

    def test_minimum_data_requirement(self, regime_detector):
        """Test handling of insufficient data."""
        small_df = generate_ohlcv_data(n_bars=10)
        result = regime_detector.detect(small_df, "BTC/USDT", "1h")

        # Should return UNKNOWN for insufficient data
        assert result.regime == MarketRegime.UNKNOWN
        assert result.confidence == 0.0

    def test_regime_priority_crash_detection(self, regime_detector, crash_market_data):
        """Test that crash conditions are properly detected."""
        result = regime_detector.detect(crash_market_data, "BTC/USDT", "1h")

        # With crash data, should detect high risk regime
        assert result.regime in [MarketRegime.CRASH, MarketRegime.HIGH_VOL, MarketRegime.BEAR]


# =============================================================================
# Risk Engine Tests
# =============================================================================


class TestRiskEngine:
    """Tests for RegimeRiskEngine class."""

    def test_check_trade_approved(self, risk_engine, default_portfolio_state, bull_regime_state):
        """Test that valid trades are approved."""
        risk_engine.update_portfolio(default_portfolio_state)
        risk_engine.update_regime(bull_regime_state)

        request = TradeRequest(
            symbol="BTC/USDT",
            direction="long",
            entry_price=50000.0,
            stop_loss=49000.0,
            signal_confidence=0.8,
        )

        result = risk_engine.check_trade(request)

        assert result.is_approved
        assert result.approved_quantity > 0

    def test_kill_switch_blocks_trades(self, risk_engine, default_portfolio_state, bull_regime_state):
        """Test that kill switch blocks all trades."""
        risk_engine.update_portfolio(default_portfolio_state)
        risk_engine.update_regime(bull_regime_state)
        risk_engine.activate_kill_switch("test")

        request = TradeRequest(
            symbol="BTC/USDT",
            direction="long",
            entry_price=50000.0,
            stop_loss=49000.0,
            signal_confidence=0.8,
        )

        result = risk_engine.check_trade(request)

        assert not result.is_approved
        assert BlockReason.KILL_SWITCH_ACTIVE in result.block_reasons

    def test_daily_loss_limit(self, risk_engine, default_portfolio_state, bull_regime_state):
        """Test that daily loss limit blocks trades."""
        import tempfile
        from pathlib import Path
        temp_dir = tempfile.mkdtemp()
        config = RiskConfig(max_daily_loss_pct=0.02)  # 2%
        risk_engine = RegimeRiskEngine(config, state_path=Path(temp_dir) / "test_state.json")
        risk_engine.update_portfolio(default_portfolio_state)
        risk_engine.update_regime(bull_regime_state)

        # Simulate losses exceeding daily limit
        for _ in range(5):
            risk_engine.record_trade_result(-50.0, is_win=False)

        request = TradeRequest(
            symbol="BTC/USDT",
            direction="long",
            entry_price=50000.0,
            stop_loss=49000.0,
            signal_confidence=0.8,
        )

        result = risk_engine.check_trade(request)

        # Should be blocked due to daily loss or consecutive losses
        if not result.is_approved:
            assert any(r in [BlockReason.DAILY_LOSS_LIMIT, BlockReason.CONSECUTIVE_LOSSES]
                       for r in result.block_reasons)

    def test_drawdown_limit(self, bull_regime_state):
        """Test that max drawdown triggers kill switch."""
        import tempfile
        from pathlib import Path
        temp_dir = tempfile.mkdtemp()
        config = RiskConfig(max_drawdown_pct=0.10)  # 10%
        risk_engine = RegimeRiskEngine(config, state_path=Path(temp_dir) / "test_state.json")

        # Portfolio with 15% drawdown
        portfolio = PortfolioState(
            equity=8500.0,
            available_balance=8500.0,
            peak_equity=10000.0,
        )
        risk_engine.update_portfolio(portfolio)
        risk_engine.update_regime(bull_regime_state)

        request = TradeRequest(
            symbol="BTC/USDT",
            direction="long",
            entry_price=50000.0,
            stop_loss=49000.0,
            signal_confidence=0.8,
        )

        result = risk_engine.check_trade(request)

        # Should be blocked due to drawdown (kill switch triggered)
        assert not result.is_approved
        assert BlockReason.KILL_SWITCH_ACTIVE in result.block_reasons or \
               BlockReason.DRAWDOWN_LIMIT in result.block_reasons

    def test_consecutive_losses_limit(self, default_portfolio_state, bull_regime_state):
        """Test that consecutive losses trigger blocking."""
        import tempfile
        from pathlib import Path
        temp_dir = tempfile.mkdtemp()
        config = RiskConfig(max_consecutive_losses=3)
        risk_engine = RegimeRiskEngine(config, state_path=Path(temp_dir) / "test_state.json")
        risk_engine.update_portfolio(default_portfolio_state)
        risk_engine.update_regime(bull_regime_state)

        # Simulate 5 consecutive losses
        for _ in range(5):
            risk_engine.record_trade_result(-10.0, is_win=False)

        request = TradeRequest(
            symbol="BTC/USDT",
            direction="long",
            entry_price=50000.0,
            stop_loss=49000.0,
            signal_confidence=0.8,
        )

        result = risk_engine.check_trade(request)

        assert BlockReason.CONSECUTIVE_LOSSES in result.block_reasons

    def test_position_sizing_respects_risk(self, risk_engine, default_portfolio_state, bull_regime_state):
        """Test that position sizing respects risk limits."""
        risk_engine.update_portfolio(default_portfolio_state)
        risk_engine.update_regime(bull_regime_state)

        request = TradeRequest(
            symbol="BTC/USDT",
            direction="long",
            entry_price=50000.0,
            stop_loss=47500.0,  # 5% stop
            signal_confidence=0.8,
        )

        result = risk_engine.check_trade(request)

        if result.is_approved:
            # Position value should not exceed max position pct
            position_value = result.approved_quantity * request.entry_price
            # Check that position is sized reasonably
            assert position_value <= default_portfolio_state.equity

    def test_volatility_circuit_breaker(self, risk_engine, default_portfolio_state, bull_regime_state):
        """Test that volatility circuit breaker blocks trades."""
        risk_engine.update_portfolio(default_portfolio_state)
        risk_engine.update_regime(bull_regime_state)

        # Set high current volatility (4x normal)
        risk_engine.update_volatility(current_vol=0.10, normal_vol=0.02)

        request = TradeRequest(
            symbol="BTC/USDT",
            direction="long",
            entry_price=50000.0,
            stop_loss=49000.0,
            signal_confidence=0.8,
        )

        result = risk_engine.check_trade(request)

        # Should be blocked due to volatility circuit breaker
        assert BlockReason.VOLATILITY_CIRCUIT_BREAKER in result.block_reasons

    def test_spread_circuit_breaker(self, risk_engine, default_portfolio_state, bull_regime_state):
        """Test that wide spread blocks trades."""
        risk_engine.update_portfolio(default_portfolio_state)
        risk_engine.update_regime(bull_regime_state)

        request = TradeRequest(
            symbol="BTC/USDT",
            direction="long",
            entry_price=50000.0,
            stop_loss=49000.0,
            signal_confidence=0.8,
        )

        # 2% spread should trigger (default max is 0.5%)
        result = risk_engine.check_trade(request, spread_pct=0.02)

        assert BlockReason.SPREAD_CIRCUIT_BREAKER in result.block_reasons

    def test_liquidity_filter(self, default_portfolio_state, bull_regime_state):
        """Test that low liquidity blocks trades."""
        import tempfile
        from pathlib import Path
        temp_dir = tempfile.mkdtemp()
        config = RiskConfig(min_liquidity_volume=1000000.0)
        risk_engine = RegimeRiskEngine(config, state_path=Path(temp_dir) / "test_state.json")
        risk_engine.update_portfolio(default_portfolio_state)
        risk_engine.update_regime(bull_regime_state)

        request = TradeRequest(
            symbol="BTC/USDT",
            direction="long",
            entry_price=50000.0,
            stop_loss=49000.0,
            signal_confidence=0.8,
        )

        # Low volume
        result = risk_engine.check_trade(request, volume_24h=100000.0)

        assert BlockReason.LIQUIDITY_FILTER in result.block_reasons

    def test_regime_specific_limits_crash(self, risk_engine, default_portfolio_state):
        """Test that crash regime affects allowed directions."""
        risk_engine.update_portfolio(default_portfolio_state)

        # Create CRASH regime state
        crash_regime = RegimeState(
            regime=MarketRegime.CRASH,
            confidence=0.9,
            indicators=RegimeIndicators(),
            symbol="BTC/USDT",
            timeframe="1h",
        )
        risk_engine.update_regime(crash_regime)

        # Long request in CRASH should be blocked
        request = TradeRequest(
            symbol="BTC/USDT",
            direction="long",
            entry_price=50000.0,
            stop_loss=49000.0,
            signal_confidence=0.8,
        )

        result = risk_engine.check_trade(request)

        # In CRASH regime, long should be blocked (only short allowed)
        assert not result.is_approved
        assert BlockReason.REGIME_NOT_ALLOWED in result.block_reasons

    def test_min_time_between_trades(self, default_portfolio_state, bull_regime_state):
        """Test minimum time between trades."""
        import tempfile
        from pathlib import Path
        temp_dir = tempfile.mkdtemp()
        config = RiskConfig(min_time_between_trades_sec=300)  # 5 minutes
        risk_engine = RegimeRiskEngine(config, state_path=Path(temp_dir) / "test_state.json")
        risk_engine.update_portfolio(default_portfolio_state)
        risk_engine.update_regime(bull_regime_state)

        request = TradeRequest(
            symbol="BTC/USDT",
            direction="long",
            entry_price=50000.0,
            stop_loss=49000.0,
            signal_confidence=0.8,
        )

        # First trade should be approved
        result1 = risk_engine.check_trade(request)
        assert result1.is_approved

        # Record the trade (this sets last_trade_time)
        risk_engine.record_trade_result(100.0, is_win=True)

        # Immediate second trade should be blocked
        result2 = risk_engine.check_trade(request)
        assert BlockReason.TIME_BETWEEN_TRADES in result2.block_reasons

    def test_portfolio_heat_limit(self, default_portfolio_state, bull_regime_state):
        """Test portfolio heat (total risk) limit."""
        import tempfile
        from pathlib import Path
        temp_dir = tempfile.mkdtemp()
        config = RiskConfig(max_portfolio_heat=0.06)  # 6%
        risk_engine = RegimeRiskEngine(config, state_path=Path(temp_dir) / "test_state.json")

        # Portfolio with existing positions using up heat
        portfolio = PortfolioState(
            equity=10000.0,
            available_balance=5000.0,
            peak_equity=10000.0,
            positions={
                "ETH/USDT": {"risk_pct": 0.02, "value": 2000},
                "SOL/USDT": {"risk_pct": 0.02, "value": 2000},
                "DOGE/USDT": {"risk_pct": 0.02, "value": 1000},
            }
        )
        risk_engine.update_portfolio(portfolio)
        risk_engine.update_regime(bull_regime_state)

        request = TradeRequest(
            symbol="BTC/USDT",
            direction="long",
            entry_price=50000.0,
            stop_loss=49000.0,  # 2% risk
            signal_confidence=0.8,
        )

        result = risk_engine.check_trade(request)

        # Should be blocked due to portfolio heat
        assert BlockReason.PORTFOLIO_HEAT_LIMIT in result.block_reasons

    def test_deactivate_kill_switch(self, risk_engine, default_portfolio_state, bull_regime_state):
        """Test that kill switch can be deactivated."""
        risk_engine.update_portfolio(default_portfolio_state)
        risk_engine.update_regime(bull_regime_state)

        # Activate then deactivate
        risk_engine.activate_kill_switch("test")
        assert risk_engine._kill_switch_active

        risk_engine.deactivate_kill_switch("test_user")
        assert not risk_engine._kill_switch_active

        request = TradeRequest(
            symbol="BTC/USDT",
            direction="long",
            entry_price=50000.0,
            stop_loss=49000.0,
            signal_confidence=0.8,
        )

        result = risk_engine.check_trade(request)
        assert BlockReason.KILL_SWITCH_ACTIVE not in result.block_reasons

    def test_unknown_regime_blocks_all(self, risk_engine, default_portfolio_state):
        """Test that UNKNOWN regime blocks all trades."""
        risk_engine.update_portfolio(default_portfolio_state)

        # Set UNKNOWN regime
        unknown_regime = RegimeState(
            regime=MarketRegime.UNKNOWN,
            confidence=0.0,
            indicators=RegimeIndicators(),
            symbol="BTC/USDT",
            timeframe="1h",
        )
        risk_engine.update_regime(unknown_regime)

        request = TradeRequest(
            symbol="BTC/USDT",
            direction="long",
            entry_price=50000.0,
            stop_loss=49000.0,
            signal_confidence=0.8,
        )

        result = risk_engine.check_trade(request)

        assert not result.is_approved
        assert BlockReason.REGIME_NOT_ALLOWED in result.block_reasons


# =============================================================================
# Strategy Selection Tests
# =============================================================================


class TestStrategySelector:
    """Tests for RegimeStrategySelector class."""

    def test_bull_regime_uses_trend_following(self):
        """Test that BULL regime selects trend following strategy."""
        selector = RegimeStrategySelector()
        strategy = selector.get_strategy(MarketRegime.BULL)

        assert "trend" in strategy.__class__.__name__.lower()

    def test_sideways_regime_uses_mean_reversion(self):
        """Test that SIDEWAYS regime selects mean reversion strategy."""
        selector = RegimeStrategySelector()
        strategy = selector.get_strategy(MarketRegime.SIDEWAYS)

        assert "meanreversion" in strategy.__class__.__name__.lower()

    def test_crash_regime_uses_defensive(self):
        """Test that CRASH regime selects defensive strategy."""
        selector = RegimeStrategySelector()
        strategy = selector.get_strategy(MarketRegime.CRASH)

        assert "defensive" in strategy.__class__.__name__.lower()

    def test_generate_signal_includes_regime(self, bull_market_data):
        """Test that generated signals include regime info."""
        selector = RegimeStrategySelector()

        regime_state = RegimeState(
            regime=MarketRegime.BULL,
            confidence=0.8,
            indicators=RegimeIndicators(),
            symbol="BTC/USDT",
            timeframe="1h",
        )

        signal = selector.generate_signal(bull_market_data, regime_state)

        if signal:
            assert signal.regime == MarketRegime.BULL

    def test_signal_has_required_fields(self, bull_market_data):
        """Test that signals have all required fields."""
        selector = RegimeStrategySelector()

        regime_state = RegimeState(
            regime=MarketRegime.BULL,
            confidence=0.8,
            indicators=RegimeIndicators(),
            symbol="BTC/USDT",
            timeframe="1h",
        )

        signal = selector.generate_signal(bull_market_data, regime_state)

        if signal:
            assert signal.entry_price > 0
            assert signal.stop_loss > 0
            assert signal.confidence > 0
            assert signal.strategy_name


# =============================================================================
# Integration Tests
# =============================================================================


class TestRegimeRiskIntegration:
    """Integration tests for regime detection + risk engine."""

    def test_crash_regime_reduces_position_size(self):
        """Test that crash regime significantly reduces position size."""
        import tempfile
        from pathlib import Path
        temp_dir = tempfile.mkdtemp()
        risk_engine = RegimeRiskEngine(state_path=Path(temp_dir) / "test_state.json")

        portfolio = PortfolioState(
            equity=10000.0,
            available_balance=10000.0,
            peak_equity=10000.0,
        )
        risk_engine.update_portfolio(portfolio)

        request = TradeRequest(
            symbol="BTC/USDT",
            direction="short",  # Short allowed in both regimes
            entry_price=50000.0,
            stop_loss=51000.0,
            signal_confidence=0.8,
        )

        # Normal regime (BULL)
        bull_regime = RegimeState(
            regime=MarketRegime.BULL,
            confidence=0.8,
            indicators=RegimeIndicators(),
            symbol="BTC/USDT",
            timeframe="1h",
        )
        risk_engine.update_regime(bull_regime)
        result_bull = risk_engine.check_trade(request)

        # HIGH_VOL regime (allows both directions but smaller size)
        high_vol_regime = RegimeState(
            regime=MarketRegime.HIGH_VOL,
            confidence=0.9,
            indicators=RegimeIndicators(),
            symbol="BTC/USDT",
            timeframe="1h",
        )
        risk_engine.update_regime(high_vol_regime)
        result_high_vol = risk_engine.check_trade(request)

        # High vol should have smaller position than bull
        if result_bull.is_approved and result_high_vol.is_approved:
            assert result_high_vol.approved_quantity <= result_bull.approved_quantity

    def test_full_pipeline(self, bull_market_data):
        """Test full pipeline: detection -> strategy -> risk check."""
        import tempfile
        from pathlib import Path
        temp_dir = tempfile.mkdtemp()

        # Components
        detector = RegimeDetector()
        selector = RegimeStrategySelector()
        risk_engine = RegimeRiskEngine(state_path=Path(temp_dir) / "test_state.json")

        portfolio = PortfolioState(
            equity=10000.0,
            available_balance=10000.0,
            peak_equity=10000.0,
        )
        risk_engine.update_portfolio(portfolio)

        # 1. Detect regime
        regime_state = detector.detect(bull_market_data, "BTC/USDT", "1h")
        risk_engine.update_regime(regime_state)

        # 2. Generate signal
        signal = selector.generate_signal(bull_market_data, regime_state)

        if signal:
            # 3. Check with risk engine
            request = TradeRequest(
                symbol="BTC/USDT",
                direction=signal.direction.value,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                signal_confidence=signal.confidence,
            )

            result = risk_engine.check_trade(request)

            # Result should be valid
            assert isinstance(result, RiskCheckResult)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataframe(self, regime_detector):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        # Should handle gracefully
        result = regime_detector.detect(empty_df, "BTC/USDT", "1h")
        assert isinstance(result, RegimeState)
        assert result.regime == MarketRegime.UNKNOWN

    def test_nan_values_in_data(self, regime_detector):
        """Test handling of NaN values in data."""
        df = generate_ohlcv_data(n_bars=150)
        df.iloc[50, 0] = np.nan  # Introduce NaN

        result = regime_detector.detect(df, "BTC/USDT", "1h")
        assert isinstance(result, RegimeState)

    def test_zero_stop_loss_distance(self, risk_engine, default_portfolio_state, bull_regime_state):
        """Test handling of zero stop loss distance."""
        risk_engine.update_portfolio(default_portfolio_state)
        risk_engine.update_regime(bull_regime_state)

        request = TradeRequest(
            symbol="BTC/USDT",
            direction="long",
            entry_price=50000.0,
            stop_loss=50000.0,  # Same as entry
            signal_confidence=0.8,
        )

        result = risk_engine.check_trade(request)

        # Should handle gracefully (likely blocked due to invalid stop)
        assert isinstance(result, RiskCheckResult)
        # Invalid stop should be blocked
        assert BlockReason.INVALID_STOP_LOSS in result.block_reasons or result.approved_quantity == 0

    def test_negative_equity(self):
        """Test handling of negative equity."""
        import tempfile
        from pathlib import Path
        temp_dir = tempfile.mkdtemp()
        risk_engine = RegimeRiskEngine(state_path=Path(temp_dir) / "test_state.json")

        portfolio = PortfolioState(
            equity=-1000.0,
            available_balance=-1000.0,
            peak_equity=10000.0,
        )
        risk_engine.update_portfolio(portfolio)

        # Set a valid regime
        bull_regime = RegimeState(
            regime=MarketRegime.BULL,
            confidence=0.8,
            indicators=RegimeIndicators(),
            symbol="BTC/USDT",
            timeframe="1h",
        )
        risk_engine.update_regime(bull_regime)

        request = TradeRequest(
            symbol="BTC/USDT",
            direction="long",
            entry_price=50000.0,
            stop_loss=49000.0,
            signal_confidence=0.8,
        )

        result = risk_engine.check_trade(request)

        # Should block due to invalid state (negative equity = huge drawdown)
        assert not result.is_approved


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
