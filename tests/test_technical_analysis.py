"""
Tests for technical analysis module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from bot.technical_analysis import (
    CandlePattern,
    PatternResult,
    CandlestickAnalyzer,
    DivergenceType,
    DivergenceResult,
    DivergenceDetector,
    SupportResistanceLevel,
    ConfluenceZone,
    ConfluenceDetector,
    TechnicalAnalysisResult,
    TechnicalAnalyzer,
)


class TestCandlePattern:
    """Test CandlePattern enum."""

    def test_bullish_patterns_exist(self):
        """Test bullish patterns exist."""
        assert CandlePattern.HAMMER.value == "hammer"
        assert CandlePattern.INVERTED_HAMMER.value == "inverted_hammer"
        assert CandlePattern.BULLISH_ENGULFING.value == "bullish_engulfing"
        assert CandlePattern.MORNING_STAR.value == "morning_star"

    def test_bearish_patterns_exist(self):
        """Test bearish patterns exist."""
        assert CandlePattern.SHOOTING_STAR.value == "shooting_star"
        assert CandlePattern.HANGING_MAN.value == "hanging_man"
        assert CandlePattern.BEARISH_ENGULFING.value == "bearish_engulfing"
        assert CandlePattern.EVENING_STAR.value == "evening_star"

    def test_neutral_patterns_exist(self):
        """Test neutral patterns exist."""
        assert CandlePattern.DOJI.value == "doji"
        assert CandlePattern.SPINNING_TOP.value == "spinning_top"


class TestPatternResult:
    """Test PatternResult dataclass."""

    def test_pattern_result_creation(self):
        """Test creating a pattern result."""
        result = PatternResult(
            pattern=CandlePattern.HAMMER,
            direction="bullish",
            strength=0.8,
            bar_index=10,
            description="Hammer at support",
        )
        assert result.pattern == CandlePattern.HAMMER
        assert result.direction == "bullish"
        assert result.strength == 0.8
        assert result.bar_index == 10


class TestCandlestickAnalyzer:
    """Test CandlestickAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create candlestick analyzer."""
        return CandlestickAnalyzer()

    @pytest.fixture
    def sample_ohlcv(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        n = 50
        dates = pd.date_range("2024-01-01", periods=n, freq="h")

        base_price = 100.0
        returns = np.random.randn(n) * 0.02
        prices = base_price * np.exp(np.cumsum(returns))

        return pd.DataFrame(
            {
                "open": prices * (1 + np.random.randn(n) * 0.005),
                "high": prices * (1 + np.abs(np.random.randn(n)) * 0.01),
                "low": prices * (1 - np.abs(np.random.randn(n)) * 0.01),
                "close": prices,
                "volume": np.random.randint(1000, 10000, n),
            },
            index=dates,
        )

    def test_analyzer_creation(self, analyzer):
        """Test analyzer is created."""
        assert analyzer is not None
        assert analyzer.body_threshold == 0.3
        assert analyzer.doji_threshold == 0.1

    def test_custom_thresholds(self):
        """Test custom thresholds."""
        analyzer = CandlestickAnalyzer(body_threshold=0.4, doji_threshold=0.05)
        assert analyzer.body_threshold == 0.4
        assert analyzer.doji_threshold == 0.05

    def test_analyze_returns_list(self, analyzer, sample_ohlcv):
        """Test analyze returns list."""
        patterns = analyzer.analyze(sample_ohlcv)
        assert isinstance(patterns, list)

    def test_analyze_insufficient_data(self, analyzer):
        """Test analyze with insufficient data."""
        short_df = pd.DataFrame(
            {
                "open": [100, 101],
                "high": [102, 103],
                "low": [99, 100],
                "close": [101, 102],
            }
        )
        patterns = analyzer.analyze(short_df)
        assert patterns == []

    def test_get_candle_metrics(self, analyzer, sample_ohlcv):
        """Test candle metrics calculation."""
        metrics = analyzer._get_candle_metrics(sample_ohlcv, 10)

        assert "open" in metrics
        assert "high" in metrics
        assert "low" in metrics
        assert "close" in metrics
        assert "body" in metrics
        assert "body_pct" in metrics
        assert "is_bullish" in metrics
        assert "is_doji" in metrics

    def test_doji_with_specific_data(self, analyzer):
        """Test with specific doji-like candle data."""
        df = pd.DataFrame(
            {
                "open": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
                "high": [102.0, 102.0, 102.0, 102.0, 102.0, 102.0, 102.0, 102.0, 102.0, 102.0],
                "low": [98.0, 98.0, 98.0, 98.0, 98.0, 98.0, 98.0, 98.0, 98.0, 98.0],
                "close": [100.1, 100.1, 100.1, 100.1, 100.1, 100.1, 100.1, 100.1, 100.1, 100.1],
            }
        )
        patterns = analyzer.analyze(df)
        # With very small body, these should be detected as doji
        assert isinstance(patterns, list)


class TestDivergenceType:
    """Test DivergenceType enum."""

    def test_all_types_exist(self):
        """Test all divergence types exist."""
        assert DivergenceType.REGULAR_BULLISH.value == "regular_bullish"
        assert DivergenceType.REGULAR_BEARISH.value == "regular_bearish"
        assert DivergenceType.HIDDEN_BULLISH.value == "hidden_bullish"
        assert DivergenceType.HIDDEN_BEARISH.value == "hidden_bearish"


class TestDivergenceResult:
    """Test DivergenceResult dataclass."""

    def test_result_creation(self):
        """Test creating divergence result."""
        result = DivergenceResult(
            divergence_type=DivergenceType.REGULAR_BULLISH,
            indicator="rsi",
            strength=0.75,
            price_point1=(10, 100.0),
            price_point2=(20, 110.0),
            indicator_point1=(10, 70.0),
            indicator_point2=(20, 65.0),
            description="Regular bullish divergence on RSI",
        )
        assert result.divergence_type == DivergenceType.REGULAR_BULLISH
        assert result.indicator == "rsi"
        assert result.strength == 0.75


class TestDivergenceDetector:
    """Test DivergenceDetector class."""

    @pytest.fixture
    def detector(self):
        """Create divergence detector."""
        return DivergenceDetector()

    @pytest.fixture
    def sample_ohlcv(self):
        """Create sample OHLCV data with clear trend."""
        np.random.seed(42)
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="h")

        # Create price series with clear trend
        trend = np.linspace(100, 120, n)
        noise = np.random.randn(n) * 0.5
        prices = trend + noise

        return pd.DataFrame(
            {
                "open": prices - 0.5,
                "high": prices + 1.0,
                "low": prices - 1.0,
                "close": prices,
                "volume": np.random.randint(1000, 10000, n),
            },
            index=dates,
        )

    def test_detector_creation(self, detector):
        """Test detector is created."""
        assert detector is not None
        # Just check the detector exists

    def test_detect_returns_list(self, detector, sample_ohlcv):
        """Test detect returns list."""
        divergences = detector.detect_all(sample_ohlcv)
        assert isinstance(divergences, list)

    def test_detect_insufficient_data(self, detector):
        """Test with insufficient data."""
        short_df = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [102, 103, 104],
                "low": [99, 100, 101],
                "close": [101, 102, 103],
            }
        )
        divergences = detector.detect_all(short_df)
        assert divergences == []


class TestSupportResistanceLevel:
    """Test SupportResistanceLevel dataclass."""

    def test_level_creation(self):
        """Test creating S/R level."""
        level = SupportResistanceLevel(
            price=100.0,
            level_type="support",
            strength=5,  # Number of touches
            sources=["ema_20", "pivot"],
        )
        assert level.price == 100.0
        assert level.level_type == "support"
        assert level.strength == 5
        assert "ema_20" in level.sources


class TestConfluenceZone:
    """Test ConfluenceZone dataclass."""

    def test_zone_creation(self):
        """Test creating confluence zone."""
        sr_level = SupportResistanceLevel(
            price=100.0,
            level_type="support",
            strength=3,
            sources=["ema_20"],
        )
        zone = ConfluenceZone(
            price_low=99.0,
            price_high=101.0,
            center=100.0,
            zone_type="support",
            strength=0.9,
            levels=[sr_level],
            description="Strong support zone",
        )
        assert zone.price_low == 99.0
        assert zone.price_high == 101.0
        assert zone.center == 100.0
        assert zone.strength == 0.9
        assert len(zone.levels) == 1


class TestConfluenceDetector:
    """Test ConfluenceDetector class."""

    @pytest.fixture
    def detector(self):
        """Create confluence detector."""
        return ConfluenceDetector()

    @pytest.fixture
    def sample_ohlcv(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="h")

        base_price = 100.0
        returns = np.random.randn(n) * 0.01
        prices = base_price * np.exp(np.cumsum(returns))

        return pd.DataFrame(
            {
                "open": prices * (1 + np.random.randn(n) * 0.002),
                "high": prices * (1 + np.abs(np.random.randn(n)) * 0.005),
                "low": prices * (1 - np.abs(np.random.randn(n)) * 0.005),
                "close": prices,
                "volume": np.random.randint(1000, 10000, n),
            },
            index=dates,
        )

    def test_detector_creation(self, detector):
        """Test detector is created."""
        assert detector is not None
        assert detector.zone_tolerance_pct == 0.5

    def test_custom_tolerance(self):
        """Test custom tolerance."""
        detector = ConfluenceDetector(zone_tolerance_pct=1.0)
        assert detector.zone_tolerance_pct == 1.0

    def test_detect_returns_list(self, detector, sample_ohlcv):
        """Test detect returns list."""
        zones = detector.detect(sample_ohlcv)
        assert isinstance(zones, list)

    def test_detect_insufficient_data(self, detector):
        """Test with insufficient data."""
        short_df = pd.DataFrame(
            {
                "open": [100] * 10,
                "high": [102] * 10,
                "low": [98] * 10,
                "close": [101] * 10,
            }
        )
        zones = detector.detect(short_df)
        assert isinstance(zones, list)


class TestTechnicalAnalysisResult:
    """Test TechnicalAnalysisResult dataclass."""

    def test_result_creation(self):
        """Test creating analysis result."""
        result = TechnicalAnalysisResult(
            candlestick_patterns=[],
            divergences=[],
            confluence_zones=[],
            nearest_support=None,
            nearest_resistance=None,
            signal_boost=0.5,
            summary="Bullish bias",
        )
        assert result.signal_boost == 0.5
        assert result.summary == "Bullish bias"

    def test_result_with_data(self):
        """Test result with actual data."""
        pattern = PatternResult(
            pattern=CandlePattern.HAMMER,
            direction="bullish",
            strength=0.8,
            bar_index=10,
            description="test",
        )
        sr_level = SupportResistanceLevel(
            price=100.0,
            level_type="support",
            strength=3,
            sources=["ema_20"],
        )
        zone = ConfluenceZone(
            price_low=99.0,
            price_high=101.0,
            center=100.0,
            zone_type="support",
            strength=0.8,
            levels=[sr_level],
            description="Support zone",
        )

        result = TechnicalAnalysisResult(
            candlestick_patterns=[pattern],
            divergences=[],
            confluence_zones=[zone],
            nearest_support=zone,
            nearest_resistance=None,
            signal_boost=0.3,
            summary="Test summary",
        )
        assert len(result.candlestick_patterns) == 1
        assert result.nearest_support is not None


class TestTechnicalAnalyzer:
    """Test TechnicalAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create technical analyzer."""
        return TechnicalAnalyzer()

    @pytest.fixture
    def sample_ohlcv(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="h")

        base_price = 100.0
        returns = np.random.randn(n) * 0.01
        prices = base_price * np.exp(np.cumsum(returns))

        return pd.DataFrame(
            {
                "open": prices * (1 + np.random.randn(n) * 0.002),
                "high": prices * (1 + np.abs(np.random.randn(n)) * 0.005),
                "low": prices * (1 - np.abs(np.random.randn(n)) * 0.005),
                "close": prices,
                "volume": np.random.randint(1000, 10000, n),
            },
            index=dates,
        )

    def test_analyzer_creation(self, analyzer):
        """Test analyzer is created."""
        assert analyzer is not None
        assert analyzer.candle_analyzer is not None
        assert analyzer.divergence_detector is not None
        assert analyzer.confluence_detector is not None

    def test_analyze_returns_result(self, analyzer, sample_ohlcv):
        """Test analyze returns TechnicalAnalysisResult."""
        result = analyzer.analyze(sample_ohlcv)
        assert isinstance(result, TechnicalAnalysisResult)

    def test_analyze_result_has_all_fields(self, analyzer, sample_ohlcv):
        """Test result has all expected fields."""
        result = analyzer.analyze(sample_ohlcv)

        assert hasattr(result, "candlestick_patterns")
        assert hasattr(result, "divergences")
        assert hasattr(result, "confluence_zones")
        assert hasattr(result, "nearest_support")
        assert hasattr(result, "nearest_resistance")
        assert hasattr(result, "signal_boost")
        assert hasattr(result, "summary")

    def test_analyze_short_data(self, analyzer):
        """Test analyze with short data."""
        # Create very short data
        short_df = pd.DataFrame(
            {
                "open": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "high": [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
                "low": [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
                "close": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            }
        )
        result = analyzer.analyze(short_df)

        # Should handle short data gracefully
        assert isinstance(result, TechnicalAnalysisResult)
