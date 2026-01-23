"""Tests for bot.drawdown_analysis module."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import pytest

from bot.drawdown_analysis import (
    DrawdownPeriod,
    DrawdownAnalysis,
    DrawdownAnalyzer,
)


class TestDrawdownPeriod:
    """Tests for DrawdownPeriod dataclass."""

    def test_creation(self) -> None:
        """Test DrawdownPeriod creation."""
        now = datetime.now()
        period = DrawdownPeriod(
            start_date=now,
            end_date=now + timedelta(days=5),
            recovery_date=now + timedelta(days=10),
            peak_value=10000.0,
            trough_value=9000.0,
            drawdown_percent=10.0,
            duration_days=5.0,
            recovery_days=5.0,
            is_recovered=True,
            peak_index=0,
            trough_index=5,
        )

        assert period.peak_value == 10000.0
        assert period.trough_value == 9000.0
        assert period.drawdown_percent == 10.0
        assert period.is_recovered is True


class TestDrawdownAnalysis:
    """Tests for DrawdownAnalysis dataclass."""

    def test_creation(self) -> None:
        """Test DrawdownAnalysis creation."""
        analysis = DrawdownAnalysis(
            max_drawdown=15.5,
            max_drawdown_date=datetime.now(),
            max_drawdown_duration=10.0,
            avg_drawdown=5.0,
            avg_recovery_time=3.0,
            current_drawdown=2.0,
            current_drawdown_start=None,
            drawdown_periods=[],
            underwater_curve=[],
            drawdown_distribution={},
            recovery_factor=2.5,
            ulcer_index=3.0,
            pain_index=2.0,
            calmar_ratio=1.5,
            time_in_drawdown_percent=30.0,
        )

        assert analysis.max_drawdown == 15.5
        assert analysis.recovery_factor == 2.5


class TestDrawdownAnalyzer:
    """Tests for DrawdownAnalyzer class."""

    @pytest.fixture
    def analyzer(self) -> DrawdownAnalyzer:
        """Create analyzer instance."""
        return DrawdownAnalyzer()

    @pytest.fixture
    def sample_equity(self) -> List[float]:
        """Create sample equity curve with drawdowns."""
        # Equity curve: 10000 -> 10500 -> 9500 -> 9000 -> 10000 -> 10500
        return [
            10000,
            10100,
            10200,
            10300,
            10400,
            10500,  # Growth
            10400,
            10200,
            10000,
            9800,
            9600,
            9500,  # Drawdown 1
            9600,
            9700,
            9800,
            9900,
            10000,  # Recovery
            10100,
            10200,
            10300,
            10400,
            10500,  # Growth
            10400,
            10200,
            10000,
            9800,
            9500,
            9200,
            9000,  # Drawdown 2
            9200,
            9400,
            9600,
            9800,
            10000,  # Recovery
            10100,
            10200,
            10300,
            10400,
            10500,
            10600,  # Growth
        ]

    def test_init(self, analyzer: DrawdownAnalyzer) -> None:
        """Test analyzer initialization."""
        assert analyzer._equity_curve == []
        assert analyzer._dates == []

    def test_load_equity_curve(self, analyzer: DrawdownAnalyzer) -> None:
        """Test loading equity curve from values."""
        equity = [10000, 10100, 10200, 10300]

        analyzer.load_equity_curve(equity)

        assert analyzer._equity_curve == equity
        assert len(analyzer._dates) == 4

    def test_load_equity_curve_with_dates(self, analyzer: DrawdownAnalyzer) -> None:
        """Test loading equity curve with custom dates."""
        equity = [10000, 10100, 10200]
        dates = [
            datetime(2024, 1, 1),
            datetime(2024, 1, 2),
            datetime(2024, 1, 3),
        ]

        analyzer.load_equity_curve(equity, dates)

        assert analyzer._dates == dates

    def test_load_equity_from_json_list(
        self,
        analyzer: DrawdownAnalyzer,
        tmp_path: Path,
    ) -> None:
        """Test loading equity from JSON with list format."""
        json_path = tmp_path / "equity.json"
        equity = [10000, 10100, 10200, 10300]

        with open(json_path, "w") as f:
            json.dump(equity, f)

        analyzer.load_equity_from_json(str(json_path))

        assert analyzer._equity_curve == equity

    def test_load_equity_from_json_dict_curve(
        self,
        analyzer: DrawdownAnalyzer,
        tmp_path: Path,
    ) -> None:
        """Test loading equity from JSON with dict format."""
        json_path = tmp_path / "equity.json"
        data = {"curve": [10000, 10100, 10200]}

        with open(json_path, "w") as f:
            json.dump(data, f)

        analyzer.load_equity_from_json(str(json_path))

        assert analyzer._equity_curve == [10000, 10100, 10200]

    def test_load_equity_from_json_dict_equity(
        self,
        analyzer: DrawdownAnalyzer,
        tmp_path: Path,
    ) -> None:
        """Test loading equity from JSON with equity key."""
        json_path = tmp_path / "equity.json"
        data = {"equity": [10000, 10100]}

        with open(json_path, "w") as f:
            json.dump(data, f)

        analyzer.load_equity_from_json(str(json_path))

        assert analyzer._equity_curve == [10000, 10100]

    def test_load_equity_from_json_with_timestamps(
        self,
        analyzer: DrawdownAnalyzer,
        tmp_path: Path,
    ) -> None:
        """Test loading equity from JSON with timestamps."""
        json_path = tmp_path / "equity.json"
        data = {
            "curve": [
                {"value": 10000, "timestamp": "2024-01-01T10:00:00"},
                {"value": 10100, "timestamp": "2024-01-02T10:00:00"},
            ]
        }

        with open(json_path, "w") as f:
            json.dump(data, f)

        analyzer.load_equity_from_json(str(json_path))

        assert analyzer._equity_curve == [10000, 10100]
        assert len(analyzer._dates) == 2

    def test_load_equity_from_json_nonexistent(
        self,
        analyzer: DrawdownAnalyzer,
    ) -> None:
        """Test loading from non-existent file."""
        analyzer.load_equity_from_json("/nonexistent/path.json")

        assert analyzer._equity_curve == []

    def test_analyze_empty(self, analyzer: DrawdownAnalyzer) -> None:
        """Test analysis with empty data."""
        result = analyzer.analyze()

        assert result.max_drawdown == 0
        assert result.max_drawdown_date is None
        assert result.drawdown_periods == []

    def test_analyze_single_value(self, analyzer: DrawdownAnalyzer) -> None:
        """Test analysis with single value."""
        analyzer.load_equity_curve([10000])

        result = analyzer.analyze()

        assert result.max_drawdown == 0

    def test_analyze_no_drawdown(self, analyzer: DrawdownAnalyzer) -> None:
        """Test analysis with only upward movement."""
        equity = [10000, 10100, 10200, 10300, 10400, 10500]
        analyzer.load_equity_curve(equity)

        result = analyzer.analyze()

        assert result.max_drawdown == 0
        assert result.current_drawdown == 0

    def test_analyze_with_drawdown(
        self,
        analyzer: DrawdownAnalyzer,
        sample_equity: List[float],
    ) -> None:
        """Test analysis with drawdowns."""
        analyzer.load_equity_curve(sample_equity)

        result = analyzer.analyze()

        assert result.max_drawdown > 0
        assert len(result.drawdown_periods) > 0
        assert len(result.underwater_curve) == len(sample_equity)
        assert result.time_in_drawdown_percent > 0

    def test_analyze_current_drawdown(self, analyzer: DrawdownAnalyzer) -> None:
        """Test detection of current ongoing drawdown."""
        # End in a drawdown
        equity = [10000, 10500, 10000, 9500]
        analyzer.load_equity_curve(equity)

        result = analyzer.analyze()

        assert result.current_drawdown > 0

    def test_analyze_recovery_factor(self, analyzer: DrawdownAnalyzer) -> None:
        """Test recovery factor calculation."""
        # Start at 10000, draw down, recover to 11000
        equity = [10000, 10500, 9000, 9500, 10000, 10500, 11000]
        analyzer.load_equity_curve(equity)

        result = analyzer.analyze()

        assert result.recovery_factor != 0

    def test_analyze_ulcer_index(
        self,
        analyzer: DrawdownAnalyzer,
        sample_equity: List[float],
    ) -> None:
        """Test ulcer index calculation."""
        analyzer.load_equity_curve(sample_equity)

        result = analyzer.analyze()

        assert result.ulcer_index >= 0

    def test_analyze_pain_index(
        self,
        analyzer: DrawdownAnalyzer,
        sample_equity: List[float],
    ) -> None:
        """Test pain index calculation."""
        analyzer.load_equity_curve(sample_equity)

        result = analyzer.analyze()

        assert result.pain_index >= 0

    def test_analyze_calmar_ratio_with_annual_return(
        self,
        analyzer: DrawdownAnalyzer,
        sample_equity: List[float],
    ) -> None:
        """Test Calmar ratio with provided annual return."""
        analyzer.load_equity_curve(sample_equity)

        result = analyzer.analyze(annual_return=20.0)

        # If max_drawdown > 0, calmar_ratio should be calculated
        if result.max_drawdown > 0:
            assert result.calmar_ratio != 0

    def test_drawdown_distribution(
        self,
        analyzer: DrawdownAnalyzer,
        sample_equity: List[float],
    ) -> None:
        """Test drawdown distribution calculation."""
        analyzer.load_equity_curve(sample_equity)

        result = analyzer.analyze()

        assert "0-1%" in result.drawdown_distribution
        assert "1-5%" in result.drawdown_distribution
        assert "5-10%" in result.drawdown_distribution
        assert "10-20%" in result.drawdown_distribution
        assert "20-30%" in result.drawdown_distribution
        assert "30%+" in result.drawdown_distribution

    def test_drawdown_periods_have_correct_structure(
        self,
        analyzer: DrawdownAnalyzer,
        sample_equity: List[float],
    ) -> None:
        """Test that drawdown periods have correct structure."""
        analyzer.load_equity_curve(sample_equity)

        result = analyzer.analyze()

        for period in result.drawdown_periods:
            assert period.start_date is not None
            assert period.peak_value > 0
            assert period.trough_value > 0
            assert period.drawdown_percent >= 0

    def test_get_top_drawdowns(
        self,
        analyzer: DrawdownAnalyzer,
        sample_equity: List[float],
    ) -> None:
        """Test getting top N drawdowns."""
        analyzer.load_equity_curve(sample_equity)

        top_5 = analyzer.get_top_drawdowns(n=5)

        assert len(top_5) <= 5
        # Should be sorted by drawdown_percent descending
        for i in range(len(top_5) - 1):
            assert top_5[i]["drawdown_percent"] >= top_5[i + 1]["drawdown_percent"]

    def test_get_top_drawdowns_with_limit(
        self,
        analyzer: DrawdownAnalyzer,
    ) -> None:
        """Test getting top drawdowns with smaller limit."""
        equity = [10000, 10500, 9500, 10000, 10200, 9800, 10000]
        analyzer.load_equity_curve(equity)

        top_1 = analyzer.get_top_drawdowns(n=1)

        assert len(top_1) <= 1

    def test_to_api_response(
        self,
        analyzer: DrawdownAnalyzer,
        sample_equity: List[float],
    ) -> None:
        """Test API response format."""
        analyzer.load_equity_curve(sample_equity)
        analysis = analyzer.analyze()

        response = analyzer.to_api_response(analysis)

        assert "summary" in response
        assert "risk_metrics" in response
        assert "distribution" in response
        assert "underwater_curve" in response
        assert "top_drawdowns" in response
        assert "total_drawdown_periods" in response
        assert "recovered_periods" in response

        # Check summary structure
        summary = response["summary"]
        assert "max_drawdown" in summary
        assert "avg_drawdown" in summary
        assert "current_drawdown" in summary

        # Check risk metrics
        risk = response["risk_metrics"]
        assert "recovery_factor" in risk
        assert "ulcer_index" in risk
        assert "pain_index" in risk
        assert "calmar_ratio" in risk

    def test_underwater_curve_limited_in_api_response(
        self,
        analyzer: DrawdownAnalyzer,
    ) -> None:
        """Test that underwater curve is limited to 100 entries in API response."""
        # Create long equity curve
        equity = [10000 + i for i in range(200)]
        analyzer.load_equity_curve(equity)
        analysis = analyzer.analyze()

        response = analyzer.to_api_response(analysis)

        assert len(response["underwater_curve"]) <= 100

    def test_ongoing_drawdown_in_periods(
        self,
        analyzer: DrawdownAnalyzer,
    ) -> None:
        """Test that ongoing drawdown is captured in periods."""
        # End with an ongoing drawdown
        equity = [10000, 10500, 10000, 9500, 9000]
        analyzer.load_equity_curve(equity)

        result = analyzer.analyze()

        # Should have at least one period
        assert len(result.drawdown_periods) >= 1

        # Last period should be unrecovered
        unrecovered = [p for p in result.drawdown_periods if not p.is_recovered]
        assert len(unrecovered) >= 1

    def test_max_drawdown_calculation(
        self,
        analyzer: DrawdownAnalyzer,
    ) -> None:
        """Test accurate max drawdown calculation."""
        # Peak at 10500, trough at 8925 = 15% drawdown
        equity = [10000, 10500, 10000, 9500, 9000, 8925, 9000, 10000]
        analyzer.load_equity_curve(equity)

        result = analyzer.analyze()

        # Max DD should be 15% (from 10500 to 8925)
        expected_max_dd = (10500 - 8925) / 10500 * 100
        assert abs(result.max_drawdown - expected_max_dd) < 0.1

    def test_avg_recovery_time(
        self,
        analyzer: DrawdownAnalyzer,
    ) -> None:
        """Test average recovery time calculation."""
        # Create equity with clear recovery patterns
        dates = [datetime(2024, 1, i) for i in range(1, 11)]
        equity = [10000, 10500, 9500, 9800, 10100, 10500, 10000, 10200, 10500, 10600]

        analyzer.load_equity_curve(equity, dates)

        result = analyzer.analyze()

        # Should have some recovery time calculated
        assert result.avg_recovery_time >= 0

    def test_time_in_drawdown_percent(
        self,
        analyzer: DrawdownAnalyzer,
        sample_equity: List[float],
    ) -> None:
        """Test time in drawdown percentage calculation."""
        analyzer.load_equity_curve(sample_equity)

        result = analyzer.analyze()

        # Should be between 0 and 100
        assert 0 <= result.time_in_drawdown_percent <= 100

    def test_max_drawdown_duration(
        self,
        analyzer: DrawdownAnalyzer,
    ) -> None:
        """Test max drawdown duration calculation."""
        dates = [datetime(2024, 1, i) for i in range(1, 16)]
        # 14 day drawdown period
        equity = [
            10000,
            10500,
            10200,
            9800,
            9400,
            9000,
            8800,
            9000,
            9200,
            9400,
            9600,
            9800,
            10000,
            10200,
            10500,
        ]

        analyzer.load_equity_curve(equity, dates)

        result = analyzer.analyze()

        # Should have calculated duration
        assert result.max_drawdown_duration >= 0

    def test_underwater_curve_structure(
        self,
        analyzer: DrawdownAnalyzer,
    ) -> None:
        """Test underwater curve has correct structure."""
        equity = [10000, 10500, 10000, 9500, 10000]
        analyzer.load_equity_curve(equity)

        result = analyzer.analyze()

        assert len(result.underwater_curve) == 5
        for point in result.underwater_curve:
            assert "date" in point
            assert "drawdown" in point
            assert "equity" in point
            assert "peak" in point
            assert isinstance(point["drawdown"], (int, float))

    def test_constant_equity_no_drawdown(
        self,
        analyzer: DrawdownAnalyzer,
    ) -> None:
        """Test with constant equity curve."""
        equity = [10000] * 10
        analyzer.load_equity_curve(equity)

        result = analyzer.analyze()

        assert result.max_drawdown == 0
        assert result.current_drawdown == 0

    def test_single_large_drawdown(
        self,
        analyzer: DrawdownAnalyzer,
    ) -> None:
        """Test with single large drawdown."""
        # 50% drawdown
        equity = [10000, 11000, 12000, 6000, 7000, 8000, 9000, 10000, 11000, 12000]
        analyzer.load_equity_curve(equity)

        result = analyzer.analyze()

        # Max drawdown should be 50% (6000 from 12000)
        expected_dd = (12000 - 6000) / 12000 * 100
        assert abs(result.max_drawdown - expected_dd) < 0.1

    def test_multiple_small_drawdowns(
        self,
        analyzer: DrawdownAnalyzer,
    ) -> None:
        """Test with multiple small drawdowns."""
        equity = [
            10000,
            10100,
            10000,
            10100,
            10050,  # Small oscillations
            10100,
            10000,
            10100,
            10050,
            10100,
        ]
        analyzer.load_equity_curve(equity)

        result = analyzer.analyze()

        # Should have small max drawdown
        assert result.max_drawdown < 2.0

    def test_recovery_factor_positive_return(
        self,
        analyzer: DrawdownAnalyzer,
    ) -> None:
        """Test recovery factor with positive overall return."""
        equity = [10000, 10500, 9500, 10000, 10500, 11000]
        analyzer.load_equity_curve(equity)

        result = analyzer.analyze()

        # Recovery factor = total_return / max_drawdown
        # Should be positive when total return is positive
        if result.max_drawdown > 0:
            assert result.recovery_factor != 0

    def test_recovery_factor_negative_return(
        self,
        analyzer: DrawdownAnalyzer,
    ) -> None:
        """Test recovery factor with negative overall return."""
        equity = [10000, 10500, 9500, 9000, 8500]
        analyzer.load_equity_curve(equity)

        result = analyzer.analyze()

        # Recovery factor should be negative when total return is negative
        if result.max_drawdown > 0:
            assert result.recovery_factor < 0

    def test_distribution_sum_equals_data_points(
        self,
        analyzer: DrawdownAnalyzer,
    ) -> None:
        """Test that distribution sums to total data points."""
        equity = [10000, 9500, 9000, 9500, 10000]
        analyzer.load_equity_curve(equity)

        result = analyzer.analyze()

        total = sum(result.drawdown_distribution.values())
        assert total == len(equity)

    def test_calmar_ratio_zero_drawdown(
        self,
        analyzer: DrawdownAnalyzer,
    ) -> None:
        """Test Calmar ratio when max drawdown is zero."""
        equity = [10000, 10100, 10200, 10300, 10400]
        analyzer.load_equity_curve(equity)

        result = analyzer.analyze()

        # With zero drawdown, Calmar ratio should be 0
        assert result.calmar_ratio == 0

    def test_api_response_top_drawdowns_sorted(
        self,
        analyzer: DrawdownAnalyzer,
        sample_equity: List[float],
    ) -> None:
        """Test that top drawdowns in API response are sorted."""
        analyzer.load_equity_curve(sample_equity)
        analysis = analyzer.analyze()

        response = analyzer.to_api_response(analysis)

        top_drawdowns = response["top_drawdowns"]
        for i in range(len(top_drawdowns) - 1):
            assert top_drawdowns[i]["drawdown_percent"] >= top_drawdowns[i + 1]["drawdown_percent"]

    def test_api_response_recovered_periods_count(
        self,
        analyzer: DrawdownAnalyzer,
        sample_equity: List[float],
    ) -> None:
        """Test recovered periods count in API response."""
        analyzer.load_equity_curve(sample_equity)
        analysis = analyzer.analyze()

        response = analyzer.to_api_response(analysis)

        # Recovered periods should be <= total periods
        assert response["recovered_periods"] <= response["total_drawdown_periods"]

    def test_drawdown_period_indices(
        self,
        analyzer: DrawdownAnalyzer,
    ) -> None:
        """Test that drawdown period indices are valid."""
        equity = [10000, 10500, 10000, 9500, 10000, 10500]
        analyzer.load_equity_curve(equity)

        result = analyzer.analyze()

        for period in result.drawdown_periods:
            assert 0 <= period.peak_index < len(equity)
            assert 0 <= period.trough_index < len(equity)
            assert period.trough_index >= period.peak_index

    def test_load_equity_from_json_with_date_key(
        self,
        analyzer: DrawdownAnalyzer,
        tmp_path: Path,
    ) -> None:
        """Test loading equity from JSON with date key instead of timestamp."""
        json_path = tmp_path / "equity.json"
        data = {
            "curve": [
                {"equity": 10000, "date": "2024-01-01T10:00:00"},
                {"equity": 10100, "date": "2024-01-02T10:00:00"},
            ]
        }

        with open(json_path, "w") as f:
            json.dump(data, f)

        analyzer.load_equity_from_json(str(json_path))

        assert analyzer._equity_curve == [10000, 10100]

    def test_large_equity_curve(
        self,
        analyzer: DrawdownAnalyzer,
    ) -> None:
        """Test with large equity curve."""
        import numpy as np

        np.random.seed(42)
        equity = [10000]
        for _ in range(999):
            equity.append(equity[-1] * (1 + np.random.normal(0.001, 0.02)))

        analyzer.load_equity_curve(equity)

        result = analyzer.analyze()

        assert len(result.underwater_curve) == 1000
        assert result.max_drawdown >= 0

    def test_negative_returns_only(
        self,
        analyzer: DrawdownAnalyzer,
    ) -> None:
        """Test with only negative returns."""
        equity = [10000, 9500, 9000, 8500, 8000, 7500]
        analyzer.load_equity_curve(equity)

        result = analyzer.analyze()

        # Should have ongoing drawdown
        assert result.current_drawdown > 0
        assert result.current_drawdown_start is not None

    def test_drawdown_percent_accuracy(
        self,
        analyzer: DrawdownAnalyzer,
    ) -> None:
        """Test drawdown percentage calculation accuracy."""
        # 20% drawdown: peak at 10000, trough at 8000
        equity = [9000, 10000, 9000, 8000, 9000, 10000]
        analyzer.load_equity_curve(equity)

        result = analyzer.analyze()

        expected_dd = (10000 - 8000) / 10000 * 100
        assert abs(result.max_drawdown - expected_dd) < 0.1

    def test_underwater_curve_drawdown_values(
        self,
        analyzer: DrawdownAnalyzer,
    ) -> None:
        """Test underwater curve drawdown values are non-negative."""
        equity = [10000, 10500, 10000, 9500, 10000, 10500]
        analyzer.load_equity_curve(equity)

        result = analyzer.analyze()

        for point in result.underwater_curve:
            assert point["drawdown"] >= 0

    def test_peak_value_tracking(
        self,
        analyzer: DrawdownAnalyzer,
    ) -> None:
        """Test that peak values in underwater curve are correct."""
        equity = [10000, 10500, 10000, 9500, 10000, 10500, 11000]
        analyzer.load_equity_curve(equity)

        result = analyzer.analyze()

        # Last point should have peak of 11000
        assert result.underwater_curve[-1]["peak"] == 11000

    def test_current_drawdown_at_peak(
        self,
        analyzer: DrawdownAnalyzer,
    ) -> None:
        """Test current drawdown is zero when at peak."""
        equity = [10000, 9500, 10000, 10500, 11000]
        analyzer.load_equity_curve(equity)

        result = analyzer.analyze()

        # Ending at all-time high
        assert result.current_drawdown == 0


class TestDrawdownEdgeCases:
    """Additional edge case tests for DrawdownAnalyzer."""

    def test_two_point_equity_curve(self) -> None:
        """Test with minimum two points."""
        analyzer = DrawdownAnalyzer()
        equity = [10000, 9000]
        analyzer.load_equity_curve(equity)

        result = analyzer.analyze()

        # Should calculate 10% drawdown
        assert result.max_drawdown == 10.0

    def test_alternating_up_down(self) -> None:
        """Test with alternating up/down movements."""
        analyzer = DrawdownAnalyzer()
        equity = [10000, 10100, 10000, 10100, 10000, 10100]
        analyzer.load_equity_curve(equity)

        result = analyzer.analyze()

        # Small drawdowns only
        assert result.max_drawdown < 2.0

    def test_sharp_v_recovery(self) -> None:
        """Test sharp V-shaped recovery."""
        analyzer = DrawdownAnalyzer()
        dates = [datetime(2024, 1, i) for i in range(1, 8)]
        equity = [10000, 10500, 8000, 10500, 11000, 11500, 12000]
        analyzer.load_equity_curve(equity, dates)

        result = analyzer.analyze()

        # Should have one recovered period
        recovered = [p for p in result.drawdown_periods if p.is_recovered]
        assert len(recovered) >= 1

    def test_double_bottom(self) -> None:
        """Test double bottom pattern."""
        analyzer = DrawdownAnalyzer()
        equity = [10000, 10500, 9000, 9500, 9000, 10000, 10500, 11000]
        analyzer.load_equity_curve(equity)

        result = analyzer.analyze()

        # Max drawdown should be from 10500 to 9000
        expected_dd = (10500 - 9000) / 10500 * 100
        assert abs(result.max_drawdown - expected_dd) < 0.5

    def test_very_small_drawdown(self) -> None:
        """Test with very small drawdown below threshold."""
        analyzer = DrawdownAnalyzer()
        equity = [10000, 10001, 10000, 10002, 10001, 10003]
        analyzer.load_equity_curve(equity)

        result = analyzer.analyze()

        # Very small drawdowns may not be counted as periods
        assert result.max_drawdown < 0.1
