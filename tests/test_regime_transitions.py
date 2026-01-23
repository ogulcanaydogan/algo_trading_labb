"""
Tests for Regime Transitions Module.

Tests the regime transition analysis functionality including
transition matrix calculations, regime state tracking, and predictions.
"""

import pytest
import tempfile
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

from bot.regime_transitions import (
    RegimeState,
    Transition,
    TransitionMatrix,
    RegimeTransitionAnalyzer,
)


class TestRegimeState:
    """Tests for RegimeState dataclass."""

    def test_regime_state_creation(self):
        """Test creating a RegimeState."""
        state = RegimeState(
            name="BULL",
            description="Strong upward trend",
            color="#4CAF50",
            avg_duration_hours=24.0,
            frequency=30.0,
            avg_return=0.5,
            avg_volatility=0.2,
        )

        assert state.name == "BULL"
        assert state.description == "Strong upward trend"
        assert state.color == "#4CAF50"
        assert state.avg_duration_hours == 24.0
        assert state.frequency == 30.0

    def test_regime_state_defaults(self):
        """Test RegimeState default values."""
        state = RegimeState(
            name="TEST",
            description="Test regime",
            color="#000000",
        )

        assert state.avg_duration_hours == 0.0
        assert state.frequency == 0.0
        assert state.avg_return == 0.0
        assert state.avg_volatility == 0.0


class TestTransition:
    """Tests for Transition dataclass."""

    def test_transition_creation(self):
        """Test creating a Transition."""
        now = datetime.now()
        transition = Transition(
            from_regime="BULL",
            to_regime="BEAR",
            timestamp=now,
            price_at_transition=50000.0,
            confidence=0.85,
        )

        assert transition.from_regime == "BULL"
        assert transition.to_regime == "BEAR"
        assert transition.timestamp == now
        assert transition.price_at_transition == 50000.0
        assert transition.confidence == 0.85


class TestTransitionMatrix:
    """Tests for TransitionMatrix dataclass."""

    def test_transition_matrix_creation(self):
        """Test creating a TransitionMatrix."""
        matrix = TransitionMatrix(
            matrix={
                "BULL": {"BULL": 0.7, "BEAR": 0.3},
                "BEAR": {"BULL": 0.4, "BEAR": 0.6},
            },
            regimes=["BULL", "BEAR"],
            total_transitions=100,
            transition_counts={
                "BULL": {"BULL": 70, "BEAR": 30},
                "BEAR": {"BULL": 40, "BEAR": 60},
            },
        )

        assert len(matrix.regimes) == 2
        assert matrix.total_transitions == 100
        assert matrix.matrix["BULL"]["BEAR"] == 0.3


class TestRegimeTransitionAnalyzer:
    """Tests for RegimeTransitionAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return RegimeTransitionAnalyzer()

    @pytest.fixture
    def sample_regimes(self) -> List[str]:
        """Create sample regime history."""
        return [
            "BULL",
            "BULL",
            "BULL",
            "VOLATILE",
            "VOLATILE",
            "BEAR",
            "BEAR",
            "BEAR",
            "BEAR",
            "ACCUMULATION",
            "ACCUMULATION",
            "BULL",
            "BULL",
            "BULL",
            "NEUTRAL",
        ]

    @pytest.fixture
    def sample_timestamps(self) -> List[datetime]:
        """Create sample timestamps."""
        base = datetime(2024, 1, 1)
        return [base + timedelta(hours=i) for i in range(15)]

    @pytest.fixture
    def sample_prices(self) -> List[float]:
        """Create sample prices."""
        return [
            50000,
            51000,
            52000,
            51500,
            50500,
            49000,
            48000,
            47000,
            46500,
            47000,
            47500,
            49000,
            50000,
            51000,
            51000,
        ]

    def test_init(self, analyzer):
        """Test analyzer initialization."""
        assert len(analyzer._transitions) == 0
        assert len(analyzer._regime_history) == 0
        assert len(analyzer._regime_states) == 0

    def test_regime_colors_defined(self, analyzer):
        """Test regime colors are defined."""
        assert "BULL" in analyzer.REGIME_COLORS
        assert "BEAR" in analyzer.REGIME_COLORS
        assert "VOLATILE" in analyzer.REGIME_COLORS
        assert analyzer.REGIME_COLORS["BULL"] == "#4CAF50"

    def test_regime_descriptions_defined(self, analyzer):
        """Test regime descriptions are defined."""
        assert "BULL" in analyzer.REGIME_DESCRIPTIONS
        assert "BEAR" in analyzer.REGIME_DESCRIPTIONS
        assert len(analyzer.REGIME_DESCRIPTIONS["BULL"]) > 0

    def test_load_regime_history_basic(self, analyzer, sample_regimes):
        """Test loading regime history."""
        analyzer.load_regime_history(sample_regimes)

        assert len(analyzer._regime_history) == len(sample_regimes)

    def test_load_regime_history_with_timestamps(self, analyzer, sample_regimes, sample_timestamps):
        """Test loading regime history with timestamps."""
        analyzer.load_regime_history(sample_regimes, sample_timestamps)

        assert len(analyzer._regime_history) == len(sample_regimes)
        assert analyzer._regime_history[0][0] == sample_timestamps[0]

    def test_load_regime_history_with_prices(
        self, analyzer, sample_regimes, sample_timestamps, sample_prices
    ):
        """Test loading regime history with prices."""
        analyzer.load_regime_history(sample_regimes, sample_timestamps, sample_prices)

        assert analyzer._regime_history[0][2] == sample_prices[0]

    def test_load_regime_history_empty(self, analyzer):
        """Test loading empty regime history."""
        analyzer.load_regime_history([])

        assert len(analyzer._regime_history) == 0

    def test_load_regime_history_generates_timestamps(self, analyzer, sample_regimes):
        """Test that timestamps are generated when not provided."""
        analyzer.load_regime_history(sample_regimes)

        # Should have generated timestamps
        for entry in analyzer._regime_history:
            assert entry[0] is not None

    def test_load_regime_history_generates_prices(
        self, analyzer, sample_regimes, sample_timestamps
    ):
        """Test that prices default to 0 when not provided."""
        analyzer.load_regime_history(sample_regimes, sample_timestamps)

        # All prices should be 0.0
        for entry in analyzer._regime_history:
            assert entry[2] == 0.0

    def test_detect_transitions(self, analyzer, sample_regimes, sample_timestamps, sample_prices):
        """Test transition detection."""
        analyzer.load_regime_history(sample_regimes, sample_timestamps, sample_prices)

        # Should detect transitions: BULL->VOLATILE, VOLATILE->BEAR,
        # BEAR->ACCUMULATION, ACCUMULATION->BULL, BULL->NEUTRAL
        assert len(analyzer._transitions) > 0

    def test_transition_properties(
        self, analyzer, sample_regimes, sample_timestamps, sample_prices
    ):
        """Test that transitions have correct properties."""
        analyzer.load_regime_history(sample_regimes, sample_timestamps, sample_prices)

        for transition in analyzer._transitions:
            assert transition.from_regime != transition.to_regime
            assert transition.timestamp is not None
            assert transition.confidence == 1.0

    def test_calculate_regime_states(
        self, analyzer, sample_regimes, sample_timestamps, sample_prices
    ):
        """Test regime state calculation."""
        analyzer.load_regime_history(sample_regimes, sample_timestamps, sample_prices)

        assert len(analyzer._regime_states) > 0
        assert "BULL" in analyzer._regime_states

    def test_regime_state_frequency(
        self, analyzer, sample_regimes, sample_timestamps, sample_prices
    ):
        """Test regime frequency calculation."""
        analyzer.load_regime_history(sample_regimes, sample_timestamps, sample_prices)

        total_frequency = sum(state.frequency for state in analyzer._regime_states.values())

        # Total frequency should be ~100%
        assert 99 <= total_frequency <= 101

    def test_calculate_transition_matrix(
        self, analyzer, sample_regimes, sample_timestamps, sample_prices
    ):
        """Test transition matrix calculation."""
        analyzer.load_regime_history(sample_regimes, sample_timestamps, sample_prices)

        matrix = analyzer.calculate_transition_matrix()

        assert isinstance(matrix, TransitionMatrix)
        assert len(matrix.regimes) > 0
        assert matrix.total_transitions > 0

    def test_transition_matrix_probabilities_sum_to_one(
        self, analyzer, sample_regimes, sample_timestamps, sample_prices
    ):
        """Test that transition probabilities sum to 1."""
        analyzer.load_regime_history(sample_regimes, sample_timestamps, sample_prices)

        matrix = analyzer.calculate_transition_matrix()

        for from_regime in matrix.regimes:
            prob_sum = sum(matrix.matrix[from_regime].values())
            assert abs(prob_sum - 1.0) < 0.01

    def test_transition_matrix_empty_history(self, analyzer):
        """Test transition matrix with empty history."""
        matrix = analyzer.calculate_transition_matrix()

        # Should return default regimes
        assert len(matrix.regimes) > 0
        assert matrix.total_transitions == 0

    def test_get_current_regime(self, analyzer, sample_regimes, sample_timestamps, sample_prices):
        """Test getting current regime."""
        analyzer.load_regime_history(sample_regimes, sample_timestamps, sample_prices)

        current = analyzer.get_current_regime()

        assert current == sample_regimes[-1]

    def test_get_current_regime_empty(self, analyzer):
        """Test getting current regime with empty history."""
        current = analyzer.get_current_regime()

        assert current is None

    def test_get_transition_history(
        self, analyzer, sample_regimes, sample_timestamps, sample_prices
    ):
        """Test getting transition history."""
        analyzer.load_regime_history(sample_regimes, sample_timestamps, sample_prices)

        history = analyzer.get_transition_history()

        assert len(history) > 0
        for entry in history:
            assert "from" in entry
            assert "to" in entry
            assert "timestamp" in entry
            assert "from_color" in entry
            assert "to_color" in entry

    def test_get_transition_history_limit(
        self, analyzer, sample_regimes, sample_timestamps, sample_prices
    ):
        """Test transition history limit."""
        analyzer.load_regime_history(sample_regimes, sample_timestamps, sample_prices)

        history = analyzer.get_transition_history(limit=2)

        assert len(history) <= 2

    def test_predict_next_regime(self, analyzer, sample_regimes, sample_timestamps, sample_prices):
        """Test regime prediction."""
        analyzer.load_regime_history(sample_regimes, sample_timestamps, sample_prices)

        predictions = analyzer.predict_next_regime()

        assert len(predictions) > 0
        # Probabilities should sum to 1
        if predictions:
            prob_sum = sum(predictions.values())
            assert abs(prob_sum - 1.0) < 0.01

    def test_predict_next_regime_with_override(
        self, analyzer, sample_regimes, sample_timestamps, sample_prices
    ):
        """Test regime prediction with override."""
        analyzer.load_regime_history(sample_regimes, sample_timestamps, sample_prices)

        predictions = analyzer.predict_next_regime(current_regime="BULL")

        assert len(predictions) > 0

    def test_predict_next_regime_empty_history(self, analyzer):
        """Test prediction with empty history."""
        predictions = analyzer.predict_next_regime()

        assert predictions == {}

    def test_predict_next_regime_unknown_regime(
        self, analyzer, sample_regimes, sample_timestamps, sample_prices
    ):
        """Test prediction with unknown regime."""
        analyzer.load_regime_history(sample_regimes, sample_timestamps, sample_prices)

        predictions = analyzer.predict_next_regime(current_regime="UNKNOWN_REGIME")

        assert predictions == {}

    def test_get_regime_timeline(self, analyzer, sample_regimes, sample_timestamps, sample_prices):
        """Test getting regime timeline."""
        analyzer.load_regime_history(sample_regimes, sample_timestamps, sample_prices)

        timeline = analyzer.get_regime_timeline()

        assert len(timeline) == len(sample_regimes)
        for entry in timeline:
            assert "timestamp" in entry
            assert "regime" in entry
            assert "price" in entry
            assert "color" in entry

    def test_get_regime_timeline_limit(
        self, analyzer, sample_regimes, sample_timestamps, sample_prices
    ):
        """Test regime timeline limit."""
        analyzer.load_regime_history(sample_regimes, sample_timestamps, sample_prices)

        timeline = analyzer.get_regime_timeline(limit=5)

        assert len(timeline) == 5

    def test_get_regime_distribution(
        self, analyzer, sample_regimes, sample_timestamps, sample_prices
    ):
        """Test regime distribution."""
        analyzer.load_regime_history(sample_regimes, sample_timestamps, sample_prices)

        distribution = analyzer.get_regime_distribution()

        assert len(distribution) > 0
        for regime, info in distribution.items():
            assert "name" in info
            assert "description" in info
            assert "color" in info
            assert "frequency_pct" in info
            assert "avg_duration_hours" in info

    def test_to_api_response(self, analyzer, sample_regimes, sample_timestamps, sample_prices):
        """Test API response generation."""
        analyzer.load_regime_history(sample_regimes, sample_timestamps, sample_prices)

        response = analyzer.to_api_response()

        assert "current_regime" in response
        assert "next_regime_probabilities" in response
        assert "transition_matrix" in response
        assert "regime_distribution" in response
        assert "recent_transitions" in response
        assert "timeline" in response
        assert "regime_info" in response

    def test_to_api_response_matrix_structure(
        self, analyzer, sample_regimes, sample_timestamps, sample_prices
    ):
        """Test API response transition matrix structure."""
        analyzer.load_regime_history(sample_regimes, sample_timestamps, sample_prices)

        response = analyzer.to_api_response()
        matrix = response["transition_matrix"]

        assert "regimes" in matrix
        assert "probabilities" in matrix
        assert "counts" in matrix
        assert "total_transitions" in matrix


class TestJSONLoading:
    """Tests for JSON file loading."""

    def test_load_from_json_list_format(self):
        """Test loading from JSON list format."""
        analyzer = RegimeTransitionAnalyzer()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            data = [
                {"regime": "BULL", "timestamp": "2024-01-01T10:00:00", "price": 50000},
                {"regime": "BEAR", "timestamp": "2024-01-01T11:00:00", "price": 49000},
                {"regime": "BEAR", "timestamp": "2024-01-01T12:00:00", "price": 48000},
            ]
            json.dump(data, f)
            temp_path = f.name

        try:
            analyzer.load_from_json(temp_path)

            assert len(analyzer._regime_history) == 3
            assert analyzer.get_current_regime() == "BEAR"
        finally:
            Path(temp_path).unlink()

    def test_load_from_json_dict_format(self):
        """Test loading from JSON dict format."""
        analyzer = RegimeTransitionAnalyzer()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            data = {
                "regimes": ["BULL", "VOLATILE", "BEAR"],
                "timestamps": ["2024-01-01T10:00:00", "2024-01-01T11:00:00", "2024-01-01T12:00:00"],
                "prices": [50000, 49500, 48000],
            }
            json.dump(data, f)
            temp_path = f.name

        try:
            analyzer.load_from_json(temp_path)

            assert len(analyzer._regime_history) == 3
        finally:
            Path(temp_path).unlink()

    def test_load_from_json_missing_file(self):
        """Test loading from non-existent file."""
        analyzer = RegimeTransitionAnalyzer()

        analyzer.load_from_json("/nonexistent/path/file.json")

        assert len(analyzer._regime_history) == 0

    def test_load_from_json_invalid_json(self):
        """Test loading from invalid JSON file."""
        analyzer = RegimeTransitionAnalyzer()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json {")
            temp_path = f.name

        try:
            analyzer.load_from_json(temp_path)
            assert len(analyzer._regime_history) == 0
        finally:
            Path(temp_path).unlink()

    def test_load_from_json_missing_timestamp(self):
        """Test loading from JSON with missing timestamps."""
        analyzer = RegimeTransitionAnalyzer()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            data = [
                {"regime": "BULL", "price": 50000},
                {"regime": "BEAR", "price": 49000},
            ]
            json.dump(data, f)
            temp_path = f.name

        try:
            analyzer.load_from_json(temp_path)

            # Should still load with generated timestamps
            assert len(analyzer._regime_history) == 2
        finally:
            Path(temp_path).unlink()


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_regime_no_transitions(self):
        """Test with single regime throughout history."""
        analyzer = RegimeTransitionAnalyzer()
        regimes = ["BULL"] * 10

        analyzer.load_regime_history(regimes)

        assert len(analyzer._transitions) == 0
        assert "BULL" in analyzer._regime_states
        assert analyzer._regime_states["BULL"].frequency == 100.0

    def test_alternating_regimes(self):
        """Test with rapidly alternating regimes."""
        analyzer = RegimeTransitionAnalyzer()
        regimes = ["BULL", "BEAR"] * 10

        analyzer.load_regime_history(regimes)

        # Should detect 19 transitions (20 regimes - 1)
        assert len(analyzer._transitions) == 19

    def test_all_unique_regimes(self):
        """Test with all unique regimes."""
        analyzer = RegimeTransitionAnalyzer()
        regimes = ["BULL", "BEAR", "VOLATILE", "ACCUMULATION", "NEUTRAL"]

        analyzer.load_regime_history(regimes)

        assert len(analyzer._transitions) == 4
        assert len(analyzer._regime_states) == 5

    def test_regime_return_calculation_zero_prices(self):
        """Test return calculation with zero prices."""
        analyzer = RegimeTransitionAnalyzer()
        regimes = ["BULL", "BEAR"]
        prices = [0.0, 0.0]  # Zero prices

        analyzer.load_regime_history(regimes, prices=prices)

        # Should not crash, returns should be 0
        for state in analyzer._regime_states.values():
            assert state.avg_return == 0.0

    def test_very_long_history(self):
        """Test with long regime history."""
        analyzer = RegimeTransitionAnalyzer()

        import numpy as np

        np.random.seed(42)
        choices = ["BULL", "BEAR", "VOLATILE", "NEUTRAL"]
        regimes = [choices[np.random.randint(0, len(choices))] for _ in range(1000)]

        analyzer.load_regime_history(regimes)

        matrix = analyzer.calculate_transition_matrix()
        assert matrix.total_transitions > 0

    def test_unknown_regime_handling(self):
        """Test handling of unknown regimes."""
        analyzer = RegimeTransitionAnalyzer()
        regimes = ["CUSTOM_REGIME_1", "CUSTOM_REGIME_2"]

        analyzer.load_regime_history(regimes)

        # Should still work with custom regimes
        assert len(analyzer._regime_states) == 2

        # Colors should default to unknown color
        for state in analyzer._regime_states.values():
            assert state.color == "#616161"

    def test_duration_calculation(self):
        """Test regime duration calculation."""
        analyzer = RegimeTransitionAnalyzer()
        base = datetime(2024, 1, 1)

        regimes = ["BULL", "BULL", "BULL", "BEAR", "BEAR"]
        timestamps = [base + timedelta(hours=i) for i in range(5)]

        analyzer.load_regime_history(regimes, timestamps)

        # BULL duration should be 3 hours
        bull_state = analyzer._regime_states.get("BULL")
        assert bull_state is not None
        assert bull_state.avg_duration_hours > 0

    def test_volatility_calculation(self):
        """Test volatility calculation with varying prices."""
        analyzer = RegimeTransitionAnalyzer()
        base = datetime(2024, 1, 1)

        regimes = ["VOLATILE"] * 10
        timestamps = [base + timedelta(hours=i) for i in range(10)]
        prices = [100, 105, 95, 110, 90, 108, 92, 106, 94, 100]

        analyzer.load_regime_history(regimes, timestamps, prices)

        volatile_state = analyzer._regime_states.get("VOLATILE")
        assert volatile_state is not None
        assert volatile_state.avg_volatility > 0

    def test_matrix_default_probability_for_unseen_transitions(self):
        """Test that unseen transitions have correct probability."""
        analyzer = RegimeTransitionAnalyzer()
        regimes = ["BULL", "BULL", "BULL", "BEAR"]

        analyzer.load_regime_history(regimes)

        matrix = analyzer.calculate_transition_matrix()

        # BEAR -> BEAR should be 1.0 (stay) since no transitions from BEAR observed
        if "BEAR" in matrix.matrix:
            assert matrix.matrix["BEAR"]["BEAR"] == 1.0
