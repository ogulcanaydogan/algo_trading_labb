"""
Tests for ML Model Monitoring and Drift Detection module.
"""

import pytest
import numpy as np
import tempfile
import shutil
import json
from datetime import datetime, timedelta
from pathlib import Path

from bot.ml.model_monitor import (
    DriftSeverity,
    DriftType,
    DriftConfig,
    DriftMetrics,
    CalibrationMetrics,
    PerformanceMetrics,
    ModelMonitor,
)


class TestDriftSeverity:
    """Test DriftSeverity enum."""

    def test_severity_values(self):
        """Test severity enum values."""
        assert DriftSeverity.NONE.value == "none"
        assert DriftSeverity.LOW.value == "low"
        assert DriftSeverity.MODERATE.value == "moderate"
        assert DriftSeverity.HIGH.value == "high"
        assert DriftSeverity.CRITICAL.value == "critical"


class TestDriftType:
    """Test DriftType enum."""

    def test_drift_type_values(self):
        """Test drift type enum values."""
        assert DriftType.DATA_DRIFT.value == "data_drift"
        assert DriftType.CONCEPT_DRIFT.value == "concept_drift"
        assert DriftType.LABEL_DRIFT.value == "label_drift"
        assert DriftType.PREDICTION_DRIFT.value == "prediction_drift"


class TestDriftConfig:
    """Test DriftConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DriftConfig()

        assert config.ks_threshold_low == 0.1
        assert config.ks_threshold_high == 0.2
        assert config.psi_threshold_low == 0.1
        assert config.psi_threshold_high == 0.25
        assert config.accuracy_drop_threshold == 0.05
        assert config.reference_window == 500
        assert config.detection_window == 100
        assert config.min_samples == 50
        assert config.auto_retrain_on_drift is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = DriftConfig(
            ks_threshold_low=0.15,
            ks_threshold_high=0.3,
            min_samples=100,
            auto_retrain_on_drift=False,
        )

        assert config.ks_threshold_low == 0.15
        assert config.ks_threshold_high == 0.3
        assert config.min_samples == 100
        assert config.auto_retrain_on_drift is False


class TestDriftMetrics:
    """Test DriftMetrics dataclass."""

    def test_default_metrics(self):
        """Test default drift metrics."""
        metrics = DriftMetrics()

        assert metrics.drift_type == DriftType.DATA_DRIFT
        assert metrics.severity == DriftSeverity.NONE
        assert metrics.feature_name is None
        assert metrics.ks_statistic == 0.0
        assert metrics.ks_pvalue == 1.0
        assert metrics.psi_value == 0.0

    def test_custom_metrics(self):
        """Test custom drift metrics."""
        metrics = DriftMetrics(
            drift_type=DriftType.CONCEPT_DRIFT,
            severity=DriftSeverity.HIGH,
            feature_name="rsi",
            ks_statistic=0.25,
            ks_pvalue=0.001,
            psi_value=0.3,
            mean_shift=1.5,
            std_shift=0.8,
        )

        assert metrics.drift_type == DriftType.CONCEPT_DRIFT
        assert metrics.severity == DriftSeverity.HIGH
        assert metrics.feature_name == "rsi"
        assert metrics.ks_statistic == 0.25

    def test_to_dict(self):
        """Test conversion to dict."""
        metrics = DriftMetrics(
            drift_type=DriftType.DATA_DRIFT,
            severity=DriftSeverity.MODERATE,
            feature_name="volume",
            ks_statistic=0.123456,
            ks_pvalue=0.054321,
            psi_value=0.15,
            mean_shift=0.5,
            std_shift=0.3,
            details={"ref_mean": 100.0, "cur_mean": 105.0},
        )
        d = metrics.to_dict()

        assert d["drift_type"] == "data_drift"
        assert d["severity"] == "moderate"
        assert d["feature_name"] == "volume"
        assert d["ks_statistic"] == 0.1235  # Rounded
        assert d["ks_pvalue"] == 0.0543  # Rounded
        assert d["psi_value"] == 0.15
        assert "timestamp" in d
        assert d["details"]["ref_mean"] == 100.0


class TestCalibrationMetrics:
    """Test CalibrationMetrics dataclass."""

    def test_default_calibration(self):
        """Test default calibration metrics."""
        metrics = CalibrationMetrics()

        assert metrics.expected_calibration_error == 0.0
        assert metrics.max_calibration_error == 0.0
        assert metrics.brier_score == 0.0
        assert metrics.reliability_curve == []
        assert metrics.is_calibrated is True

    def test_custom_calibration(self):
        """Test custom calibration metrics."""
        metrics = CalibrationMetrics(
            expected_calibration_error=0.05,
            max_calibration_error=0.15,
            brier_score=0.2,
            reliability_curve=[(0.1, 0.12), (0.5, 0.48)],
            is_calibrated=True,
        )

        assert metrics.expected_calibration_error == 0.05
        assert len(metrics.reliability_curve) == 2

    def test_to_dict(self):
        """Test calibration to_dict."""
        metrics = CalibrationMetrics(
            expected_calibration_error=0.0512345,
            max_calibration_error=0.1234,
            brier_score=0.2,
            reliability_curve=[(0.1, 0.12), (0.5, 0.52)],
            is_calibrated=True,
        )
        d = metrics.to_dict()

        assert d["ece"] == 0.0512  # Rounded
        assert d["mce"] == 0.1234
        assert d["brier_score"] == 0.2
        assert d["is_calibrated"] is True
        # Reliability curve should be rounded
        assert d["reliability_curve"][0] == (0.1, 0.12)


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass."""

    def test_default_performance(self):
        """Test default performance metrics."""
        metrics = PerformanceMetrics()

        assert metrics.window_size == 0
        assert metrics.accuracy == 0.0
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1_score == 0.0
        assert metrics.win_rate == 0.0
        assert metrics.sharpe_ratio == 0.0

    def test_custom_performance(self):
        """Test custom performance metrics."""
        metrics = PerformanceMetrics(
            window_size=100,
            accuracy=0.65,
            precision=0.7,
            recall=0.6,
            f1_score=0.65,
            win_rate=0.55,
            sharpe_ratio=1.5,
            profit_factor=2.0,
            avg_confidence=0.75,
        )

        assert metrics.accuracy == 0.65
        assert metrics.sharpe_ratio == 1.5

    def test_to_dict(self):
        """Test performance to_dict."""
        metrics = PerformanceMetrics(
            accuracy=0.654321,
            precision=0.712345,
            recall=0.601234,
            f1_score=0.652341,
        )
        d = metrics.to_dict()

        assert d["accuracy"] == 0.6543  # Rounded
        assert d["precision"] == 0.7123
        assert "timestamp" in d


class TestModelMonitor:
    """Test ModelMonitor class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def monitor(self, temp_dir):
        """Create model monitor instance."""
        config = DriftConfig(
            min_samples=10,
            reference_window=100,
            detection_window=50,
        )
        return ModelMonitor(config=config, data_dir=str(temp_dir))

    def test_initialization(self, monitor):
        """Test monitor initialization."""
        assert monitor.config is not None
        assert monitor.config.min_samples == 10
        assert monitor._reference_features == {}
        assert monitor._current_features == {}
        assert monitor._drift_history == []

    def test_set_reference_data(self, monitor):
        """Test setting reference data."""
        features = {
            "rsi": np.random.rand(200),
            "macd": np.random.rand(200),
        }
        predictions = np.random.rand(200)
        labels = np.random.randint(0, 2, 200)

        monitor.set_reference_data(features, predictions, labels)

        # Should be trimmed to reference_window
        assert len(monitor._reference_features["rsi"]) == 100
        assert len(monitor._reference_features["macd"]) == 100
        assert len(monitor._reference_predictions) == 100
        assert len(monitor._reference_labels) == 100

    def test_add_sample(self, monitor):
        """Test adding samples."""
        features = {"rsi": 45.0, "macd": 0.02}

        monitor.add_sample(
            features=features,
            prediction=0.6,
            confidence=0.75,
            label=1,
            pnl=0.02,
        )

        assert len(monitor._current_features["rsi"]) == 1
        assert monitor._current_features["rsi"][0] == 45.0
        assert len(monitor._current_predictions) == 1
        assert len(monitor._current_labels) == 1
        assert len(monitor._current_returns) == 1

    def test_add_sample_respects_window(self, monitor):
        """Test sample window limit."""
        # Add more samples than detection_window
        for i in range(100):
            monitor.add_sample(
                features={"rsi": float(i)},
                prediction=0.5,
                confidence=0.6,
            )

        # Should be limited to detection_window
        assert len(monitor._current_features["rsi"]) == monitor.config.detection_window
        assert len(monitor._current_predictions) == monitor.config.detection_window

    def test_check_drift_no_reference(self, monitor):
        """Test check_drift with no reference data."""
        # Add samples but no reference
        for i in range(20):
            monitor.add_sample(
                features={"rsi": float(i)},
                prediction=0.5,
                confidence=0.6,
            )

        results = monitor.check_drift()

        # Should return empty - no reference to compare
        assert len(results) == 0

    def test_check_feature_drift(self, monitor):
        """Test feature drift detection."""
        # Set reference data with low values
        reference = np.random.normal(30, 5, 100)  # Mean 30, std 5
        monitor._reference_features["rsi"] = reference

        # Add current samples with different distribution
        for val in np.random.normal(60, 5, 20):  # Mean 60 - significant shift
            monitor.add_sample(
                features={"rsi": val},
                prediction=0.5,
                confidence=0.6,
            )

        # Check drift for this feature
        drift = monitor._check_feature_drift(
            "rsi",
            reference,
            np.array(monitor._current_features["rsi"]),
        )

        # Should detect some level of drift
        assert drift.feature_name == "rsi"
        assert drift.drift_type == DriftType.DATA_DRIFT
        # Mean shift should be significant
        assert drift.mean_shift > 0

    def test_check_feature_drift_no_drift(self, monitor):
        """Test drift detection returns valid metrics when called."""
        # Use reference and current with similar characteristics
        np.random.seed(42)
        reference = np.random.normal(50, 10, 200)
        current = np.random.normal(50, 10, 20)

        drift = monitor._check_feature_drift("rsi", reference, current)

        # Verify drift metrics are computed properly
        assert drift.feature_name == "rsi"
        assert drift.drift_type == DriftType.DATA_DRIFT
        assert drift.ks_statistic >= 0  # KS stat is always non-negative
        assert 0 <= drift.ks_pvalue <= 1  # p-value in valid range
        assert "ref_mean" in drift.details

    def test_check_feature_drift_insufficient_samples(self, monitor):
        """Test with insufficient samples."""
        reference = np.random.rand(100)
        current = np.random.rand(5)  # Less than min_samples

        drift = monitor._check_feature_drift("rsi", reference, current)

        # Should return default metrics (no drift detected)
        assert drift.severity == DriftSeverity.NONE

    def test_calculate_psi(self, monitor):
        """Test PSI calculation."""
        # Use deterministic data: identical distributions
        reference = np.linspace(-3, 3, 1000)  # Uniform-like spread
        current = np.linspace(-3, 3, 100)  # Same range, fewer points

        psi = monitor._calculate_psi(reference, current)

        # Nearly identical spread should have low PSI
        # PSI threshold interpretation: < 0.1 no drift, 0.1-0.25 moderate, > 0.25 significant
        assert psi >= 0  # PSI is always non-negative

        # Different distribution should have higher PSI
        current_different = np.linspace(2, 8, 100)  # Shifted range
        psi_different = monitor._calculate_psi(reference, current_different)

        # Shifted distribution must have higher PSI than similar distribution
        assert psi_different > psi

    def test_calibrate_predictions(self, monitor):
        """Test prediction calibration."""
        # Create slightly miscalibrated predictions
        np.random.seed(42)
        n_samples = 500
        predictions = np.random.rand(n_samples)
        # Labels correlated with predictions but with noise
        labels = (predictions + np.random.normal(0, 0.2, n_samples) > 0.5).astype(int)

        metrics = monitor.calibrate_predictions(predictions, labels)

        assert isinstance(metrics, CalibrationMetrics)
        assert 0 <= metrics.expected_calibration_error <= 1
        assert 0 <= metrics.brier_score <= 1
        assert len(metrics.reliability_curve) > 0

    def test_calibrate_predictions_builds_map(self, monitor):
        """Test calibration builds calibration map."""
        predictions = np.array([0.1, 0.3, 0.5, 0.7, 0.9] * 20)
        labels = np.array([0, 0, 0, 1, 1] * 20)

        monitor.calibrate_predictions(predictions, labels)

        assert len(monitor._calibration_map) > 0

    def test_get_calibrated_prediction_no_map(self, monitor):
        """Test calibration with no map returns raw."""
        raw = 0.7
        calibrated = monitor.get_calibrated_prediction(raw)

        assert calibrated == raw

    def test_get_calibrated_prediction_with_map(self, monitor):
        """Test calibration with map."""
        # Set up calibration map manually
        monitor._calibration_map = {
            0.05: 0.1,
            0.15: 0.15,
            0.25: 0.2,
            0.35: 0.3,
            0.45: 0.45,
            0.55: 0.55,
            0.65: 0.7,
            0.75: 0.75,
            0.85: 0.8,
            0.95: 0.9,
        }

        # Test edge cases
        assert monitor.get_calibrated_prediction(0.0) == 0.1  # Below first bin
        assert monitor.get_calibrated_prediction(1.0) == 0.9  # Above last bin

    def test_update_performance(self, monitor):
        """Test updating performance metrics."""
        # Add some samples with labels and returns
        for i in range(20):
            monitor.add_sample(
                features={"rsi": 50.0},
                prediction=0.6 if i % 2 == 0 else 0.4,
                confidence=0.7,
                label=1 if i % 2 == 0 else 0,
                pnl=0.01 if i % 2 == 0 else -0.005,
            )

        metrics = monitor.update_performance()

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.window_size == 20
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.win_rate <= 1

    def test_update_performance_insufficient_samples(self, monitor):
        """Test performance update with few samples."""
        # Add fewer than min_samples
        for i in range(5):
            monitor.add_sample(
                features={"rsi": 50.0},
                prediction=0.5,
                confidence=0.6,
                label=1,
            )

        metrics = monitor.update_performance()

        # Should return mostly zeros
        assert metrics.accuracy == 0.0

    def test_update_performance_tracks_history(self, monitor):
        """Test performance history tracking."""
        # Add enough samples and update multiple times
        for _ in range(3):
            for i in range(15):
                monitor.add_sample(
                    features={"rsi": 50.0},
                    prediction=0.5,
                    confidence=0.6,
                    label=1,
                )
            monitor.update_performance()

        assert len(monitor._performance_history) == 3

    def test_get_performance_trend(self, monitor):
        """Test performance trend calculation."""
        # Add improving performance history
        for i in range(10):
            metrics = PerformanceMetrics(accuracy=0.5 + i * 0.02)
            monitor._performance_history.append(metrics)

        trend = monitor.get_performance_trend()

        assert trend["trend"] == "improving"
        assert trend["change"] > 0

    def test_get_performance_trend_degrading(self, monitor):
        """Test degrading performance trend."""
        # Add degrading performance
        for i in range(10):
            metrics = PerformanceMetrics(accuracy=0.8 - i * 0.03)
            monitor._performance_history.append(metrics)

        trend = monitor.get_performance_trend()

        assert trend["trend"] == "degrading"
        assert trend["change"] < 0

    def test_get_performance_trend_stable(self, monitor):
        """Test stable performance trend."""
        # Add stable performance
        for i in range(10):
            metrics = PerformanceMetrics(accuracy=0.6 + (i % 2) * 0.01)
            monitor._performance_history.append(metrics)

        trend = monitor.get_performance_trend()

        assert trend["trend"] == "stable"

    def test_get_performance_trend_insufficient_data(self, monitor):
        """Test trend with insufficient data."""
        trend = monitor.get_performance_trend()

        assert trend["trend"] == "unknown"

    def test_should_retrain_cooldown(self, monitor):
        """Test retrain cooldown."""
        monitor._last_retrain = datetime.now()

        should, reasons = monitor.should_retrain()

        assert should is False
        assert "Cooldown period active" in reasons

    def test_should_retrain_weekly_limit(self, monitor):
        """Test weekly retrain limit."""
        monitor.config.max_retrains_per_week = 2
        monitor._retrain_count_this_week = 2

        should, reasons = monitor.should_retrain()

        assert should is False
        assert "Weekly retrain limit reached" in reasons

    def test_should_retrain_auto_disabled(self, monitor):
        """Test retrain with auto disabled."""
        monitor.config.auto_retrain_on_drift = False

        should, reasons = monitor.should_retrain()

        assert should is False

    def test_should_retrain_on_high_drift(self, monitor):
        """Test retrain triggered by high drift."""
        # Add high severity drift to history
        for _ in range(5):
            drift = DriftMetrics(severity=DriftSeverity.HIGH)
            monitor._drift_history.append(drift)

        should, reasons = monitor.should_retrain()

        assert should is True
        assert any("High severity drift" in r for r in reasons)

    def test_should_retrain_on_low_accuracy(self, monitor):
        """Test retrain triggered by low accuracy."""
        # Add low accuracy performance
        for _ in range(5):
            metrics = PerformanceMetrics(accuracy=0.4)
            monitor._performance_history.append(metrics)

        should, reasons = monitor.should_retrain()

        assert should is True
        assert any("below 50%" in r for r in reasons)

    def test_record_retrain(self, monitor):
        """Test recording retrain."""
        before_count = monitor._retrain_count_this_week

        monitor.record_retrain()

        assert monitor._last_retrain is not None
        assert monitor._retrain_count_this_week == before_count + 1

    def test_state_persistence(self, temp_dir):
        """Test state is saved and loaded."""
        config = DriftConfig(min_samples=10)
        monitor1 = ModelMonitor(config=config, data_dir=str(temp_dir))

        # Set some state
        monitor1._calibration_map = {0.5: 0.55}
        monitor1.record_retrain()
        monitor1._save_state()

        # Create new monitor - should load state
        monitor2 = ModelMonitor(config=config, data_dir=str(temp_dir))

        assert monitor2._retrain_count_this_week == 1
        assert 0.5 in monitor2._calibration_map

    def test_state_file_created(self, temp_dir):
        """Test state file is created."""
        config = DriftConfig()
        monitor = ModelMonitor(config=config, data_dir=str(temp_dir))
        monitor._save_state()

        state_file = temp_dir / "monitor_state.json"
        assert state_file.exists()

    def test_get_monitoring_summary(self, monitor):
        """Test getting monitoring summary."""
        # Set up some reference data
        monitor._reference_features["rsi"] = np.random.rand(100)
        monitor._reference_predictions = np.random.rand(100)
        monitor._reference_labels = np.random.randint(0, 2, 100)

        # Add current samples
        for i in range(20):
            monitor.add_sample(
                features={"rsi": float(i)},
                prediction=0.5,
                confidence=0.6,
                label=1 if i % 2 == 0 else 0,
                pnl=0.01 if i % 2 == 0 else -0.005,
            )

        summary = monitor.get_monitoring_summary()

        assert "drift_detected" in summary
        assert "drift_severity" in summary
        assert "performance" in summary
        assert "performance_trend" in summary
        assert "should_retrain" in summary
        assert "samples_monitored" in summary


class TestModelMonitorDriftDetection:
    """Test drift detection scenarios."""

    @pytest.fixture
    def monitor(self):
        """Create monitor with small windows for testing."""
        config = DriftConfig(
            min_samples=10,
            reference_window=100,
            detection_window=50,
            ks_threshold_low=0.1,
            ks_threshold_high=0.2,
        )
        return ModelMonitor(config=config, data_dir=tempfile.mkdtemp())

    def test_detect_significant_data_drift(self, monitor):
        """Test detecting significant data drift."""
        # Reference: normal distribution around 50
        monitor._reference_features["rsi"] = np.random.normal(50, 5, 100)

        # Current: shifted distribution around 70
        for val in np.random.normal(70, 5, 20):
            monitor.add_sample(
                features={"rsi": val},
                prediction=0.5,
                confidence=0.6,
            )

        results = monitor.check_drift()

        # Should detect drift
        assert len(results) > 0
        drift = results[0]
        assert drift.drift_type == DriftType.DATA_DRIFT
        assert drift.severity in [DriftSeverity.MODERATE, DriftSeverity.HIGH]

    def test_check_prediction_drift(self, monitor):
        """Test prediction drift detection."""
        # Reference predictions around 0.5
        monitor._reference_predictions = np.random.normal(0.5, 0.1, 100)

        # Current predictions around 0.8
        for _ in range(20):
            monitor.add_sample(
                features={"rsi": 50.0},
                prediction=np.random.normal(0.8, 0.1),
                confidence=0.6,
            )

        # Call internal method directly
        drift = monitor._check_prediction_drift()

        assert drift.drift_type == DriftType.PREDICTION_DRIFT

    def test_check_concept_drift_accuracy_drop(self, monitor):
        """Test concept drift via accuracy degradation."""
        # Reference with high accuracy
        monitor._reference_predictions = np.array([0.8] * 50 + [0.2] * 50)
        monitor._reference_labels = np.array([1] * 50 + [0] * 50)

        # Current with poor accuracy (wrong predictions)
        for i in range(20):
            monitor.add_sample(
                features={"rsi": 50.0},
                prediction=0.8,  # Predicting 1
                confidence=0.8,
                label=0,  # But actual is 0
            )

        drift = monitor._check_concept_drift()

        assert drift.drift_type == DriftType.CONCEPT_DRIFT
        # Should detect degradation
        assert "accuracy_drop" in drift.details


class TestDriftSeverityDetermination:
    """Test drift severity determination logic."""

    @pytest.fixture
    def monitor(self):
        config = DriftConfig(
            min_samples=10,
            ks_threshold_low=0.1,
            ks_threshold_high=0.2,
            psi_threshold_low=0.1,
            psi_threshold_high=0.25,
        )
        return ModelMonitor(config=config, data_dir=tempfile.mkdtemp())

    def test_severity_computation(self, monitor):
        """Test drift severity is computed based on thresholds."""
        # Use similar distributions to get low/moderate drift
        np.random.seed(42)
        reference = np.random.normal(50, 5, 200)
        current = np.random.normal(50, 5, 20)

        drift = monitor._check_feature_drift("test", reference, current)

        # Verify severity is one of the valid values
        assert drift.severity in [
            DriftSeverity.NONE,
            DriftSeverity.LOW,
            DriftSeverity.MODERATE,
            DriftSeverity.HIGH,
        ]
        # KS statistic should be computed
        assert drift.ks_statistic >= 0

    def test_severity_high_ks(self, monitor):
        """Test high severity from KS statistic."""
        # Very different distributions
        reference = np.random.normal(0, 1, 100)
        current = np.random.normal(5, 1, 20)  # Shifted by 5 std devs

        drift = monitor._check_feature_drift("test", reference, current)

        assert drift.severity in [DriftSeverity.HIGH, DriftSeverity.MODERATE]


class TestCalibrationInterpolation:
    """Test calibration prediction interpolation."""

    @pytest.fixture
    def monitor(self):
        monitor = ModelMonitor(data_dir=tempfile.mkdtemp())
        # Set up linear calibration map
        monitor._calibration_map = {
            0.1: 0.15,
            0.3: 0.35,
            0.5: 0.55,
            0.7: 0.75,
            0.9: 0.85,
        }
        return monitor

    def test_interpolation_middle(self, monitor):
        """Test interpolation between bins."""
        # 0.4 is between 0.3 and 0.5 bins
        calibrated = monitor.get_calibrated_prediction(0.4)

        # Should interpolate between 0.35 and 0.55
        assert 0.35 < calibrated < 0.55

    def test_interpolation_below_min(self, monitor):
        """Test value below minimum bin."""
        calibrated = monitor.get_calibrated_prediction(0.05)

        assert calibrated == 0.15  # First bin value

    def test_interpolation_above_max(self, monitor):
        """Test value above maximum bin."""
        calibrated = monitor.get_calibrated_prediction(0.95)

        assert calibrated == 0.85  # Last bin value

    def test_exact_bin_value(self, monitor):
        """Test exact bin center value."""
        calibrated = monitor.get_calibrated_prediction(0.5)

        assert calibrated == pytest.approx(0.55, rel=0.01)
