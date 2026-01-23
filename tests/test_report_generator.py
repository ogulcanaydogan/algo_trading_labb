"""
Tests for report generator module.
"""

import pytest
import tempfile
import os
import shutil
from datetime import datetime
from pathlib import Path

from bot.report_generator import (
    ReportConfig,
    ReportSection,
    ReportGenerator,
)


class TestReportConfig:
    """Test ReportConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ReportConfig()
        assert config.title == "Trading Performance Report"
        assert config.period == "weekly"
        assert config.include_charts is True
        assert config.include_trades is True
        assert config.include_risk_metrics is True
        assert config.include_regime_analysis is True
        assert config.include_factor_analysis is True
        assert config.include_recommendations is True
        assert config.max_trades_to_show == 50

    def test_custom_config(self):
        """Test custom configuration."""
        config = ReportConfig(
            title="Custom Report",
            period="monthly",
            include_charts=False,
            max_trades_to_show=100,
        )
        assert config.title == "Custom Report"
        assert config.period == "monthly"
        assert config.include_charts is False
        assert config.max_trades_to_show == 100

    def test_period_options(self):
        """Test various period options."""
        for period in ["daily", "weekly", "monthly", "quarterly", "yearly", "custom"]:
            config = ReportConfig(period=period)
            assert config.period == period


class TestReportSection:
    """Test ReportSection dataclass."""

    def test_section_creation(self):
        """Test creating a report section."""
        section = ReportSection(
            title="Summary",
            content="<p>Report content</p>",
            order=1,
        )
        assert section.title == "Summary"
        assert section.content == "<p>Report content</p>"
        assert section.order == 1

    def test_section_default_order(self):
        """Test default order value."""
        section = ReportSection(
            title="Test",
            content="<p>Test</p>",
        )
        assert section.order == 0


class TestReportGenerator:
    """Test ReportGenerator class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for reports."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def generator(self, temp_dir):
        """Create report generator with temp directory."""
        return ReportGenerator(output_dir=temp_dir)

    @pytest.fixture
    def sample_trades(self):
        """Create sample trades data."""
        return [
            {
                "timestamp": "2024-01-01T10:00:00",
                "symbol": "BTC/USDT",
                "side": "buy",
                "entry_price": 50000.0,
                "exit_price": 51000.0,
                "pnl": 100.0,
                "strategy": "ema_crossover",
            },
            {
                "timestamp": "2024-01-02T10:00:00",
                "symbol": "ETH/USDT",
                "side": "buy",
                "entry_price": 3000.0,
                "exit_price": 2900.0,
                "pnl": -50.0,
                "strategy": "rsi_mean_reversion",
            },
            {
                "timestamp": "2024-01-03T10:00:00",
                "symbol": "BTC/USDT",
                "side": "sell",
                "entry_price": 52000.0,
                "exit_price": 51500.0,
                "pnl": 50.0,
                "strategy": "bollinger_bands",
            },
        ]

    @pytest.fixture
    def sample_equity(self):
        """Create sample equity curve."""
        return [10000, 10100, 10050, 10100, 10200, 10150, 10250]

    @pytest.fixture
    def sample_risk_metrics(self):
        """Create sample risk metrics."""
        return {
            "sharpe_ratio": 1.5,
            "sortino_ratio": 2.0,
            "max_drawdown": 5.5,
            "avg_drawdown": 2.3,
            "var_95": 2.1,
            "cvar_95": 3.2,
            "calmar_ratio": 1.8,
            "profit_factor": 2.5,
            "expectancy": 0.3,
            "time_in_drawdown": 25.0,
        }

    def test_generator_creation(self, generator):
        """Test generator is created."""
        assert generator is not None
        assert generator.config is not None
        assert generator.output_dir.exists()

    def test_generator_with_custom_config(self, temp_dir):
        """Test generator with custom config."""
        config = ReportConfig(
            title="Custom Report",
            period="daily",
        )
        generator = ReportGenerator(config=config, output_dir=temp_dir)
        assert generator.config.title == "Custom Report"
        assert generator.config.period == "daily"

    def test_load_data(self, generator, sample_trades, sample_equity, sample_risk_metrics):
        """Test loading data."""
        generator.load_data(
            trades=sample_trades,
            equity_curve=sample_equity,
            risk_metrics=sample_risk_metrics,
        )
        assert len(generator._data["trades"]) == 3
        assert len(generator._data["equity_curve"]) == 7
        assert generator._data["risk_metrics"]["sharpe_ratio"] == 1.5

    def test_load_empty_data(self, generator):
        """Test loading empty data."""
        generator.load_data()
        assert generator._data["trades"] == []
        assert generator._data["equity_curve"] == []
        assert generator._data["risk_metrics"] == {}

    def test_generate_report(self, generator, sample_trades, sample_equity, sample_risk_metrics):
        """Test generating a full report."""
        generator.load_data(
            trades=sample_trades,
            equity_curve=sample_equity,
            risk_metrics=sample_risk_metrics,
        )

        filepath = generator.generate_report()
        assert filepath is not None
        assert os.path.exists(filepath)
        assert filepath.endswith(".html")

    def test_report_contains_title(self, generator, sample_trades, sample_equity):
        """Test report contains title."""
        config = ReportConfig(title="My Test Report")
        generator = ReportGenerator(config=config, output_dir=str(generator.output_dir))
        generator.load_data(trades=sample_trades, equity_curve=sample_equity)

        filepath = generator.generate_report()
        with open(filepath, "r") as f:
            content = f.read()
        assert "My Test Report" in content

    def test_report_without_charts(self, temp_dir, sample_trades, sample_equity):
        """Test report without charts."""
        config = ReportConfig(include_charts=False)
        generator = ReportGenerator(config=config, output_dir=temp_dir)
        generator.load_data(trades=sample_trades, equity_curve=sample_equity)

        filepath = generator.generate_report()
        assert os.path.exists(filepath)

    def test_report_without_trades(self, temp_dir, sample_equity):
        """Test report without trades section."""
        config = ReportConfig(include_trades=False)
        generator = ReportGenerator(config=config, output_dir=temp_dir)
        generator.load_data(equity_curve=sample_equity)

        filepath = generator.generate_report()
        assert os.path.exists(filepath)

    def test_report_without_risk_metrics(self, temp_dir, sample_trades, sample_equity):
        """Test report without risk metrics section."""
        config = ReportConfig(include_risk_metrics=False)
        generator = ReportGenerator(config=config, output_dir=temp_dir)
        generator.load_data(trades=sample_trades, equity_curve=sample_equity)

        filepath = generator.generate_report()
        assert os.path.exists(filepath)

    def test_report_without_regime_analysis(self, temp_dir, sample_trades, sample_equity):
        """Test report without regime analysis."""
        config = ReportConfig(include_regime_analysis=False)
        generator = ReportGenerator(config=config, output_dir=temp_dir)
        generator.load_data(trades=sample_trades, equity_curve=sample_equity)

        filepath = generator.generate_report()
        assert os.path.exists(filepath)

    def test_report_without_recommendations(self, temp_dir, sample_trades, sample_equity):
        """Test report without recommendations."""
        config = ReportConfig(include_recommendations=False)
        generator = ReportGenerator(config=config, output_dir=temp_dir)
        generator.load_data(trades=sample_trades, equity_curve=sample_equity)

        filepath = generator.generate_report()
        assert os.path.exists(filepath)

    def test_report_with_regime_data(self, temp_dir, sample_trades, sample_equity):
        """Test report with regime analysis data."""
        config = ReportConfig(include_regime_analysis=True)
        generator = ReportGenerator(config=config, output_dir=temp_dir)
        generator.load_data(
            trades=sample_trades,
            equity_curve=sample_equity,
            regime_analysis={
                "current_regime": "bull",
                "regime_distribution": {
                    "bull": {
                        "frequency_pct": 40,
                        "avg_duration_hours": 24,
                        "avg_return_pct": 1.5,
                        "color": "#4CAF50",
                    },
                    "bear": {
                        "frequency_pct": 30,
                        "avg_duration_hours": 12,
                        "avg_return_pct": -1.2,
                        "color": "#F44336",
                    },
                    "sideways": {
                        "frequency_pct": 30,
                        "avg_duration_hours": 18,
                        "avg_return_pct": 0.2,
                        "color": "#FFC107",
                    },
                },
            },
        )

        filepath = generator.generate_report()
        assert os.path.exists(filepath)

    def test_report_with_factor_data(self, temp_dir, sample_trades, sample_equity):
        """Test report with factor analysis data."""
        config = ReportConfig(include_factor_analysis=True)
        generator = ReportGenerator(config=config, output_dir=temp_dir)
        generator.load_data(
            trades=sample_trades,
            equity_curve=sample_equity,
            factor_analysis={
                "summary": {"alpha": 0.05, "r_squared": 0.65},
                "factors": [
                    {
                        "name": "Market",
                        "beta": 1.2,
                        "contribution": 60.0,
                        "t_stat": 2.5,
                        "is_significant": True,
                    },
                    {
                        "name": "Momentum",
                        "beta": 0.3,
                        "contribution": 15.0,
                        "t_stat": 1.8,
                        "is_significant": False,
                    },
                ],
            },
        )

        filepath = generator.generate_report()
        assert os.path.exists(filepath)

    def test_get_report_metadata(self, generator, sample_trades, sample_equity):
        """Test getting report metadata."""
        generator.load_data(trades=sample_trades, equity_curve=sample_equity)
        generator.generate_report()

        metadata = generator.get_report_metadata()
        assert "title" in metadata
        assert "period" in metadata
        assert "generated_at" in metadata
        assert "sections" in metadata
        assert "output_dir" in metadata


class TestReportEdgeCases:
    """Test edge cases for report generation."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_empty_trades_list(self, temp_dir):
        """Test with empty trades list."""
        generator = ReportGenerator(output_dir=temp_dir)
        generator.load_data(trades=[], equity_curve=[10000, 10100])

        filepath = generator.generate_report()
        assert os.path.exists(filepath)

    def test_single_equity_point(self, temp_dir):
        """Test with single equity point."""
        generator = ReportGenerator(output_dir=temp_dir)
        generator.load_data(equity_curve=[10000])

        filepath = generator.generate_report()
        assert os.path.exists(filepath)

    def test_empty_equity_curve(self, temp_dir):
        """Test with empty equity curve."""
        generator = ReportGenerator(output_dir=temp_dir)
        generator.load_data(equity_curve=[])

        filepath = generator.generate_report()
        assert os.path.exists(filepath)

    def test_all_winning_trades(self, temp_dir):
        """Test with all winning trades."""
        generator = ReportGenerator(output_dir=temp_dir)
        generator.load_data(
            trades=[
                {"pnl": 100, "timestamp": "2024-01-01"},
                {"pnl": 50, "timestamp": "2024-01-02"},
                {"pnl": 75, "timestamp": "2024-01-03"},
            ],
            equity_curve=[10000, 10100, 10150, 10225],
        )

        filepath = generator.generate_report()
        assert os.path.exists(filepath)

    def test_all_losing_trades(self, temp_dir):
        """Test with all losing trades."""
        generator = ReportGenerator(output_dir=temp_dir)
        generator.load_data(
            trades=[
                {"pnl": -100, "timestamp": "2024-01-01"},
                {"pnl": -50, "timestamp": "2024-01-02"},
                {"pnl": -75, "timestamp": "2024-01-03"},
            ],
            equity_curve=[10000, 9900, 9850, 9775],
        )

        filepath = generator.generate_report()
        assert os.path.exists(filepath)

    def test_zero_starting_equity(self, temp_dir):
        """Test with zero starting equity."""
        generator = ReportGenerator(output_dir=temp_dir)
        generator.load_data(equity_curve=[0, 100, 200])

        filepath = generator.generate_report()
        assert os.path.exists(filepath)

    def test_high_sharpe_ratio_insight(self, temp_dir):
        """Test insights with high Sharpe ratio."""
        generator = ReportGenerator(output_dir=temp_dir)
        generator.load_data(
            trades=[{"pnl": 100}],
            equity_curve=[10000, 10100],
            risk_metrics={"sharpe_ratio": 2.0, "max_drawdown": 5},
        )

        filepath = generator.generate_report()
        with open(filepath, "r") as f:
            content = f.read()
        assert os.path.exists(filepath)

    def test_low_sharpe_ratio_insight(self, temp_dir):
        """Test insights with low Sharpe ratio."""
        generator = ReportGenerator(output_dir=temp_dir)
        generator.load_data(
            trades=[{"pnl": 10}],
            equity_curve=[10000, 10010],
            risk_metrics={"sharpe_ratio": 0.3, "max_drawdown": 15},
        )

        filepath = generator.generate_report()
        assert os.path.exists(filepath)

    def test_high_drawdown_insight(self, temp_dir):
        """Test insights with high drawdown."""
        generator = ReportGenerator(output_dir=temp_dir)
        generator.load_data(
            trades=[{"pnl": -100}],
            equity_curve=[10000, 9000],
            risk_metrics={"sharpe_ratio": 1.0, "max_drawdown": 25},
        )

        filepath = generator.generate_report()
        assert os.path.exists(filepath)

    def test_max_trades_limit(self, temp_dir):
        """Test max trades limit."""
        config = ReportConfig(max_trades_to_show=5)
        generator = ReportGenerator(config=config, output_dir=temp_dir)

        trades = [{"pnl": i, "timestamp": f"2024-01-{i:02d}"} for i in range(1, 20)]
        generator.load_data(trades=trades, equity_curve=[10000, 10100])

        filepath = generator.generate_report()
        assert os.path.exists(filepath)
