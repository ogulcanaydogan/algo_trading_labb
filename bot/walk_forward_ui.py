"""
Walk-Forward Optimization UI Data Provider.

Provides data structures and utilities for the walk-forward
visualization dashboard panel.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class WindowSummary:
    """Summary of a single walk-forward window for UI display."""
    window_id: int
    train_period: str
    test_period: str
    train_accuracy: float
    test_accuracy: float
    sharpe_ratio: float
    profit_factor: float
    win_rate: float
    is_profitable: bool
    training_time: float


@dataclass
class WalkForwardUIData:
    """Complete walk-forward data formatted for UI consumption."""
    model_type: str
    symbol: str
    total_windows: int
    profitable_windows: int
    robustness_score: float
    consistency_score: float
    overfitting_score: float
    is_robust: bool
    aggregate_metrics: Dict[str, float]
    windows: List[WindowSummary]
    equity_curve: List[Dict[str, Any]]
    accuracy_progression: List[Dict[str, float]]
    sharpe_progression: List[Dict[str, float]]
    train_vs_test_comparison: List[Dict[str, float]]
    validation_period: Dict[str, str]
    config: Dict[str, Any]


class WalkForwardUIProvider:
    """
    Provides walk-forward validation data for dashboard visualization.

    Reads results from saved JSON files and transforms them into
    UI-friendly formats suitable for charting and display.
    """

    def __init__(self, results_dir: str = "data/walk_forward_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def get_latest_results(self, symbol: Optional[str] = None) -> Optional[WalkForwardUIData]:
        """
        Get the most recent walk-forward results.

        Args:
            symbol: Optional symbol filter (e.g., 'BTC_USDT')

        Returns:
            WalkForwardUIData or None if no results found
        """
        # Find result files
        pattern = f"{symbol.replace('/', '_')}*.json" if symbol else "*.json"
        result_files = sorted(self.results_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)

        if not result_files:
            logger.warning(f"No walk-forward results found in {self.results_dir}")
            return None

        # Load most recent
        return self.load_results(result_files[0])

    def get_all_results(self) -> List[Dict[str, Any]]:
        """Get summary of all available walk-forward results."""
        results = []

        for path in sorted(self.results_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                with open(path) as f:
                    data = json.load(f)

                summary = data.get("summary", {})
                results.append({
                    "filename": path.name,
                    "model_type": summary.get("model_type", "unknown"),
                    "symbol": summary.get("symbol", "unknown"),
                    "total_windows": summary.get("total_windows", 0),
                    "profitable_windows": summary.get("profitable_windows", 0),
                    "is_robust": summary.get("is_robust", False),
                    "robustness_score": summary.get("robustness_score", 0),
                    "timestamp": path.stat().st_mtime,
                    "date": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                })
            except Exception as e:
                logger.error(f"Error reading {path}: {e}")

        return results

    def load_results(self, path: Path) -> Optional[WalkForwardUIData]:
        """Load and transform walk-forward results from a JSON file."""
        try:
            with open(path) as f:
                data = json.load(f)

            return self._transform_to_ui_data(data)
        except Exception as e:
            logger.error(f"Error loading results from {path}: {e}")
            return None

    def _transform_to_ui_data(self, data: Dict[str, Any]) -> WalkForwardUIData:
        """Transform raw JSON data to UI format."""
        summary = data.get("summary", {})
        aggregate = data.get("aggregate_metrics", {})
        windows = data.get("windows", [])
        config = data.get("config", {})

        # Transform windows
        window_summaries = []
        equity_curve = []
        accuracy_progression = []
        sharpe_progression = []
        train_vs_test = []

        cumulative_pnl = 0.0

        for w in windows:
            window_id = w.get("window_id", 0)
            test_metrics = w.get("test_metrics", {})
            train_metrics = w.get("train_metrics", {})

            # Window summary
            window_summaries.append(WindowSummary(
                window_id=window_id,
                train_period=f"{w.get('train_start', 'N/A')[:10]} to {w.get('train_end', 'N/A')[:10]}",
                test_period=f"{w.get('test_start', 'N/A')[:10]} to {w.get('test_end', 'N/A')[:10]}",
                train_accuracy=train_metrics.get("accuracy", 0),
                test_accuracy=test_metrics.get("accuracy", 0),
                sharpe_ratio=test_metrics.get("sharpe_ratio", 0),
                profit_factor=test_metrics.get("profit_factor", 0),
                win_rate=test_metrics.get("win_rate", 0),
                is_profitable=w.get("is_profitable", False),
                training_time=w.get("training_time_seconds", 0),
            ))

            # Equity curve (simulated based on Sharpe ratio direction)
            sharpe = test_metrics.get("sharpe_ratio", 0)
            pnl_estimate = sharpe * 0.02  # Rough estimate
            cumulative_pnl += pnl_estimate

            equity_curve.append({
                "window": window_id + 1,
                "period": w.get("test_end", "")[:10],
                "pnl": round(pnl_estimate * 100, 2),
                "cumulative": round(cumulative_pnl * 100, 2),
            })

            # Accuracy progression
            accuracy_progression.append({
                "window": window_id + 1,
                "train_accuracy": round(train_metrics.get("accuracy", 0) * 100, 1),
                "test_accuracy": round(test_metrics.get("accuracy", 0) * 100, 1),
            })

            # Sharpe progression
            sharpe_progression.append({
                "window": window_id + 1,
                "sharpe": round(test_metrics.get("sharpe_ratio", 0), 2),
            })

            # Train vs Test comparison
            train_vs_test.append({
                "window": window_id + 1,
                "train_f1": round(train_metrics.get("f1_score", 0), 3),
                "test_f1": round(test_metrics.get("f1_score", 0), 3),
                "overfitting_gap": round(
                    train_metrics.get("accuracy", 0) - test_metrics.get("accuracy", 0), 3
                ),
            })

        return WalkForwardUIData(
            model_type=summary.get("model_type", "unknown"),
            symbol=summary.get("symbol", "unknown"),
            total_windows=summary.get("total_windows", 0),
            profitable_windows=summary.get("profitable_windows", 0),
            robustness_score=summary.get("robustness_score", 0),
            consistency_score=summary.get("consistency_score", 0),
            overfitting_score=summary.get("overfitting_score", 0),
            is_robust=summary.get("is_robust", False),
            aggregate_metrics={
                "mean_accuracy": round(aggregate.get("mean_accuracy", 0) * 100, 1),
                "mean_sharpe": round(aggregate.get("mean_sharpe_ratio", 0), 2),
                "mean_profit_factor": round(aggregate.get("mean_profit_factor", 0), 2),
                "mean_win_rate": round(aggregate.get("mean_win_rate", 0) * 100, 1),
                "std_accuracy": round(aggregate.get("std_accuracy", 0) * 100, 1),
                "std_sharpe": round(aggregate.get("std_sharpe_ratio", 0), 2),
            },
            windows=window_summaries,
            equity_curve=equity_curve,
            accuracy_progression=accuracy_progression,
            sharpe_progression=sharpe_progression,
            train_vs_test_comparison=train_vs_test,
            validation_period={
                "start": summary.get("start_date", "N/A"),
                "end": summary.get("end_date", "N/A"),
            },
            config=config,
        )

    def get_comparison_data(self, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get comparison data across multiple models/symbols.

        Args:
            symbols: Optional list of symbols to compare

        Returns:
            Comparison data for visualization
        """
        all_results = self.get_all_results()

        if symbols:
            all_results = [r for r in all_results if r["symbol"] in symbols]

        comparison = {
            "models": [],
            "robustness_scores": [],
            "profitable_ratios": [],
            "timestamps": [],
        }

        for result in all_results[:10]:  # Last 10 results
            comparison["models"].append(f"{result['symbol']}_{result['model_type']}")
            comparison["robustness_scores"].append(result["robustness_score"])

            ratio = result["profitable_windows"] / result["total_windows"] if result["total_windows"] > 0 else 0
            comparison["profitable_ratios"].append(round(ratio * 100, 1))
            comparison["timestamps"].append(result["date"])

        return comparison

    def to_api_response(self, ui_data: WalkForwardUIData) -> Dict[str, Any]:
        """Convert UI data to API response format."""
        return {
            "summary": {
                "model_type": ui_data.model_type,
                "symbol": ui_data.symbol,
                "total_windows": ui_data.total_windows,
                "profitable_windows": ui_data.profitable_windows,
                "robustness_score": ui_data.robustness_score,
                "consistency_score": ui_data.consistency_score,
                "overfitting_score": ui_data.overfitting_score,
                "is_robust": ui_data.is_robust,
            },
            "aggregate_metrics": ui_data.aggregate_metrics,
            "validation_period": ui_data.validation_period,
            "config": ui_data.config,
            "charts": {
                "equity_curve": ui_data.equity_curve,
                "accuracy_progression": ui_data.accuracy_progression,
                "sharpe_progression": ui_data.sharpe_progression,
                "train_vs_test": ui_data.train_vs_test_comparison,
            },
            "windows": [
                {
                    "window_id": w.window_id,
                    "train_period": w.train_period,
                    "test_period": w.test_period,
                    "train_accuracy": round(w.train_accuracy * 100, 1),
                    "test_accuracy": round(w.test_accuracy * 100, 1),
                    "sharpe_ratio": round(w.sharpe_ratio, 2),
                    "profit_factor": round(w.profit_factor, 2),
                    "win_rate": round(w.win_rate * 100, 1),
                    "is_profitable": w.is_profitable,
                    "training_time": round(w.training_time, 1),
                }
                for w in ui_data.windows
            ],
        }
