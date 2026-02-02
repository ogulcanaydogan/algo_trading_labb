"""
Counterfactual Evaluation Engine.

Phase 2B Core Deliverable:
Evaluates "What if we followed RL recommendation?" without changing execution.

Computes metrics:
- delta_PnL vs baseline strategy
- hit rate / precision by regime
- incremental Sharpe (rolling weekly)
- incremental drawdown / tail risk
- turnover impact and cost sensitivity

Supports:
- per-symbol analysis
- per-regime analysis
- cost decomposition (slippage vs fees vs turnover)
- confidence-threshold sweeps

CRITICAL: This is evaluation only. No execution authority.
"""

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CounterfactualTrade:
    """Single counterfactual trade comparison."""
    timestamp: datetime
    symbol: str
    regime: str

    # RL recommendation
    rl_action: str
    rl_confidence: float
    rl_primary_agent: str

    # Actual decision
    actual_action: str
    actual_pnl: float

    # Execution costs
    slippage_cost: float = 0.0
    fee_cost: float = 0.0
    spread_cost: float = 0.0

    # Context
    preservation_level: str = "normal"
    gate_approved: bool = True
    gate_rejection_reason: str = ""

    # Computed
    followed_rl: bool = False
    counterfactual_pnl: float = 0.0  # What would have happened


@dataclass
class RegimeMetrics:
    """Metrics for a specific regime."""
    regime: str
    total_decisions: int = 0
    rl_correct: int = 0
    rl_wrong: int = 0
    rl_hit_rate: float = 0.0
    avg_rl_confidence: float = 0.0
    avg_actual_pnl: float = 0.0
    avg_counterfactual_pnl: float = 0.0
    delta_pnl: float = 0.0
    total_turnover: int = 0


@dataclass
class CostDecomposition:
    """Breakdown of execution costs."""
    total_slippage: float = 0.0
    total_fees: float = 0.0
    total_spread: float = 0.0
    total_cost: float = 0.0
    avg_cost_per_trade: float = 0.0
    cost_as_pct_of_pnl: float = 0.0


@dataclass
class ConfidenceThresholdResult:
    """Result of analyzing a specific confidence threshold."""
    threshold: float
    trades_above_threshold: int
    avg_pnl_above: float
    avg_pnl_below: float
    incremental_sharpe: float
    hit_rate_above: float


@dataclass
class CounterfactualReport:
    """Complete counterfactual analysis report."""
    timestamp: datetime = field(default_factory=datetime.now)
    period_start: datetime = field(default_factory=datetime.now)
    period_end: datetime = field(default_factory=datetime.now)
    analysis_weeks: int = 0

    # Overall metrics
    total_decisions: int = 0
    rl_agreement_rate: float = 0.0
    rl_hit_rate: float = 0.0

    # P&L comparison
    total_actual_pnl: float = 0.0
    total_counterfactual_pnl: float = 0.0
    delta_pnl: float = 0.0
    incremental_sharpe: float = 0.0

    # Risk metrics
    actual_max_drawdown: float = 0.0
    counterfactual_max_drawdown: float = 0.0
    actual_volatility: float = 0.0
    counterfactual_volatility: float = 0.0

    # Per-symbol analysis
    by_symbol: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Per-regime analysis
    by_regime: Dict[str, RegimeMetrics] = field(default_factory=dict)

    # Cost decomposition
    costs: CostDecomposition = field(default_factory=CostDecomposition)

    # Confidence threshold analysis
    threshold_analysis: List[ConfidenceThresholdResult] = field(default_factory=list)

    # Recommendations
    recommended_symbols: List[str] = field(default_factory=list)
    recommended_regimes: List[str] = field(default_factory=list)
    safe_activation_threshold: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "period": {
                "start": self.period_start.isoformat(),
                "end": self.period_end.isoformat(),
                "weeks": self.analysis_weeks,
            },
            "overall": {
                "total_decisions": self.total_decisions,
                "rl_agreement_rate": round(self.rl_agreement_rate, 4),
                "rl_hit_rate": round(self.rl_hit_rate, 4),
            },
            "pnl_comparison": {
                "actual_total": round(self.total_actual_pnl, 2),
                "counterfactual_total": round(self.total_counterfactual_pnl, 2),
                "delta_pnl": round(self.delta_pnl, 2),
                "incremental_sharpe": round(self.incremental_sharpe, 4),
            },
            "risk_metrics": {
                "actual_max_drawdown": round(self.actual_max_drawdown, 4),
                "counterfactual_max_drawdown": round(self.counterfactual_max_drawdown, 4),
                "actual_volatility": round(self.actual_volatility, 4),
                "counterfactual_volatility": round(self.counterfactual_volatility, 4),
            },
            "by_symbol": self.by_symbol,
            "by_regime": {
                k: {
                    "total_decisions": v.total_decisions,
                    "rl_hit_rate": round(v.rl_hit_rate, 4),
                    "avg_rl_confidence": round(v.avg_rl_confidence, 4),
                    "delta_pnl": round(v.delta_pnl, 2),
                }
                for k, v in self.by_regime.items()
            },
            "costs": {
                "total_slippage": round(self.costs.total_slippage, 2),
                "total_fees": round(self.costs.total_fees, 2),
                "total_spread": round(self.costs.total_spread, 2),
                "total_cost": round(self.costs.total_cost, 2),
                "avg_cost_per_trade": round(self.costs.avg_cost_per_trade, 4),
                "cost_as_pct_of_pnl": round(self.costs.cost_as_pct_of_pnl, 4),
            },
            "threshold_analysis": [
                {
                    "threshold": t.threshold,
                    "trades_above": t.trades_above_threshold,
                    "avg_pnl_above": round(t.avg_pnl_above, 4),
                    "incremental_sharpe": round(t.incremental_sharpe, 4),
                    "hit_rate_above": round(t.hit_rate_above, 4),
                }
                for t in self.threshold_analysis
            ],
            "recommendations": {
                "symbols": self.recommended_symbols,
                "regimes": self.recommended_regimes,
                "safe_activation_threshold": round(self.safe_activation_threshold, 2),
            },
        }

    def print_summary(self):
        """Print human-readable summary."""
        print("\n" + "=" * 70)
        print("COUNTERFACTUAL EVALUATION REPORT")
        print("=" * 70)
        print(f"Period: {self.period_start.date()} to {self.period_end.date()} ({self.analysis_weeks} weeks)")
        print(f"Total Decisions Analyzed: {self.total_decisions}")

        print(f"\n{'─' * 70}")
        print("RL PERFORMANCE")
        print(f"{'─' * 70}")
        print(f"  Agreement Rate (RL matched actual): {self.rl_agreement_rate:.1%}")
        print(f"  RL Hit Rate (when followed): {self.rl_hit_rate:.1%}")

        print(f"\n{'─' * 70}")
        print("P&L COMPARISON")
        print(f"{'─' * 70}")
        print(f"  Actual Total P&L:        ${self.total_actual_pnl:,.2f}")
        print(f"  Counterfactual P&L:      ${self.total_counterfactual_pnl:,.2f}")
        print(f"  Delta (CF - Actual):     ${self.delta_pnl:,.2f}")
        print(f"  Incremental Sharpe:      {self.incremental_sharpe:.4f}")

        print(f"\n{'─' * 70}")
        print("RISK METRICS")
        print(f"{'─' * 70}")
        print(f"  Actual Max Drawdown:     {self.actual_max_drawdown:.2%}")
        print(f"  Counterfactual Max DD:   {self.counterfactual_max_drawdown:.2%}")

        print(f"\n{'─' * 70}")
        print("COST ANALYSIS")
        print(f"{'─' * 70}")
        print(f"  Total Slippage:          ${self.costs.total_slippage:,.2f}")
        print(f"  Total Fees:              ${self.costs.total_fees:,.2f}")
        print(f"  Avg Cost/Trade:          ${self.costs.avg_cost_per_trade:.4f}")

        if self.by_symbol:
            print(f"\n{'─' * 70}")
            print("BY SYMBOL")
            print(f"{'─' * 70}")
            for symbol, data in self.by_symbol.items():
                delta = data.get("delta_pnl", 0)
                marker = "[+]" if delta > 0 else "[-]"
                print(f"  {marker} {symbol}: delta=${delta:,.2f}, hit_rate={data.get('hit_rate', 0):.1%}")

        if self.threshold_analysis:
            print(f"\n{'─' * 70}")
            print("CONFIDENCE THRESHOLD ANALYSIS")
            print(f"{'─' * 70}")
            for t in self.threshold_analysis:
                print(f"  >{t.threshold:.0%}: {t.trades_above_threshold} trades, "
                      f"Sharpe={t.incremental_sharpe:.3f}, hit={t.hit_rate_above:.1%}")

        print(f"\n{'─' * 70}")
        print("RECOMMENDATIONS")
        print(f"{'─' * 70}")
        print(f"  Safe Activation Threshold: {self.safe_activation_threshold:.0%} confidence")
        print(f"  Recommended Symbols: {', '.join(self.recommended_symbols) or 'None yet'}")
        print(f"  Recommended Regimes: {', '.join(self.recommended_regimes) or 'None yet'}")
        print("=" * 70)


class CounterfactualEvaluator:
    """
    Evaluates counterfactual performance of RL recommendations.

    CRITICAL: This is EVALUATION ONLY. No execution authority.
    """

    def __init__(
        self,
        report_dir: Path = Path("data/reports"),
        min_decisions_for_analysis: int = 10,
        confidence_thresholds: List[float] = None,
    ):
        """
        Initialize evaluator.

        Args:
            report_dir: Directory for saving reports
            min_decisions_for_analysis: Minimum decisions needed for valid analysis
            confidence_thresholds: Thresholds to analyze (default: [0.5, 0.6, 0.7, 0.8, 0.9])
        """
        self.report_dir = report_dir
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.min_decisions = min_decisions_for_analysis
        self.confidence_thresholds = confidence_thresholds or [0.5, 0.6, 0.7, 0.8, 0.9]

        logger.info(f"CounterfactualEvaluator initialized, reports at {self.report_dir}")

    def evaluate_from_database(
        self,
        db,  # LearningDatabase instance
        weeks: int = 4,
        symbol: Optional[str] = None,
    ) -> CounterfactualReport:
        """
        Evaluate counterfactual performance from database records.

        Args:
            db: LearningDatabase instance with rl_recommendations data
            weeks: Number of weeks to analyze
            symbol: Optional symbol filter

        Returns:
            CounterfactualReport with full analysis
        """
        # Fetch recommendations with outcomes
        recs = db.get_rl_recommendations(
            symbol=symbol,
            days_lookback=weeks * 7,
            limit=10000,
            only_counterfactuals=True,
        )

        if len(recs) < self.min_decisions:
            logger.warning(
                f"Insufficient data: {len(recs)} decisions < {self.min_decisions} minimum"
            )
            return CounterfactualReport(
                total_decisions=len(recs),
                analysis_weeks=weeks,
            )

        # Convert to CounterfactualTrade objects
        trades = self._convert_records(recs)

        return self.evaluate(trades, weeks=weeks)

    def evaluate(
        self,
        trades: List[CounterfactualTrade],
        weeks: int = 4,
    ) -> CounterfactualReport:
        """
        Run full counterfactual evaluation on trade data.

        Args:
            trades: List of CounterfactualTrade objects
            weeks: Analysis period in weeks

        Returns:
            CounterfactualReport with complete analysis
        """
        if not trades:
            return CounterfactualReport(analysis_weeks=weeks)

        report = CounterfactualReport(
            period_start=min(t.timestamp for t in trades),
            period_end=max(t.timestamp for t in trades),
            analysis_weeks=weeks,
            total_decisions=len(trades),
        )

        # Calculate agreement and hit rates
        self._calculate_agreement_metrics(trades, report)

        # Calculate P&L comparison
        self._calculate_pnl_metrics(trades, report)

        # Calculate risk metrics
        self._calculate_risk_metrics(trades, report)

        # Per-symbol analysis
        self._analyze_by_symbol(trades, report)

        # Per-regime analysis
        self._analyze_by_regime(trades, report)

        # Cost decomposition
        self._analyze_costs(trades, report)

        # Confidence threshold sweep
        self._analyze_confidence_thresholds(trades, report)

        # Generate recommendations
        self._generate_recommendations(report)

        return report

    def _convert_records(self, recs: List[Dict]) -> List[CounterfactualTrade]:
        """Convert database records to CounterfactualTrade objects."""
        trades = []
        for rec in recs:
            # Parse timestamp
            try:
                ts = datetime.fromisoformat(rec.get("timestamp", ""))
            except (ValueError, TypeError):
                ts = datetime.now()

            # Determine if RL was followed
            rl_action = rec.get("suggested_action", "hold").lower()
            actual_action = rec.get("actual_action", "hold")
            if actual_action:
                actual_action = actual_action.lower()

            # Map actions for comparison
            action_map = {
                "buy": "long", "long": "long",
                "sell": "short", "short": "short",
                "hold": "hold", "cover": "hold",
            }
            rl_mapped = action_map.get(rl_action, rl_action)
            actual_mapped = action_map.get(actual_action, actual_action) if actual_action else "hold"

            followed = rl_mapped == actual_mapped

            # Calculate counterfactual P&L
            actual_pnl = rec.get("actual_pnl") or 0.0

            # Simple counterfactual: if followed, same PnL; if not, inverse sign
            # (This is a simplification - real implementation would need trade-level data)
            if followed:
                cf_pnl = actual_pnl
            else:
                # If RL said opposite, counterfactual is inverse
                # If RL said hold when we traded, counterfactual is 0
                if rl_mapped == "hold":
                    cf_pnl = 0.0
                else:
                    cf_pnl = -actual_pnl * 0.8  # Conservative estimate

            trade = CounterfactualTrade(
                timestamp=ts,
                symbol=rec.get("symbol", "UNKNOWN"),
                regime=rec.get("regime", "unknown"),
                rl_action=rl_action,
                rl_confidence=rec.get("action_confidence", 0.5),
                rl_primary_agent=rec.get("primary_agent", ""),
                actual_action=actual_action or "hold",
                actual_pnl=actual_pnl,
                followed_rl=followed,
                counterfactual_pnl=cf_pnl,
                preservation_level=rec.get("preservation_level", "normal"),
                gate_approved=rec.get("was_applied", False),
            )
            trades.append(trade)

        return trades

    def _calculate_agreement_metrics(
        self,
        trades: List[CounterfactualTrade],
        report: CounterfactualReport,
    ):
        """Calculate RL agreement and hit rate metrics."""
        followed = [t for t in trades if t.followed_rl]
        report.rl_agreement_rate = len(followed) / len(trades) if trades else 0.0

        # Hit rate: when RL was followed, how often was it profitable?
        if followed:
            profitable = [t for t in followed if t.actual_pnl > 0]
            report.rl_hit_rate = len(profitable) / len(followed)

    def _calculate_pnl_metrics(
        self,
        trades: List[CounterfactualTrade],
        report: CounterfactualReport,
    ):
        """Calculate P&L comparison metrics."""
        report.total_actual_pnl = sum(t.actual_pnl for t in trades)
        report.total_counterfactual_pnl = sum(t.counterfactual_pnl for t in trades)
        report.delta_pnl = report.total_counterfactual_pnl - report.total_actual_pnl

        # Calculate incremental Sharpe
        if len(trades) > 1:
            actual_returns = [t.actual_pnl for t in trades]
            cf_returns = [t.counterfactual_pnl for t in trades]
            incremental_returns = [cf - actual for cf, actual in zip(cf_returns, actual_returns)]

            if np.std(incremental_returns) > 0:
                report.incremental_sharpe = (
                    np.mean(incremental_returns) / np.std(incremental_returns)
                ) * np.sqrt(252 / 7)  # Annualize assuming weekly data

    def _calculate_risk_metrics(
        self,
        trades: List[CounterfactualTrade],
        report: CounterfactualReport,
    ):
        """Calculate drawdown and volatility metrics."""
        # Sort by timestamp
        sorted_trades = sorted(trades, key=lambda t: t.timestamp)

        # Calculate cumulative equity curves
        actual_equity = [0.0]
        cf_equity = [0.0]

        for t in sorted_trades:
            actual_equity.append(actual_equity[-1] + t.actual_pnl)
            cf_equity.append(cf_equity[-1] + t.counterfactual_pnl)

        # Max drawdown
        report.actual_max_drawdown = self._max_drawdown(actual_equity)
        report.counterfactual_max_drawdown = self._max_drawdown(cf_equity)

        # Volatility (of returns)
        actual_returns = [t.actual_pnl for t in sorted_trades]
        cf_returns = [t.counterfactual_pnl for t in sorted_trades]

        report.actual_volatility = np.std(actual_returns) if actual_returns else 0.0
        report.counterfactual_volatility = np.std(cf_returns) if cf_returns else 0.0

    def _max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown from equity curve."""
        if not equity_curve:
            return 0.0

        peak = equity_curve[0]
        max_dd = 0.0

        for value in equity_curve:
            if value > peak:
                peak = value
            if peak > 0:
                dd = (peak - value) / peak
                max_dd = max(max_dd, dd)

        return max_dd

    def _analyze_by_symbol(
        self,
        trades: List[CounterfactualTrade],
        report: CounterfactualReport,
    ):
        """Analyze performance by symbol."""
        by_symbol = defaultdict(list)
        for t in trades:
            by_symbol[t.symbol].append(t)

        for symbol, symbol_trades in by_symbol.items():
            actual_pnl = sum(t.actual_pnl for t in symbol_trades)
            cf_pnl = sum(t.counterfactual_pnl for t in symbol_trades)
            followed = [t for t in symbol_trades if t.followed_rl]
            profitable_followed = [t for t in followed if t.actual_pnl > 0]

            report.by_symbol[symbol] = {
                "total_decisions": len(symbol_trades),
                "actual_pnl": round(actual_pnl, 2),
                "counterfactual_pnl": round(cf_pnl, 2),
                "delta_pnl": round(cf_pnl - actual_pnl, 2),
                "agreement_rate": round(len(followed) / len(symbol_trades), 4) if symbol_trades else 0,
                "hit_rate": round(len(profitable_followed) / len(followed), 4) if followed else 0,
                "avg_confidence": round(np.mean([t.rl_confidence for t in symbol_trades]), 4),
            }

    def _analyze_by_regime(
        self,
        trades: List[CounterfactualTrade],
        report: CounterfactualReport,
    ):
        """Analyze performance by market regime."""
        by_regime = defaultdict(list)
        for t in trades:
            by_regime[t.regime].append(t)

        for regime, regime_trades in by_regime.items():
            metrics = RegimeMetrics(regime=regime)
            metrics.total_decisions = len(regime_trades)

            followed = [t for t in regime_trades if t.followed_rl]
            metrics.rl_correct = len([t for t in followed if t.actual_pnl > 0])
            metrics.rl_wrong = len(followed) - metrics.rl_correct
            metrics.rl_hit_rate = metrics.rl_correct / len(followed) if followed else 0

            metrics.avg_rl_confidence = np.mean([t.rl_confidence for t in regime_trades])
            metrics.avg_actual_pnl = np.mean([t.actual_pnl for t in regime_trades])
            metrics.avg_counterfactual_pnl = np.mean([t.counterfactual_pnl for t in regime_trades])
            metrics.delta_pnl = sum(t.counterfactual_pnl - t.actual_pnl for t in regime_trades)

            report.by_regime[regime] = metrics

    def _analyze_costs(
        self,
        trades: List[CounterfactualTrade],
        report: CounterfactualReport,
    ):
        """Decompose execution costs."""
        report.costs.total_slippage = sum(t.slippage_cost for t in trades)
        report.costs.total_fees = sum(t.fee_cost for t in trades)
        report.costs.total_spread = sum(t.spread_cost for t in trades)
        report.costs.total_cost = (
            report.costs.total_slippage +
            report.costs.total_fees +
            report.costs.total_spread
        )

        if trades:
            report.costs.avg_cost_per_trade = report.costs.total_cost / len(trades)

        if report.total_actual_pnl != 0:
            report.costs.cost_as_pct_of_pnl = abs(
                report.costs.total_cost / report.total_actual_pnl
            )

    def _analyze_confidence_thresholds(
        self,
        trades: List[CounterfactualTrade],
        report: CounterfactualReport,
    ):
        """Analyze performance at different confidence thresholds."""
        for threshold in self.confidence_thresholds:
            above = [t for t in trades if t.rl_confidence >= threshold]
            below = [t for t in trades if t.rl_confidence < threshold]

            if not above:
                continue

            followed_above = [t for t in above if t.followed_rl]
            profitable_above = [t for t in followed_above if t.actual_pnl > 0]

            # Calculate incremental Sharpe for trades above threshold
            if len(above) > 1:
                incr_returns = [t.counterfactual_pnl - t.actual_pnl for t in above]
                if np.std(incr_returns) > 0:
                    incr_sharpe = (np.mean(incr_returns) / np.std(incr_returns)) * np.sqrt(52)
                else:
                    incr_sharpe = 0.0
            else:
                incr_sharpe = 0.0

            result = ConfidenceThresholdResult(
                threshold=threshold,
                trades_above_threshold=len(above),
                avg_pnl_above=np.mean([t.counterfactual_pnl for t in above]) if above else 0,
                avg_pnl_below=np.mean([t.counterfactual_pnl for t in below]) if below else 0,
                incremental_sharpe=incr_sharpe,
                hit_rate_above=len(profitable_above) / len(followed_above) if followed_above else 0,
            )
            report.threshold_analysis.append(result)

    def _generate_recommendations(self, report: CounterfactualReport):
        """Generate recommendations based on analysis."""
        # Find symbols with positive delta P&L
        for symbol, data in report.by_symbol.items():
            if data.get("delta_pnl", 0) > 0 and data.get("hit_rate", 0) > 0.5:
                report.recommended_symbols.append(symbol)

        # Find regimes with positive delta P&L
        for regime, metrics in report.by_regime.items():
            if metrics.delta_pnl > 0 and metrics.rl_hit_rate > 0.5:
                report.recommended_regimes.append(regime)

        # Find safe activation threshold
        # Look for threshold where incremental Sharpe becomes positive
        safe_threshold = 0.9  # Default to high threshold
        for result in sorted(report.threshold_analysis, key=lambda x: x.threshold, reverse=True):
            if result.incremental_sharpe > 0 and result.hit_rate_above > 0.55:
                safe_threshold = result.threshold
                break

        report.safe_activation_threshold = safe_threshold

    def save_report(self, report: CounterfactualReport, filename: Optional[str] = None):
        """Save report to JSON file."""
        if filename is None:
            filename = f"counterfactual_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = self.report_dir / filename
        with open(filepath, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        logger.info(f"Counterfactual report saved to {filepath}")
        return filepath

    def generate_weekly_report(
        self,
        db,  # LearningDatabase instance
        symbol: Optional[str] = None,
    ) -> Tuple[CounterfactualReport, Path]:
        """
        Generate and save weekly counterfactual report.

        Args:
            db: LearningDatabase instance
            symbol: Optional symbol filter

        Returns:
            Tuple of (report, filepath)
        """
        report = self.evaluate_from_database(db, weeks=1, symbol=symbol)

        # Save with dated filename
        filename = f"weekly_counterfactual_{datetime.now().strftime('%Y%m%d')}.json"
        filepath = self.save_report(report, filename)

        return report, filepath


def run_counterfactual_evaluation(
    db_path: Optional[str] = None,
    weeks: int = 4,
    symbol: Optional[str] = None,
) -> CounterfactualReport:
    """
    Run counterfactual evaluation from command line.

    Args:
        db_path: Path to learning database
        weeks: Number of weeks to analyze
        symbol: Optional symbol filter

    Returns:
        CounterfactualReport
    """
    from bot.learning.learning_database import get_learning_database

    db = get_learning_database(db_path) if db_path else get_learning_database()
    evaluator = CounterfactualEvaluator()

    report = evaluator.evaluate_from_database(db, weeks=weeks, symbol=symbol)
    report.print_summary()

    evaluator.save_report(report)

    return report


if __name__ == "__main__":
    print("\nRunning Counterfactual Evaluation...")
    run_counterfactual_evaluation(weeks=4)
