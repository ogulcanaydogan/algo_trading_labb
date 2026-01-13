"""
Paper Trading vs Backtest Comparison.

Compares paper trading results with backtest expectations
to validate strategy performance and identify discrepancies.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DiscrepancyType(Enum):
    """Types of discrepancies between paper and backtest."""
    SLIPPAGE = "slippage"
    TIMING = "timing"
    SIGNAL_MISMATCH = "signal_mismatch"
    EXECUTION_GAP = "execution_gap"
    DATA_DIVERGENCE = "data_divergence"
    REGIME_SHIFT = "regime_shift"


@dataclass
class TradeComparison:
    """Comparison of a single trade pair (paper vs backtest)."""
    symbol: str
    paper_entry_time: datetime
    backtest_entry_time: datetime
    paper_entry_price: float
    backtest_entry_price: float
    paper_exit_price: Optional[float]
    backtest_exit_price: Optional[float]
    paper_pnl: float
    backtest_pnl: float
    pnl_difference: float
    pnl_difference_pct: float
    entry_slippage: float
    exit_slippage: float
    timing_difference_seconds: float
    matched: bool
    discrepancy_types: List[DiscrepancyType]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "paper_entry_time": self.paper_entry_time.isoformat(),
            "backtest_entry_time": self.backtest_entry_time.isoformat(),
            "paper_entry_price": round(self.paper_entry_price, 4),
            "backtest_entry_price": round(self.backtest_entry_price, 4),
            "paper_exit_price": round(self.paper_exit_price, 4) if self.paper_exit_price else None,
            "backtest_exit_price": round(self.backtest_exit_price, 4) if self.backtest_exit_price else None,
            "paper_pnl": round(self.paper_pnl, 2),
            "backtest_pnl": round(self.backtest_pnl, 2),
            "pnl_difference": round(self.pnl_difference, 2),
            "pnl_difference_pct": round(self.pnl_difference_pct, 4),
            "entry_slippage": round(self.entry_slippage, 4),
            "exit_slippage": round(self.exit_slippage, 4),
            "timing_difference_seconds": round(self.timing_difference_seconds, 2),
            "matched": self.matched,
            "discrepancy_types": [d.value for d in self.discrepancy_types],
        }


@dataclass
class ComparisonMetrics:
    """Aggregate comparison metrics."""
    total_paper_pnl: float
    total_backtest_pnl: float
    pnl_difference: float
    pnl_difference_pct: float
    paper_win_rate: float
    backtest_win_rate: float
    win_rate_difference: float
    paper_sharpe: float
    backtest_sharpe: float
    sharpe_difference: float
    avg_slippage: float
    avg_timing_diff: float
    correlation: float
    tracking_error: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_paper_pnl": round(self.total_paper_pnl, 2),
            "total_backtest_pnl": round(self.total_backtest_pnl, 2),
            "pnl_difference": round(self.pnl_difference, 2),
            "pnl_difference_pct": round(self.pnl_difference_pct, 4),
            "paper_win_rate": round(self.paper_win_rate, 4),
            "backtest_win_rate": round(self.backtest_win_rate, 4),
            "win_rate_difference": round(self.win_rate_difference, 4),
            "paper_sharpe": round(self.paper_sharpe, 2),
            "backtest_sharpe": round(self.backtest_sharpe, 2),
            "sharpe_difference": round(self.sharpe_difference, 2),
            "avg_slippage": round(self.avg_slippage, 4),
            "avg_timing_diff": round(self.avg_timing_diff, 2),
            "correlation": round(self.correlation, 4),
            "tracking_error": round(self.tracking_error, 4),
        }


@dataclass
class DiscrepancyAnalysis:
    """Analysis of discrepancies."""
    discrepancy_type: DiscrepancyType
    count: int
    total_pnl_impact: float
    avg_pnl_impact: float
    description: str
    examples: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "discrepancy_type": self.discrepancy_type.value,
            "count": self.count,
            "total_pnl_impact": round(self.total_pnl_impact, 2),
            "avg_pnl_impact": round(self.avg_pnl_impact, 2),
            "description": self.description,
            "examples": self.examples[:3],
        }


@dataclass
class ValidationReport:
    """Complete paper vs backtest validation report."""
    period_start: datetime
    period_end: datetime
    paper_trades_count: int
    backtest_trades_count: int
    matched_trades_count: int
    metrics: ComparisonMetrics
    trade_comparisons: List[TradeComparison]
    discrepancy_analysis: List[DiscrepancyAnalysis]
    validation_passed: bool
    validation_score: float  # 0-100
    recommendations: List[str]
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "paper_trades_count": self.paper_trades_count,
            "backtest_trades_count": self.backtest_trades_count,
            "matched_trades_count": self.matched_trades_count,
            "metrics": self.metrics.to_dict(),
            "trade_comparisons": [t.to_dict() for t in self.trade_comparisons[:20]],
            "discrepancy_analysis": [d.to_dict() for d in self.discrepancy_analysis],
            "validation_passed": self.validation_passed,
            "validation_score": round(self.validation_score, 1),
            "recommendations": self.recommendations,
            "generated_at": self.generated_at.isoformat(),
        }


@dataclass
class PaperTrade:
    """Paper trading trade record."""
    trade_id: str
    symbol: str
    action: str
    quantity: float
    entry_price: float
    exit_price: Optional[float]
    entry_time: datetime
    exit_time: Optional[datetime]
    pnl: float
    signal: str
    confidence: float


@dataclass
class BacktestTrade:
    """Backtest trade record."""
    symbol: str
    action: str
    quantity: float
    entry_price: float
    exit_price: Optional[float]
    entry_time: datetime
    exit_time: Optional[datetime]
    pnl: float
    signal: str


class PaperBacktestComparator:
    """
    Compares paper trading results with backtest.

    Features:
    - Trade matching between paper and backtest
    - Slippage analysis
    - P&L divergence tracking
    - Validation scoring
    - Discrepancy categorization

    Usage:
        comparator = PaperBacktestComparator()

        # Load paper trading results
        comparator.load_paper_trades(paper_trades)

        # Run backtest for same period
        comparator.run_backtest(strategy, start_date, end_date)

        # Generate comparison report
        report = comparator.generate_report()
    """

    def __init__(
        self,
        data_dir: str = "data/paper_vs_backtest",
        slippage_threshold: float = 0.001,  # 0.1%
        timing_threshold_seconds: float = 60,  # 1 minute
        pnl_tolerance: float = 0.05,  # 5%
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.slippage_threshold = slippage_threshold
        self.timing_threshold = timing_threshold_seconds
        self.pnl_tolerance = pnl_tolerance

        self.paper_trades: List[PaperTrade] = []
        self.backtest_trades: List[BacktestTrade] = []

    def load_paper_trades(
        self,
        trades: List[Dict[str, Any]],
    ) -> int:
        """
        Load paper trading results.

        Args:
            trades: List of trade dictionaries

        Returns:
            Number of trades loaded
        """
        self.paper_trades = []

        for t in trades:
            try:
                entry_time = t.get("entry_time")
                if isinstance(entry_time, str):
                    entry_time = datetime.fromisoformat(entry_time)

                exit_time = t.get("exit_time")
                if isinstance(exit_time, str):
                    exit_time = datetime.fromisoformat(exit_time)

                trade = PaperTrade(
                    trade_id=t.get("trade_id", str(len(self.paper_trades))),
                    symbol=t["symbol"],
                    action=t["action"],
                    quantity=t["quantity"],
                    entry_price=t["entry_price"],
                    exit_price=t.get("exit_price"),
                    entry_time=entry_time,
                    exit_time=exit_time,
                    pnl=t.get("pnl", 0),
                    signal=t.get("signal", "unknown"),
                    confidence=t.get("confidence", 0.5),
                )
                self.paper_trades.append(trade)
            except (KeyError, ValueError) as e:
                logger.warning(f"Failed to parse paper trade: {e}")

        logger.info(f"Loaded {len(self.paper_trades)} paper trades")
        return len(self.paper_trades)

    def load_backtest_trades(
        self,
        trades: List[Dict[str, Any]],
    ) -> int:
        """
        Load backtest results.

        Args:
            trades: List of backtest trade dictionaries

        Returns:
            Number of trades loaded
        """
        self.backtest_trades = []

        for t in trades:
            try:
                entry_time = t.get("entry_time")
                if isinstance(entry_time, str):
                    entry_time = datetime.fromisoformat(entry_time)

                exit_time = t.get("exit_time")
                if isinstance(exit_time, str):
                    exit_time = datetime.fromisoformat(exit_time)

                trade = BacktestTrade(
                    symbol=t["symbol"],
                    action=t["action"],
                    quantity=t["quantity"],
                    entry_price=t["entry_price"],
                    exit_price=t.get("exit_price"),
                    entry_time=entry_time,
                    exit_time=exit_time,
                    pnl=t.get("pnl", 0),
                    signal=t.get("signal", "unknown"),
                )
                self.backtest_trades.append(trade)
            except (KeyError, ValueError) as e:
                logger.warning(f"Failed to parse backtest trade: {e}")

        logger.info(f"Loaded {len(self.backtest_trades)} backtest trades")
        return len(self.backtest_trades)

    def _match_trades(self) -> List[Tuple[Optional[PaperTrade], Optional[BacktestTrade]]]:
        """Match paper trades with backtest trades."""
        matched = []
        used_backtest = set()

        for paper in self.paper_trades:
            best_match = None
            best_score = float('inf')

            for i, backtest in enumerate(self.backtest_trades):
                if i in used_backtest:
                    continue

                if paper.symbol != backtest.symbol:
                    continue

                if paper.action != backtest.action:
                    continue

                # Calculate match score based on timing
                time_diff = abs((paper.entry_time - backtest.entry_time).total_seconds())

                if time_diff < best_score and time_diff < 3600:  # Within 1 hour
                    best_score = time_diff
                    best_match = (i, backtest)

            if best_match:
                matched.append((paper, best_match[1]))
                used_backtest.add(best_match[0])
            else:
                matched.append((paper, None))

        # Add unmatched backtest trades
        for i, backtest in enumerate(self.backtest_trades):
            if i not in used_backtest:
                matched.append((None, backtest))

        return matched

    def _compare_trade_pair(
        self,
        paper: PaperTrade,
        backtest: BacktestTrade,
    ) -> TradeComparison:
        """Compare a matched paper-backtest trade pair."""
        discrepancies = []

        # Entry slippage
        entry_slippage = (paper.entry_price - backtest.entry_price) / backtest.entry_price
        if abs(entry_slippage) > self.slippage_threshold:
            discrepancies.append(DiscrepancyType.SLIPPAGE)

        # Exit slippage
        if paper.exit_price and backtest.exit_price:
            exit_slippage = (paper.exit_price - backtest.exit_price) / backtest.exit_price
            if abs(exit_slippage) > self.slippage_threshold:
                if DiscrepancyType.SLIPPAGE not in discrepancies:
                    discrepancies.append(DiscrepancyType.SLIPPAGE)
        else:
            exit_slippage = 0

        # Timing difference
        timing_diff = (paper.entry_time - backtest.entry_time).total_seconds()
        if abs(timing_diff) > self.timing_threshold:
            discrepancies.append(DiscrepancyType.TIMING)

        # P&L difference
        pnl_diff = paper.pnl - backtest.pnl
        pnl_diff_pct = pnl_diff / abs(backtest.pnl) if backtest.pnl != 0 else 0

        if abs(pnl_diff_pct) > self.pnl_tolerance:
            if DiscrepancyType.SLIPPAGE not in discrepancies:
                discrepancies.append(DiscrepancyType.EXECUTION_GAP)

        return TradeComparison(
            symbol=paper.symbol,
            paper_entry_time=paper.entry_time,
            backtest_entry_time=backtest.entry_time,
            paper_entry_price=paper.entry_price,
            backtest_entry_price=backtest.entry_price,
            paper_exit_price=paper.exit_price,
            backtest_exit_price=backtest.exit_price,
            paper_pnl=paper.pnl,
            backtest_pnl=backtest.pnl,
            pnl_difference=pnl_diff,
            pnl_difference_pct=pnl_diff_pct,
            entry_slippage=entry_slippage,
            exit_slippage=exit_slippage,
            timing_difference_seconds=timing_diff,
            matched=True,
            discrepancy_types=discrepancies,
        )

    def _calculate_metrics(
        self,
        comparisons: List[TradeComparison],
    ) -> ComparisonMetrics:
        """Calculate aggregate comparison metrics."""
        paper_pnls = [c.paper_pnl for c in comparisons]
        backtest_pnls = [c.backtest_pnl for c in comparisons]

        total_paper = sum(paper_pnls)
        total_backtest = sum(backtest_pnls)
        pnl_diff = total_paper - total_backtest
        pnl_diff_pct = pnl_diff / abs(total_backtest) if total_backtest != 0 else 0

        paper_wins = sum(1 for p in paper_pnls if p > 0)
        backtest_wins = sum(1 for p in backtest_pnls if p > 0)
        n = len(comparisons) if comparisons else 1

        paper_wr = paper_wins / n
        backtest_wr = backtest_wins / n

        # Sharpe ratios
        paper_returns = np.array(paper_pnls)
        backtest_returns = np.array(backtest_pnls)

        paper_sharpe = np.mean(paper_returns) / (np.std(paper_returns) + 1e-8) * np.sqrt(252)
        backtest_sharpe = np.mean(backtest_returns) / (np.std(backtest_returns) + 1e-8) * np.sqrt(252)

        # Slippage and timing
        avg_slippage = np.mean([abs(c.entry_slippage) for c in comparisons]) if comparisons else 0
        avg_timing = np.mean([abs(c.timing_difference_seconds) for c in comparisons]) if comparisons else 0

        # Correlation
        if len(paper_pnls) > 1:
            correlation = np.corrcoef(paper_pnls, backtest_pnls)[0, 1]
        else:
            correlation = 1.0

        # Tracking error
        diff_returns = paper_returns - backtest_returns
        tracking_error = np.std(diff_returns) * np.sqrt(252)

        return ComparisonMetrics(
            total_paper_pnl=total_paper,
            total_backtest_pnl=total_backtest,
            pnl_difference=pnl_diff,
            pnl_difference_pct=pnl_diff_pct,
            paper_win_rate=paper_wr,
            backtest_win_rate=backtest_wr,
            win_rate_difference=paper_wr - backtest_wr,
            paper_sharpe=paper_sharpe,
            backtest_sharpe=backtest_sharpe,
            sharpe_difference=paper_sharpe - backtest_sharpe,
            avg_slippage=avg_slippage,
            avg_timing_diff=avg_timing,
            correlation=correlation,
            tracking_error=tracking_error,
        )

    def _analyze_discrepancies(
        self,
        comparisons: List[TradeComparison],
    ) -> List[DiscrepancyAnalysis]:
        """Analyze discrepancies by type."""
        discrepancy_groups: Dict[DiscrepancyType, List[TradeComparison]] = {}

        for comp in comparisons:
            for dtype in comp.discrepancy_types:
                if dtype not in discrepancy_groups:
                    discrepancy_groups[dtype] = []
                discrepancy_groups[dtype].append(comp)

        analyses = []

        descriptions = {
            DiscrepancyType.SLIPPAGE: "Price execution differs between paper and backtest",
            DiscrepancyType.TIMING: "Entry/exit timing differs significantly",
            DiscrepancyType.SIGNAL_MISMATCH: "Trading signals don't match between systems",
            DiscrepancyType.EXECUTION_GAP: "P&L differs more than expected from slippage alone",
            DiscrepancyType.DATA_DIVERGENCE: "Market data differs between paper and backtest",
            DiscrepancyType.REGIME_SHIFT: "Market regime changed during execution",
        }

        for dtype, comps in discrepancy_groups.items():
            pnl_impacts = [abs(c.pnl_difference) for c in comps]

            analysis = DiscrepancyAnalysis(
                discrepancy_type=dtype,
                count=len(comps),
                total_pnl_impact=sum(pnl_impacts),
                avg_pnl_impact=np.mean(pnl_impacts),
                description=descriptions.get(dtype, "Unknown discrepancy"),
                examples=[c.to_dict() for c in comps[:3]],
            )
            analyses.append(analysis)

        # Sort by impact
        analyses.sort(key=lambda x: x.total_pnl_impact, reverse=True)

        return analyses

    def _calculate_validation_score(
        self,
        metrics: ComparisonMetrics,
        comparisons: List[TradeComparison],
    ) -> Tuple[bool, float]:
        """
        Calculate validation score.

        Returns:
            Tuple of (passed, score)
        """
        score = 100.0

        # Penalize P&L divergence
        pnl_penalty = min(30, abs(metrics.pnl_difference_pct) * 200)
        score -= pnl_penalty

        # Penalize win rate divergence
        wr_penalty = min(20, abs(metrics.win_rate_difference) * 100)
        score -= wr_penalty

        # Penalize high slippage
        slippage_penalty = min(15, metrics.avg_slippage * 1500)
        score -= slippage_penalty

        # Penalize low correlation
        if metrics.correlation < 0.9:
            corr_penalty = (0.9 - metrics.correlation) * 50
            score -= corr_penalty

        # Penalize high tracking error
        tracking_penalty = min(15, metrics.tracking_error * 50)
        score -= tracking_penalty

        score = max(0, min(100, score))
        passed = score >= 70

        return passed, score

    def generate_report(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> ValidationReport:
        """
        Generate comparison report.

        Args:
            start_date: Start of comparison period
            end_date: End of comparison period

        Returns:
            ValidationReport with full analysis
        """
        # Match trades
        matched = self._match_trades()

        # Compare matched pairs
        comparisons = []
        for paper, backtest in matched:
            if paper and backtest:
                comp = self._compare_trade_pair(paper, backtest)
                comparisons.append(comp)

        # Calculate metrics
        metrics = self._calculate_metrics(comparisons)

        # Analyze discrepancies
        discrepancy_analysis = self._analyze_discrepancies(comparisons)

        # Calculate validation score
        passed, score = self._calculate_validation_score(metrics, comparisons)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            metrics, discrepancy_analysis, score
        )

        # Determine period
        if start_date is None and self.paper_trades:
            start_date = min(t.entry_time for t in self.paper_trades)
        if end_date is None and self.paper_trades:
            end_date = max(t.entry_time for t in self.paper_trades)

        start_date = start_date or datetime.now() - timedelta(days=30)
        end_date = end_date or datetime.now()

        return ValidationReport(
            period_start=start_date,
            period_end=end_date,
            paper_trades_count=len(self.paper_trades),
            backtest_trades_count=len(self.backtest_trades),
            matched_trades_count=len(comparisons),
            metrics=metrics,
            trade_comparisons=comparisons,
            discrepancy_analysis=discrepancy_analysis,
            validation_passed=passed,
            validation_score=score,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self,
        metrics: ComparisonMetrics,
        discrepancies: List[DiscrepancyAnalysis],
        score: float,
    ) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        if metrics.avg_slippage > 0.002:
            recommendations.append(
                f"High slippage detected ({metrics.avg_slippage:.2%}). "
                "Consider using limit orders or improving execution timing."
            )

        if metrics.avg_timing_diff > 120:
            recommendations.append(
                f"Significant timing differences ({metrics.avg_timing_diff:.0f}s). "
                "Review signal generation timing and order execution."
            )

        if abs(metrics.pnl_difference_pct) > 0.1:
            recommendations.append(
                f"P&L divergence ({metrics.pnl_difference_pct:.1%}) exceeds tolerance. "
                "Investigate execution quality and market data accuracy."
            )

        if metrics.correlation < 0.8:
            recommendations.append(
                f"Low correlation ({metrics.correlation:.2f}) between paper and backtest. "
                "Check for data synchronization issues or regime changes."
            )

        for disc in discrepancies[:2]:
            if disc.total_pnl_impact > 100:
                recommendations.append(
                    f"{disc.discrepancy_type.value}: {disc.description} "
                    f"(Impact: ${disc.total_pnl_impact:.2f})"
                )

        if score < 70:
            recommendations.append(
                "Validation score below threshold. "
                "Review all discrepancies before going live."
            )

        return recommendations[:5]

    def save_report(
        self,
        report: ValidationReport,
        filename: Optional[str] = None,
    ) -> Path:
        """Save report to disk."""
        filename = filename or f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.data_dir / filename

        with open(filepath, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        logger.info(f"Saved validation report to {filepath}")
        return filepath
