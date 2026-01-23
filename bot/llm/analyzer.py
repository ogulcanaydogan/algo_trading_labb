"""
Performance Analyzer with LLM-enhanced insights.

Analyzes trading performance and generates actionable reports.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from .prompts import PERFORMANCE_ANALYZER_PROMPT


@dataclass
class AnalysisReport:
    """Comprehensive performance analysis report."""

    summary: str
    key_metrics: Dict[str, float]
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    risk_assessment: str
    strategy_scores: Dict[str, float]
    improvement_priority: List[str]
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "summary": self.summary,
            "key_metrics": self.key_metrics,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "recommendations": self.recommendations,
            "risk_assessment": self.risk_assessment,
            "strategy_scores": self.strategy_scores,
            "improvement_priority": self.improvement_priority,
            "generated_at": self.generated_at.isoformat(),
        }

    def to_markdown(self) -> str:
        """Generate markdown report."""
        md = f"""# Performance Analysis Report
Generated: {self.generated_at.strftime("%Y-%m-%d %H:%M")}

## Summary
{self.summary}

## Key Metrics
| Metric | Value |
|--------|-------|
"""
        for metric, value in self.key_metrics.items():
            md += f"| {metric} | {value:.4f} |\n"

        md += "\n## Strengths\n"
        for s in self.strengths:
            md += f"- {s}\n"

        md += "\n## Areas for Improvement\n"
        for w in self.weaknesses:
            md += f"- {w}\n"

        md += "\n## Recommendations\n"
        for i, r in enumerate(self.recommendations, 1):
            md += f"{i}. {r}\n"

        md += f"\n## Risk Assessment\n{self.risk_assessment}\n"

        md += "\n## Improvement Priority\n"
        for i, p in enumerate(self.improvement_priority, 1):
            md += f"{i}. {p}\n"

        return md


class PerformanceAnalyzer:
    """
    Analyzes trading performance and generates insights.

    Features:
    - Statistical analysis of trades
    - Strategy comparison
    - Risk metrics calculation
    - LLM-enhanced recommendations (if available)
    """

    def __init__(self, llm_advisor: Optional[Any] = None):
        """
        Initialize analyzer.

        Args:
            llm_advisor: Optional LLMAdvisor for enhanced insights
        """
        self.llm_advisor = llm_advisor

    def analyze(
        self,
        trades: List[Dict],
        equity_curve: List[Dict],
        strategy_name: str = "unknown",
    ) -> AnalysisReport:
        """
        Perform comprehensive performance analysis.

        Args:
            trades: List of trade dictionaries
            equity_curve: List of equity points
            strategy_name: Name of the strategy analyzed

        Returns:
            AnalysisReport with insights and recommendations
        """
        if not trades:
            return self._empty_report()

        # Calculate metrics
        metrics = self._calculate_metrics(trades, equity_curve)

        # Identify strengths and weaknesses
        strengths = self._identify_strengths(metrics, trades)
        weaknesses = self._identify_weaknesses(metrics, trades)

        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, trades)

        # Risk assessment
        risk_assessment = self._assess_risk(metrics, trades)

        # Strategy scoring
        strategy_scores = self._score_strategy(metrics)

        # Priority ranking
        priority = self._prioritize_improvements(weaknesses, metrics)

        # Generate summary
        summary = self._generate_summary(metrics, strategy_name)

        return AnalysisReport(
            summary=summary,
            key_metrics=metrics,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
            risk_assessment=risk_assessment,
            strategy_scores=strategy_scores,
            improvement_priority=priority,
        )

    def _calculate_metrics(
        self,
        trades: List[Dict],
        equity_curve: List[Dict],
    ) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        pnls = [t.get("pnl", 0) for t in trades]
        pnl_pcts = [t.get("pnl_pct", 0) for t in trades]

        wins = [p for p in pnls if p > 0]
        losses = [abs(p) for p in pnls if p < 0]

        # Basic metrics
        total_pnl = sum(pnls)
        total_trades = len(trades)
        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Average metrics
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        avg_trade = np.mean(pnls) if pnls else 0

        # Risk metrics
        profit_factor = sum(wins) / sum(losses) if losses else float("inf")
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        # Sharpe ratio (simplified)
        returns_std = np.std(pnl_pcts) if len(pnl_pcts) > 1 else 1
        sharpe_ratio = np.mean(pnl_pcts) / returns_std if returns_std > 0 else 0

        # Drawdown analysis
        max_drawdown, max_drawdown_pct = self._calculate_drawdown(equity_curve)

        # Win/loss streaks
        max_win_streak, max_loss_streak = self._calculate_streaks(pnls)

        # Recovery factor
        recovery_factor = total_pnl / max_drawdown if max_drawdown > 0 else 0

        return {
            "total_pnl": total_pnl,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "avg_trade": avg_trade,
            "profit_factor": min(profit_factor, 100),  # Cap for display
            "expectancy": expectancy,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown_pct,
            "max_win_streak": max_win_streak,
            "max_loss_streak": max_loss_streak,
            "recovery_factor": recovery_factor,
        }

    def _calculate_drawdown(
        self,
        equity_curve: List[Dict],
    ) -> tuple[float, float]:
        """Calculate maximum drawdown."""
        if not equity_curve:
            return 0.0, 0.0

        balances = [e.get("balance", 0) for e in equity_curve]
        peak = balances[0]
        max_dd = 0
        max_dd_pct = 0

        for balance in balances:
            if balance > peak:
                peak = balance
            drawdown = peak - balance
            drawdown_pct = (drawdown / peak * 100) if peak > 0 else 0

            if drawdown > max_dd:
                max_dd = drawdown
                max_dd_pct = drawdown_pct

        return max_dd, max_dd_pct

    def _calculate_streaks(self, pnls: List[float]) -> tuple[int, int]:
        """Calculate maximum win and loss streaks."""
        if not pnls:
            return 0, 0

        max_win = 0
        max_loss = 0
        current_win = 0
        current_loss = 0

        for pnl in pnls:
            if pnl > 0:
                current_win += 1
                current_loss = 0
                max_win = max(max_win, current_win)
            else:
                current_loss += 1
                current_win = 0
                max_loss = max(max_loss, current_loss)

        return max_win, max_loss

    def _identify_strengths(
        self,
        metrics: Dict[str, float],
        trades: List[Dict],
    ) -> List[str]:
        """Identify strategy strengths."""
        strengths = []

        if metrics["win_rate"] > 0.55:
            strengths.append(f"Strong win rate ({metrics['win_rate']:.1%})")

        if metrics["profit_factor"] > 1.5:
            strengths.append(f"Good profit factor ({metrics['profit_factor']:.2f})")

        if metrics["sharpe_ratio"] > 1.0:
            strengths.append(
                f"Excellent risk-adjusted returns (Sharpe: {metrics['sharpe_ratio']:.2f})"
            )

        if metrics["max_drawdown_pct"] < 10:
            strengths.append(f"Low drawdown ({metrics['max_drawdown_pct']:.1f}%)")

        if metrics["recovery_factor"] > 2:
            strengths.append(f"Strong recovery factor ({metrics['recovery_factor']:.2f})")

        if metrics["max_win_streak"] > 5:
            strengths.append(f"Consistent winning streaks (max: {metrics['max_win_streak']})")

        if not strengths:
            strengths.append("Strategy is functioning but needs optimization")

        return strengths

    def _identify_weaknesses(
        self,
        metrics: Dict[str, float],
        trades: List[Dict],
    ) -> List[str]:
        """Identify strategy weaknesses."""
        weaknesses = []

        if metrics["win_rate"] < 0.45:
            weaknesses.append(
                f"Low win rate ({metrics['win_rate']:.1%}) - entry signals need refinement"
            )

        if metrics["profit_factor"] < 1.0:
            weaknesses.append(f"Unprofitable (profit factor: {metrics['profit_factor']:.2f})")

        if metrics["sharpe_ratio"] < 0.5:
            weaknesses.append(f"Poor risk-adjusted returns (Sharpe: {metrics['sharpe_ratio']:.2f})")

        if metrics["max_drawdown_pct"] > 20:
            weaknesses.append(
                f"High drawdown ({metrics['max_drawdown_pct']:.1f}%) - risk management needs improvement"
            )

        if metrics["max_loss_streak"] > 5:
            weaknesses.append(f"Extended losing streaks (max: {metrics['max_loss_streak']})")

        if metrics["avg_loss"] > metrics["avg_win"] * 1.5:
            weaknesses.append("Losses are significantly larger than wins - check stop losses")

        return weaknesses

    def _generate_recommendations(
        self,
        metrics: Dict[str, float],
        trades: List[Dict],
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Win rate improvements
        if metrics["win_rate"] < 0.5:
            recommendations.append("Increase entry confidence threshold to filter weak signals")
            recommendations.append("Add trend filter to avoid counter-trend trades")

        # Risk management
        if metrics["max_drawdown_pct"] > 15:
            recommendations.append(
                f"Reduce position size by {min(50, metrics['max_drawdown_pct'])}%"
            )
            recommendations.append("Implement maximum daily loss limit")

        # Profit optimization
        if metrics["avg_win"] < metrics["avg_loss"]:
            recommendations.append("Widen take-profit targets to improve reward/risk ratio")
            recommendations.append("Consider trailing stops to lock in profits")

        # Consistency
        if metrics["sharpe_ratio"] < 0.5:
            recommendations.append("Focus on higher probability setups only")
            recommendations.append("Consider market regime filter to avoid unfavorable conditions")

        if not recommendations:
            recommendations.append("Continue current approach with regular monitoring")
            recommendations.append("Consider scaling position size gradually")

        return recommendations[:6]  # Top 6 recommendations

    def _assess_risk(
        self,
        metrics: Dict[str, float],
        trades: List[Dict],
    ) -> str:
        """Generate risk assessment text."""
        risk_score = 0

        # Drawdown risk
        if metrics["max_drawdown_pct"] > 20:
            risk_score += 3
        elif metrics["max_drawdown_pct"] > 10:
            risk_score += 1

        # Consistency risk
        if metrics["max_loss_streak"] > 5:
            risk_score += 2
        elif metrics["max_loss_streak"] > 3:
            risk_score += 1

        # Profitability risk
        if metrics["profit_factor"] < 1.0:
            risk_score += 3
        elif metrics["profit_factor"] < 1.2:
            risk_score += 1

        # Risk level classification
        if risk_score >= 5:
            level = "HIGH"
            advice = "Significant risk of capital loss. Reduce exposure and review strategy fundamentals."
        elif risk_score >= 3:
            level = "MODERATE"
            advice = "Acceptable risk levels but monitor closely. Consider tighter stops."
        else:
            level = "LOW"
            advice = "Risk is well-managed. Maintain current approach."

        return f"Risk Level: {level}. {advice} Max drawdown: {metrics['max_drawdown_pct']:.1f}%, Loss streak risk: {metrics['max_loss_streak']} trades."

    def _score_strategy(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Score strategy on multiple dimensions."""
        # Profitability (0-100)
        profit_score = min(100, max(0, metrics["profit_factor"] * 30))

        # Consistency (0-100)
        consistency_score = min(100, metrics["win_rate"] * 100 + metrics["sharpe_ratio"] * 20)

        # Risk management (0-100)
        risk_score = max(0, 100 - metrics["max_drawdown_pct"] * 3)

        # Efficiency (0-100)
        efficiency_score = min(100, metrics["expectancy"] * 500 + 50)

        # Overall score
        overall = (
            profit_score * 0.3
            + consistency_score * 0.25
            + risk_score * 0.25
            + efficiency_score * 0.2
        )

        return {
            "profitability": round(profit_score, 1),
            "consistency": round(consistency_score, 1),
            "risk_management": round(risk_score, 1),
            "efficiency": round(efficiency_score, 1),
            "overall": round(overall, 1),
        }

    def _prioritize_improvements(
        self,
        weaknesses: List[str],
        metrics: Dict[str, float],
    ) -> List[str]:
        """Prioritize improvements by impact."""
        priority = []

        # Most critical first
        if metrics["profit_factor"] < 1.0:
            priority.append("FIX: Strategy is losing money - review entry/exit logic")

        if metrics["max_drawdown_pct"] > 20:
            priority.append("URGENT: Reduce drawdown with better position sizing")

        if metrics["win_rate"] < 0.4:
            priority.append("HIGH: Improve entry signal quality")

        if metrics["sharpe_ratio"] < 0.5:
            priority.append("MEDIUM: Optimize risk-adjusted returns")

        if metrics["avg_loss"] > metrics["avg_win"]:
            priority.append("MEDIUM: Improve stop loss placement")

        if not priority:
            priority.append("LOW: Continue optimization and monitoring")

        return priority

    def _generate_summary(
        self,
        metrics: Dict[str, float],
        strategy_name: str,
    ) -> str:
        """Generate executive summary."""
        profitability = "profitable" if metrics["profit_factor"] > 1 else "unprofitable"
        win_rate_quality = (
            "high"
            if metrics["win_rate"] > 0.55
            else "moderate"
            if metrics["win_rate"] > 0.45
            else "low"
        )

        return (
            f"The {strategy_name} strategy is {profitability} with a {win_rate_quality} win rate "
            f"({metrics['win_rate']:.1%}). Profit factor: {metrics['profit_factor']:.2f}, "
            f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}. Maximum drawdown was "
            f"{metrics['max_drawdown_pct']:.1f}% across {int(metrics['total_trades'])} trades."
        )

    def _empty_report(self) -> AnalysisReport:
        """Return empty report when no trades available."""
        return AnalysisReport(
            summary="No trades to analyze.",
            key_metrics={},
            strengths=[],
            weaknesses=["No trading data available"],
            recommendations=["Execute some trades first"],
            risk_assessment="Unable to assess - no data",
            strategy_scores={},
            improvement_priority=["Start trading to generate performance data"],
        )
