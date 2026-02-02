"""
BTC-USD Performance Diagnosis Tool.

Phase 2B Requirement:
Diagnose the ~149% return degradation observed in BTC-USD extended backtest
vs ETH-USD's ~10% degradation.

Attribution Categories:
1. Turnover costs (trading frequency)
2. Slippage costs (market impact)
3. Spread costs (bid-ask spread)
4. Strategy behavior (regime-specific issues)

Output: Actionable diagnosis report with recommendations.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TradingMetrics:
    """Metrics from backtest/paper trading."""
    symbol: str
    period_days: int

    # Returns
    gross_return_pct: float = 0.0
    net_return_pct: float = 0.0
    degradation_pct: float = 0.0  # Gross - Net

    # Trade counts
    total_trades: int = 0
    trades_per_day: float = 0.0
    win_rate: float = 0.5

    # Costs breakdown
    total_slippage_cost: float = 0.0
    total_spread_cost: float = 0.0
    total_fee_cost: float = 0.0
    total_costs: float = 0.0

    avg_slippage_bps: float = 0.0
    avg_spread_bps: float = 0.0
    avg_fee_bps: float = 0.0

    # Position metrics
    avg_position_size: float = 0.0
    avg_holding_time_hours: float = 0.0

    # Strategy breakdown
    strategy_trades: Dict[str, int] = field(default_factory=dict)
    strategy_pnl: Dict[str, float] = field(default_factory=dict)

    # Regime breakdown
    regime_trades: Dict[str, int] = field(default_factory=dict)
    regime_pnl: Dict[str, float] = field(default_factory=dict)


@dataclass
class DegradationAttribution:
    """Attribution of performance degradation to specific causes."""
    total_degradation_pct: float

    # Absolute attribution (adds up to total)
    slippage_attribution_pct: float = 0.0
    spread_attribution_pct: float = 0.0
    fee_attribution_pct: float = 0.0
    turnover_excess_pct: float = 0.0  # Cost from excessive trading

    # Strategy-specific
    worst_strategy: str = ""
    worst_strategy_drag_pct: float = 0.0

    # Regime-specific
    worst_regime: str = ""
    worst_regime_drag_pct: float = 0.0

    # Comparison metrics
    turnover_ratio_vs_baseline: float = 1.0  # vs ETH or benchmark
    slippage_ratio_vs_baseline: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_degradation_pct": round(self.total_degradation_pct, 2),
            "attribution": {
                "slippage": round(self.slippage_attribution_pct, 2),
                "spread": round(self.spread_attribution_pct, 2),
                "fees": round(self.fee_attribution_pct, 2),
                "excess_turnover": round(self.turnover_excess_pct, 2),
            },
            "worst_performers": {
                "strategy": self.worst_strategy,
                "strategy_drag_pct": round(self.worst_strategy_drag_pct, 2),
                "regime": self.worst_regime,
                "regime_drag_pct": round(self.worst_regime_drag_pct, 2),
            },
            "vs_baseline": {
                "turnover_ratio": round(self.turnover_ratio_vs_baseline, 2),
                "slippage_ratio": round(self.slippage_ratio_vs_baseline, 2),
            },
        }


@dataclass
class DiagnosisRecommendation:
    """Single actionable recommendation."""
    category: str  # turnover, slippage, spread, strategy, regime
    priority: str  # critical, high, medium, low
    issue: str
    impact_pct: float
    recommendation: str
    implementation_notes: str = ""


@dataclass
class BTCDiagnosisReport:
    """Complete diagnosis report."""
    timestamp: datetime = field(default_factory=datetime.now)
    symbol: str = "BTC-USD"

    # Core metrics
    btc_metrics: Optional[TradingMetrics] = None
    baseline_metrics: Optional[TradingMetrics] = None  # ETH-USD for comparison

    # Attribution
    attribution: Optional[DegradationAttribution] = None

    # Recommendations
    recommendations: List[DiagnosisRecommendation] = field(default_factory=list)

    # Summary
    primary_cause: str = ""
    estimated_recoverable_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "summary": {
                "primary_cause": self.primary_cause,
                "estimated_recoverable_pct": round(self.estimated_recoverable_pct, 2),
                "total_degradation_pct": (
                    self.attribution.total_degradation_pct
                    if self.attribution else 0.0
                ),
            },
            "btc_metrics": {
                "gross_return_pct": self.btc_metrics.gross_return_pct if self.btc_metrics else 0,
                "net_return_pct": self.btc_metrics.net_return_pct if self.btc_metrics else 0,
                "trades_per_day": self.btc_metrics.trades_per_day if self.btc_metrics else 0,
                "avg_holding_time_hours": self.btc_metrics.avg_holding_time_hours if self.btc_metrics else 0,
            } if self.btc_metrics else {},
            "baseline_comparison": {
                "baseline_symbol": self.baseline_metrics.symbol if self.baseline_metrics else "N/A",
                "baseline_degradation_pct": self.baseline_metrics.degradation_pct if self.baseline_metrics else 0,
                "turnover_multiple": (
                    self.btc_metrics.trades_per_day / max(0.1, self.baseline_metrics.trades_per_day)
                    if self.btc_metrics and self.baseline_metrics else 1.0
                ),
            } if self.baseline_metrics else {},
            "attribution": self.attribution.to_dict() if self.attribution else {},
            "recommendations": [
                {
                    "category": r.category,
                    "priority": r.priority,
                    "issue": r.issue,
                    "impact_pct": round(r.impact_pct, 2),
                    "recommendation": r.recommendation,
                    "implementation_notes": r.implementation_notes,
                }
                for r in self.recommendations
            ],
        }


class BTCDiagnosisTool:
    """
    Diagnostic tool for analyzing BTC-USD performance degradation.

    Compares against baseline (ETH-USD) and attributes degradation to
    specific causes with actionable recommendations.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("data/reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def analyze_from_backtest_results(
        self,
        btc_results: Dict[str, Any],
        eth_results: Optional[Dict[str, Any]] = None,
    ) -> BTCDiagnosisReport:
        """
        Analyze backtest results and generate diagnosis.

        Args:
            btc_results: BTC-USD backtest results dict
            eth_results: ETH-USD backtest results for comparison (optional)

        Returns:
            Complete diagnosis report
        """
        # Parse metrics
        btc_metrics = self._parse_backtest_results(btc_results, "BTC-USD")
        eth_metrics = (
            self._parse_backtest_results(eth_results, "ETH-USD")
            if eth_results else None
        )

        # Calculate attribution
        attribution = self._calculate_attribution(btc_metrics, eth_metrics)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            btc_metrics, eth_metrics, attribution
        )

        # Determine primary cause
        primary_cause = self._determine_primary_cause(attribution)

        # Estimate recoverable return
        recoverable = self._estimate_recoverable_return(attribution)

        report = BTCDiagnosisReport(
            btc_metrics=btc_metrics,
            baseline_metrics=eth_metrics,
            attribution=attribution,
            recommendations=recommendations,
            primary_cause=primary_cause,
            estimated_recoverable_pct=recoverable,
        )

        logger.info(
            f"BTC diagnosis complete: degradation={attribution.total_degradation_pct:.1f}%, "
            f"primary_cause={primary_cause}, recoverable={recoverable:.1f}%"
        )

        return report

    def _parse_backtest_results(
        self,
        results: Dict[str, Any],
        symbol: str,
    ) -> TradingMetrics:
        """Parse backtest results into metrics."""
        period_days = results.get("period_days", 365)

        # Extract key values with defaults
        gross_return = results.get("gross_return_pct", 0.0)
        net_return = results.get("net_return_pct", 0.0)
        total_trades = results.get("total_trades", 0)

        # Costs
        slippage = results.get("total_slippage_cost", 0.0)
        spread = results.get("total_spread_cost", 0.0)
        fees = results.get("total_fee_cost", 0.0)
        total_costs = results.get("total_costs", slippage + spread + fees)

        # Averages
        avg_trade_value = results.get("avg_trade_value", 1000)

        metrics = TradingMetrics(
            symbol=symbol,
            period_days=period_days,
            gross_return_pct=gross_return,
            net_return_pct=net_return,
            degradation_pct=gross_return - net_return,
            total_trades=total_trades,
            trades_per_day=total_trades / max(1, period_days),
            win_rate=results.get("win_rate", 0.5),
            total_slippage_cost=slippage,
            total_spread_cost=spread,
            total_fee_cost=fees,
            total_costs=total_costs,
            avg_slippage_bps=(slippage / max(1, total_trades * avg_trade_value)) * 10000,
            avg_spread_bps=(spread / max(1, total_trades * avg_trade_value)) * 10000,
            avg_fee_bps=(fees / max(1, total_trades * avg_trade_value)) * 10000,
            avg_position_size=results.get("avg_position_size", 0.0),
            avg_holding_time_hours=results.get("avg_holding_time_hours", 0.0),
            strategy_trades=results.get("strategy_trades", {}),
            strategy_pnl=results.get("strategy_pnl", {}),
            regime_trades=results.get("regime_trades", {}),
            regime_pnl=results.get("regime_pnl", {}),
        )

        return metrics

    def _calculate_attribution(
        self,
        btc: TradingMetrics,
        eth: Optional[TradingMetrics],
    ) -> DegradationAttribution:
        """Calculate degradation attribution."""
        total_deg = btc.degradation_pct

        # Direct cost attribution
        starting_capital = 10000  # Assume $10k for percentage calculation
        slippage_attr = (btc.total_slippage_cost / starting_capital) * 100
        spread_attr = (btc.total_spread_cost / starting_capital) * 100
        fee_attr = (btc.total_fee_cost / starting_capital) * 100

        # Excess turnover (compared to baseline)
        baseline_trades_per_day = eth.trades_per_day if eth else 5.0  # Default baseline
        turnover_ratio = btc.trades_per_day / max(0.1, baseline_trades_per_day)

        # Estimate excess cost from over-trading
        if turnover_ratio > 1.0:
            excess_trades = (turnover_ratio - 1.0) * baseline_trades_per_day * btc.period_days
            excess_cost_per_trade = (
                btc.avg_slippage_bps + btc.avg_spread_bps + btc.avg_fee_bps
            ) / 10000 * btc.avg_position_size
            turnover_excess = (excess_trades * excess_cost_per_trade / starting_capital) * 100
        else:
            turnover_excess = 0.0

        # Find worst strategy
        worst_strategy = ""
        worst_strategy_drag = 0.0
        for strategy, pnl in btc.strategy_pnl.items():
            if pnl < worst_strategy_drag:
                worst_strategy_drag = pnl
                worst_strategy = strategy

        # Find worst regime
        worst_regime = ""
        worst_regime_drag = 0.0
        for regime, pnl in btc.regime_pnl.items():
            if pnl < worst_regime_drag:
                worst_regime_drag = pnl
                worst_regime = regime

        # Slippage ratio vs baseline
        baseline_slippage = eth.avg_slippage_bps if eth else 3.0
        slippage_ratio = btc.avg_slippage_bps / max(0.1, baseline_slippage)

        return DegradationAttribution(
            total_degradation_pct=total_deg,
            slippage_attribution_pct=slippage_attr,
            spread_attribution_pct=spread_attr,
            fee_attribution_pct=fee_attr,
            turnover_excess_pct=turnover_excess,
            worst_strategy=worst_strategy,
            worst_strategy_drag_pct=(worst_strategy_drag / starting_capital) * 100 if worst_strategy else 0,
            worst_regime=worst_regime,
            worst_regime_drag_pct=(worst_regime_drag / starting_capital) * 100 if worst_regime else 0,
            turnover_ratio_vs_baseline=turnover_ratio,
            slippage_ratio_vs_baseline=slippage_ratio,
        )

    def _generate_recommendations(
        self,
        btc: TradingMetrics,
        eth: Optional[TradingMetrics],
        attr: DegradationAttribution,
    ) -> List[DiagnosisRecommendation]:
        """Generate actionable recommendations."""
        recs = []

        # Turnover recommendation
        if attr.turnover_ratio_vs_baseline > 1.5:
            recs.append(DiagnosisRecommendation(
                category="turnover",
                priority="critical" if attr.turnover_ratio_vs_baseline > 2.0 else "high",
                issue=f"BTC trading {attr.turnover_ratio_vs_baseline:.1f}x more frequently than baseline",
                impact_pct=attr.turnover_excess_pct,
                recommendation="Increase minimum holding time and confidence threshold for BTC entries",
                implementation_notes=(
                    "Add symbol-specific holding time constraint in TradeGate. "
                    "Consider 4+ hour minimum for BTC vs 1-2 hours for alts."
                ),
            ))

        # Slippage recommendation
        if attr.slippage_ratio_vs_baseline > 1.3:
            recs.append(DiagnosisRecommendation(
                category="slippage",
                priority="high",
                issue=f"BTC slippage {attr.slippage_ratio_vs_baseline:.1f}x higher than baseline",
                impact_pct=attr.slippage_attribution_pct,
                recommendation="Reduce position sizes or use TWAP execution for BTC",
                implementation_notes=(
                    "Current avg slippage: {:.1f} bps. "
                    "Target: <5 bps. Consider splitting large orders."
                ).format(btc.avg_slippage_bps),
            ))

        # Position size recommendation
        if btc.avg_position_size > 5000:  # > $5k positions
            recs.append(DiagnosisRecommendation(
                category="slippage",
                priority="medium",
                issue="Large position sizes contributing to market impact",
                impact_pct=attr.slippage_attribution_pct * 0.5,
                recommendation="Cap BTC position size at 3% of daily volume or $3,000 max",
                implementation_notes=(
                    "Square-root market impact scales with position size. "
                    "Smaller positions = disproportionately lower slippage."
                ),
            ))

        # Strategy-specific recommendations
        if attr.worst_strategy and attr.worst_strategy_drag_pct < -5:
            recs.append(DiagnosisRecommendation(
                category="strategy",
                priority="high",
                issue=f"Strategy '{attr.worst_strategy}' underperforming on BTC",
                impact_pct=abs(attr.worst_strategy_drag_pct),
                recommendation=f"Reduce {attr.worst_strategy} weight for BTC specifically",
                implementation_notes=(
                    "Use StrategyWeightingAdvisor to reduce weight. "
                    "Consider symbol-specific strategy preferences."
                ),
            ))

        # Regime-specific recommendations
        if attr.worst_regime and attr.worst_regime_drag_pct < -5:
            recs.append(DiagnosisRecommendation(
                category="regime",
                priority="medium",
                issue=f"Poor BTC performance in '{attr.worst_regime}' regime",
                impact_pct=abs(attr.worst_regime_drag_pct),
                recommendation=f"Avoid or reduce BTC trading during {attr.worst_regime} regime",
                implementation_notes=(
                    "Add regime filter in signal generator. "
                    "Consider holding during unfavorable regimes."
                ),
            ))

        # Holding time recommendation
        if btc.avg_holding_time_hours < 2:
            recs.append(DiagnosisRecommendation(
                category="turnover",
                priority="high",
                issue=f"Very short holding time ({btc.avg_holding_time_hours:.1f}h) increases costs",
                impact_pct=attr.turnover_excess_pct * 0.7,
                recommendation="Increase minimum holding time to 4+ hours for BTC",
                implementation_notes=(
                    "Short-term noise trading on BTC is unprofitable after costs. "
                    "Higher timeframe signals have better cost/reward."
                ),
            ))

        # Fee optimization
        if btc.avg_fee_bps > 10:
            recs.append(DiagnosisRecommendation(
                category="spread",
                priority="medium",
                issue=f"High trading fees ({btc.avg_fee_bps:.1f} bps)",
                impact_pct=attr.fee_attribution_pct,
                recommendation="Use limit orders instead of market orders",
                implementation_notes=(
                    "Maker fees are typically 0-5 bps vs 10-25 bps for takers. "
                    "Passive execution can save 50%+ in fees."
                ),
            ))

        # Sort by priority and impact
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        recs.sort(key=lambda r: (priority_order.get(r.priority, 3), -r.impact_pct))

        return recs

    def _determine_primary_cause(self, attr: DegradationAttribution) -> str:
        """Determine the primary cause of degradation."""
        causes = {
            "excessive_turnover": attr.turnover_excess_pct + (attr.turnover_ratio_vs_baseline - 1) * 10,
            "high_slippage": attr.slippage_attribution_pct * attr.slippage_ratio_vs_baseline,
            "spread_costs": attr.spread_attribution_pct,
            "strategy_mismatch": abs(attr.worst_strategy_drag_pct),
            "regime_sensitivity": abs(attr.worst_regime_drag_pct),
        }

        primary = max(causes, key=causes.get)

        cause_descriptions = {
            "excessive_turnover": "Excessive trading frequency driving costs",
            "high_slippage": "Market impact from position sizes",
            "spread_costs": "Bid-ask spread costs",
            "strategy_mismatch": f"Strategy '{attr.worst_strategy}' not suited for BTC",
            "regime_sensitivity": f"Poor performance in {attr.worst_regime} market conditions",
        }

        return cause_descriptions.get(primary, "Unknown")

    def _estimate_recoverable_return(self, attr: DegradationAttribution) -> float:
        """Estimate how much return could be recovered with optimizations."""
        # Recoverable from turnover reduction
        turnover_recoverable = min(
            attr.turnover_excess_pct * 0.7,  # Can recover 70% of excess turnover cost
            attr.total_degradation_pct * 0.3,  # But cap at 30% of total
        )

        # Recoverable from slippage reduction
        slippage_recoverable = min(
            attr.slippage_attribution_pct * 0.4,  # Can recover 40% with better execution
            attr.total_degradation_pct * 0.2,
        )

        # Recoverable from strategy optimization
        strategy_recoverable = min(
            abs(attr.worst_strategy_drag_pct) * 0.5,  # Can recover 50% with reweighting
            attr.total_degradation_pct * 0.15,
        )

        total_recoverable = turnover_recoverable + slippage_recoverable + strategy_recoverable

        # Cap at realistic maximum
        return min(total_recoverable, attr.total_degradation_pct * 0.5)

    def save_report(
        self,
        report: BTCDiagnosisReport,
        filename: Optional[str] = None,
    ) -> Path:
        """Save diagnosis report to JSON file."""
        if filename is None:
            filename = f"btc_diagnosis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        path = self.output_dir / filename

        with open(path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        logger.info(f"Diagnosis report saved to {path}")
        return path

    def generate_markdown_report(self, report: BTCDiagnosisReport) -> str:
        """Generate markdown-formatted diagnosis report."""
        lines = [
            "# BTC-USD Performance Diagnosis Report",
            "",
            f"**Generated:** {report.timestamp.isoformat()}",
            "",
            "## Executive Summary",
            "",
            f"- **Total Return Degradation:** {report.attribution.total_degradation_pct:.1f}%"
            if report.attribution else "",
            f"- **Primary Cause:** {report.primary_cause}",
            f"- **Estimated Recoverable Return:** {report.estimated_recoverable_pct:.1f}%",
            "",
        ]

        if report.btc_metrics:
            lines.extend([
                "## BTC-USD Metrics",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Gross Return | {report.btc_metrics.gross_return_pct:.1f}% |",
                f"| Net Return | {report.btc_metrics.net_return_pct:.1f}% |",
                f"| Trades per Day | {report.btc_metrics.trades_per_day:.1f} |",
                f"| Avg Holding Time | {report.btc_metrics.avg_holding_time_hours:.1f}h |",
                f"| Win Rate | {report.btc_metrics.win_rate:.1%} |",
                f"| Avg Slippage | {report.btc_metrics.avg_slippage_bps:.1f} bps |",
                "",
            ])

        if report.baseline_metrics:
            btc_tpd = report.btc_metrics.trades_per_day if report.btc_metrics else 0
            lines.extend([
                "## Comparison vs ETH-USD (Baseline)",
                "",
                "| Metric | BTC-USD | ETH-USD | Ratio |",
                "|--------|---------|---------|-------|",
                f"| Degradation | {report.btc_metrics.degradation_pct:.1f}% | {report.baseline_metrics.degradation_pct:.1f}% | {report.btc_metrics.degradation_pct / max(0.1, report.baseline_metrics.degradation_pct):.1f}x |"
                if report.btc_metrics else "",
                f"| Trades/Day | {btc_tpd:.1f} | {report.baseline_metrics.trades_per_day:.1f} | {btc_tpd / max(0.1, report.baseline_metrics.trades_per_day):.1f}x |",
                "",
            ])

        if report.attribution:
            lines.extend([
                "## Degradation Attribution",
                "",
                "| Cause | Impact |",
                "|-------|--------|",
                f"| Slippage | {report.attribution.slippage_attribution_pct:.1f}% |",
                f"| Spread | {report.attribution.spread_attribution_pct:.1f}% |",
                f"| Fees | {report.attribution.fee_attribution_pct:.1f}% |",
                f"| Excess Turnover | {report.attribution.turnover_excess_pct:.1f}% |",
                "",
            ])

        if report.recommendations:
            lines.extend([
                "## Recommendations",
                "",
            ])

            for i, rec in enumerate(report.recommendations, 1):
                priority_emoji = {
                    "critical": "🔴",
                    "high": "🟠",
                    "medium": "🟡",
                    "low": "🟢",
                }.get(rec.priority, "⚪")

                lines.extend([
                    f"### {i}. {priority_emoji} [{rec.priority.upper()}] {rec.category.title()}",
                    "",
                    f"**Issue:** {rec.issue}",
                    "",
                    f"**Impact:** {rec.impact_pct:.1f}% return drag",
                    "",
                    f"**Recommendation:** {rec.recommendation}",
                    "",
                    f"*Implementation:* {rec.implementation_notes}",
                    "",
                ])

        return "\n".join(lines)

    def run_full_diagnosis(
        self,
        btc_results: Dict[str, Any],
        eth_results: Optional[Dict[str, Any]] = None,
        save_json: bool = True,
        save_markdown: bool = True,
    ) -> Tuple[BTCDiagnosisReport, Optional[Path], Optional[Path]]:
        """
        Run full diagnosis and save reports.

        Returns:
            (report, json_path, markdown_path)
        """
        report = self.analyze_from_backtest_results(btc_results, eth_results)

        json_path = None
        md_path = None

        if save_json:
            json_path = self.save_report(report)

        if save_markdown:
            md_content = self.generate_markdown_report(report)
            md_path = self.output_dir / f"btc_diagnosis_{datetime.now().strftime('%Y%m%d')}.md"
            with open(md_path, "w") as f:
                f.write(md_content)
            logger.info(f"Markdown report saved to {md_path}")

        return report, json_path, md_path


# =============================================================================
# Example backtest results for testing
# =============================================================================

def get_example_btc_results() -> Dict[str, Any]:
    """Example BTC results showing 149% degradation."""
    return {
        "period_days": 365,
        "gross_return_pct": 180.0,
        "net_return_pct": 31.0,  # 149% degradation
        "total_trades": 3650,  # ~10 trades/day
        "win_rate": 0.48,
        "total_slippage_cost": 8500.0,
        "total_spread_cost": 3200.0,
        "total_fee_cost": 4500.0,
        "avg_trade_value": 2000,
        "avg_position_size": 2000,
        "avg_holding_time_hours": 1.5,
        "strategy_trades": {
            "TrendFollower": 800,
            "MeanReversion": 1200,
            "MomentumTrader": 600,
            "Scalper": 1050,
        },
        "strategy_pnl": {
            "TrendFollower": 2500,
            "MeanReversion": -1500,
            "MomentumTrader": 1000,
            "Scalper": -800,
        },
        "regime_trades": {
            "bull": 1200,
            "bear": 800,
            "sideways": 1400,
            "volatile": 250,
        },
        "regime_pnl": {
            "bull": 3000,
            "bear": -500,
            "sideways": -200,
            "volatile": -1100,
        },
    }


def get_example_eth_results() -> Dict[str, Any]:
    """Example ETH results showing only 10% degradation."""
    return {
        "period_days": 365,
        "gross_return_pct": 95.0,
        "net_return_pct": 85.0,  # ~10% degradation
        "total_trades": 1825,  # ~5 trades/day
        "win_rate": 0.53,
        "total_slippage_cost": 2100.0,
        "total_spread_cost": 1800.0,
        "total_fee_cost": 2200.0,
        "avg_trade_value": 1500,
        "avg_position_size": 1500,
        "avg_holding_time_hours": 4.2,
        "strategy_trades": {
            "TrendFollower": 600,
            "MeanReversion": 500,
            "MomentumTrader": 400,
            "Scalper": 325,
        },
        "strategy_pnl": {
            "TrendFollower": 4000,
            "MeanReversion": 2000,
            "MomentumTrader": 1500,
            "Scalper": 500,
        },
        "regime_trades": {
            "bull": 700,
            "bear": 400,
            "sideways": 600,
            "volatile": 125,
        },
        "regime_pnl": {
            "bull": 5000,
            "bear": 1000,
            "sideways": 1500,
            "volatile": 500,
        },
    }


if __name__ == "__main__":
    # Run example diagnosis
    logging.basicConfig(level=logging.INFO)

    tool = BTCDiagnosisTool()

    report, json_path, md_path = tool.run_full_diagnosis(
        btc_results=get_example_btc_results(),
        eth_results=get_example_eth_results(),
    )

    print("\n" + tool.generate_markdown_report(report))
