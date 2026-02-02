"""
Failure Analyzer.

Analyzes every losing trade to identify:
- Was exit timing optimal?
- Was entry premature?
- Did regime match strategy?
- Generates improvement recommendations

This is the core of the meta-learning system.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from enum import Enum
import statistics

logger = logging.getLogger(__name__)


class FailureCategory(Enum):
    """Categories of trade failures."""
    PREMATURE_ENTRY = "premature_entry"
    LATE_ENTRY = "late_entry"
    PREMATURE_EXIT = "premature_exit"
    LATE_EXIT = "late_exit"
    WRONG_DIRECTION = "wrong_direction"
    WRONG_REGIME = "wrong_regime"
    OVERLEVERAGED = "overleveraged"
    POOR_TIMING = "poor_timing"
    STOPPED_OUT = "stopped_out"
    NEWS_SHOCK = "news_shock"
    SQUEEZE = "squeeze"
    CORRELATION_FAILURE = "correlation_failure"
    UNKNOWN = "unknown"


@dataclass
class TradeAnalysis:
    """Analysis of a single trade."""
    trade_id: str
    symbol: str
    is_loss: bool
    pnl: float
    pnl_pct: float

    # Entry analysis
    entry_score: float  # 0-1, how good was entry
    entry_issues: List[str]
    optimal_entry_offset: float  # How much earlier/later would have been better

    # Exit analysis
    exit_score: float  # 0-1, how good was exit
    exit_issues: List[str]
    optimal_exit_offset: float  # How much earlier/later would have been better

    # Regime analysis
    regime_match: bool  # Did strategy match regime?
    regime_at_entry: str
    regime_at_exit: str

    # Failure categories
    failure_categories: List[FailureCategory]

    # Recommendations
    recommendations: List[str]

    # Context
    leverage_used: float
    was_short: bool
    hold_time_hours: float

    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class FailurePattern:
    """A recurring failure pattern."""
    pattern_id: str
    category: FailureCategory
    frequency: int
    symbols_affected: List[str]
    regimes_affected: List[str]
    avg_loss: float
    description: str
    mitigation: str
    last_occurrence: datetime


@dataclass
class ImprovementRecommendation:
    """Recommended improvement based on analysis."""
    category: str
    priority: int  # 1-10, higher = more important
    description: str
    expected_improvement: float  # Expected % improvement
    affected_trades_pct: float
    action: str  # Specific action to take


class FailureAnalyzer:
    """
    Analyzes trade failures to identify patterns and improvements.

    Features:
    - Entry timing analysis
    - Exit timing analysis
    - Regime-strategy matching
    - Pattern detection
    - Improvement recommendations
    """

    # Strategy-regime compatibility
    STRATEGY_REGIME_MATCH = {
        "TREND_FOLLOWER": ["BULL", "STRONG_BULL", "BEAR", "STRONG_BEAR"],
        "MEAN_REVERSION": ["SIDEWAYS", "LOW_VOL"],
        "MOMENTUM_TRADER": ["STRONG_BULL", "STRONG_BEAR", "CRASH"],
        "SHORT_SPECIALIST": ["BEAR", "STRONG_BEAR", "CRASH"],
        "SCALPER": ["SIDEWAYS", "LOW_VOL", "HIGH_VOL"],
    }

    def __init__(
        self,
        lookback_trades: int = 100,
        min_pattern_frequency: int = 3,
    ):
        self.lookback_trades = lookback_trades
        self.min_pattern_frequency = min_pattern_frequency

        # Trade analysis history
        self._analyses: List[TradeAnalysis] = []

        # Pattern tracking
        self._failure_patterns: Dict[str, FailurePattern] = {}

        # Statistics
        self._failure_counts: Dict[FailureCategory, int] = defaultdict(int)
        self._regime_failure_counts: Dict[str, int] = defaultdict(int)
        self._symbol_failure_counts: Dict[str, int] = defaultdict(int)

        logger.info("FailureAnalyzer initialized")

    def analyze_trade(
        self,
        trade_id: str,
        symbol: str,
        pnl: float,
        pnl_pct: float,
        entry_price: float,
        exit_price: float,
        entry_time: datetime,
        exit_time: datetime,
        regime_at_entry: str,
        regime_at_exit: str,
        strategy: str,
        leverage: float,
        was_short: bool,
        price_history: Optional[List[Dict]] = None,
        news_during_trade: Optional[List[Dict]] = None,
        rsi_at_entry: float = 50.0,
        rsi_at_exit: float = 50.0,
        volume_at_entry: float = 1.0,
    ) -> TradeAnalysis:
        """
        Analyze a completed trade.

        Args:
            trade_id: Unique trade identifier
            symbol: Trading symbol
            pnl: Absolute P&L
            pnl_pct: Percentage P&L
            entry_price: Entry price
            exit_price: Exit price
            entry_time: Entry timestamp
            exit_time: Exit timestamp
            regime_at_entry: Market regime at entry
            regime_at_exit: Market regime at exit
            strategy: Strategy used
            leverage: Leverage used
            was_short: Whether it was a short position
            price_history: Optional price history during trade
            news_during_trade: Optional news events during trade
            rsi_at_entry: RSI at entry
            rsi_at_exit: RSI at exit
            volume_at_entry: Volume ratio at entry

        Returns:
            TradeAnalysis with detailed findings
        """
        is_loss = pnl < 0
        hold_time = (exit_time - entry_time).total_seconds() / 3600

        entry_issues = []
        exit_issues = []
        failure_categories = []
        recommendations = []

        # 1. Entry Analysis
        entry_score = self._analyze_entry(
            symbol, entry_price, regime_at_entry, strategy,
            rsi_at_entry, volume_at_entry, was_short, entry_issues
        )

        # Calculate optimal entry offset
        optimal_entry_offset = self._calculate_optimal_entry_offset(
            price_history, entry_price, was_short
        ) if price_history else 0.0

        # 2. Exit Analysis
        exit_score = self._analyze_exit(
            symbol, exit_price, regime_at_exit, pnl_pct,
            rsi_at_exit, was_short, exit_issues
        )

        # Calculate optimal exit offset
        optimal_exit_offset = self._calculate_optimal_exit_offset(
            price_history, exit_price, was_short
        ) if price_history else 0.0

        # 3. Regime Match Analysis
        regime_match = self._check_regime_match(strategy, regime_at_entry)
        if not regime_match:
            failure_categories.append(FailureCategory.WRONG_REGIME)
            recommendations.append(
                f"Avoid {strategy} in {regime_at_entry} regime"
            )

        # 4. Identify Failure Categories
        if is_loss:
            failure_categories.extend(
                self._identify_failure_categories(
                    pnl_pct, entry_score, exit_score,
                    optimal_entry_offset, optimal_exit_offset,
                    leverage, was_short, regime_at_entry, regime_at_exit,
                    news_during_trade
                )
            )

        # 5. Generate Recommendations
        recommendations.extend(
            self._generate_recommendations(
                failure_categories, symbol, regime_at_entry,
                leverage, was_short, entry_issues, exit_issues
            )
        )

        # Create analysis
        analysis = TradeAnalysis(
            trade_id=trade_id,
            symbol=symbol,
            is_loss=is_loss,
            pnl=pnl,
            pnl_pct=pnl_pct,
            entry_score=entry_score,
            entry_issues=entry_issues,
            optimal_entry_offset=optimal_entry_offset,
            exit_score=exit_score,
            exit_issues=exit_issues,
            optimal_exit_offset=optimal_exit_offset,
            regime_match=regime_match,
            regime_at_entry=regime_at_entry,
            regime_at_exit=regime_at_exit,
            failure_categories=failure_categories,
            recommendations=recommendations,
            leverage_used=leverage,
            was_short=was_short,
            hold_time_hours=hold_time,
        )

        # Store and update statistics
        self._analyses.append(analysis)
        if len(self._analyses) > self.lookback_trades:
            self._analyses = self._analyses[-self.lookback_trades:]

        if is_loss:
            for cat in failure_categories:
                self._failure_counts[cat] += 1
            self._regime_failure_counts[regime_at_entry] += 1
            self._symbol_failure_counts[symbol] += 1

        # Detect patterns
        self._update_patterns()

        logger.info(
            f"Analyzed trade {trade_id}: {'LOSS' if is_loss else 'WIN'} "
            f"({pnl_pct*100:.2f}%), categories: {[c.value for c in failure_categories]}"
        )

        return analysis

    def _analyze_entry(
        self,
        symbol: str,
        entry_price: float,
        regime: str,
        strategy: str,
        rsi: float,
        volume: float,
        was_short: bool,
        issues: List[str],
    ) -> float:
        """Analyze entry quality. Returns score 0-1."""
        score = 0.7  # Base score

        # RSI analysis
        if was_short:
            if rsi < 40:
                score -= 0.2
                issues.append(f"Shorted when RSI low ({rsi:.1f})")
            elif rsi > 65:
                score += 0.1
        else:
            if rsi > 70:
                score -= 0.2
                issues.append(f"Bought when overbought ({rsi:.1f})")
            elif rsi < 35:
                score += 0.1

        # Volume analysis
        if volume < 0.5:
            score -= 0.1
            issues.append("Low volume at entry")
        elif volume > 2.0:
            score += 0.1

        # Regime analysis
        if not self._check_regime_match(strategy, regime):
            score -= 0.2
            issues.append(f"Strategy-regime mismatch: {strategy} in {regime}")

        return max(0.0, min(1.0, score))

    def _analyze_exit(
        self,
        symbol: str,
        exit_price: float,
        regime: str,
        pnl_pct: float,
        rsi: float,
        was_short: bool,
        issues: List[str],
    ) -> float:
        """Analyze exit quality. Returns score 0-1."""
        score = 0.7  # Base score

        # RSI at exit
        if was_short:
            if rsi < 30 and pnl_pct < 0:
                score += 0.1  # Good to exit short at oversold
            elif rsi > 50 and pnl_pct > 0:
                score -= 0.1
                issues.append("Covered short early, RSI still high")
        else:
            if rsi > 70 and pnl_pct > 0:
                score += 0.1  # Good to take profit at overbought
            elif rsi < 50 and pnl_pct > 0:
                score -= 0.1
                issues.append("Sold early, RSI still had room")

        # Loss exits
        if pnl_pct < -0.05:
            score -= 0.2
            issues.append("Large loss - consider tighter stops")
        elif pnl_pct < -0.02:
            score -= 0.1

        return max(0.0, min(1.0, score))

    def _calculate_optimal_entry_offset(
        self,
        price_history: List[Dict],
        entry_price: float,
        was_short: bool,
    ) -> float:
        """Calculate how much better entry timing could have been."""
        if not price_history or len(price_history) < 2:
            return 0.0

        prices = [p.get("close", p.get("price", entry_price)) for p in price_history]

        if was_short:
            # For shorts, best entry was at highest price
            best_price = max(prices)
            optimal_offset = (best_price - entry_price) / entry_price
        else:
            # For longs, best entry was at lowest price
            best_price = min(prices)
            optimal_offset = (entry_price - best_price) / entry_price

        return optimal_offset

    def _calculate_optimal_exit_offset(
        self,
        price_history: List[Dict],
        exit_price: float,
        was_short: bool,
    ) -> float:
        """Calculate how much better exit timing could have been."""
        if not price_history or len(price_history) < 2:
            return 0.0

        prices = [p.get("close", p.get("price", exit_price)) for p in price_history]

        if was_short:
            # For shorts, best exit was at lowest price
            best_price = min(prices)
            optimal_offset = (exit_price - best_price) / exit_price
        else:
            # For longs, best exit was at highest price
            best_price = max(prices)
            optimal_offset = (best_price - exit_price) / exit_price

        return optimal_offset

    def _check_regime_match(self, strategy: str, regime: str) -> bool:
        """Check if strategy matches regime."""
        compatible_regimes = self.STRATEGY_REGIME_MATCH.get(strategy, [])
        return regime in compatible_regimes or not compatible_regimes

    def _identify_failure_categories(
        self,
        pnl_pct: float,
        entry_score: float,
        exit_score: float,
        entry_offset: float,
        exit_offset: float,
        leverage: float,
        was_short: bool,
        regime_entry: str,
        regime_exit: str,
        news: Optional[List[Dict]],
    ) -> List[FailureCategory]:
        """Identify failure categories for a losing trade."""
        categories = []

        # Entry timing issues
        if entry_offset > 0.02:  # Could have entered 2%+ better
            if entry_offset > 0.05:
                categories.append(FailureCategory.PREMATURE_ENTRY)
            else:
                categories.append(FailureCategory.POOR_TIMING)

        # Exit timing issues
        if exit_offset > 0.02:  # Could have exited 2%+ better
            if pnl_pct < -0.03:  # Lost >3%, probably late exit
                categories.append(FailureCategory.LATE_EXIT)
            else:
                categories.append(FailureCategory.PREMATURE_EXIT)

        # Direction issues
        if was_short and regime_entry in ["BULL", "STRONG_BULL"]:
            categories.append(FailureCategory.WRONG_DIRECTION)
        elif not was_short and regime_entry in ["BEAR", "STRONG_BEAR", "CRASH"]:
            categories.append(FailureCategory.WRONG_DIRECTION)

        # Regime change
        if regime_entry != regime_exit:
            if regime_exit in ["CRASH", "HIGH_VOL"] and regime_entry not in ["CRASH", "HIGH_VOL"]:
                categories.append(FailureCategory.NEWS_SHOCK)

        # Overleveraged
        if leverage > 2.0 and abs(pnl_pct) > 0.05:
            categories.append(FailureCategory.OVERLEVERAGED)

        # Stopped out (loss close to typical stop level)
        if -0.035 <= pnl_pct <= -0.025:
            categories.append(FailureCategory.STOPPED_OUT)

        # Squeeze (shorts)
        if was_short and pnl_pct < -0.05:
            categories.append(FailureCategory.SQUEEZE)

        # News shock
        if news and len(news) > 0:
            critical_news = [n for n in news if n.get("urgency", 0) >= 7]
            if critical_news:
                categories.append(FailureCategory.NEWS_SHOCK)

        # Default
        if not categories:
            categories.append(FailureCategory.UNKNOWN)

        return categories

    def _generate_recommendations(
        self,
        categories: List[FailureCategory],
        symbol: str,
        regime: str,
        leverage: float,
        was_short: bool,
        entry_issues: List[str],
        exit_issues: List[str],
    ) -> List[str]:
        """Generate recommendations based on failure analysis."""
        recommendations = []

        for cat in categories:
            if cat == FailureCategory.PREMATURE_ENTRY:
                recommendations.append("Wait for confirmation before entry")
                recommendations.append("Use limit orders instead of market orders")

            elif cat == FailureCategory.LATE_EXIT:
                recommendations.append("Implement trailing stops")
                recommendations.append("Take partial profits earlier")

            elif cat == FailureCategory.WRONG_DIRECTION:
                if was_short:
                    recommendations.append(f"Avoid shorting in {regime} regime")
                else:
                    recommendations.append(f"Avoid longs in {regime} regime")

            elif cat == FailureCategory.OVERLEVERAGED:
                recommendations.append(f"Reduce leverage from {leverage:.1f}x")
                recommendations.append("Cap leverage at 2x for this setup")

            elif cat == FailureCategory.SQUEEZE:
                recommendations.append("Check short interest before shorting")
                recommendations.append("Use tighter stops on short positions")

            elif cat == FailureCategory.NEWS_SHOCK:
                recommendations.append("Check event calendar before trading")
                recommendations.append("Reduce position size before events")

        return recommendations

    def _update_patterns(self):
        """Update failure patterns based on recent analyses."""
        recent_losses = [a for a in self._analyses if a.is_loss]

        if len(recent_losses) < self.min_pattern_frequency:
            return

        # Count categories
        category_counts = defaultdict(list)
        for analysis in recent_losses:
            for cat in analysis.failure_categories:
                category_counts[cat].append(analysis)

        # Update patterns
        for cat, analyses in category_counts.items():
            if len(analyses) >= self.min_pattern_frequency:
                pattern_id = f"{cat.value}_{datetime.now().strftime('%Y%m')}"

                self._failure_patterns[pattern_id] = FailurePattern(
                    pattern_id=pattern_id,
                    category=cat,
                    frequency=len(analyses),
                    symbols_affected=list(set(a.symbol for a in analyses)),
                    regimes_affected=list(set(a.regime_at_entry for a in analyses)),
                    avg_loss=statistics.mean(a.pnl for a in analyses),
                    description=self._get_pattern_description(cat),
                    mitigation=self._get_pattern_mitigation(cat),
                    last_occurrence=max(a.timestamp for a in analyses),
                )

    def _get_pattern_description(self, category: FailureCategory) -> str:
        """Get description for a failure pattern."""
        descriptions = {
            FailureCategory.PREMATURE_ENTRY: "Entering positions before confirmation signals",
            FailureCategory.LATE_EXIT: "Holding losing positions too long",
            FailureCategory.WRONG_DIRECTION: "Trading against the dominant trend",
            FailureCategory.OVERLEVERAGED: "Using excessive leverage for market conditions",
            FailureCategory.SQUEEZE: "Short positions getting squeezed",
            FailureCategory.NEWS_SHOCK: "Losses due to unexpected news events",
            FailureCategory.WRONG_REGIME: "Strategy not suited for market regime",
        }
        return descriptions.get(category, "Unclassified failure pattern")

    def _get_pattern_mitigation(self, category: FailureCategory) -> str:
        """Get mitigation strategy for a failure pattern."""
        mitigations = {
            FailureCategory.PREMATURE_ENTRY: "Wait for 2+ confirmation signals before entry",
            FailureCategory.LATE_EXIT: "Implement trailing stops at 2% from peak",
            FailureCategory.WRONG_DIRECTION: "Check regime before entry, avoid counter-trend trades",
            FailureCategory.OVERLEVERAGED: "Cap leverage at 2x, reduce further in high volatility",
            FailureCategory.SQUEEZE: "Limit short exposure to 20% of portfolio",
            FailureCategory.NEWS_SHOCK: "Reduce positions 24h before scheduled events",
            FailureCategory.WRONG_REGIME: "Add regime filter to strategy entry conditions",
        }
        return mitigations.get(category, "Further analysis required")

    def get_improvement_recommendations(
        self,
        min_priority: int = 5,
    ) -> List[ImprovementRecommendation]:
        """
        Get prioritized improvement recommendations.

        Returns recommendations based on pattern analysis.
        """
        recommendations = []
        total_losses = sum(1 for a in self._analyses if a.is_loss)

        if total_losses == 0:
            return recommendations

        for pattern in self._failure_patterns.values():
            affected_pct = pattern.frequency / total_losses

            # Calculate priority based on frequency and loss amount
            priority = min(10, int(
                affected_pct * 5 + abs(pattern.avg_loss / 100) * 5
            ))

            if priority >= min_priority:
                recommendations.append(ImprovementRecommendation(
                    category=pattern.category.value,
                    priority=priority,
                    description=pattern.description,
                    expected_improvement=affected_pct * 0.5,  # Assume 50% fixable
                    affected_trades_pct=affected_pct,
                    action=pattern.mitigation,
                ))

        # Sort by priority
        recommendations.sort(key=lambda r: r.priority, reverse=True)

        return recommendations

    def get_symbol_analysis(self, symbol: str) -> Dict:
        """Get analysis summary for a specific symbol."""
        symbol_analyses = [a for a in self._analyses if a.symbol == symbol]

        if not symbol_analyses:
            return {"symbol": symbol, "trades": 0}

        losses = [a for a in symbol_analyses if a.is_loss]
        wins = [a for a in symbol_analyses if not a.is_loss]

        # Category breakdown
        category_counts = defaultdict(int)
        for analysis in losses:
            for cat in analysis.failure_categories:
                category_counts[cat.value] += 1

        return {
            "symbol": symbol,
            "trades": len(symbol_analyses),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(symbol_analyses) if symbol_analyses else 0,
            "avg_loss_pct": statistics.mean(a.pnl_pct for a in losses) if losses else 0,
            "avg_win_pct": statistics.mean(a.pnl_pct for a in wins) if wins else 0,
            "avg_entry_score": statistics.mean(a.entry_score for a in symbol_analyses),
            "avg_exit_score": statistics.mean(a.exit_score for a in symbol_analyses),
            "failure_categories": dict(category_counts),
            "common_issues": self._get_common_issues(losses),
        }

    def _get_common_issues(self, analyses: List[TradeAnalysis]) -> List[str]:
        """Get most common issues from analyses."""
        all_issues = []
        for a in analyses:
            all_issues.extend(a.entry_issues)
            all_issues.extend(a.exit_issues)

        if not all_issues:
            return []

        issue_counts = defaultdict(int)
        for issue in all_issues:
            issue_counts[issue] += 1

        sorted_issues = sorted(
            issue_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [issue for issue, count in sorted_issues[:5]]

    def get_stats(self) -> Dict:
        """Get overall failure analysis statistics."""
        total = len(self._analyses)
        losses = sum(1 for a in self._analyses if a.is_loss)

        return {
            "total_analyzed": total,
            "losses": losses,
            "loss_rate": losses / total if total > 0 else 0,
            "patterns_detected": len(self._failure_patterns),
            "top_failure_categories": dict(
                sorted(
                    self._failure_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
            ),
            "worst_regimes": dict(
                sorted(
                    self._regime_failure_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
            ),
            "worst_symbols": dict(
                sorted(
                    self._symbol_failure_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
            ),
        }


# Singleton
_failure_analyzer: Optional[FailureAnalyzer] = None


def get_failure_analyzer() -> FailureAnalyzer:
    """Get or create the FailureAnalyzer singleton."""
    global _failure_analyzer
    if _failure_analyzer is None:
        _failure_analyzer = FailureAnalyzer()
    return _failure_analyzer
