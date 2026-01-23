"""
Trade Journal and Analysis System.

This module provides:
- Trade journaling with pattern recognition
- Learning from winning/losing trades
- Identifying optimal conditions
- Exit analysis (premature exits, missed profits)
- Time-based analysis (best trading hours, days)
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import statistics


@dataclass
class TradeAnalysis:
    """Analysis of a single trade."""

    symbol: str
    side: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    hold_time_minutes: float
    exit_reason: str

    # Context at entry
    regime: str = ""
    confidence: float = 0.0
    volatility: str = ""  # low, medium, high

    # Post-trade analysis
    max_favorable_excursion: float = 0.0  # MFE - best unrealized gain
    max_adverse_excursion: float = 0.0  # MAE - worst unrealized loss
    efficiency: float = 0.0  # actual profit / MFE

    # Learning tags
    tags: List[str] = field(default_factory=list)
    lessons: List[str] = field(default_factory=list)


@dataclass
class PatternInsight:
    """Insight from pattern analysis."""

    pattern_name: str
    description: str
    occurrences: int
    win_rate: float
    avg_pnl: float
    confidence: float  # How confident we are in this pattern
    recommendation: str


@dataclass
class TradeJournalSummary:
    """Summary of trade journal analysis."""

    # Time analysis
    best_trading_hours: List[int] = field(default_factory=list)
    worst_trading_hours: List[int] = field(default_factory=list)
    best_trading_days: List[str] = field(default_factory=list)
    worst_trading_days: List[str] = field(default_factory=list)

    # Exit analysis
    premature_exits: int = 0  # Exited before reaching MFE
    optimal_exits: int = 0
    avg_profit_left_on_table: float = 0.0  # MFE - actual profit

    # Regime analysis
    best_regime: str = ""
    worst_regime: str = ""
    regime_performance: Dict[str, Dict] = field(default_factory=dict)

    # Pattern insights
    patterns: List[PatternInsight] = field(default_factory=list)

    # Key lessons
    top_lessons: List[str] = field(default_factory=list)
    areas_to_improve: List[str] = field(default_factory=list)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    calculated_at: str = ""

    def to_dict(self) -> Dict:
        result = asdict(self)
        result["patterns"] = [asdict(p) for p in self.patterns]
        return result


class TradeJournal:
    """
    Trade journal for tracking and learning from trades.
    """

    def __init__(self, journal_path: str = "data/unified_trading/trade_journal.json"):
        self.journal_path = Path(journal_path)
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: List[TradeAnalysis] = []
        self._load_journal()

    def _load_journal(self):
        """Load existing journal entries."""
        if self.journal_path.exists():
            try:
                with open(self.journal_path) as f:
                    data = json.load(f)
                    self.entries = [TradeAnalysis(**e) for e in data]
            except Exception:
                self.entries = []

    def _save_journal(self):
        """Save journal to file."""
        try:
            with open(self.journal_path, "w") as f:
                json.dump([asdict(e) for e in self.entries], f, indent=2)
        except Exception as e:
            print(f"Error saving journal: {e}")

    def add_trade(self, trade: Dict, context: Optional[Dict] = None) -> TradeAnalysis:
        """
        Add a trade to the journal with analysis.

        Args:
            trade: Trade dict with pnl, entry/exit prices, etc.
            context: Optional context (regime, confidence, etc.)
        """
        analysis = TradeAnalysis(
            symbol=trade.get("symbol", "unknown"),
            side=trade.get("side", trade.get("action", "")).lower(),
            entry_time=trade.get("entry_time", trade.get("timestamp", "")),
            exit_time=trade.get("exit_time", ""),
            entry_price=trade.get("entry_price", trade.get("price", 0)),
            exit_price=trade.get("exit_price", 0),
            pnl=trade.get("pnl", 0),
            pnl_pct=trade.get("pnl_pct", 0),
            hold_time_minutes=trade.get("hold_time_minutes", 0),
            exit_reason=trade.get("exit_reason", ""),
        )

        # Add context
        if context:
            analysis.regime = context.get("regime", "")
            analysis.confidence = context.get("confidence", 0)
            analysis.volatility = context.get("volatility", "")

        # Calculate MFE/MAE if available
        if "max_profit" in trade:
            analysis.max_favorable_excursion = trade["max_profit"]
        if "max_loss" in trade:
            analysis.max_adverse_excursion = trade["max_loss"]

        # Calculate efficiency
        if analysis.max_favorable_excursion > 0:
            analysis.efficiency = analysis.pnl / analysis.max_favorable_excursion

        # Auto-tag the trade
        analysis.tags = self._auto_tag_trade(analysis)

        # Generate lessons
        analysis.lessons = self._generate_lessons(analysis)

        self.entries.append(analysis)
        self._save_journal()

        return analysis

    def _auto_tag_trade(self, trade: TradeAnalysis) -> List[str]:
        """Auto-generate tags based on trade characteristics."""
        tags = []

        # Outcome tags
        if trade.pnl > 0:
            tags.append("winner")
            if trade.pnl_pct > 2:
                tags.append("big_winner")
        else:
            tags.append("loser")
            if trade.pnl_pct < -2:
                tags.append("big_loser")

        # Hold time tags
        if trade.hold_time_minutes < 15:
            tags.append("quick_trade")
        elif trade.hold_time_minutes > 120:
            tags.append("long_hold")

        # Exit tags
        if trade.exit_reason:
            tags.append(f"exit_{trade.exit_reason}")

        # Efficiency tags
        if trade.efficiency > 0.8:
            tags.append("efficient_exit")
        elif trade.efficiency < 0.3 and trade.pnl > 0:
            tags.append("premature_exit")

        # Regime tags
        if trade.regime:
            tags.append(f"regime_{trade.regime}")

        # Confidence tags
        if trade.confidence > 0.7:
            tags.append("high_confidence")
        elif trade.confidence < 0.5:
            tags.append("low_confidence")

        return tags

    def _generate_lessons(self, trade: TradeAnalysis) -> List[str]:
        """Generate learning lessons from trade."""
        lessons = []

        # Premature exit lesson
        if trade.efficiency < 0.5 and trade.pnl > 0:
            lessons.append(
                f"Premature exit: captured only {trade.efficiency:.0%} of potential profit. "
                f"Consider widening take-profit or using trailing stops."
            )

        # Big loser lesson
        if trade.pnl_pct < -2:
            lessons.append(
                f"Large loss ({trade.pnl_pct:.1f}%). "
                f"Review stop-loss placement and position sizing."
            )

        # Quick loss lesson
        if trade.pnl < 0 and trade.hold_time_minutes < 10:
            lessons.append(
                f"Quick stop-out in {trade.hold_time_minutes:.0f} min. "
                f"Entry timing may need improvement."
            )

        # Low confidence winner
        if trade.pnl > 0 and trade.confidence < 0.5:
            lessons.append(
                f"Winner despite low confidence ({trade.confidence:.0%}). "
                f"Market was favorable; don't rely on luck."
            )

        # High confidence loser
        if trade.pnl < 0 and trade.confidence > 0.7:
            lessons.append(
                f"Loss despite high confidence ({trade.confidence:.0%}). "
                f"Review signal quality and market conditions."
            )

        return lessons

    def analyze_patterns(self) -> List[PatternInsight]:
        """Identify patterns in winning and losing trades."""
        patterns = []

        if len(self.entries) < 10:
            return patterns

        # Pattern 1: Time of day analysis
        hour_perf = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0})
        for entry in self.entries:
            try:
                hour = datetime.fromisoformat(entry.entry_time).hour
                if entry.pnl > 0:
                    hour_perf[hour]["wins"] += 1
                else:
                    hour_perf[hour]["losses"] += 1
                hour_perf[hour]["pnl"] += entry.pnl
            except Exception:
                pass

        # Find best/worst hours
        for hour, data in hour_perf.items():
            total = data["wins"] + data["losses"]
            if total >= 3:
                win_rate = data["wins"] / total
                if win_rate > 0.65:
                    patterns.append(
                        PatternInsight(
                            pattern_name=f"strong_hour_{hour}",
                            description=f"Trading at {hour}:00 shows {win_rate:.0%} win rate",
                            occurrences=total,
                            win_rate=win_rate,
                            avg_pnl=data["pnl"] / total,
                            confidence=min(1.0, total / 20),
                            recommendation=f"Prioritize trades around {hour}:00",
                        )
                    )
                elif win_rate < 0.35:
                    patterns.append(
                        PatternInsight(
                            pattern_name=f"weak_hour_{hour}",
                            description=f"Trading at {hour}:00 shows only {win_rate:.0%} win rate",
                            occurrences=total,
                            win_rate=win_rate,
                            avg_pnl=data["pnl"] / total,
                            confidence=min(1.0, total / 20),
                            recommendation=f"Avoid or reduce trading at {hour}:00",
                        )
                    )

        # Pattern 2: Regime performance
        regime_perf = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0})
        for entry in self.entries:
            if entry.regime:
                if entry.pnl > 0:
                    regime_perf[entry.regime]["wins"] += 1
                else:
                    regime_perf[entry.regime]["losses"] += 1
                regime_perf[entry.regime]["pnl"] += entry.pnl

        for regime, data in regime_perf.items():
            total = data["wins"] + data["losses"]
            if total >= 5:
                win_rate = data["wins"] / total
                if win_rate > 0.6:
                    patterns.append(
                        PatternInsight(
                            pattern_name=f"strong_regime_{regime}",
                            description=f"Strong performance in {regime} regime ({win_rate:.0%} win rate)",
                            occurrences=total,
                            win_rate=win_rate,
                            avg_pnl=data["pnl"] / total,
                            confidence=min(1.0, total / 15),
                            recommendation=f"Increase position size in {regime} regime",
                        )
                    )
                elif win_rate < 0.4:
                    patterns.append(
                        PatternInsight(
                            pattern_name=f"weak_regime_{regime}",
                            description=f"Weak performance in {regime} regime ({win_rate:.0%} win rate)",
                            occurrences=total,
                            win_rate=win_rate,
                            avg_pnl=data["pnl"] / total,
                            confidence=min(1.0, total / 15),
                            recommendation=f"Reduce trading or skip {regime} regime",
                        )
                    )

        # Pattern 3: Confidence threshold analysis
        high_conf = [e for e in self.entries if e.confidence > 0.65]
        low_conf = [e for e in self.entries if e.confidence < 0.55 and e.confidence > 0]

        if len(high_conf) >= 5:
            wins = len([e for e in high_conf if e.pnl > 0])
            win_rate = wins / len(high_conf)
            avg_pnl = statistics.mean([e.pnl for e in high_conf])
            patterns.append(
                PatternInsight(
                    pattern_name="high_confidence_trades",
                    description=f"High confidence (>65%) trades: {win_rate:.0%} win rate",
                    occurrences=len(high_conf),
                    win_rate=win_rate,
                    avg_pnl=avg_pnl,
                    confidence=min(1.0, len(high_conf) / 20),
                    recommendation="Trust high confidence signals"
                    if win_rate > 0.55
                    else "Review confidence calibration",
                )
            )

        if len(low_conf) >= 5:
            wins = len([e for e in low_conf if e.pnl > 0])
            win_rate = wins / len(low_conf)
            avg_pnl = statistics.mean([e.pnl for e in low_conf])
            if win_rate < 0.45:
                patterns.append(
                    PatternInsight(
                        pattern_name="low_confidence_trades",
                        description=f"Low confidence (<55%) trades underperform: {win_rate:.0%} win rate",
                        occurrences=len(low_conf),
                        win_rate=win_rate,
                        avg_pnl=avg_pnl,
                        confidence=min(1.0, len(low_conf) / 20),
                        recommendation="Raise confidence threshold to 0.55+",
                    )
                )

        # Pattern 4: Hold time analysis
        quick_trades = [e for e in self.entries if e.hold_time_minutes < 30]
        long_trades = [e for e in self.entries if e.hold_time_minutes > 60]

        if len(quick_trades) >= 5 and len(long_trades) >= 5:
            quick_wr = len([e for e in quick_trades if e.pnl > 0]) / len(quick_trades)
            long_wr = len([e for e in long_trades if e.pnl > 0]) / len(long_trades)

            if quick_wr > long_wr + 0.15:
                patterns.append(
                    PatternInsight(
                        pattern_name="quick_trades_better",
                        description=f"Quick trades (<30min) outperform: {quick_wr:.0%} vs {long_wr:.0%}",
                        occurrences=len(quick_trades),
                        win_rate=quick_wr,
                        avg_pnl=statistics.mean([e.pnl for e in quick_trades]),
                        confidence=0.7,
                        recommendation="Consider tighter take-profits for faster exits",
                    )
                )
            elif long_wr > quick_wr + 0.15:
                patterns.append(
                    PatternInsight(
                        pattern_name="long_trades_better",
                        description=f"Longer holds (>60min) outperform: {long_wr:.0%} vs {quick_wr:.0%}",
                        occurrences=len(long_trades),
                        win_rate=long_wr,
                        avg_pnl=statistics.mean([e.pnl for e in long_trades]),
                        confidence=0.7,
                        recommendation="Be patient; avoid premature exits",
                    )
                )

        return patterns

    def get_summary(self) -> TradeJournalSummary:
        """Generate comprehensive journal summary with insights."""
        summary = TradeJournalSummary()
        summary.calculated_at = datetime.now().isoformat()

        if len(self.entries) < 5:
            summary.recommendations.append("Need more trades for meaningful analysis (min 10)")
            return summary

        # Time analysis
        hour_perf = defaultdict(lambda: {"wins": 0, "losses": 0})
        day_perf = defaultdict(lambda: {"wins": 0, "losses": 0})

        for entry in self.entries:
            try:
                dt = datetime.fromisoformat(entry.entry_time)
                hour = dt.hour
                day = dt.strftime("%A")

                if entry.pnl > 0:
                    hour_perf[hour]["wins"] += 1
                    day_perf[day]["wins"] += 1
                else:
                    hour_perf[hour]["losses"] += 1
                    day_perf[day]["losses"] += 1
            except Exception:
                pass

        # Best/worst hours
        hour_rates = {
            h: d["wins"] / (d["wins"] + d["losses"])
            for h, d in hour_perf.items()
            if d["wins"] + d["losses"] >= 2
        }
        if hour_rates:
            sorted_hours = sorted(hour_rates.items(), key=lambda x: x[1], reverse=True)
            summary.best_trading_hours = [h for h, r in sorted_hours[:3] if r > 0.5]
            summary.worst_trading_hours = [h for h, r in sorted_hours[-3:] if r < 0.5]

        # Best/worst days
        day_rates = {
            d: data["wins"] / (data["wins"] + data["losses"])
            for d, data in day_perf.items()
            if data["wins"] + data["losses"] >= 2
        }
        if day_rates:
            sorted_days = sorted(day_rates.items(), key=lambda x: x[1], reverse=True)
            summary.best_trading_days = [d for d, r in sorted_days[:2] if r > 0.5]
            summary.worst_trading_days = [d for d, r in sorted_days[-2:] if r < 0.5]

        # Exit analysis
        winners_with_mfe = [e for e in self.entries if e.pnl > 0 and e.max_favorable_excursion > 0]
        if winners_with_mfe:
            summary.premature_exits = len([e for e in winners_with_mfe if e.efficiency < 0.5])
            summary.optimal_exits = len([e for e in winners_with_mfe if e.efficiency > 0.7])
            left_on_table = [e.max_favorable_excursion - e.pnl for e in winners_with_mfe]
            summary.avg_profit_left_on_table = (
                statistics.mean(left_on_table) if left_on_table else 0
            )

        # Regime analysis
        regime_perf = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0})
        for entry in self.entries:
            if entry.regime:
                if entry.pnl > 0:
                    regime_perf[entry.regime]["wins"] += 1
                else:
                    regime_perf[entry.regime]["losses"] += 1
                regime_perf[entry.regime]["pnl"] += entry.pnl

        for regime, data in regime_perf.items():
            total = data["wins"] + data["losses"]
            summary.regime_performance[regime] = {
                "trades": total,
                "win_rate": data["wins"] / total if total > 0 else 0,
                "total_pnl": data["pnl"],
            }

        if regime_perf:
            best = max(regime_perf.items(), key=lambda x: x[1]["pnl"])
            worst = min(regime_perf.items(), key=lambda x: x[1]["pnl"])
            summary.best_regime = best[0]
            summary.worst_regime = worst[0]

        # Get patterns
        summary.patterns = self.analyze_patterns()

        # Compile lessons
        all_lessons = []
        for entry in self.entries[-20:]:  # Last 20 trades
            all_lessons.extend(entry.lessons)

        # Count lesson frequency
        lesson_counts = defaultdict(int)
        for lesson in all_lessons:
            # Simplify lesson to category
            if "premature" in lesson.lower():
                lesson_counts["Premature exits"] += 1
            elif "large loss" in lesson.lower():
                lesson_counts["Position sizing"] += 1
            elif "quick stop" in lesson.lower():
                lesson_counts["Entry timing"] += 1
            elif "confidence" in lesson.lower():
                lesson_counts["Signal quality"] += 1

        summary.top_lessons = [
            f"{k}: {v} occurrences"
            for k, v in sorted(lesson_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        ]

        # Generate recommendations
        recommendations = []

        # Win rate based
        total = len(self.entries)
        wins = len([e for e in self.entries if e.pnl > 0])
        win_rate = wins / total if total > 0 else 0

        if win_rate < 0.45:
            recommendations.append("Win rate below 45% - consider raising confidence threshold")
        elif win_rate > 0.65:
            recommendations.append(
                "Strong win rate - can slightly lower confidence threshold to increase trade frequency"
            )

        # Exit efficiency based
        if summary.premature_exits > summary.optimal_exits:
            recommendations.append(
                "Many premature exits - consider trailing stops or wider take-profits"
            )

        # Regime based
        if (
            summary.worst_regime
            and summary.regime_performance.get(summary.worst_regime, {}).get("win_rate", 1) < 0.4
        ):
            recommendations.append(
                f"Poor performance in {summary.worst_regime} regime - consider skipping or reducing size"
            )

        # Pattern based
        for pattern in summary.patterns:
            if pattern.confidence > 0.6 and pattern.recommendation:
                recommendations.append(pattern.recommendation)

        summary.recommendations = recommendations[:5]  # Top 5

        return summary

    def get_recent_lessons(self, n: int = 10) -> List[str]:
        """Get lessons from last N trades."""
        lessons = []
        for entry in self.entries[-n:]:
            lessons.extend(entry.lessons)
        return lessons


# Standalone function for quick analysis
def analyze_trades(
    trades_file: str = "data/ml_paper_trading_enhanced/trades.json",
) -> TradeJournalSummary:
    """
    Quick function to analyze trades and get insights.
    """
    journal = TradeJournal()

    # Load trades
    path = Path(trades_file)
    if path.exists():
        try:
            with open(path) as f:
                trades = json.load(f)

            # Add closed trades to journal
            for trade in trades:
                if "pnl" in trade:
                    journal.add_trade(trade)
        except Exception as e:
            print(f"Error loading trades: {e}")

    return journal.get_summary()


if __name__ == "__main__":
    # Test with sample trades
    journal = TradeJournal(journal_path="data/test_journal.json")

    sample_trades = [
        {
            "symbol": "BTC/USDT",
            "side": "long",
            "entry_time": "2026-01-15T10:30:00",
            "exit_time": "2026-01-15T11:15:00",
            "entry_price": 94000,
            "exit_price": 94800,
            "pnl": 80,
            "pnl_pct": 0.85,
            "hold_time_minutes": 45,
            "exit_reason": "take_profit",
            "max_profit": 120,
        },
        {
            "symbol": "ETH/USDT",
            "side": "long",
            "entry_time": "2026-01-15T14:00:00",
            "exit_time": "2026-01-15T14:20:00",
            "entry_price": 3100,
            "exit_price": 3050,
            "pnl": -50,
            "pnl_pct": -1.6,
            "hold_time_minutes": 20,
            "exit_reason": "stop_loss",
        },
    ]

    for trade in sample_trades:
        analysis = journal.add_trade(trade, {"regime": "bull", "confidence": 0.65})
        print(f"\nTrade: {trade['symbol']} - Tags: {analysis.tags}")
        for lesson in analysis.lessons:
            print(f"  Lesson: {lesson}")

    summary = journal.get_summary()
    print("\n=== Journal Summary ===")
    print(f"Best hours: {summary.best_trading_hours}")
    print(f"Recommendations: {summary.recommendations}")
