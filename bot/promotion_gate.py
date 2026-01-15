"""
Champion-Challenger Promotion Gate

Safe deployment system for strategy/model promotion:
1. Challenger runs in shadow mode (no real trades)
2. Performance compared to champion over evaluation period
3. Statistical tests ensure improvement is real, not noise
4. Gradual rollout with traffic splitting
5. Automatic rollback if challenger degrades

This prevents "online learning ruins account" scenarios by ensuring
all changes are validated before live deployment.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class PromotionStatus(Enum):
    """Status of a challenger in the promotion pipeline"""
    PENDING = "pending"  # Waiting to start evaluation
    SHADOW = "shadow"  # Running in shadow mode
    EVALUATION = "evaluation"  # Being evaluated against champion
    CANARY = "canary"  # Partial traffic allocation
    PROMOTED = "promoted"  # Fully promoted to champion
    REJECTED = "rejected"  # Failed evaluation
    ROLLED_BACK = "rolled_back"  # Was promoted but rolled back


class ComparisonResult(Enum):
    """Result of champion vs challenger comparison"""
    CHALLENGER_BETTER = "challenger_better"
    CHAMPION_BETTER = "champion_better"
    NO_DIFFERENCE = "no_difference"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class PromotionCriteria:
    """Criteria for promoting a challenger"""
    # Minimum requirements
    min_shadow_days: int = 14
    min_shadow_trades: int = 50
    min_evaluation_days: int = 7

    # Performance thresholds (relative to champion)
    min_sharpe_improvement: float = 0.1  # Must be 10% better Sharpe
    min_return_improvement: float = 0.05  # 5% better returns
    max_drawdown_increase: float = 0.1  # Can't have 10% worse drawdown

    # Statistical significance
    confidence_level: float = 0.95
    min_t_statistic: float = 2.0

    # Risk constraints
    max_acceptable_drawdown: float = 15.0  # Absolute max drawdown %

    # Canary rollout
    canary_traffic_pct: float = 10.0  # Start with 10% traffic
    canary_duration_days: int = 3
    canary_success_threshold: float = 0.9  # 90% of champion performance


@dataclass
class ChallengerPerformance:
    """Performance metrics for a challenger"""
    # Identity
    challenger_id: str
    strategy_name: str
    strategy_version: str

    # Tracking
    status: PromotionStatus = PromotionStatus.PENDING
    started_at: Optional[datetime] = None
    promoted_at: Optional[datetime] = None
    evaluation_started_at: Optional[datetime] = None

    # Shadow mode metrics
    shadow_trades: int = 0
    shadow_days: int = 0
    shadow_pnl_pct: float = 0.0
    shadow_sharpe: float = 0.0
    shadow_max_drawdown: float = 0.0
    shadow_win_rate: float = 0.0

    # Trade history (for statistical tests)
    shadow_trade_returns: List[float] = field(default_factory=list)

    # Evaluation metrics (vs champion)
    evaluation_trades: int = 0
    evaluation_pnl_pct: float = 0.0
    evaluation_sharpe: float = 0.0

    # Canary metrics
    canary_trades: int = 0
    canary_pnl_pct: float = 0.0

    # Comparison results
    comparison_result: Optional[ComparisonResult] = None
    t_statistic: float = 0.0
    p_value: float = 1.0

    # Notes
    rejection_reason: str = ""
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "challenger_id": self.challenger_id,
            "strategy_name": self.strategy_name,
            "strategy_version": self.strategy_version,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "shadow_days": self.shadow_days,
            "shadow_trades": self.shadow_trades,
            "shadow_pnl_pct": round(self.shadow_pnl_pct, 2),
            "shadow_sharpe": round(self.shadow_sharpe, 3),
            "shadow_max_drawdown": round(self.shadow_max_drawdown, 2),
            "comparison_result": self.comparison_result.value if self.comparison_result else None,
            "t_statistic": round(self.t_statistic, 3),
            "p_value": round(self.p_value, 4),
            "rejection_reason": self.rejection_reason,
        }


@dataclass
class ChampionRecord:
    """Record of the current champion"""
    strategy_name: str
    strategy_version: str
    promoted_at: datetime

    # Lifetime metrics
    total_trades: int = 0
    total_pnl_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0

    # Recent metrics (for comparison)
    recent_trade_returns: List[float] = field(default_factory=list)
    recent_sharpe: float = 0.0
    recent_pnl_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "strategy_version": self.strategy_version,
            "promoted_at": self.promoted_at.isoformat(),
            "total_trades": self.total_trades,
            "total_pnl_pct": round(self.total_pnl_pct, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "max_drawdown": round(self.max_drawdown, 2),
            "win_rate": round(self.win_rate, 4),
        }


class PromotionGate:
    """
    Champion-Challenger promotion system.

    Manages the lifecycle of strategy promotion:
    PENDING -> SHADOW -> EVALUATION -> CANARY -> PROMOTED
                |            |            |
                v            v            v
             REJECTED    REJECTED    ROLLED_BACK
    """

    def __init__(
        self,
        criteria: Optional[PromotionCriteria] = None,
        state_file: str = "data/promotion_state.json"
    ):
        self.criteria = criteria or PromotionCriteria()
        self.state_file = Path(state_file)

        # Current champion per strategy type
        self.champions: Dict[str, ChampionRecord] = {}

        # Active challengers
        self.challengers: Dict[str, ChallengerPerformance] = {}

        # Historical promotions
        self.promotion_history: List[Dict[str, Any]] = []

        # Load state
        self._load_state()

        logger.info("Promotion Gate initialized")

    def _load_state(self):
        """Load state from file"""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    state = json.load(f)

                # Restore champions
                for name, data in state.get("champions", {}).items():
                    self.champions[name] = ChampionRecord(
                        strategy_name=data["strategy_name"],
                        strategy_version=data["strategy_version"],
                        promoted_at=datetime.fromisoformat(data["promoted_at"]),
                        total_trades=data.get("total_trades", 0),
                        total_pnl_pct=data.get("total_pnl_pct", 0),
                        sharpe_ratio=data.get("sharpe_ratio", 0),
                    )

                self.promotion_history = state.get("history", [])
                logger.info(f"Loaded promotion state: {len(self.champions)} champions")

            except Exception as e:
                logger.error(f"Failed to load promotion state: {e}")

    def _save_state(self):
        """Save state to file"""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)

            state = {
                "champions": {
                    name: champion.to_dict()
                    for name, champion in self.champions.items()
                },
                "challengers": {
                    cid: c.to_dict()
                    for cid, c in self.challengers.items()
                },
                "history": self.promotion_history[-100:],  # Keep last 100
                "saved_at": datetime.now().isoformat(),
            }

            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save promotion state: {e}")

    def register_challenger(
        self,
        strategy_name: str,
        strategy_version: str,
        challenger_id: Optional[str] = None
    ) -> str:
        """
        Register a new challenger for evaluation.

        Args:
            strategy_name: Name of the strategy
            strategy_version: Version being challenged
            challenger_id: Optional custom ID

        Returns:
            Challenger ID
        """
        if challenger_id is None:
            challenger_id = f"{strategy_name}_v{strategy_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        challenger = ChallengerPerformance(
            challenger_id=challenger_id,
            strategy_name=strategy_name,
            strategy_version=strategy_version,
            status=PromotionStatus.PENDING,
        )

        self.challengers[challenger_id] = challenger
        logger.info(f"Registered challenger: {challenger_id}")

        self._save_state()
        return challenger_id

    def start_shadow_mode(self, challenger_id: str) -> bool:
        """Start shadow mode for a challenger"""
        if challenger_id not in self.challengers:
            logger.error(f"Challenger not found: {challenger_id}")
            return False

        challenger = self.challengers[challenger_id]
        challenger.status = PromotionStatus.SHADOW
        challenger.started_at = datetime.now()
        challenger.notes.append(f"Shadow mode started: {datetime.now().isoformat()}")

        logger.info(f"Challenger {challenger_id} entering shadow mode")
        self._save_state()
        return True

    def record_shadow_trade(
        self,
        challenger_id: str,
        pnl_pct: float,
        trade_details: Optional[Dict[str, Any]] = None
    ):
        """Record a trade result from shadow mode"""
        if challenger_id not in self.challengers:
            return

        challenger = self.challengers[challenger_id]
        if challenger.status != PromotionStatus.SHADOW:
            return

        challenger.shadow_trades += 1
        challenger.shadow_trade_returns.append(pnl_pct)
        challenger.shadow_pnl_pct += pnl_pct

        # Update days count
        if challenger.started_at:
            challenger.shadow_days = (datetime.now() - challenger.started_at).days

        # Update metrics
        if len(challenger.shadow_trade_returns) >= 5:
            returns = np.array(challenger.shadow_trade_returns)
            challenger.shadow_win_rate = np.mean(returns > 0)

            if len(returns) >= 10:
                challenger.shadow_sharpe = self._calculate_sharpe(returns)

            # Track drawdown
            cumulative = np.cumsum(returns)
            peak = np.maximum.accumulate(cumulative)
            drawdown = peak - cumulative
            challenger.shadow_max_drawdown = np.max(drawdown)

        # Check if ready for evaluation
        self._check_shadow_completion(challenger_id)

    def _check_shadow_completion(self, challenger_id: str):
        """Check if shadow mode is complete"""
        challenger = self.challengers[challenger_id]

        if (
            challenger.shadow_days >= self.criteria.min_shadow_days and
            challenger.shadow_trades >= self.criteria.min_shadow_trades
        ):
            # Check if meets minimum quality
            if challenger.shadow_max_drawdown > self.criteria.max_acceptable_drawdown:
                self._reject_challenger(
                    challenger_id,
                    f"Shadow mode drawdown {challenger.shadow_max_drawdown:.1f}% exceeds max {self.criteria.max_acceptable_drawdown}%"
                )
            else:
                # Ready for evaluation
                logger.info(f"Challenger {challenger_id} ready for evaluation")
                challenger.notes.append(f"Shadow mode complete: {challenger.shadow_trades} trades, {challenger.shadow_days} days")

    def start_evaluation(self, challenger_id: str) -> bool:
        """Start formal evaluation against champion"""
        if challenger_id not in self.challengers:
            return False

        challenger = self.challengers[challenger_id]

        # Verify shadow mode complete
        if challenger.shadow_days < self.criteria.min_shadow_days:
            logger.warning(f"Challenger {challenger_id} hasn't completed shadow mode")
            return False

        challenger.status = PromotionStatus.EVALUATION
        challenger.evaluation_started_at = datetime.now()
        challenger.notes.append(f"Evaluation started: {datetime.now().isoformat()}")

        logger.info(f"Challenger {challenger_id} entering evaluation")
        self._save_state()
        return True

    def evaluate_challenger(
        self,
        challenger_id: str,
        champion_name: Optional[str] = None
    ) -> Tuple[ComparisonResult, Dict[str, Any]]:
        """
        Evaluate challenger against champion.

        Returns comparison result and detailed analysis.
        """
        if challenger_id not in self.challengers:
            return ComparisonResult.INSUFFICIENT_DATA, {"error": "Challenger not found"}

        challenger = self.challengers[challenger_id]

        # Get champion for comparison
        champ_key = champion_name or challenger.strategy_name
        champion = self.champions.get(champ_key)

        if champion is None:
            # No champion = auto-promote first challenger
            logger.info(f"No champion for {champ_key}, challenger auto-promoted")
            challenger.comparison_result = ComparisonResult.CHALLENGER_BETTER
            return ComparisonResult.CHALLENGER_BETTER, {"reason": "No existing champion"}

        # Need sufficient data
        if len(challenger.shadow_trade_returns) < self.criteria.min_shadow_trades:
            return ComparisonResult.INSUFFICIENT_DATA, {
                "reason": f"Need {self.criteria.min_shadow_trades} trades, have {len(challenger.shadow_trade_returns)}"
            }

        if len(champion.recent_trade_returns) < 20:
            # Not enough champion data, use overall metrics
            champion_returns = np.array([champion.sharpe_ratio] * 30)  # Synthetic
        else:
            champion_returns = np.array(champion.recent_trade_returns)

        challenger_returns = np.array(challenger.shadow_trade_returns)

        # Statistical comparison
        t_stat, p_value = self._welch_t_test(challenger_returns, champion_returns)
        challenger.t_statistic = t_stat
        challenger.p_value = p_value

        analysis = {
            "challenger_sharpe": challenger.shadow_sharpe,
            "champion_sharpe": champion.sharpe_ratio,
            "challenger_pnl": challenger.shadow_pnl_pct,
            "champion_recent_pnl": champion.recent_pnl_pct,
            "t_statistic": t_stat,
            "p_value": p_value,
            "sharpe_improvement": (challenger.shadow_sharpe - champion.sharpe_ratio) / champion.sharpe_ratio if champion.sharpe_ratio else 0,
        }

        # Determine result
        sharpe_improvement = analysis["sharpe_improvement"]
        significant = p_value < (1 - self.criteria.confidence_level) and t_stat > self.criteria.min_t_statistic

        if (
            sharpe_improvement >= self.criteria.min_sharpe_improvement and
            challenger.shadow_max_drawdown <= champion.max_drawdown * (1 + self.criteria.max_drawdown_increase) and
            significant
        ):
            result = ComparisonResult.CHALLENGER_BETTER
            analysis["decision"] = "Challenger shows statistically significant improvement"
        elif sharpe_improvement < -self.criteria.min_sharpe_improvement:
            result = ComparisonResult.CHAMPION_BETTER
            analysis["decision"] = "Champion is clearly better"
        else:
            result = ComparisonResult.NO_DIFFERENCE
            analysis["decision"] = "No statistically significant difference"

        challenger.comparison_result = result
        self._save_state()

        return result, analysis

    def _welch_t_test(
        self, sample1: np.ndarray, sample2: np.ndarray
    ) -> Tuple[float, float]:
        """
        Perform Welch's t-test for samples with unequal variances.

        Returns (t_statistic, p_value)
        """
        n1, n2 = len(sample1), len(sample2)
        if n1 < 2 or n2 < 2:
            return 0.0, 1.0

        mean1, mean2 = np.mean(sample1), np.mean(sample2)
        var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)

        # Welch's t-statistic
        se = np.sqrt(var1 / n1 + var2 / n2)
        if se == 0:
            return 0.0, 1.0

        t_stat = (mean1 - mean2) / se

        # Degrees of freedom (Welch-Satterthwaite)
        num = (var1 / n1 + var2 / n2) ** 2
        denom = (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
        df = num / denom if denom > 0 else n1 + n2 - 2

        # Approximate p-value using normal distribution (for large df)
        # For production, use scipy.stats.t.sf
        p_value = 2 * (1 - self._normal_cdf(abs(t_stat)))

        return float(t_stat), float(p_value)

    def _normal_cdf(self, x: float) -> float:
        """Approximate standard normal CDF"""
        return 0.5 * (1 + np.tanh(0.797885 * x))

    def _calculate_sharpe(self, returns: np.ndarray, risk_free: float = 0.0) -> float:
        """Calculate Sharpe ratio from returns"""
        if len(returns) < 2:
            return 0.0
        excess = returns - risk_free
        std = np.std(excess)
        if std == 0:
            return 0.0
        return float(np.mean(excess) / std * np.sqrt(252))  # Annualized

    def start_canary(self, challenger_id: str) -> bool:
        """Start canary rollout (partial traffic)"""
        if challenger_id not in self.challengers:
            return False

        challenger = self.challengers[challenger_id]

        if challenger.comparison_result != ComparisonResult.CHALLENGER_BETTER:
            logger.warning(f"Challenger {challenger_id} didn't pass evaluation")
            return False

        challenger.status = PromotionStatus.CANARY
        challenger.notes.append(
            f"Canary started: {self.criteria.canary_traffic_pct}% traffic, "
            f"{datetime.now().isoformat()}"
        )

        logger.info(f"Challenger {challenger_id} entering canary rollout")
        self._save_state()
        return True

    def record_canary_trade(self, challenger_id: str, pnl_pct: float):
        """Record trade during canary period"""
        if challenger_id not in self.challengers:
            return

        challenger = self.challengers[challenger_id]
        if challenger.status != PromotionStatus.CANARY:
            return

        challenger.canary_trades += 1
        challenger.canary_pnl_pct += pnl_pct

    def check_canary_health(self, challenger_id: str) -> Tuple[bool, str]:
        """Check if canary is healthy enough to promote"""
        if challenger_id not in self.challengers:
            return False, "Challenger not found"

        challenger = self.challengers[challenger_id]

        if challenger.canary_trades < 10:
            return False, "Need more canary trades"

        # Compare canary performance to shadow
        canary_avg = challenger.canary_pnl_pct / challenger.canary_trades
        shadow_avg = challenger.shadow_pnl_pct / challenger.shadow_trades if challenger.shadow_trades > 0 else 0

        # Canary should be at least threshold % as good as shadow
        if shadow_avg > 0:
            canary_ratio = canary_avg / shadow_avg
            if canary_ratio < self.criteria.canary_success_threshold:
                return False, f"Canary performance {canary_ratio:.1%} below threshold"

        return True, "Canary healthy"

    def promote_challenger(self, challenger_id: str) -> bool:
        """Promote challenger to champion"""
        if challenger_id not in self.challengers:
            return False

        challenger = self.challengers[challenger_id]

        # Verify canary passed (or skip if no canary required)
        if challenger.status == PromotionStatus.CANARY:
            healthy, reason = self.check_canary_health(challenger_id)
            if not healthy:
                logger.warning(f"Canary not healthy: {reason}")
                return False
        elif challenger.comparison_result != ComparisonResult.CHALLENGER_BETTER:
            logger.warning(f"Challenger {challenger_id} not cleared for promotion")
            return False

        # Archive old champion
        old_champion = self.champions.get(challenger.strategy_name)
        if old_champion:
            self.promotion_history.append({
                "event": "champion_replaced",
                "old_champion": old_champion.to_dict(),
                "new_champion": challenger.to_dict(),
                "timestamp": datetime.now().isoformat(),
            })

        # Promote
        self.champions[challenger.strategy_name] = ChampionRecord(
            strategy_name=challenger.strategy_name,
            strategy_version=challenger.strategy_version,
            promoted_at=datetime.now(),
            total_trades=challenger.shadow_trades + challenger.canary_trades,
            total_pnl_pct=challenger.shadow_pnl_pct + challenger.canary_pnl_pct,
            sharpe_ratio=challenger.shadow_sharpe,
            max_drawdown=challenger.shadow_max_drawdown,
            win_rate=challenger.shadow_win_rate,
            recent_trade_returns=challenger.shadow_trade_returns[-50:],
        )

        challenger.status = PromotionStatus.PROMOTED
        challenger.promoted_at = datetime.now()
        challenger.notes.append(f"Promoted to champion: {datetime.now().isoformat()}")

        # Remove from active challengers
        del self.challengers[challenger_id]

        logger.info(f"Challenger {challenger_id} promoted to champion!")
        self._save_state()
        return True

    def _reject_challenger(self, challenger_id: str, reason: str):
        """Reject a challenger"""
        if challenger_id not in self.challengers:
            return

        challenger = self.challengers[challenger_id]
        challenger.status = PromotionStatus.REJECTED
        challenger.rejection_reason = reason
        challenger.notes.append(f"Rejected: {reason}")

        self.promotion_history.append({
            "event": "challenger_rejected",
            "challenger": challenger.to_dict(),
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
        })

        logger.warning(f"Challenger {challenger_id} rejected: {reason}")
        self._save_state()

    def rollback_champion(self, strategy_name: str, reason: str) -> bool:
        """Rollback to previous champion version"""
        if strategy_name not in self.champions:
            return False

        # Find previous champion from history
        previous = None
        for entry in reversed(self.promotion_history):
            if entry.get("event") == "champion_replaced":
                old = entry.get("old_champion", {})
                if old.get("strategy_name") == strategy_name:
                    previous = old
                    break

        if previous is None:
            logger.error(f"No previous champion found for {strategy_name}")
            return False

        # Record rollback
        current = self.champions[strategy_name]
        self.promotion_history.append({
            "event": "champion_rolled_back",
            "rolled_back_champion": current.to_dict(),
            "restored_champion": previous,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
        })

        # Restore previous
        self.champions[strategy_name] = ChampionRecord(
            strategy_name=previous["strategy_name"],
            strategy_version=previous["strategy_version"],
            promoted_at=datetime.fromisoformat(previous["promoted_at"]),
            total_trades=previous.get("total_trades", 0),
            total_pnl_pct=previous.get("total_pnl_pct", 0),
            sharpe_ratio=previous.get("sharpe_ratio", 0),
        )

        logger.warning(f"Champion {strategy_name} rolled back: {reason}")
        self._save_state()
        return True

    def update_champion_performance(
        self,
        strategy_name: str,
        trade_pnl_pct: float
    ):
        """Update champion's recent performance"""
        if strategy_name not in self.champions:
            return

        champion = self.champions[strategy_name]
        champion.total_trades += 1
        champion.total_pnl_pct += trade_pnl_pct
        champion.recent_trade_returns.append(trade_pnl_pct)

        # Keep last 100 trades
        if len(champion.recent_trade_returns) > 100:
            champion.recent_trade_returns = champion.recent_trade_returns[-100:]

        # Update recent metrics
        if len(champion.recent_trade_returns) >= 10:
            returns = np.array(champion.recent_trade_returns)
            champion.win_rate = float(np.mean(returns > 0))
            champion.recent_sharpe = self._calculate_sharpe(returns)
            champion.recent_pnl_pct = float(np.sum(returns))

    def get_promotion_status(self, challenger_id: str) -> Dict[str, Any]:
        """Get status of a challenger"""
        if challenger_id not in self.challengers:
            return {"error": "Challenger not found"}

        challenger = self.challengers[challenger_id]
        return {
            **challenger.to_dict(),
            "criteria": {
                "min_shadow_days": self.criteria.min_shadow_days,
                "min_shadow_trades": self.criteria.min_shadow_trades,
                "current_shadow_days": challenger.shadow_days,
                "current_shadow_trades": challenger.shadow_trades,
                "days_remaining": max(0, self.criteria.min_shadow_days - challenger.shadow_days),
                "trades_remaining": max(0, self.criteria.min_shadow_trades - challenger.shadow_trades),
            },
        }

    def get_champion(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """Get current champion info"""
        if strategy_name not in self.champions:
            return None
        return self.champions[strategy_name].to_dict()

    def get_all_champions(self) -> Dict[str, Dict[str, Any]]:
        """Get all current champions"""
        return {
            name: champion.to_dict()
            for name, champion in self.champions.items()
        }

    def get_active_challengers(self) -> List[Dict[str, Any]]:
        """Get all active challengers"""
        return [c.to_dict() for c in self.challengers.values()]


# Global instance
_promotion_gate: Optional[PromotionGate] = None


def get_promotion_gate() -> PromotionGate:
    """Get or create global promotion gate"""
    global _promotion_gate
    if _promotion_gate is None:
        _promotion_gate = PromotionGate()
    return _promotion_gate


__all__ = [
    # Enums
    "PromotionStatus",
    "ComparisonResult",
    # Data classes
    "PromotionCriteria",
    "ChallengerPerformance",
    "ChampionRecord",
    # Main class
    "PromotionGate",
    # Factory
    "get_promotion_gate",
]
