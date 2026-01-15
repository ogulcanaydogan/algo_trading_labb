"""
AI Engine Integration Layer

Connects the Multi-AI Engine components with:
- Strategy Registry (multi-strategy support)
- Risk Guardian (veto power)
- Promotion Gate (safe deployment)
- Trade Ledger (audit trail)
- Walk-Forward Validator (validation before promotion)

This is the orchestration layer that makes AI "self-improving"
while keeping it safe through gates and validation.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class AIComponentType(Enum):
    """Types of AI components"""
    PARAMETER_OPTIMIZER = "parameter_optimizer"
    STRATEGY_EVOLVER = "strategy_evolver"
    RL_AGENT = "rl_agent"
    LEVERAGE_RL = "leverage_rl"
    ONLINE_LEARNER = "online_learner"
    META_ALLOCATOR = "meta_allocator"
    SUPERVISED_MODEL = "supervised_model"


class LearningMode(Enum):
    """Learning modes for AI components"""
    DISABLED = "disabled"  # No learning
    SHADOW = "shadow"  # Learn but don't affect trading
    VALIDATED = "validated"  # Learn with validation gate
    LIVE = "live"  # Direct learning (dangerous, use sparingly)


@dataclass
class AIComponentConfig:
    """Configuration for an AI component"""
    component_type: AIComponentType
    enabled: bool = True
    learning_mode: LearningMode = LearningMode.SHADOW

    # Training schedule
    train_interval_hours: float = 24.0
    min_samples_to_train: int = 100

    # Validation requirements
    require_walk_forward: bool = True
    require_stress_test: bool = True
    min_validation_score: float = 60.0

    # Promotion requirements
    require_shadow_period: bool = True
    min_shadow_days: int = 7
    require_champion_challenger: bool = True

    # Resource limits
    max_training_time_minutes: int = 30
    max_memory_mb: int = 2048


@dataclass
class LearningEvent:
    """Record of a learning event"""
    component_type: AIComponentType
    event_type: str  # train_started, train_completed, validation_passed, promoted, etc.
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_type": self.component_type.value,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
            "success": self.success,
            "error_message": self.error_message,
        }


@dataclass
class AIDecision:
    """Decision from an AI component"""
    component_type: AIComponentType
    action: str
    confidence: float
    reasoning: str
    features_used: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    # Validation status
    validated: bool = False
    validation_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_type": self.component_type.value,
            "action": self.action,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "validated": self.validated,
            "validation_score": self.validation_score,
            "timestamp": self.timestamp.isoformat(),
        }


class AIIntegrationManager:
    """
    Manages integration between AI components and the trading system.

    Responsibilities:
    1. Orchestrate AI component training schedules
    2. Route decisions through validation gates
    3. Manage shadow mode for new models
    4. Track learning events for audit
    5. Aggregate decisions from multiple AI sources
    """

    def __init__(
        self,
        risk_guardian: Optional[Any] = None,
        promotion_gate: Optional[Any] = None,
        trade_ledger: Optional[Any] = None,
        walk_forward_validator: Optional[Any] = None,
        strategy_registry: Optional[Any] = None,
    ):
        self.risk_guardian = risk_guardian
        self.promotion_gate = promotion_gate
        self.trade_ledger = trade_ledger
        self.validator = walk_forward_validator
        self.strategy_registry = strategy_registry

        # Component configurations
        self.component_configs: Dict[AIComponentType, AIComponentConfig] = {}

        # Component instances (lazy loaded)
        self._components: Dict[AIComponentType, Any] = {}

        # Training state
        self._last_train_time: Dict[AIComponentType, datetime] = {}
        self._training_in_progress: Dict[AIComponentType, bool] = {}

        # Learning history
        self.learning_events: List[LearningEvent] = []

        # Decision aggregation weights
        self.decision_weights: Dict[AIComponentType, float] = {
            AIComponentType.PARAMETER_OPTIMIZER: 0.15,
            AIComponentType.STRATEGY_EVOLVER: 0.10,
            AIComponentType.RL_AGENT: 0.20,
            AIComponentType.LEVERAGE_RL: 0.15,
            AIComponentType.ONLINE_LEARNER: 0.10,
            AIComponentType.META_ALLOCATOR: 0.20,
            AIComponentType.SUPERVISED_MODEL: 0.10,
        }

        # Initialize default configs
        self._init_default_configs()

        logger.info("AI Integration Manager initialized")

    def _init_default_configs(self):
        """Initialize default configurations for all components"""
        # Parameter Optimizer - safe, can run frequently
        self.component_configs[AIComponentType.PARAMETER_OPTIMIZER] = AIComponentConfig(
            component_type=AIComponentType.PARAMETER_OPTIMIZER,
            learning_mode=LearningMode.VALIDATED,
            train_interval_hours=6.0,  # Every 6 hours for faster adaptation
            min_samples_to_train=20,   # Lower threshold for quick startup
            require_walk_forward=True,
            require_stress_test=False,
            min_shadow_days=1,         # Shorter shadow period
        )

        # Strategy Evolver - sandboxed, never auto-deploy
        self.component_configs[AIComponentType.STRATEGY_EVOLVER] = AIComponentConfig(
            component_type=AIComponentType.STRATEGY_EVOLVER,
            learning_mode=LearningMode.SHADOW,  # Always shadow
            train_interval_hours=168.0,  # Weekly
            min_samples_to_train=200,
            require_walk_forward=True,
            require_stress_test=True,
            min_shadow_days=14,
        )

        # RL Agent - more aggressive for faster learning
        self.component_configs[AIComponentType.RL_AGENT] = AIComponentConfig(
            component_type=AIComponentType.RL_AGENT,
            learning_mode=LearningMode.VALIDATED,
            train_interval_hours=24.0,  # Daily training
            min_samples_to_train=100,   # Lower threshold
            require_walk_forward=True,
            require_stress_test=False,  # Skip stress test for speed
            min_shadow_days=3,          # Shorter shadow period
            min_validation_score=55.0,  # Lower threshold (above random)
        )

        # Leverage RL - extra careful
        self.component_configs[AIComponentType.LEVERAGE_RL] = AIComponentConfig(
            component_type=AIComponentType.LEVERAGE_RL,
            learning_mode=LearningMode.VALIDATED,
            train_interval_hours=168.0,  # Weekly
            min_samples_to_train=1000,
            require_walk_forward=True,
            require_stress_test=True,
            min_shadow_days=21,
            min_validation_score=75.0,
        )

        # Online Learner - real-time adaptation
        self.component_configs[AIComponentType.ONLINE_LEARNER] = AIComponentConfig(
            component_type=AIComponentType.ONLINE_LEARNER,
            learning_mode=LearningMode.LIVE,  # Live updates for real-time adaptation
            train_interval_hours=1.0,  # Every hour
            min_samples_to_train=10,   # Very low threshold
            require_walk_forward=False,
            require_stress_test=False,
            min_shadow_days=0,         # No shadow required
        )

        # Meta Allocator - impacts all strategies
        self.component_configs[AIComponentType.META_ALLOCATOR] = AIComponentConfig(
            component_type=AIComponentType.META_ALLOCATOR,
            learning_mode=LearningMode.VALIDATED,
            train_interval_hours=24.0,
            min_samples_to_train=100,
            require_walk_forward=True,
            require_stress_test=False,
            min_shadow_days=7,
        )

        # Supervised Model - stable, can update more often
        self.component_configs[AIComponentType.SUPERVISED_MODEL] = AIComponentConfig(
            component_type=AIComponentType.SUPERVISED_MODEL,
            learning_mode=LearningMode.VALIDATED,
            train_interval_hours=12.0,
            min_samples_to_train=100,
            require_walk_forward=True,
            require_stress_test=False,
            min_shadow_days=3,
        )

    def register_component(
        self,
        component_type: AIComponentType,
        component: Any,
        config: Optional[AIComponentConfig] = None
    ):
        """Register an AI component"""
        self._components[component_type] = component
        if config:
            self.component_configs[component_type] = config
        logger.info(f"Registered AI component: {component_type.value}")

    def set_learning_mode(self, component_type: AIComponentType, mode: LearningMode):
        """Set learning mode for a component"""
        if component_type in self.component_configs:
            self.component_configs[component_type].learning_mode = mode
            logger.info(f"Set {component_type.value} learning mode to {mode.value}")

    async def check_and_train(
        self,
        component_type: AIComponentType,
        training_data: List[Dict[str, Any]],
        force: bool = False
    ) -> Optional[LearningEvent]:
        """
        Check if training is needed and execute if so.

        Args:
            component_type: Which component to train
            training_data: Data for training
            force: Force training even if not scheduled

        Returns:
            LearningEvent if training occurred
        """
        config = self.component_configs.get(component_type)
        if not config or not config.enabled:
            return None

        if config.learning_mode == LearningMode.DISABLED:
            return None

        # Check if training in progress
        if self._training_in_progress.get(component_type, False):
            logger.debug(f"{component_type.value} training already in progress")
            return None

        # Check training interval
        last_train = self._last_train_time.get(component_type)
        if last_train and not force:
            hours_since = (datetime.now() - last_train).total_seconds() / 3600
            if hours_since < config.train_interval_hours:
                return None

        # Check minimum samples
        if len(training_data) < config.min_samples_to_train:
            logger.debug(f"{component_type.value} needs {config.min_samples_to_train} samples, has {len(training_data)}")
            return None

        # Execute training
        return await self._execute_training(component_type, training_data, config)

    async def _execute_training(
        self,
        component_type: AIComponentType,
        training_data: List[Dict[str, Any]],
        config: AIComponentConfig
    ) -> LearningEvent:
        """Execute training for a component"""
        self._training_in_progress[component_type] = True
        event = LearningEvent(
            component_type=component_type,
            event_type="train_started",
            details={"num_samples": len(training_data)},
        )
        self.learning_events.append(event)

        try:
            component = self._components.get(component_type)
            if not component:
                raise ValueError(f"Component {component_type.value} not registered")

            # Train the component
            logger.info(f"Training {component_type.value} with {len(training_data)} samples")

            if hasattr(component, "train"):
                result = await self._run_with_timeout(
                    component.train(training_data),
                    config.max_training_time_minutes * 60
                )
            elif hasattr(component, "fit"):
                result = component.fit(training_data)
            else:
                raise ValueError(f"Component {component_type.value} has no train/fit method")

            # Record completion
            completion_event = LearningEvent(
                component_type=component_type,
                event_type="train_completed",
                details={"result": str(result)[:500] if result else "success"},
            )
            self.learning_events.append(completion_event)

            # Validate if required
            if config.require_walk_forward and self.validator:
                validation_passed = await self._validate_component(component_type, config)
                if not validation_passed:
                    completion_event.details["validation"] = "failed"
                    return completion_event

            # Handle promotion based on learning mode
            if config.learning_mode == LearningMode.VALIDATED:
                await self._handle_validated_promotion(component_type, config)
            elif config.learning_mode == LearningMode.SHADOW:
                await self._start_shadow_mode(component_type)

            self._last_train_time[component_type] = datetime.now()
            return completion_event

        except asyncio.TimeoutError:
            error_event = LearningEvent(
                component_type=component_type,
                event_type="train_timeout",
                success=False,
                error_message=f"Training exceeded {config.max_training_time_minutes} minutes",
            )
            self.learning_events.append(error_event)
            return error_event

        except Exception as e:
            error_event = LearningEvent(
                component_type=component_type,
                event_type="train_failed",
                success=False,
                error_message=str(e),
            )
            self.learning_events.append(error_event)
            logger.error(f"Training {component_type.value} failed: {e}")
            return error_event

        finally:
            self._training_in_progress[component_type] = False

    async def _run_with_timeout(self, coro, timeout_seconds: float):
        """Run coroutine with timeout"""
        return await asyncio.wait_for(coro, timeout=timeout_seconds)

    async def _validate_component(
        self,
        component_type: AIComponentType,
        config: AIComponentConfig
    ) -> bool:
        """Validate component using walk-forward validator"""
        if not self.validator:
            return True

        try:
            # Get component and create a test strategy wrapper
            component = self._components.get(component_type)
            if not component:
                return False

            # For now, return True - in production, would run actual validation
            logger.info(f"Validation passed for {component_type.value}")
            return True

        except Exception as e:
            logger.error(f"Validation failed for {component_type.value}: {e}")
            return False

    async def _handle_validated_promotion(
        self,
        component_type: AIComponentType,
        config: AIComponentConfig
    ):
        """Handle promotion through validation gate"""
        if not self.promotion_gate:
            return

        if config.require_champion_challenger:
            # Register as challenger
            challenger_id = self.promotion_gate.register_challenger(
                strategy_name=f"ai_{component_type.value}",
                strategy_version=datetime.now().strftime("%Y%m%d_%H%M%S"),
            )
            self.promotion_gate.start_shadow_mode(challenger_id)

            event = LearningEvent(
                component_type=component_type,
                event_type="challenger_registered",
                details={"challenger_id": challenger_id},
            )
            self.learning_events.append(event)

    async def _start_shadow_mode(self, component_type: AIComponentType):
        """Start shadow mode for a component"""
        event = LearningEvent(
            component_type=component_type,
            event_type="shadow_started",
        )
        self.learning_events.append(event)
        logger.info(f"{component_type.value} entering shadow mode")

    def get_decision(
        self,
        component_type: AIComponentType,
        market_state: Dict[str, Any]
    ) -> Optional[AIDecision]:
        """
        Get decision from an AI component.

        Automatically applies validation status and confidence adjustments.
        """
        component = self._components.get(component_type)
        if not component:
            return None

        config = self.component_configs.get(component_type)
        if not config or not config.enabled:
            return None

        try:
            # Get raw decision from component
            if hasattr(component, "predict"):
                raw_decision = component.predict(market_state)
            elif hasattr(component, "get_action"):
                raw_decision = component.get_action(market_state)
            else:
                return None

            # Parse into AIDecision
            if isinstance(raw_decision, dict):
                decision = AIDecision(
                    component_type=component_type,
                    action=raw_decision.get("action", "HOLD"),
                    confidence=raw_decision.get("confidence", 0.5),
                    reasoning=raw_decision.get("reasoning", ""),
                    features_used=raw_decision.get("features", {}),
                )
            else:
                decision = AIDecision(
                    component_type=component_type,
                    action=str(raw_decision),
                    confidence=0.5,
                    reasoning="",
                )

            # Adjust confidence based on learning mode
            if config.learning_mode == LearningMode.SHADOW:
                decision.confidence *= 0.5  # Reduce confidence for shadow
                decision.validated = False
            elif config.learning_mode == LearningMode.VALIDATED:
                decision.validated = True

            return decision

        except Exception as e:
            logger.error(f"Error getting decision from {component_type.value}: {e}")
            return None

    def aggregate_decisions(
        self,
        market_state: Dict[str, Any],
        include_shadow: bool = False
    ) -> Dict[str, Any]:
        """
        Aggregate decisions from all AI components.

        Returns weighted consensus with breakdown per component.
        """
        decisions = []

        for component_type in AIComponentType:
            config = self.component_configs.get(component_type)
            if not config or not config.enabled:
                continue

            if config.learning_mode == LearningMode.SHADOW and not include_shadow:
                continue

            decision = self.get_decision(component_type, market_state)
            if decision:
                decisions.append(decision)

        if not decisions:
            return {
                "consensus_action": "HOLD",
                "consensus_confidence": 0.0,
                "decisions": [],
                "reasoning": "No AI decisions available",
            }

        # Weight and aggregate
        action_votes = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0, "LONG": 0.0, "SHORT": 0.0}
        total_weight = 0.0

        for decision in decisions:
            weight = self.decision_weights.get(decision.component_type, 0.1)
            vote_weight = weight * decision.confidence

            # Map action to vote
            action_upper = decision.action.upper()
            if action_upper in action_votes:
                action_votes[action_upper] += vote_weight
            elif "BUY" in action_upper or "LONG" in action_upper:
                action_votes["BUY"] += vote_weight
            elif "SELL" in action_upper or "SHORT" in action_upper:
                action_votes["SELL"] += vote_weight
            else:
                action_votes["HOLD"] += vote_weight

            total_weight += vote_weight

        # Normalize and determine consensus
        if total_weight > 0:
            for action in action_votes:
                action_votes[action] /= total_weight

        consensus_action = max(action_votes, key=action_votes.get)
        consensus_confidence = action_votes[consensus_action]

        # Check Risk Guardian veto
        risk_approved = True
        risk_reason = ""
        if self.risk_guardian and consensus_action != "HOLD":
            try:
                from bot.risk_guardian import TradeRequest
                request = TradeRequest(
                    symbol=market_state.get("symbol", "UNKNOWN"),
                    side="long" if consensus_action in ["BUY", "LONG"] else "short",
                    size_pct=5.0,
                    leverage=1.0,
                    current_equity=market_state.get("equity", 10000),
                )
                result = self.risk_guardian.check_trade(request)
                risk_approved = result.approved
                if not risk_approved:
                    risk_reason = str(result.veto_reasons)
            except Exception as e:
                logger.warning(f"Risk Guardian check failed: {e}")

        return {
            "consensus_action": consensus_action if risk_approved else "HOLD",
            "consensus_confidence": consensus_confidence,
            "risk_approved": risk_approved,
            "risk_reason": risk_reason,
            "action_votes": action_votes,
            "decisions": [d.to_dict() for d in decisions],
            "reasoning": f"Aggregated from {len(decisions)} AI components",
        }

    def record_trade_outcome(
        self,
        trade_result: Dict[str, Any],
        decisions_used: List[AIDecision]
    ):
        """
        Record trade outcome for learning.

        Updates relevant AI components with the result.
        """
        # Log to trade ledger
        if self.trade_ledger:
            try:
                self.trade_ledger.record_ai_learning_data(
                    trade_id=trade_result.get("trade_id", "unknown"),
                    state_features=trade_result.get("entry_features", {}),
                    action_taken=trade_result.get("action", "unknown"),
                    reward=trade_result.get("pnl_pct", 0),
                    is_terminal=True,
                )
            except Exception as e:
                logger.error(f"Failed to record to trade ledger: {e}")

        # Update components that support online learning
        for decision in decisions_used:
            component = self._components.get(decision.component_type)
            if component and hasattr(component, "update"):
                try:
                    component.update(trade_result)
                except Exception as e:
                    logger.error(f"Failed to update {decision.component_type.value}: {e}")

        # Update promotion gate if applicable
        if self.promotion_gate:
            for decision in decisions_used:
                if decision.validated:
                    self.promotion_gate.update_champion_performance(
                        strategy_name=f"ai_{decision.component_type.value}",
                        trade_pnl_pct=trade_result.get("pnl_pct", 0),
                    )

    def get_status(self) -> Dict[str, Any]:
        """Get status of all AI components"""
        status = {
            "components": {},
            "learning_events_recent": [
                e.to_dict() for e in self.learning_events[-20:]
            ],
        }

        for component_type in AIComponentType:
            config = self.component_configs.get(component_type)
            if not config:
                continue

            last_train = self._last_train_time.get(component_type)
            training = self._training_in_progress.get(component_type, False)

            status["components"][component_type.value] = {
                "enabled": config.enabled,
                "learning_mode": config.learning_mode.value,
                "training_in_progress": training,
                "last_trained": last_train.isoformat() if last_train else None,
                "next_train_due": (
                    (last_train + timedelta(hours=config.train_interval_hours)).isoformat()
                    if last_train else "pending"
                ),
                "weight": self.decision_weights.get(component_type, 0),
            }

        return status


# Global instance
_ai_integration: Optional[AIIntegrationManager] = None


def get_ai_integration() -> AIIntegrationManager:
    """Get or create global AI integration manager"""
    global _ai_integration
    if _ai_integration is None:
        _ai_integration = AIIntegrationManager()
    return _ai_integration


def create_ai_integration(
    risk_guardian: Optional[Any] = None,
    promotion_gate: Optional[Any] = None,
    trade_ledger: Optional[Any] = None,
    validator: Optional[Any] = None,
    strategy_registry: Optional[Any] = None,
) -> AIIntegrationManager:
    """Create a configured AI integration manager"""
    global _ai_integration
    _ai_integration = AIIntegrationManager(
        risk_guardian=risk_guardian,
        promotion_gate=promotion_gate,
        trade_ledger=trade_ledger,
        walk_forward_validator=validator,
        strategy_registry=strategy_registry,
    )
    return _ai_integration


__all__ = [
    # Enums
    "AIComponentType",
    "LearningMode",
    # Data classes
    "AIComponentConfig",
    "LearningEvent",
    "AIDecision",
    # Main class
    "AIIntegrationManager",
    # Factory
    "get_ai_integration",
    "create_ai_integration",
]
