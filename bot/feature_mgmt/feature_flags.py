"""
Feature Flags System for Runtime Feature Control.

Provides dynamic feature toggling, gradual rollouts,
and targeting rules for trading system components.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)


class FlagType(Enum):
    """Feature flag types."""

    BOOLEAN = "boolean"
    PERCENTAGE = "percentage"  # Gradual rollout
    VARIANT = "variant"  # Multiple variants
    JSON = "json"  # Complex config


class RolloutStrategy(Enum):
    """Rollout strategy for gradual deployment."""

    ALL = "all"
    NONE = "none"
    PERCENTAGE = "percentage"
    USER_LIST = "user_list"
    ENVIRONMENT = "environment"


@dataclass
class TargetingRule:
    """Rule for targeting specific users/entities."""

    attribute: str  # e.g., "user_id", "environment", "symbol"
    operator: str  # "eq", "neq", "in", "not_in", "contains", "gt", "lt"
    value: Any

    def matches(self, context: Dict[str, Any]) -> bool:
        """Check if rule matches context."""
        actual = context.get(self.attribute)
        if actual is None:
            return False

        if self.operator == "eq":
            return actual == self.value
        elif self.operator == "neq":
            return actual != self.value
        elif self.operator == "in":
            return actual in self.value
        elif self.operator == "not_in":
            return actual not in self.value
        elif self.operator == "contains":
            return self.value in str(actual)
        elif self.operator == "gt":
            return actual > self.value
        elif self.operator == "lt":
            return actual < self.value
        elif self.operator == "gte":
            return actual >= self.value
        elif self.operator == "lte":
            return actual <= self.value
        else:
            return False

    def to_dict(self) -> Dict:
        return {
            "attribute": self.attribute,
            "operator": self.operator,
            "value": self.value,
        }


@dataclass
class FeatureFlag:
    """A feature flag definition."""

    key: str
    name: str
    description: str = ""
    flag_type: FlagType = FlagType.BOOLEAN
    enabled: bool = False
    default_value: Any = False

    # Rollout configuration
    rollout_strategy: RolloutStrategy = RolloutStrategy.ALL
    rollout_percentage: float = 100.0  # 0-100
    allowed_users: Set[str] = field(default_factory=set)
    allowed_environments: Set[str] = field(default_factory=set)

    # Targeting rules
    targeting_rules: List[TargetingRule] = field(default_factory=list)

    # Variants (for variant type)
    variants: Dict[str, Any] = field(default_factory=dict)
    variant_weights: Dict[str, float] = field(default_factory=dict)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    owner: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "key": self.key,
            "name": self.name,
            "description": self.description,
            "flag_type": self.flag_type.value,
            "enabled": self.enabled,
            "default_value": self.default_value,
            "rollout_strategy": self.rollout_strategy.value,
            "rollout_percentage": self.rollout_percentage,
            "allowed_users": list(self.allowed_users),
            "allowed_environments": list(self.allowed_environments),
            "targeting_rules": [r.to_dict() for r in self.targeting_rules],
            "variants": self.variants,
            "variant_weights": self.variant_weights,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "owner": self.owner,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "FeatureFlag":
        """Create flag from dictionary."""
        rules = [TargetingRule(**r) for r in data.get("targeting_rules", [])]

        return cls(
            key=data["key"],
            name=data["name"],
            description=data.get("description", ""),
            flag_type=FlagType(data.get("flag_type", "boolean")),
            enabled=data.get("enabled", False),
            default_value=data.get("default_value", False),
            rollout_strategy=RolloutStrategy(data.get("rollout_strategy", "all")),
            rollout_percentage=data.get("rollout_percentage", 100.0),
            allowed_users=set(data.get("allowed_users", [])),
            allowed_environments=set(data.get("allowed_environments", [])),
            targeting_rules=rules,
            variants=data.get("variants", {}),
            variant_weights=data.get("variant_weights", {}),
            owner=data.get("owner", ""),
            tags=data.get("tags", []),
        )


@dataclass
class EvaluationResult:
    """Result of flag evaluation."""

    flag_key: str
    enabled: bool
    value: Any
    variant: Optional[str] = None
    reason: str = ""
    evaluation_time_ms: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "flag_key": self.flag_key,
            "enabled": self.enabled,
            "value": self.value,
            "variant": self.variant,
            "reason": self.reason,
            "evaluation_time_ms": round(self.evaluation_time_ms, 3),
        }


class FeatureFlagManager:
    """
    Feature Flag Management System.

    Features:
    - Boolean, percentage, and variant flags
    - Targeting rules for fine-grained control
    - Gradual rollout support
    - File-based and in-memory storage
    - Thread-safe operations
    """

    def __init__(self, storage_path: Optional[str] = None, environment: str = "development"):
        self._flags: Dict[str, FeatureFlag] = {}
        self._storage_path = storage_path
        self._environment = environment
        self._lock = threading.RLock()
        self._listeners: List[Callable[[str, bool], None]] = []
        self._evaluation_cache: Dict[str, EvaluationResult] = {}
        self._cache_ttl = 60  # seconds

        if storage_path:
            self._load_from_file()

    def _load_from_file(self):
        """Load flags from storage file."""
        if not self._storage_path:
            return

        path = Path(self._storage_path)
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                    for flag_data in data.get("flags", []):
                        flag = FeatureFlag.from_dict(flag_data)
                        self._flags[flag.key] = flag
                logger.info(f"Loaded {len(self._flags)} feature flags from {path}")
            except Exception as e:
                logger.error(f"Failed to load feature flags: {e}")

    def _save_to_file(self):
        """Save flags to storage file."""
        if not self._storage_path:
            return

        try:
            path = Path(self._storage_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "flags": [f.to_dict() for f in self._flags.values()],
                "updated_at": datetime.now().isoformat(),
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save feature flags: {e}")

    def create_flag(
        self,
        key: str,
        name: str,
        description: str = "",
        flag_type: FlagType = FlagType.BOOLEAN,
        default_value: Any = False,
        enabled: bool = False,
        **kwargs,
    ) -> FeatureFlag:
        """Create a new feature flag."""
        with self._lock:
            if key in self._flags:
                raise ValueError(f"Flag already exists: {key}")

            flag = FeatureFlag(
                key=key,
                name=name,
                description=description,
                flag_type=flag_type,
                default_value=default_value,
                enabled=enabled,
                **kwargs,
            )
            self._flags[key] = flag
            self._save_to_file()
            logger.info(f"Created feature flag: {key}")
            return flag

    def update_flag(self, key: str, **updates) -> FeatureFlag:
        """Update an existing feature flag."""
        with self._lock:
            if key not in self._flags:
                raise ValueError(f"Flag not found: {key}")

            flag = self._flags[key]
            for attr, value in updates.items():
                if hasattr(flag, attr):
                    setattr(flag, attr, value)
            flag.updated_at = datetime.now()

            # Clear cache for this flag
            self._clear_cache(key)

            self._save_to_file()
            self._notify_listeners(key, flag.enabled)
            logger.info(f"Updated feature flag: {key}")
            return flag

    def delete_flag(self, key: str):
        """Delete a feature flag."""
        with self._lock:
            if key not in self._flags:
                raise ValueError(f"Flag not found: {key}")

            del self._flags[key]
            self._clear_cache(key)
            self._save_to_file()
            logger.info(f"Deleted feature flag: {key}")

    def get_flag(self, key: str) -> Optional[FeatureFlag]:
        """Get a feature flag by key."""
        return self._flags.get(key)

    def list_flags(
        self, tag: Optional[str] = None, enabled_only: bool = False
    ) -> List[FeatureFlag]:
        """List all flags, optionally filtered."""
        flags = list(self._flags.values())

        if tag:
            flags = [f for f in flags if tag in f.tags]

        if enabled_only:
            flags = [f for f in flags if f.enabled]

        return flags

    def is_enabled(
        self, key: str, context: Optional[Dict[str, Any]] = None, default: bool = False
    ) -> bool:
        """
        Check if a feature flag is enabled.

        Args:
            key: Flag key
            context: Evaluation context (user_id, environment, etc.)
            default: Default value if flag not found

        Returns:
            True if feature is enabled for this context
        """
        result = self.evaluate(key, context)
        if result is None:
            return default
        return result.enabled

    def evaluate(
        self, key: str, context: Optional[Dict[str, Any]] = None
    ) -> Optional[EvaluationResult]:
        """
        Evaluate a feature flag with full context.

        Args:
            key: Flag key
            context: Evaluation context

        Returns:
            EvaluationResult with enabled status and value
        """
        start_time = time.perf_counter()
        context = context or {}

        # Add environment to context
        context.setdefault("environment", self._environment)

        # Check cache
        cache_key = self._get_cache_key(key, context)
        cached = self._evaluation_cache.get(cache_key)
        if cached:
            return cached

        flag = self._flags.get(key)
        if not flag:
            return None

        # Evaluate
        enabled = False
        value = flag.default_value
        variant = None
        reason = "default"

        if not flag.enabled:
            reason = "flag_disabled"
        else:
            # Check environment
            if (
                flag.allowed_environments
                and context.get("environment") not in flag.allowed_environments
            ):
                reason = "environment_not_allowed"
            # Check user allowlist
            elif flag.rollout_strategy == RolloutStrategy.USER_LIST:
                user_id = context.get("user_id")
                if user_id and user_id in flag.allowed_users:
                    enabled = True
                    reason = "user_allowlist"
            # Check targeting rules
            elif flag.targeting_rules:
                all_match = all(r.matches(context) for r in flag.targeting_rules)
                if all_match:
                    enabled = True
                    reason = "targeting_rules"
            # Check percentage rollout
            elif flag.rollout_strategy == RolloutStrategy.PERCENTAGE:
                enabled = self._check_percentage(key, context, flag.rollout_percentage)
                reason = "percentage_rollout"
            # Default: all
            elif flag.rollout_strategy == RolloutStrategy.ALL:
                enabled = True
                reason = "all"

        # Get value based on type
        if enabled:
            if flag.flag_type == FlagType.BOOLEAN:
                value = True
            elif flag.flag_type == FlagType.VARIANT:
                variant = self._select_variant(key, context, flag)
                value = flag.variants.get(variant, flag.default_value)
            elif flag.flag_type == FlagType.JSON:
                value = flag.default_value
            elif flag.flag_type == FlagType.PERCENTAGE:
                value = flag.rollout_percentage

        eval_time = (time.perf_counter() - start_time) * 1000

        result = EvaluationResult(
            flag_key=key,
            enabled=enabled,
            value=value,
            variant=variant,
            reason=reason,
            evaluation_time_ms=eval_time,
        )

        # Cache result
        self._evaluation_cache[cache_key] = result

        return result

    def _check_percentage(self, key: str, context: Dict[str, Any], percentage: float) -> bool:
        """Check if entity falls within percentage rollout."""
        # Use user_id or generate from context
        entity_id = context.get("user_id") or json.dumps(context, sort_keys=True)
        hash_input = f"{key}:{entity_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        bucket = (hash_value % 10000) / 100  # 0-100

        return bucket < percentage

    def _select_variant(self, key: str, context: Dict[str, Any], flag: FeatureFlag) -> str:
        """Select a variant based on weights."""
        if not flag.variants:
            return ""

        entity_id = context.get("user_id") or json.dumps(context, sort_keys=True)
        hash_input = f"{key}:variant:{entity_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        bucket = (hash_value % 10000) / 10000  # 0-1

        # Normalize weights
        total_weight = sum(flag.variant_weights.values())
        if total_weight == 0:
            # Equal weights
            variants = list(flag.variants.keys())
            idx = hash_value % len(variants)
            return variants[idx]

        cumulative = 0.0
        for variant, weight in flag.variant_weights.items():
            cumulative += weight / total_weight
            if bucket < cumulative:
                return variant

        return list(flag.variants.keys())[-1]

    def _get_cache_key(self, key: str, context: Dict[str, Any]) -> str:
        """Generate cache key."""
        context_str = json.dumps(context, sort_keys=True)
        return f"{key}:{hashlib.md5(context_str.encode()).hexdigest()[:8]}"

    def _clear_cache(self, key: Optional[str] = None):
        """Clear evaluation cache."""
        if key:
            self._evaluation_cache = {
                k: v for k, v in self._evaluation_cache.items() if not k.startswith(f"{key}:")
            }
        else:
            self._evaluation_cache.clear()

    def add_listener(self, callback: Callable[[str, bool], None]):
        """Add a listener for flag changes."""
        self._listeners.append(callback)

    def _notify_listeners(self, key: str, enabled: bool):
        """Notify listeners of flag change."""
        for listener in self._listeners:
            try:
                listener(key, enabled)
            except Exception as e:
                logger.error(f"Listener error: {e}")

    def enable(self, key: str):
        """Enable a flag."""
        self.update_flag(key, enabled=True)

    def disable(self, key: str):
        """Disable a flag."""
        self.update_flag(key, enabled=False)

    def set_percentage(self, key: str, percentage: float):
        """Set rollout percentage."""
        self.update_flag(
            key, rollout_strategy=RolloutStrategy.PERCENTAGE, rollout_percentage=percentage
        )

    def add_to_allowlist(self, key: str, user_id: str):
        """Add user to flag allowlist."""
        flag = self._flags.get(key)
        if flag:
            flag.allowed_users.add(user_id)
            self._save_to_file()

    def remove_from_allowlist(self, key: str, user_id: str):
        """Remove user from flag allowlist."""
        flag = self._flags.get(key)
        if flag:
            flag.allowed_users.discard(user_id)
            self._save_to_file()


# Trading-specific feature flags
class TradingFeatureFlags:
    """
    Pre-defined feature flags for trading system.

    Common flags:
    - Strategy toggles
    - Risk limits
    - Exchange features
    - ML model versions
    """

    # Flag keys
    LIVE_TRADING = "trading.live_enabled"
    PAPER_TRADING = "trading.paper_enabled"
    ML_SIGNALS = "ml.signals_enabled"
    ML_MODEL_V2 = "ml.model_v2"
    RISK_LIMITS_STRICT = "risk.strict_limits"
    ADVANCED_ORDER_TYPES = "execution.advanced_orders"
    WEBSOCKET_STREAMING = "data.websocket_enabled"
    MULTI_EXCHANGE = "exchange.multi_enabled"

    def __init__(self, manager: Optional[FeatureFlagManager] = None):
        self.manager = manager or FeatureFlagManager()
        self._initialize_defaults()

    def _initialize_defaults(self):
        """Initialize default trading flags."""
        defaults = [
            (self.LIVE_TRADING, "Live Trading", "Enable live trading execution", False),
            (self.PAPER_TRADING, "Paper Trading", "Enable paper trading mode", True),
            (self.ML_SIGNALS, "ML Signals", "Enable ML-based trading signals", True),
            (self.ML_MODEL_V2, "ML Model V2", "Use v2 ML model for predictions", False),
            (self.RISK_LIMITS_STRICT, "Strict Risk Limits", "Enable strict risk limits", True),
            (self.ADVANCED_ORDER_TYPES, "Advanced Orders", "Enable iceberg/TWAP orders", False),
            (self.WEBSOCKET_STREAMING, "WebSocket Data", "Enable WebSocket data streaming", True),
            (self.MULTI_EXCHANGE, "Multi-Exchange", "Enable multi-exchange trading", False),
        ]

        for key, name, desc, enabled in defaults:
            if not self.manager.get_flag(key):
                try:
                    self.manager.create_flag(
                        key=key,
                        name=name,
                        description=desc,
                        enabled=enabled,
                        tags=["trading", "core"],
                    )
                except ValueError:
                    pass  # Flag already exists

    def is_live_trading_enabled(self, context: Optional[Dict] = None) -> bool:
        return self.manager.is_enabled(self.LIVE_TRADING, context)

    def is_ml_enabled(self, context: Optional[Dict] = None) -> bool:
        return self.manager.is_enabled(self.ML_SIGNALS, context)

    def get_ml_model_version(self, context: Optional[Dict] = None) -> str:
        if self.manager.is_enabled(self.ML_MODEL_V2, context):
            return "v2"
        return "v1"


def create_feature_flag_manager(
    storage_path: Optional[str] = None, environment: Optional[str] = None
) -> FeatureFlagManager:
    """Factory function to create feature flag manager."""
    env = environment or os.getenv("ENVIRONMENT", "development")
    return FeatureFlagManager(storage_path=storage_path, environment=env)


def create_trading_flags(manager: Optional[FeatureFlagManager] = None) -> TradingFeatureFlags:
    """Factory function to create trading feature flags."""
    return TradingFeatureFlags(manager)
