"""
Feature Flags API Router.

Provides REST API for managing feature flags:
- List all flags
- Get flag status
- Update flag status
- Evaluate flags for context

Usage:
    from api.feature_flags_router import router as feature_flags_router
    app.include_router(feature_flags_router, tags=["feature-flags"])
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/flags")


# Pydantic models for API
class FlagResponse(BaseModel):
    """Response model for a feature flag."""

    key: str
    name: str
    description: str
    enabled: bool
    flag_type: str
    default_value: Any
    rollout_percentage: float
    variants: List[str]


class FlagUpdateRequest(BaseModel):
    """Request model for updating a flag."""

    enabled: Optional[bool] = None
    rollout_percentage: Optional[float] = Field(None, ge=0, le=100)
    default_value: Optional[Any] = None


class FlagEvaluationRequest(BaseModel):
    """Request model for evaluating flags."""

    context: Dict[str, Any] = Field(default_factory=dict)
    flags: Optional[List[str]] = None  # Specific flags to evaluate, or all


class FlagEvaluationResponse(BaseModel):
    """Response model for flag evaluation."""

    results: Dict[str, Any]
    context_hash: str


def _get_flag_manager():
    """Get the feature flag manager."""
    try:
        from bot.feature_mgmt.feature_flags import get_flag_manager

        return get_flag_manager()
    except ImportError:
        return None


@router.get("", response_model=List[FlagResponse])
async def list_flags():
    """
    List all feature flags.

    Returns:
        List of all defined feature flags with their current status
    """
    manager = _get_flag_manager()
    if manager is None:
        raise HTTPException(
            status_code=503,
            detail="Feature flag system not available",
        )

    flags = []
    for key, flag in manager._flags.items():
        flags.append(
            FlagResponse(
                key=flag.key,
                name=flag.name,
                description=flag.description,
                enabled=flag.enabled,
                flag_type=flag.flag_type.value,
                default_value=flag.default_value,
                rollout_percentage=flag.rollout_percentage,
                variants=list(flag.variants.keys()) if flag.variants else [],
            )
        )

    return flags


@router.get("/{flag_key}", response_model=FlagResponse)
async def get_flag(flag_key: str):
    """
    Get a specific feature flag.

    Args:
        flag_key: The flag key to retrieve

    Returns:
        Flag details
    """
    manager = _get_flag_manager()
    if manager is None:
        raise HTTPException(
            status_code=503,
            detail="Feature flag system not available",
        )

    flag = manager._flags.get(flag_key)
    if flag is None:
        raise HTTPException(
            status_code=404,
            detail=f"Flag '{flag_key}' not found",
        )

    return FlagResponse(
        key=flag.key,
        name=flag.name,
        description=flag.description,
        enabled=flag.enabled,
        flag_type=flag.flag_type.value,
        default_value=flag.default_value,
        rollout_percentage=flag.rollout_percentage,
        variants=list(flag.variants.keys()) if flag.variants else [],
    )


@router.patch("/{flag_key}", response_model=FlagResponse)
async def update_flag(flag_key: str, update: FlagUpdateRequest):
    """
    Update a feature flag.

    Args:
        flag_key: The flag key to update
        update: Update parameters

    Returns:
        Updated flag details
    """
    manager = _get_flag_manager()
    if manager is None:
        raise HTTPException(
            status_code=503,
            detail="Feature flag system not available",
        )

    flag = manager._flags.get(flag_key)
    if flag is None:
        raise HTTPException(
            status_code=404,
            detail=f"Flag '{flag_key}' not found",
        )

    # Update fields
    if update.enabled is not None:
        flag.enabled = update.enabled
        logger.info(f"Flag '{flag_key}' enabled set to {update.enabled}")

    if update.rollout_percentage is not None:
        flag.rollout_percentage = update.rollout_percentage
        logger.info(f"Flag '{flag_key}' rollout set to {update.rollout_percentage}%")

    if update.default_value is not None:
        flag.default_value = update.default_value
        logger.info(f"Flag '{flag_key}' default value updated")

    # Persist changes
    manager._save_flags()

    return FlagResponse(
        key=flag.key,
        name=flag.name,
        description=flag.description,
        enabled=flag.enabled,
        flag_type=flag.flag_type.value,
        default_value=flag.default_value,
        rollout_percentage=flag.rollout_percentage,
        variants=list(flag.variants.keys()) if flag.variants else [],
    )


@router.post("/{flag_key}/enable")
async def enable_flag(flag_key: str):
    """
    Enable a feature flag.

    Args:
        flag_key: The flag key to enable

    Returns:
        Success message
    """
    manager = _get_flag_manager()
    if manager is None:
        raise HTTPException(
            status_code=503,
            detail="Feature flag system not available",
        )

    if flag_key not in manager._flags:
        raise HTTPException(
            status_code=404,
            detail=f"Flag '{flag_key}' not found",
        )

    manager.enable_flag(flag_key)
    return {"message": f"Flag '{flag_key}' enabled", "enabled": True}


@router.post("/{flag_key}/disable")
async def disable_flag(flag_key: str):
    """
    Disable a feature flag.

    Args:
        flag_key: The flag key to disable

    Returns:
        Success message
    """
    manager = _get_flag_manager()
    if manager is None:
        raise HTTPException(
            status_code=503,
            detail="Feature flag system not available",
        )

    if flag_key not in manager._flags:
        raise HTTPException(
            status_code=404,
            detail=f"Flag '{flag_key}' not found",
        )

    manager.disable_flag(flag_key)
    return {"message": f"Flag '{flag_key}' disabled", "enabled": False}


@router.post("/evaluate", response_model=FlagEvaluationResponse)
async def evaluate_flags(request: FlagEvaluationRequest):
    """
    Evaluate feature flags for a given context.

    Args:
        request: Context and optional list of flags to evaluate

    Returns:
        Evaluation results for each flag
    """
    manager = _get_flag_manager()
    if manager is None:
        raise HTTPException(
            status_code=503,
            detail="Feature flag system not available",
        )

    context = request.context
    flags_to_evaluate = request.flags or list(manager._flags.keys())

    results = {}
    for flag_key in flags_to_evaluate:
        if flag_key in manager._flags:
            results[flag_key] = manager.evaluate(flag_key, context)

    # Generate context hash for caching
    import hashlib
    import json
    context_hash = hashlib.md5(
        json.dumps(context, sort_keys=True).encode()
    ).hexdigest()[:8]

    return FlagEvaluationResponse(
        results=results,
        context_hash=context_hash,
    )


@router.get("/evaluate/{flag_key}")
async def evaluate_single_flag(
    flag_key: str,
    user_id: Optional[str] = Query(None),
    environment: Optional[str] = Query(None),
    symbol: Optional[str] = Query(None),
):
    """
    Evaluate a single flag with context from query parameters.

    Args:
        flag_key: Flag to evaluate
        user_id: Optional user ID for context
        environment: Optional environment for context
        symbol: Optional trading symbol for context

    Returns:
        Evaluation result
    """
    manager = _get_flag_manager()
    if manager is None:
        raise HTTPException(
            status_code=503,
            detail="Feature flag system not available",
        )

    if flag_key not in manager._flags:
        raise HTTPException(
            status_code=404,
            detail=f"Flag '{flag_key}' not found",
        )

    # Build context from query params
    context = {}
    if user_id:
        context["user_id"] = user_id
    if environment:
        context["environment"] = environment
    if symbol:
        context["symbol"] = symbol

    result = manager.evaluate(flag_key, context)

    return {
        "flag_key": flag_key,
        "result": result,
        "context": context,
    }


@router.get("/stats")
async def get_flag_stats():
    """
    Get feature flag statistics.

    Returns:
        Statistics about flag usage and evaluation
    """
    manager = _get_flag_manager()
    if manager is None:
        raise HTTPException(
            status_code=503,
            detail="Feature flag system not available",
        )

    total_flags = len(manager._flags)
    enabled_flags = sum(1 for f in manager._flags.values() if f.enabled)
    disabled_flags = total_flags - enabled_flags

    # Group by type
    by_type = {}
    for flag in manager._flags.values():
        flag_type = flag.flag_type.value
        by_type[flag_type] = by_type.get(flag_type, 0) + 1

    return {
        "total_flags": total_flags,
        "enabled_flags": enabled_flags,
        "disabled_flags": disabled_flags,
        "by_type": by_type,
    }
