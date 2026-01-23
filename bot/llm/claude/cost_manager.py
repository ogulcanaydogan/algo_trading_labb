"""
Cost Manager for Claude API usage.

Tracks spending, enforces daily budgets, and provides usage analytics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class UsageRecord:
    """Single API usage record."""

    timestamp: datetime
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost: float
    purpose: str  # e.g., "market_analysis", "trade_explanation"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost": self.cost,
            "purpose": self.purpose,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UsageRecord":
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class UsageStats:
    """Usage statistics."""

    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    requests_today: int = 0
    cost_today: float = 0.0
    budget_remaining_today: float = 0.0
    by_purpose: Dict[str, float] = field(default_factory=dict)


# Pricing per 1M tokens (as of 2024)
MODEL_PRICING = {
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 1.00, "output": 5.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    # Default fallback
    "default": {"input": 3.00, "output": 15.00},
}


class CostManager:
    """
    Manages Claude API costs and budgets.

    Features:
    - Daily budget enforcement
    - Usage tracking and persistence
    - Cost estimation before requests
    - Usage analytics by purpose

    Usage:
        manager = CostManager(daily_budget=5.0)
        if manager.can_spend(estimated_cost=0.01):
            # Make API call
            manager.record_usage(model, input_tokens, output_tokens, "analysis")
    """

    def __init__(
        self,
        daily_budget: float = 5.0,
        data_dir: str = "data/claude_usage",
    ):
        self.daily_budget = daily_budget
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.usage_file = self.data_dir / "usage_history.json"
        self.usage_history: List[UsageRecord] = []

        self._load_history()

    def _load_history(self) -> None:
        """Load usage history from disk."""
        if self.usage_file.exists():
            with open(self.usage_file, "r") as f:
                data = json.load(f)
                self.usage_history = [UsageRecord.from_dict(r) for r in data]

    def _save_history(self) -> None:
        """Save usage history to disk."""
        data = [r.to_dict() for r in self.usage_history]
        with open(self.usage_file, "w") as f:
            json.dump(data, f, indent=2)

    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Calculate cost for a request."""
        pricing = MODEL_PRICING.get(model, MODEL_PRICING["default"])
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    def estimate_cost(
        self,
        model: str,
        input_text: str,
        estimated_output_tokens: int = 500,
    ) -> float:
        """
        Estimate cost before making a request.

        Args:
            model: Model name
            input_text: Input text
            estimated_output_tokens: Expected output tokens

        Returns:
            Estimated cost in USD
        """
        # Rough estimate: ~4 chars per token for English
        estimated_input_tokens = len(input_text) // 4
        return self.calculate_cost(model, estimated_input_tokens, estimated_output_tokens)

    def get_today_usage(self) -> float:
        """Get total spending today."""
        today = date.today()
        return sum(r.cost for r in self.usage_history if r.timestamp.date() == today)

    def get_budget_remaining(self) -> float:
        """Get remaining budget for today."""
        return max(0, self.daily_budget - self.get_today_usage())

    def can_spend(self, estimated_cost: float) -> bool:
        """
        Check if estimated cost is within budget.

        Args:
            estimated_cost: Estimated cost of request

        Returns:
            True if within budget
        """
        remaining = self.get_budget_remaining()
        return estimated_cost <= remaining

    def record_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        purpose: str,
    ) -> UsageRecord:
        """
        Record API usage.

        Args:
            model: Model used
            input_tokens: Input tokens
            output_tokens: Output tokens
            purpose: Purpose of request

        Returns:
            UsageRecord created
        """
        cost = self.calculate_cost(model, input_tokens, output_tokens)

        record = UsageRecord(
            timestamp=datetime.now(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost=cost,
            purpose=purpose,
        )

        self.usage_history.append(record)
        self._save_history()

        return record

    def get_stats(self) -> UsageStats:
        """Get usage statistics."""
        today = date.today()
        today_records = [r for r in self.usage_history if r.timestamp.date() == today]

        by_purpose: Dict[str, float] = {}
        for r in self.usage_history:
            by_purpose[r.purpose] = by_purpose.get(r.purpose, 0) + r.cost

        return UsageStats(
            total_requests=len(self.usage_history),
            total_input_tokens=sum(r.input_tokens for r in self.usage_history),
            total_output_tokens=sum(r.output_tokens for r in self.usage_history),
            total_cost=sum(r.cost for r in self.usage_history),
            requests_today=len(today_records),
            cost_today=sum(r.cost for r in today_records),
            budget_remaining_today=self.get_budget_remaining(),
            by_purpose=by_purpose,
        )

    def get_daily_report(self, target_date: Optional[date] = None) -> Dict[str, Any]:
        """Get daily usage report."""
        target = target_date or date.today()
        records = [r for r in self.usage_history if r.timestamp.date() == target]

        if not records:
            return {
                "date": target.isoformat(),
                "total_requests": 0,
                "total_cost": 0,
                "budget": self.daily_budget,
                "utilization": 0,
            }

        by_model: Dict[str, float] = {}
        by_purpose: Dict[str, float] = {}
        for r in records:
            by_model[r.model] = by_model.get(r.model, 0) + r.cost
            by_purpose[r.purpose] = by_purpose.get(r.purpose, 0) + r.cost

        total_cost = sum(r.cost for r in records)

        return {
            "date": target.isoformat(),
            "total_requests": len(records),
            "total_cost": round(total_cost, 4),
            "budget": self.daily_budget,
            "utilization": round(total_cost / self.daily_budget * 100, 1),
            "by_model": by_model,
            "by_purpose": by_purpose,
        }

    def clear_old_history(self, days_to_keep: int = 30) -> int:
        """
        Clear usage history older than specified days.

        Returns:
            Number of records removed
        """
        cutoff = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff = cutoff.replace(day=cutoff.day - days_to_keep)

        original_count = len(self.usage_history)
        self.usage_history = [r for r in self.usage_history if r.timestamp >= cutoff]
        self._save_history()

        return original_count - len(self.usage_history)
