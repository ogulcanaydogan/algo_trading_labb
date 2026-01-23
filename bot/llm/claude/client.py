"""
Claude API Client.

Provides a wrapper around the Anthropic API with cost management,
retry logic, and trading-specific utilities.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from .cost_manager import CostManager

try:
    import anthropic

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


@dataclass
class ClaudeResponse:
    """Response from Claude API."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    stop_reason: str


class ClaudeClient:
    """
    Client for Claude API with cost management.

    Features:
    - Automatic cost tracking
    - Budget enforcement
    - Retry with exponential backoff
    - Model selection (haiku for cheap, sonnet/opus for quality)

    Usage:
        client = ClaudeClient(daily_budget=5.0)
        response = client.complete("Analyze this market data...")
    """

    # Model options
    MODELS = {
        "haiku": "claude-3-haiku-20240307",
        "sonnet": "claude-3-5-sonnet-20241022",
        "opus": "claude-3-opus-20240229",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        daily_budget: float = 5.0,
        default_model: str = "sonnet",
        data_dir: str = "data/claude_usage",
    ):
        if not HAS_ANTHROPIC:
            raise ImportError(
                "anthropic library not installed. Install with: pip install anthropic"
            )

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No API key provided. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.default_model = self.MODELS.get(default_model, default_model)
        self.cost_manager = CostManager(daily_budget=daily_budget, data_dir=data_dir)

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        purpose: str = "general",
        force: bool = False,
    ) -> Optional[ClaudeResponse]:
        """
        Get a completion from Claude.

        Args:
            prompt: User prompt
            system: System prompt
            model: Model to use (haiku/sonnet/opus or full name)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            purpose: Purpose for cost tracking
            force: If True, ignore budget limits

        Returns:
            ClaudeResponse or None if budget exceeded
        """
        model_name = self.MODELS.get(model, model) if model else self.default_model

        # Estimate cost
        estimated_cost = self.cost_manager.estimate_cost(
            model_name,
            prompt + (system or ""),
            max_tokens,
        )

        if not force and not self.cost_manager.can_spend(estimated_cost):
            remaining = self.cost_manager.get_budget_remaining()
            print(f"Budget exceeded. Remaining: ${remaining:.4f}, Estimated: ${estimated_cost:.4f}")
            return None

        # Build messages
        messages = [{"role": "user", "content": prompt}]

        # Make request with retries
        response = self._make_request_with_retry(
            model=model_name,
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        if response is None:
            return None

        # Record usage
        self.cost_manager.record_usage(
            model=model_name,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            purpose=purpose,
        )

        cost = self.cost_manager.calculate_cost(
            model_name,
            response.usage.input_tokens,
            response.usage.output_tokens,
        )

        return ClaudeResponse(
            content=response.content[0].text,
            model=model_name,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost=cost,
            stop_reason=response.stop_reason,
        )

    def _make_request_with_retry(
        self,
        model: str,
        messages: List[Dict[str, str]],
        system: Optional[str],
        max_tokens: int,
        temperature: float,
        max_retries: int = 3,
    ) -> Optional[Any]:
        """Make API request with exponential backoff retry."""
        for attempt in range(max_retries):
            try:
                kwargs = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                if system:
                    kwargs["system"] = system

                return self.client.messages.create(**kwargs)

            except anthropic.RateLimitError:
                wait_time = 2**attempt
                print(f"Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)

            except anthropic.APIConnectionError as e:
                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    print(f"Connection error, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    print(f"Connection error after {max_retries} attempts: {e}")
                    return None

            except anthropic.APIError as e:
                print(f"API error: {e}")
                return None

        return None

    def chat(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        purpose: str = "chat",
    ) -> Optional[ClaudeResponse]:
        """
        Multi-turn chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content'
            system: System prompt
            model: Model to use
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            purpose: Purpose for cost tracking

        Returns:
            ClaudeResponse or None
        """
        model_name = self.MODELS.get(model, model) if model else self.default_model

        # Estimate cost from all messages
        all_text = " ".join(m["content"] for m in messages) + (system or "")
        estimated_cost = self.cost_manager.estimate_cost(model_name, all_text, max_tokens)

        if not self.cost_manager.can_spend(estimated_cost):
            print(f"Budget exceeded for chat")
            return None

        response = self._make_request_with_retry(
            model=model_name,
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        if response is None:
            return None

        self.cost_manager.record_usage(
            model=model_name,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            purpose=purpose,
        )

        cost = self.cost_manager.calculate_cost(
            model_name,
            response.usage.input_tokens,
            response.usage.output_tokens,
        )

        return ClaudeResponse(
            content=response.content[0].text,
            model=model_name,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost=cost,
            stop_reason=response.stop_reason,
        )

    def health_check(self) -> Dict[str, Any]:
        """
        Check API health and budget status.

        Returns:
            Health status dict
        """
        stats = self.cost_manager.get_stats()

        # Try a minimal API call to verify connectivity
        api_healthy = False
        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            api_healthy = response is not None

            # Record this minimal usage
            self.cost_manager.record_usage(
                model="claude-3-haiku-20240307",
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                purpose="health_check",
            )
        except Exception as e:
            print(f"Health check failed: {e}")

        return {
            "api_healthy": api_healthy,
            "budget_remaining": stats.budget_remaining_today,
            "cost_today": stats.cost_today,
            "total_requests_today": stats.requests_today,
            "daily_budget": self.cost_manager.daily_budget,
        }

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        stats = self.cost_manager.get_stats()
        return {
            "total_requests": stats.total_requests,
            "total_cost": round(stats.total_cost, 4),
            "cost_today": round(stats.cost_today, 4),
            "budget_remaining": round(stats.budget_remaining_today, 4),
            "requests_today": stats.requests_today,
            "by_purpose": {k: round(v, 4) for k, v in stats.by_purpose.items()},
        }

    def set_daily_budget(self, budget: float) -> None:
        """Update daily budget."""
        self.cost_manager.daily_budget = budget
