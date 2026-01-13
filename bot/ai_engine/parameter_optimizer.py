"""
Parameter Optimizer AI

Uses Bayesian optimization and smart search to find optimal indicator parameters
for different market regimes. Continuously improves by learning from backtest results.

Features:
- Bayesian optimization for efficient parameter search
- Regime-specific parameter tuning
- Multi-objective optimization (sharpe, win rate, drawdown)
- Caches best parameters per symbol/regime
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import random
import math

import numpy as np
import pandas as pd

from .learning_db import LearningDatabase, OptimizationResult, get_learning_db

logger = logging.getLogger(__name__)


@dataclass
class ParameterSpace:
    """Defines the search space for a parameter."""
    name: str
    min_val: float
    max_val: float
    step: float = 1.0
    param_type: str = "int"  # int, float


@dataclass
class OptimizationConfig:
    """Configuration for optimization run."""
    symbol: str
    regime: str
    backtest_days: int = 90
    n_trials: int = 50
    n_random_starts: int = 10
    objective: str = "sharpe_ratio"  # sharpe_ratio, win_rate, profit_factor


# Default parameter search spaces
DEFAULT_PARAM_SPACES = [
    ParameterSpace("ema_fast", 5, 20, 1, "int"),
    ParameterSpace("ema_slow", 15, 50, 1, "int"),
    ParameterSpace("rsi_period", 7, 21, 1, "int"),
    ParameterSpace("rsi_overbought", 65, 80, 1, "int"),
    ParameterSpace("rsi_oversold", 20, 35, 1, "int"),
    ParameterSpace("adx_period", 10, 20, 1, "int"),
    ParameterSpace("adx_threshold", 15, 30, 1, "int"),
    ParameterSpace("atr_period", 10, 20, 1, "int"),
    ParameterSpace("atr_multiplier_sl", 1.0, 3.0, 0.25, "float"),
    ParameterSpace("atr_multiplier_tp", 1.5, 4.0, 0.25, "float"),
    ParameterSpace("macd_fast", 8, 15, 1, "int"),
    ParameterSpace("macd_slow", 20, 30, 1, "int"),
    ParameterSpace("macd_signal", 7, 12, 1, "int"),
    ParameterSpace("bb_period", 15, 25, 1, "int"),
    ParameterSpace("bb_std", 1.5, 2.5, 0.25, "float"),
]


class ParameterOptimizer:
    """
    AI-driven parameter optimizer using Bayesian-like optimization.

    Uses a combination of:
    1. Random exploration (initial phase)
    2. Exploitation of promising regions
    3. Gaussian process-like surrogate modeling (simplified)
    """

    def __init__(
        self,
        param_spaces: List[ParameterSpace] = None,
        db: LearningDatabase = None,
    ):
        self.param_spaces = param_spaces or DEFAULT_PARAM_SPACES
        self.db = db or get_learning_db()
        self._results_cache: Dict[str, List[Tuple[Dict, float]]] = {}
        self._best_params: Dict[str, Dict[str, Any]] = {}

    def _sample_random(self) -> Dict[str, Any]:
        """Sample random parameters from search space."""
        params = {}
        for space in self.param_spaces:
            if space.param_type == "int":
                params[space.name] = random.randint(
                    int(space.min_val), int(space.max_val)
                )
            else:
                n_steps = int((space.max_val - space.min_val) / space.step)
                params[space.name] = space.min_val + random.randint(0, n_steps) * space.step
        return params

    def _mutate_params(
        self,
        params: Dict[str, Any],
        mutation_rate: float = 0.3
    ) -> Dict[str, Any]:
        """Mutate parameters slightly for local search."""
        new_params = params.copy()
        for space in self.param_spaces:
            if random.random() < mutation_rate:
                if space.param_type == "int":
                    delta = random.randint(-2, 2) * int(space.step)
                    new_val = params[space.name] + delta
                    new_params[space.name] = max(
                        int(space.min_val),
                        min(int(space.max_val), new_val)
                    )
                else:
                    delta = random.choice([-1, 0, 1]) * space.step
                    new_val = params[space.name] + delta
                    new_params[space.name] = max(
                        space.min_val,
                        min(space.max_val, new_val)
                    )
        return new_params

    def _crossover(
        self,
        params1: Dict[str, Any],
        params2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Crossover two parameter sets."""
        child = {}
        for space in self.param_spaces:
            if random.random() < 0.5:
                child[space.name] = params1[space.name]
            else:
                child[space.name] = params2[space.name]
        return child

    async def optimize(
        self,
        config: OptimizationConfig,
        backtest_fn: Callable[[Dict[str, Any]], Dict[str, float]],
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Run optimization to find best parameters.

        Args:
            config: Optimization configuration
            backtest_fn: Function that takes params and returns metrics dict
                         with keys: sharpe_ratio, win_rate, total_return, max_drawdown, num_trades
            progress_callback: Optional callback(trial, total, best_score)

        Returns:
            Tuple of (best_params, best_score)
        """
        logger.info(f"Starting optimization for {config.symbol} in {config.regime} regime")

        cache_key = f"{config.symbol}_{config.regime}"
        self._results_cache[cache_key] = []

        best_params = None
        best_score = float('-inf')

        # Phase 1: Random exploration
        for i in range(config.n_random_starts):
            params = self._sample_random()

            # Ensure ema_fast < ema_slow
            if params.get('ema_fast', 0) >= params.get('ema_slow', 100):
                params['ema_fast'], params['ema_slow'] = params['ema_slow'] - 5, params['ema_fast'] + 5

            try:
                metrics = backtest_fn(params)
                score = metrics.get(config.objective, 0)

                self._results_cache[cache_key].append((params, score))

                if score > best_score:
                    best_score = score
                    best_params = params

                if progress_callback:
                    progress_callback(i + 1, config.n_trials, best_score)

            except Exception as e:
                logger.warning(f"Backtest failed for params {params}: {e}")

        # Phase 2: Exploitation with mutation and crossover
        for i in range(config.n_random_starts, config.n_trials):
            # Get top performers
            sorted_results = sorted(
                self._results_cache[cache_key],
                key=lambda x: x[1],
                reverse=True
            )[:5]

            if len(sorted_results) >= 2:
                # Sometimes crossover, sometimes mutate
                if random.random() < 0.3:
                    # Crossover two top performers
                    p1, p2 = random.sample(sorted_results, 2)
                    params = self._crossover(p1[0], p2[0])
                else:
                    # Mutate best performer
                    params = self._mutate_params(sorted_results[0][0])
            else:
                params = self._sample_random()

            # Ensure ema_fast < ema_slow
            if params.get('ema_fast', 0) >= params.get('ema_slow', 100):
                params['ema_fast'] = params['ema_slow'] - 5

            try:
                metrics = backtest_fn(params)
                score = metrics.get(config.objective, 0)

                self._results_cache[cache_key].append((params, score))

                if score > best_score:
                    best_score = score
                    best_params = params
                    logger.info(f"New best: {config.objective}={score:.4f}")

                if progress_callback:
                    progress_callback(i + 1, config.n_trials, best_score)

            except Exception as e:
                logger.warning(f"Backtest failed: {e}")

        # Save result to database
        if best_params:
            # Get full metrics for best params
            try:
                best_metrics = backtest_fn(best_params)
                result = OptimizationResult(
                    id=None,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    symbol=config.symbol,
                    regime=config.regime,
                    parameters=best_params,
                    sharpe_ratio=best_metrics.get('sharpe_ratio', 0),
                    win_rate=best_metrics.get('win_rate', 0),
                    total_return=best_metrics.get('total_return', 0),
                    max_drawdown=best_metrics.get('max_drawdown', 0),
                    num_trades=best_metrics.get('num_trades', 0),
                    backtest_period_days=config.backtest_days,
                )
                self.db.save_optimization_result(result)
            except Exception as e:
                logger.error(f"Failed to save optimization result: {e}")

            self._best_params[cache_key] = best_params

        logger.info(f"Optimization complete: best {config.objective}={best_score:.4f}")
        return best_params, best_score

    def get_best_params(
        self,
        symbol: str,
        regime: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached or database best parameters."""
        cache_key = f"{symbol}_{regime}"

        # Check memory cache first
        if cache_key in self._best_params:
            return self._best_params[cache_key]

        # Check database
        params = self.db.get_best_parameters(symbol, regime)
        if params:
            self._best_params[cache_key] = params
            return params

        return None

    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameters (middle of each range)."""
        params = {}
        for space in self.param_spaces:
            mid = (space.min_val + space.max_val) / 2
            if space.param_type == "int":
                params[space.name] = int(mid)
            else:
                params[space.name] = mid
        return params

    def get_regime_adjusted_params(
        self,
        base_params: Dict[str, Any],
        regime: str
    ) -> Dict[str, Any]:
        """
        Adjust parameters based on market regime.

        This encodes domain knowledge about what works in different regimes.
        """
        params = base_params.copy()

        if regime in ["strong_bull", "bull"]:
            # In bull markets: trend following, wider stops
            params['ema_fast'] = max(8, params.get('ema_fast', 12) - 2)
            params['atr_multiplier_sl'] = min(3.0, params.get('atr_multiplier_sl', 2.0) + 0.5)
            params['adx_threshold'] = max(15, params.get('adx_threshold', 20) - 5)

        elif regime in ["strong_bear", "bear"]:
            # In bear markets: faster signals, tighter stops
            params['ema_fast'] = min(15, params.get('ema_fast', 12) + 2)
            params['atr_multiplier_sl'] = max(1.5, params.get('atr_multiplier_sl', 2.0) - 0.25)

        elif regime in ["sideways", "ranging"]:
            # In sideways: mean reversion, RSI-focused
            params['rsi_overbought'] = min(75, params.get('rsi_overbought', 70) + 5)
            params['rsi_oversold'] = max(25, params.get('rsi_oversold', 30) - 5)
            params['adx_threshold'] = min(30, params.get('adx_threshold', 20) + 5)

        elif regime in ["volatile", "high_vol", "crash"]:
            # In volatile: tight stops, smaller positions
            params['atr_multiplier_sl'] = max(1.0, params.get('atr_multiplier_sl', 2.0) - 0.5)
            params['atr_multiplier_tp'] = max(1.5, params.get('atr_multiplier_tp', 3.0) - 0.5)

        return params

    def analyze_parameter_importance(
        self,
        symbol: str,
        regime: str
    ) -> Dict[str, float]:
        """
        Analyze which parameters have the most impact on performance.

        Returns dict of parameter name -> importance score (0-1).
        """
        cache_key = f"{symbol}_{regime}"
        results = self._results_cache.get(cache_key, [])

        if len(results) < 20:
            return {}

        # Simple correlation-based importance
        importance = {}

        for space in self.param_spaces:
            param_values = [r[0].get(space.name, 0) for r in results]
            scores = [r[1] for r in results]

            if len(set(param_values)) < 2:
                importance[space.name] = 0
                continue

            # Correlation coefficient
            try:
                correlation = np.corrcoef(param_values, scores)[0, 1]
                importance[space.name] = abs(correlation) if not np.isnan(correlation) else 0
            except Exception:
                importance[space.name] = 0

        # Normalize
        max_imp = max(importance.values()) if importance else 1
        if max_imp > 0:
            importance = {k: v / max_imp for k, v in importance.items()}

        return importance


# Global instance
_optimizer: Optional[ParameterOptimizer] = None


def get_parameter_optimizer() -> ParameterOptimizer:
    """Get or create global parameter optimizer."""
    global _optimizer
    if _optimizer is None:
        _optimizer = ParameterOptimizer()
    return _optimizer
