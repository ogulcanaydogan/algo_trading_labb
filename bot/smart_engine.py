"""
Smart Trading Engine - Unified AI-Enhanced Trading System.

Integrates all components:
- Market regime detection
- ML-based predictions
- Multi-strategy selection
- LLM-powered analysis
- Risk-adjusted execution
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import pandas as pd

from .strategy import StrategyConfig, compute_indicators, generate_signal
from .state import BotState, StateStore
from .trading import TradingManager

# ML modules
from .ml import MLPredictor, MarketRegimeClassifier, MarketRegime, RegimeAnalysis

# Auto-retraining
from .ml import (
    AutoRetrainingScheduler,
    ModelHealth,
    ModelHealthStatus,
    PerformanceMetrics,
)

# Deep Learning model selector (optional)
try:
    from .ml.registry import ModelRegistry, ModelSelector, ModelSelectionStrategy
    from .ml.models.base import ModelPrediction
    HAS_DL_MODELS = True
except ImportError:
    HAS_DL_MODELS = False
    ModelSelector = None
    ModelRegistry = None

# Strategy library
from .strategies import StrategySelector, StrategySignal

# LLM modules
from .llm import LLMAdvisor, PerformanceAnalyzer, AnalysisReport


@dataclass
class TradingDecision:
    """Final trading decision from the smart engine."""
    action: str  # LONG, SHORT, FLAT
    confidence: float
    source: str  # Which component made the decision
    regime: str
    regime_confidence: float
    ml_probability: float
    strategy_signal: str
    strategy_confidence: float
    position_size_multiplier: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reasoning: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "action": self.action,
            "confidence": round(self.confidence, 4),
            "source": self.source,
            "regime": self.regime,
            "regime_confidence": round(self.regime_confidence, 4),
            "ml_probability": round(self.ml_probability, 4),
            "strategy_signal": self.strategy_signal,
            "strategy_confidence": round(self.strategy_confidence, 4),
            "position_size_multiplier": round(self.position_size_multiplier, 2),
            "stop_loss": round(self.stop_loss, 2) if self.stop_loss else None,
            "take_profit": round(self.take_profit, 2) if self.take_profit else None,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class EngineConfig:
    """Configuration for the Smart Trading Engine."""
    # Decision weights
    ml_weight: float = 0.35
    strategy_weight: float = 0.35
    regime_weight: float = 0.30

    # Thresholds
    min_confidence: float = 0.4
    ml_override_threshold: float = 0.8  # ML can override if very confident

    # ML settings
    use_ml: bool = True
    ml_model_type: str = "random_forest"
    auto_retrain_days: int = 30

    # Auto-retraining settings
    enable_auto_retraining: bool = True
    retraining_check_hours: float = 6
    min_accuracy_threshold: float = 0.52
    min_sharpe_threshold: float = 0.5

    # Deep Learning settings
    use_deep_learning: bool = True
    dl_model_selection: str = "regime_based"  # single_best, regime_based, ensemble
    dl_model_priority: List[str] = None  # Model type priority, e.g., ["lstm", "transformer"]

    # Strategy settings
    use_multi_strategy: bool = True
    strategy_agreement_threshold: float = 0.5

    # LLM settings
    use_llm: bool = True
    llm_model: str = "llama3"

    # Risk settings
    max_position_multiplier: float = 2.0
    min_position_multiplier: float = 0.25


class SmartTradingEngine:
    """
    Unified Smart Trading Engine.

    Combines:
    1. Market Regime Detection - Identifies bull/bear/sideways/volatile
    2. ML Predictions - XGBoost/RandomForest probability estimates
    3. Strategy Selection - Auto-selects best strategy for regime
    4. LLM Analysis - AI-powered suggestions and explanations
    5. Risk Management - Dynamic position sizing based on conditions

    Decision Flow:
    1. Analyze market regime
    2. Get ML prediction
    3. Get strategy signals (multiple strategies vote)
    4. Combine signals with weighted voting
    5. Apply regime-based risk adjustments
    6. Generate final decision with reasoning
    """

    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        strategy_config: Optional[StrategyConfig] = None,
        data_dir: str = "data",
    ):
        self.config = config or EngineConfig()
        self.strategy_config = strategy_config or StrategyConfig()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.regime_classifier = MarketRegimeClassifier()
        self.strategy_selector = StrategySelector(
            use_multi_strategy=self.config.use_multi_strategy,
            min_agreement=self.config.strategy_agreement_threshold,
        )

        # ML predictor (lazy initialization)
        self._ml_predictor: Optional[MLPredictor] = None
        self._ml_last_trained: Optional[datetime] = None

        # LLM advisor (lazy initialization)
        self._llm_advisor: Optional[LLMAdvisor] = None
        self._performance_analyzer: Optional[PerformanceAnalyzer] = None

        # Deep Learning model selector (lazy initialization)
        self._model_registry: Optional[ModelRegistry] = None
        self._model_selector: Optional[ModelSelector] = None

        # Auto-retraining scheduler (lazy initialization)
        self._auto_retrainer: Optional[AutoRetrainingScheduler] = None

        # State tracking
        self._last_regime: Optional[MarketRegime] = None
        self._decision_history: List[TradingDecision] = []

    @property
    def model_selector(self) -> Optional[ModelSelector]:
        """Lazy load model selector for deep learning models."""
        if not HAS_DL_MODELS or not self.config.use_deep_learning:
            return None

        if self._model_selector is None:
            self._model_registry = ModelRegistry(
                registry_dir=str(self.data_dir / "model_registry")
            )
            strategy_map = {
                "single_best": ModelSelectionStrategy.SINGLE_BEST,
                "regime_based": ModelSelectionStrategy.REGIME_BASED,
                "ensemble": ModelSelectionStrategy.ENSEMBLE,
                "adaptive": ModelSelectionStrategy.ADAPTIVE,
            }
            strategy = strategy_map.get(
                self.config.dl_model_selection,
                ModelSelectionStrategy.REGIME_BASED
            )
            self._model_selector = ModelSelector(
                registry=self._model_registry,
                strategy=strategy,
            )
        return self._model_selector

    @property
    def ml_predictor(self) -> Optional[MLPredictor]:
        """Lazy load ML predictor."""
        if self._ml_predictor is None and self.config.use_ml:
            self._ml_predictor = MLPredictor(
                model_type=self.config.ml_model_type,
                model_dir=str(self.data_dir / "models"),
            )
            # Try to load existing model
            self._ml_predictor.load("smart_engine")
        return self._ml_predictor

    @property
    def llm_advisor(self) -> Optional[LLMAdvisor]:
        """Lazy load LLM advisor."""
        if self._llm_advisor is None and self.config.use_llm:
            self._llm_advisor = LLMAdvisor(model=self.config.llm_model)
        return self._llm_advisor

    @property
    def performance_analyzer(self) -> PerformanceAnalyzer:
        """Get performance analyzer."""
        if self._performance_analyzer is None:
            self._performance_analyzer = PerformanceAnalyzer(self.llm_advisor)
        return self._performance_analyzer

    @property
    def auto_retrainer(self) -> Optional[AutoRetrainingScheduler]:
        """Lazy load auto-retraining scheduler."""
        if not self.config.enable_auto_retraining:
            return None

        if self._auto_retrainer is None:
            self._auto_retrainer = AutoRetrainingScheduler(
                check_interval_hours=self.config.retraining_check_hours,
                data_dir=str(self.data_dir / "auto_retraining"),
            )
        return self._auto_retrainer

    def check_model_health(self, symbol: str) -> Dict[str, Any]:
        """
        Check the health of the ML model for a symbol.

        Returns:
            Dict with health status, metrics, and recommendations
        """
        if not self.config.use_ml or not self.auto_retrainer:
            return {
                "status": "disabled",
                "reason": "ML or auto-retraining is disabled",
            }

        # Check health via auto-retrainer
        health = self.auto_retrainer.check_model_health(
            symbol=symbol,
            model_type=self.config.ml_model_type,
        )

        # Generate recommendations based on health
        recommendations = []
        if health.needs_retraining:
            recommendations.append(f"Model should be retrained: {health.reason}")
        if health.accuracy and health.accuracy < self.config.min_accuracy_threshold:
            recommendations.append(
                f"Accuracy ({health.accuracy:.1%}) below threshold ({self.config.min_accuracy_threshold:.1%})"
            )
        if health.sharpe_ratio and health.sharpe_ratio < self.config.min_sharpe_threshold:
            recommendations.append(
                f"Sharpe ratio ({health.sharpe_ratio:.2f}) below threshold ({self.config.min_sharpe_threshold:.2f})"
            )
        if health.days_since_trained and health.days_since_trained > self.config.auto_retrain_days:
            recommendations.append(
                f"Model is {health.days_since_trained} days old, consider retraining"
            )

        return {
            "symbol": symbol,
            "model_type": self.config.ml_model_type,
            "status": health.status.value,
            "accuracy": health.accuracy,
            "sharpe_ratio": health.sharpe_ratio,
            "win_rate": health.win_rate,
            "days_since_trained": health.days_since_trained,
            "drift_detected": health.drift_detected,
            "needs_retraining": health.needs_retraining,
            "reason": health.reason,
            "recommendations": recommendations,
            "last_checked": health.last_checked.isoformat(),
        }

    def start_auto_retraining(self, symbols: List[str]) -> bool:
        """
        Start the auto-retraining scheduler for given symbols.

        Args:
            symbols: List of symbols to monitor

        Returns:
            True if started successfully
        """
        if not self.auto_retrainer:
            return False

        # Register models
        for symbol in symbols:
            model = self.ml_predictor
            if model:
                self.auto_retrainer.register_model(
                    symbol=symbol,
                    model_type=self.config.ml_model_type,
                    model=model,
                    created_at=self._ml_last_trained,
                )

        # Start scheduler
        self.auto_retrainer.start()
        return True

    def stop_auto_retraining(self) -> None:
        """Stop the auto-retraining scheduler."""
        if self.auto_retrainer:
            self.auto_retrainer.stop()

    def record_trading_performance(
        self,
        symbol: str,
        accuracy: float,
        sharpe_ratio: float,
        win_rate: float,
        profit_factor: float,
        max_drawdown: float,
    ) -> None:
        """
        Record trading performance metrics for model health tracking.

        Call this periodically (e.g., daily) to track model performance.
        """
        if not self.auto_retrainer:
            return

        metrics = PerformanceMetrics(
            accuracy=accuracy,
            precision=accuracy,  # Simplified
            recall=accuracy,
            f1_score=accuracy,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
        )

        self.auto_retrainer.record_performance(
            symbol=symbol,
            model_type=self.config.ml_model_type,
            metrics=metrics,
        )

    def analyze(self, ohlcv: pd.DataFrame) -> TradingDecision:
        """
        Analyze market and generate trading decision.

        Args:
            ohlcv: OHLCV DataFrame with at least 200 rows

        Returns:
            TradingDecision with action, confidence, and reasoning
        """
        reasoning = []

        # 1. Analyze market regime
        regime_analysis = self.regime_classifier.classify(ohlcv)
        self._last_regime = regime_analysis.regime
        reasoning.append(f"Market regime: {regime_analysis.regime.value} ({regime_analysis.confidence:.0%} confidence)")
        reasoning.extend(regime_analysis.reasoning)

        # 2. Get ML prediction (if available)
        ml_action = "FLAT"
        ml_confidence = 0.33
        ml_probability = 0.33

        if self.ml_predictor and self.ml_predictor.is_trained:
            try:
                ml_result = self.ml_predictor.predict(ohlcv)
                ml_action = ml_result.action
                ml_confidence = ml_result.confidence
                ml_probability = getattr(ml_result, f"probability_{ml_action.lower()}", 0.33)
                reasoning.append(f"ML model predicts {ml_action} ({ml_confidence:.0%})")
            except Exception as e:
                reasoning.append(f"ML prediction failed: {e}")

        # 3. Get strategy signals
        strategy_result = self.strategy_selector.select_and_generate(ohlcv)
        strategy_signal = strategy_result.primary_signal
        reasoning.append(f"Strategy ({strategy_result.selected_strategy}): {strategy_signal.decision} ({strategy_signal.confidence:.0%})")

        if strategy_result.supporting_strategies:
            reasoning.append(f"Supporting strategies: {', '.join(strategy_result.supporting_strategies)}")

        # 4. Combine signals with weighted voting
        final_action, final_confidence = self._combine_signals(
            ml_action=ml_action,
            ml_confidence=ml_confidence,
            strategy_action=strategy_signal.decision,
            strategy_confidence=strategy_signal.confidence,
            regime=regime_analysis.regime,
            regime_confidence=regime_analysis.confidence,
        )

        # 5. Apply regime-based adjustments
        regime_params = self.regime_classifier.get_strategy_parameters(regime_analysis.regime)
        position_multiplier = regime_params.get("position_size_multiplier", 1.0)

        # Reduce position if confidence is low
        if final_confidence < self.config.min_confidence:
            position_multiplier *= 0.5
            reasoning.append(f"Position reduced due to low confidence ({final_confidence:.0%})")

        # Clamp position multiplier
        position_multiplier = max(
            self.config.min_position_multiplier,
            min(self.config.max_position_multiplier, position_multiplier)
        )

        # 6. Get stop loss and take profit from strategy
        stop_loss = strategy_signal.stop_loss
        take_profit = strategy_signal.take_profit

        # Adjust SL/TP based on regime
        if stop_loss and regime_params.get("stop_loss_multiplier"):
            stop_loss_dist = abs(ohlcv["close"].iloc[-1] - stop_loss)
            stop_loss_dist *= regime_params["stop_loss_multiplier"]
            if final_action == "LONG":
                stop_loss = ohlcv["close"].iloc[-1] - stop_loss_dist
            else:
                stop_loss = ohlcv["close"].iloc[-1] + stop_loss_dist

        # Create decision
        decision = TradingDecision(
            action=final_action,
            confidence=final_confidence,
            source=self._determine_decision_source(ml_confidence, strategy_signal.confidence),
            regime=regime_analysis.regime.value,
            regime_confidence=regime_analysis.confidence,
            ml_probability=ml_probability,
            strategy_signal=strategy_signal.decision,
            strategy_confidence=strategy_signal.confidence,
            position_size_multiplier=position_multiplier,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=reasoning,
        )

        # Track decision
        self._decision_history.append(decision)
        if len(self._decision_history) > 1000:
            self._decision_history = self._decision_history[-500:]

        return decision

    def _combine_signals(
        self,
        ml_action: str,
        ml_confidence: float,
        strategy_action: str,
        strategy_confidence: float,
        regime: MarketRegime,
        regime_confidence: float,
    ) -> tuple[str, float]:
        """Combine multiple signals into final decision."""

        # Special case: ML is very confident and overrides
        if ml_confidence >= self.config.ml_override_threshold:
            return ml_action, ml_confidence * 0.95

        # Calculate weighted scores for each action
        scores = {"LONG": 0.0, "SHORT": 0.0, "FLAT": 0.0}

        # ML contribution
        if self.config.use_ml:
            scores[ml_action] += ml_confidence * self.config.ml_weight

        # Strategy contribution
        scores[strategy_action] += strategy_confidence * self.config.strategy_weight

        # Regime contribution (bias towards certain actions)
        regime_bias = self._get_regime_bias(regime)
        for action, bias in regime_bias.items():
            scores[action] += bias * regime_confidence * self.config.regime_weight

        # Determine winner
        final_action = max(scores, key=scores.get)
        final_confidence = scores[final_action]

        # Normalize confidence to 0-1
        total = sum(scores.values())
        if total > 0:
            final_confidence = scores[final_action] / total

        return final_action, final_confidence

    def _get_regime_bias(self, regime: MarketRegime) -> Dict[str, float]:
        """Get action bias based on market regime."""
        bias_map = {
            MarketRegime.STRONG_BULL: {"LONG": 0.7, "SHORT": 0.1, "FLAT": 0.2},
            MarketRegime.BULL: {"LONG": 0.5, "SHORT": 0.2, "FLAT": 0.3},
            MarketRegime.SIDEWAYS: {"LONG": 0.3, "SHORT": 0.3, "FLAT": 0.4},
            MarketRegime.BEAR: {"LONG": 0.2, "SHORT": 0.5, "FLAT": 0.3},
            MarketRegime.STRONG_BEAR: {"LONG": 0.1, "SHORT": 0.6, "FLAT": 0.3},
            MarketRegime.VOLATILE: {"LONG": 0.25, "SHORT": 0.25, "FLAT": 0.5},
        }
        return bias_map.get(regime, {"LONG": 0.33, "SHORT": 0.33, "FLAT": 0.34})

    def _determine_decision_source(self, ml_conf: float, strategy_conf: float) -> str:
        """Determine which component drove the decision."""
        if ml_conf > strategy_conf + 0.2:
            return "ml_model"
        elif strategy_conf > ml_conf + 0.2:
            return "strategy"
        else:
            return "combined"

    def train_ml(self, ohlcv: pd.DataFrame) -> Dict[str, Any]:
        """
        Train or retrain the ML model.

        Args:
            ohlcv: Historical OHLCV data (recommend at least 1 year)

        Returns:
            Training metrics dictionary
        """
        if not self.config.use_ml:
            return {"error": "ML is disabled in config"}

        if self._ml_predictor is None:
            self._ml_predictor = MLPredictor(
                model_type=self.config.ml_model_type,
                model_dir=str(self.data_dir / "models"),
            )

        metrics = self._ml_predictor.train(ohlcv)
        self._ml_predictor.save("smart_engine")
        self._ml_last_trained = datetime.now()

        return {
            "accuracy": metrics.accuracy,
            "cross_val_mean": metrics.cross_val_mean,
            "cross_val_std": metrics.cross_val_std,
            "train_samples": metrics.train_samples,
            "test_samples": metrics.test_samples,
            "trained_at": self._ml_last_trained.isoformat(),
        }

    def should_retrain(self) -> bool:
        """Check if ML model should be retrained."""
        if not self._ml_last_trained:
            return True

        days_since_train = (datetime.now() - self._ml_last_trained).days
        return days_since_train >= self.config.auto_retrain_days

    def get_analysis_report(
        self,
        trades: List[Dict],
        equity_curve: List[Dict],
    ) -> AnalysisReport:
        """Get comprehensive performance analysis."""
        return self.performance_analyzer.analyze(
            trades=trades,
            equity_curve=equity_curve,
            strategy_name="smart_engine",
        )

    def get_llm_advice(
        self,
        metrics: Dict[str, float],
        recent_trades: List[Dict],
    ) -> Dict:
        """Get LLM-powered strategy advice."""
        if not self.llm_advisor:
            return {"error": "LLM is disabled"}

        advice = self.llm_advisor.get_strategy_advice(
            symbol=self.strategy_config.symbol,
            timeframe=self.strategy_config.timeframe,
            regime=self._last_regime.value if self._last_regime else "unknown",
            metrics=metrics,
            current_strategy="smart_engine",
            recent_trades=recent_trades,
        )

        return advice.to_dict()

    def get_status(self) -> Dict:
        """Get current engine status."""
        status = {
            "config": {
                "ml_enabled": self.config.use_ml,
                "ml_model_type": self.config.ml_model_type,
                "multi_strategy": self.config.use_multi_strategy,
                "llm_enabled": self.config.use_llm,
                "auto_retraining_enabled": self.config.enable_auto_retraining,
            },
            "ml_status": {
                "is_trained": self.ml_predictor.is_trained if self.ml_predictor else False,
                "last_trained": self._ml_last_trained.isoformat() if self._ml_last_trained else None,
                "should_retrain": self.should_retrain(),
            },
            "last_regime": self._last_regime.value if self._last_regime else None,
            "decisions_tracked": len(self._decision_history),
            "available_strategies": self.strategy_selector.get_available_strategies(),
        }

        # Add auto-retraining status if enabled
        if self.auto_retrainer:
            status["auto_retraining"] = {
                "running": self.auto_retrainer.is_running(),
                "check_interval_hours": self.config.retraining_check_hours,
                "registered_models": len(self.auto_retrainer._models),
                "pending_jobs": len(self.auto_retrainer._job_queue),
            }

        return status

    def explain_decision(self, decision: TradingDecision) -> str:
        """Get human-readable explanation of a decision."""
        if not self.llm_advisor:
            return self._rule_based_explanation(decision)

        # Use LLM for detailed explanation
        return self.llm_advisor.explain_trade(
            direction=decision.action,
            entry_price=decision.stop_loss or 0,  # Placeholder
            exit_price=decision.take_profit or 0,
            pnl=0,
            strategy=decision.source,
            indicators={
                "ml_probability": decision.ml_probability,
                "strategy_confidence": decision.strategy_confidence,
                "regime_confidence": decision.regime_confidence,
            },
            market_conditions=f"{decision.regime} market",
        )

    def _rule_based_explanation(self, decision: TradingDecision) -> str:
        """Generate rule-based explanation."""
        parts = [
            f"Decision: {decision.action} with {decision.confidence:.0%} confidence.",
            f"Market is in {decision.regime} regime ({decision.regime_confidence:.0%} certain).",
        ]

        if decision.ml_probability > 0.5:
            parts.append(f"ML model supports this direction ({decision.ml_probability:.0%}).")

        if decision.strategy_confidence > 0.5:
            parts.append(f"Strategy signal confirms ({decision.strategy_signal}, {decision.strategy_confidence:.0%}).")

        if decision.position_size_multiplier != 1.0:
            parts.append(f"Position size adjusted to {decision.position_size_multiplier:.1f}x based on conditions.")

        return " ".join(parts)
