"""
Enhanced Signal Processor

Integrates all advanced ML modules for comprehensive signal generation:
1. Multi-timeframe analysis
2. Regime-specific models
3. Order book analysis
4. Transfer learning insights
5. Online learning adaptation
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Try to import all enhanced modules
try:
    from bot.ml.multi_timeframe import (
        MultiTimeframeAnalyzer,
        TimeframeCascadeFilter,
        get_mtf_analyzer,
        get_mtf_signal,
    )

    MTF_AVAILABLE = True
except ImportError:
    MTF_AVAILABLE = False
    logger.debug("Multi-timeframe module not available")

try:
    from bot.ml.regime_models import (
        RegimeModelEnsemble,
        RegimeDetector,
        MarketRegime,
        create_regime_ensemble,
    )

    REGIME_MODELS_AVAILABLE = True
except ImportError:
    REGIME_MODELS_AVAILABLE = False
    logger.debug("Regime models module not available")

try:
    from bot.ml.orderbook_analyzer import (
        OrderBookAnalyzer,
        OptionsFlowAnalyzer,
        get_orderbook_analyzer,
        get_options_analyzer,
    )

    ORDERBOOK_AVAILABLE = True
except ImportError:
    ORDERBOOK_AVAILABLE = False
    logger.debug("Order book module not available")

try:
    from bot.ml.transfer_learning import TransferLearner, PretrainedModelBank

    TRANSFER_AVAILABLE = True
except ImportError:
    TRANSFER_AVAILABLE = False
    logger.debug("Transfer learning module not available")

try:
    from bot.ml.online_learning import OnlineLearner, get_adaptive_manager

    ONLINE_LEARNING_AVAILABLE = True
except ImportError:
    ONLINE_LEARNING_AVAILABLE = False
    logger.debug("Online learning module not available")

try:
    from bot.ml.rl_optimizer import RLTradingOptimizer, get_rl_optimizer

    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    logger.debug("RL optimizer module not available")

try:
    from bot.ml.alternative_data import SentimentDataFetcher

    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False
    logger.debug("Sentiment data module not available")


@dataclass
class EnhancedSignal:
    """Comprehensive signal with all enhancements."""

    symbol: str
    timestamp: datetime
    action: str
    confidence: float

    # Multi-timeframe
    mtf_alignment: float
    mtf_primary_trend: str
    mtf_entry_quality: str

    # Regime
    detected_regime: str
    regime_model_accuracy: float

    # Order book
    orderbook_imbalance: float
    whale_activity: float
    flow_toxicity: float

    # RL recommendation
    rl_action: str
    rl_confidence: float

    # Adjustments
    position_size_multiplier: float
    stop_loss_adjustment: float
    take_profit_adjustment: float

    # Metadata
    enhancements_applied: List[str]
    warnings: List[str]
    reasoning: str

    # Sentiment data (fields with defaults must come last)
    fear_greed_index: float = 50.0
    social_sentiment: float = 0.0
    news_sentiment: float = 0.0
    sentiment_composite: float = 0.0


class EnhancedSignalProcessor:
    """
    Processes signals through all enhancement modules.
    """

    def __init__(
        self,
        enable_mtf: bool = True,
        enable_regime: bool = True,
        enable_orderbook: bool = True,
        enable_rl: bool = True,
        enable_online_learning: bool = True,
        enable_sentiment: bool = True,
    ):
        self.enable_mtf = enable_mtf and MTF_AVAILABLE
        self.enable_regime = enable_regime and REGIME_MODELS_AVAILABLE
        self.enable_orderbook = enable_orderbook and ORDERBOOK_AVAILABLE
        self.enable_rl = enable_rl and RL_AVAILABLE
        self.enable_online_learning = enable_online_learning and ONLINE_LEARNING_AVAILABLE
        self.enable_sentiment = enable_sentiment and SENTIMENT_AVAILABLE

        # Initialize components
        self.mtf_analyzer = get_mtf_analyzer() if self.enable_mtf else None
        self.sentiment_fetcher = SentimentDataFetcher() if self.enable_sentiment else None
        self.mtf_filter = TimeframeCascadeFilter() if self.enable_mtf else None
        self.regime_ensembles: Dict[str, RegimeModelEnsemble] = {}
        self.orderbook_analyzers: Dict[str, OrderBookAnalyzer] = {}
        self.rl_optimizer = get_rl_optimizer() if self.enable_rl else None

        # Price/data cache
        self.price_history: Dict[str, pd.DataFrame] = {}

        logger.info(
            f"Enhanced Signal Processor initialized: "
            f"MTF={self.enable_mtf}, Regime={self.enable_regime}, "
            f"OrderBook={self.enable_orderbook}, RL={self.enable_rl}"
        )

    def update_price_history(self, symbol: str, ohlcv: pd.DataFrame):
        """Update price history for a symbol."""
        self.price_history[symbol] = ohlcv

    def update_orderbook(
        self, symbol: str, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]
    ):
        """Update order book data for a symbol."""
        if not self.enable_orderbook:
            return

        if symbol not in self.orderbook_analyzers:
            self.orderbook_analyzers[symbol] = get_orderbook_analyzer(symbol)

        self.orderbook_analyzers[symbol].process_orderbook(bids, asks)

    async def process_signal(
        self,
        symbol: str,
        base_signal: Dict[str, Any],
        current_price: float,
        features: Optional[np.ndarray] = None,
    ) -> EnhancedSignal:
        """
        Process a base signal through all enhancements.

        Args:
            symbol: Trading symbol
            base_signal: Base ML signal
            current_price: Current price
            features: Feature vector for ML models

        Returns:
            EnhancedSignal with all enhancements
        """
        enhancements_applied = []
        warnings = []
        reasoning_parts = []

        action = base_signal.get("action", "HOLD")
        confidence = base_signal.get("confidence", 0.5)

        # Initialize defaults
        mtf_alignment = 0.5
        mtf_primary_trend = "neutral"
        mtf_entry_quality = "fair"
        detected_regime = "unknown"
        regime_model_accuracy = 0.5
        orderbook_imbalance = 0.0
        whale_activity = 0.0
        flow_toxicity = 0.0
        rl_action = action
        rl_confidence = confidence
        position_size_multiplier = 1.0
        stop_loss_adjustment = 1.0
        take_profit_adjustment = 1.0

        # 1. Multi-timeframe Analysis
        if self.enable_mtf and symbol in self.price_history:
            try:
                mtf_signal = self.mtf_analyzer.analyze(symbol, self.price_history[symbol])

                mtf_alignment = mtf_signal.alignment
                mtf_primary_trend = mtf_signal.primary_trend
                mtf_entry_quality = mtf_signal.entry_quality

                # Filter signal using MTF
                should_take, filter_reason = self.mtf_filter.filter_signal(
                    base_signal, self.price_history[symbol]
                )

                if not should_take:
                    # Reduce confidence for counter-trend trades
                    confidence *= 0.6
                    warnings.append(f"MTF filter: {filter_reason}")
                else:
                    # Boost confidence for aligned trades
                    confidence = min(0.95, confidence * (1 + mtf_alignment * 0.2))

                reasoning_parts.append(f"MTF: {mtf_primary_trend} trend, {mtf_entry_quality} entry")
                enhancements_applied.append("multi_timeframe")

            except Exception as e:
                logger.warning(f"MTF analysis failed: {e}")

        # 2. Regime-Specific Model
        if self.enable_regime and symbol in self.price_history:
            try:
                if symbol not in self.regime_ensembles:
                    # Simple rule-based regime detection fallback
                    prices = self.price_history[symbol]["close"]
                    if len(prices) >= 50:
                        returns = prices.pct_change()
                        volatility = returns.rolling(20).std().iloc[-1]
                        trend = (prices.iloc[-1] - prices.iloc[-20]) / prices.iloc[-20]

                        # Classify regime based on trend and volatility
                        if volatility > 0.03:  # High volatility
                            detected_regime = "volatile"
                        elif trend > 0.05:  # Strong uptrend
                            detected_regime = "bull"
                        elif trend < -0.05:  # Strong downtrend
                            detected_regime = "bear"
                        elif abs(trend) < 0.02:  # Sideways
                            detected_regime = "sideways"
                        else:
                            detected_regime = "neutral"

                        regime_model_accuracy = 0.6  # Rule-based confidence
                        enhancements_applied.append("regime_rules")
                        reasoning_parts.append(f"Regime: {detected_regime} (rule-based)")
                else:
                    ensemble = self.regime_ensembles[symbol]
                    regime_signal = ensemble.get_regime_signal(
                        pd.DataFrame({"feature": [features]})
                        if features is not None
                        else pd.DataFrame(),
                        self.price_history[symbol]["close"],
                    )

                    detected_regime = regime_signal["regime"]
                    regime_model_accuracy = regime_signal["model_accuracy"]

                    # Adjust based on regime
                    if detected_regime in ["crash", "high_volatility"]:
                        position_size_multiplier *= 0.5
                        stop_loss_adjustment *= 1.5  # Wider stops
                        warnings.append(f"High risk regime: {detected_regime}")
                    elif detected_regime in ["bull_trend", "recovery"]:
                        position_size_multiplier *= 1.2 if action == "BUY" else 0.8
                    elif detected_regime in ["bear_trend"]:
                        position_size_multiplier *= 1.2 if action == "SELL" else 0.8

                    reasoning_parts.append(f"Regime: {detected_regime}")
                    enhancements_applied.append("regime_model")

            except Exception as e:
                logger.warning(f"Regime model failed: {e}")

        # 3. Order Book Analysis
        if self.enable_orderbook and symbol in self.orderbook_analyzers:
            try:
                ob_analyzer = self.orderbook_analyzers[symbol]
                ob_signal = ob_analyzer.get_signal()

                orderbook_imbalance = ob_signal.imbalance
                whale_activity = ob_signal.whale_activity
                flow_toxicity = ob_signal.toxicity

                # Adjust based on order flow
                if abs(orderbook_imbalance) > 0.3:
                    if (orderbook_imbalance > 0 and action == "BUY") or (
                        orderbook_imbalance < 0 and action == "SELL"
                    ):
                        confidence = min(0.95, confidence * 1.1)
                        reasoning_parts.append(f"Order flow confirms {action}")
                    else:
                        confidence *= 0.85
                        warnings.append("Order flow conflicts with signal")

                # Whale activity
                if whale_activity > 0.1:
                    reasoning_parts.append(f"Whale activity detected: {whale_activity:.0%}")

                # High toxicity = reduce size
                if flow_toxicity > 0.6:
                    position_size_multiplier *= 0.8
                    warnings.append("High flow toxicity detected")

                enhancements_applied.append("orderbook")

            except Exception as e:
                logger.warning(f"Order book analysis failed: {e}")

        # 4. RL Optimizer
        if self.enable_rl and features is not None and self.rl_optimizer.trained:
            try:
                rl_action_name, rl_position_delta, rl_conf = self.rl_optimizer.get_action(features)
                rl_action = rl_action_name.replace("STRONG_", "")
                rl_confidence = rl_conf

                # RL can override or confirm
                if rl_action == action:
                    confidence = min(0.95, confidence * 1.1)
                    reasoning_parts.append(f"RL confirms: {rl_action}")
                elif rl_action != "HOLD" and action != "HOLD":
                    # Conflicting signals
                    confidence *= 0.75
                    warnings.append(f"RL suggests {rl_action} vs ML {action}")

                enhancements_applied.append("rl_optimizer")

            except Exception as e:
                logger.warning(f"RL optimization failed: {e}")

        # 5. Sentiment Analysis
        fear_greed_index = 50.0
        social_sentiment = 0.0
        news_sentiment = 0.0
        sentiment_composite = 0.0

        if self.enable_sentiment and self.sentiment_fetcher:
            try:
                sentiment_data = await self.sentiment_fetcher.get_crypto_sentiment(symbol)

                fear_greed_index = sentiment_data.get("fear_greed") or 50.0
                social_sentiment = sentiment_data.get("social_sentiment") or 0.0
                news_sentiment = sentiment_data.get("news_sentiment") or 0.0
                sentiment_composite = sentiment_data.get("composite_score", 0.0)

                # Adjust confidence based on sentiment
                # Fear/Greed: < 25 = Extreme Fear, > 75 = Extreme Greed
                if fear_greed_index < 25:  # Extreme fear
                    if action == "BUY":
                        confidence *= 1.1  # Contrarian boost
                        reasoning_parts.append("Extreme fear - contrarian buy")
                    else:
                        confidence *= 0.9
                elif fear_greed_index > 75:  # Extreme greed
                    if action == "SELL":
                        confidence *= 1.1  # Contrarian boost
                        reasoning_parts.append("Extreme greed - contrarian sell")
                    else:
                        confidence *= 0.9

                # Sentiment composite alignment
                if (sentiment_composite > 0.3 and action == "BUY") or (
                    sentiment_composite < -0.3 and action == "SELL"
                ):
                    confidence = min(0.95, confidence * 1.05)
                    reasoning_parts.append(f"Sentiment confirms {action}")
                elif (sentiment_composite > 0.3 and action == "SELL") or (
                    sentiment_composite < -0.3 and action == "BUY"
                ):
                    warnings.append("Sentiment conflicts with signal")

                enhancements_applied.append("sentiment")

            except Exception as e:
                logger.warning(f"Sentiment analysis failed: {e}")

        # 6. Final adjustments based on entry quality
        if mtf_entry_quality == "excellent":
            position_size_multiplier *= 1.2
        elif mtf_entry_quality == "poor":
            position_size_multiplier *= 0.7
            warnings.append("Poor entry quality - reduced size")

        # Clamp values
        confidence = np.clip(confidence, 0.1, 0.95)
        position_size_multiplier = np.clip(position_size_multiplier, 0.3, 2.0)

        # Build reasoning
        reasoning = " | ".join(reasoning_parts) if reasoning_parts else "Base signal only"

        return EnhancedSignal(
            symbol=symbol,
            timestamp=datetime.now(),
            action=action,
            confidence=confidence,
            mtf_alignment=mtf_alignment,
            mtf_primary_trend=mtf_primary_trend,
            mtf_entry_quality=mtf_entry_quality,
            detected_regime=detected_regime,
            regime_model_accuracy=regime_model_accuracy,
            orderbook_imbalance=orderbook_imbalance,
            whale_activity=whale_activity,
            flow_toxicity=flow_toxicity,
            rl_action=rl_action,
            rl_confidence=rl_confidence,
            fear_greed_index=fear_greed_index,
            social_sentiment=social_sentiment,
            news_sentiment=news_sentiment,
            sentiment_composite=sentiment_composite,
            position_size_multiplier=position_size_multiplier,
            stop_loss_adjustment=stop_loss_adjustment,
            take_profit_adjustment=take_profit_adjustment,
            enhancements_applied=enhancements_applied,
            warnings=warnings,
            reasoning=reasoning,
        )

    def record_outcome(
        self,
        symbol: str,
        action: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        features: Optional[np.ndarray] = None,
    ):
        """
        Record trade outcome for learning.

        Updates online learning and RL models.
        """
        if self.enable_online_learning and ONLINE_LEARNING_AVAILABLE:
            try:
                manager = get_adaptive_manager()
                learner = manager.get_or_create(symbol)

                label = 1 if pnl > 0 else (-1 if pnl < 0 else 0)
                action_map = {"BUY": 2, "SELL": 0, "HOLD": 1}
                prediction = action_map.get(action, 1)

                if features is not None:
                    learner.update(features, label, prediction)

            except Exception as e:
                logger.warning(f"Online learning update failed: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get processor status."""
        return {
            "mtf_enabled": self.enable_mtf,
            "regime_enabled": self.enable_regime,
            "orderbook_enabled": self.enable_orderbook,
            "rl_enabled": self.enable_rl,
            "online_learning_enabled": self.enable_online_learning,
            "symbols_with_history": list(self.price_history.keys()),
            "symbols_with_orderbook": list(self.orderbook_analyzers.keys()),
            "regime_ensembles_loaded": list(self.regime_ensembles.keys()),
        }


# Global processor instance
_enhanced_processor: Optional[EnhancedSignalProcessor] = None


def get_enhanced_processor() -> EnhancedSignalProcessor:
    """Get or create enhanced signal processor."""
    global _enhanced_processor
    if _enhanced_processor is None:
        _enhanced_processor = EnhancedSignalProcessor()
    return _enhanced_processor


async def enhance_signal(
    symbol: str,
    base_signal: Dict[str, Any],
    current_price: float,
    ohlcv: Optional[pd.DataFrame] = None,
    features: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Convenience function to enhance a signal.

    Args:
        symbol: Trading symbol
        base_signal: Base ML signal
        current_price: Current price
        ohlcv: Optional OHLCV data for MTF
        features: Optional feature vector

    Returns:
        Enhanced signal dict
    """
    processor = get_enhanced_processor()

    if ohlcv is not None:
        processor.update_price_history(symbol, ohlcv)

    enhanced = await processor.process_signal(symbol, base_signal, current_price, features)

    # Convert to dict for compatibility
    return {
        "action": enhanced.action,
        "confidence": enhanced.confidence,
        "symbol": enhanced.symbol,
        "timestamp": enhanced.timestamp.isoformat(),
        "enhancements": {
            "mtf_alignment": enhanced.mtf_alignment,
            "mtf_primary_trend": enhanced.mtf_primary_trend,
            "mtf_entry_quality": enhanced.mtf_entry_quality,
            "detected_regime": enhanced.detected_regime,
            "regime_model_accuracy": enhanced.regime_model_accuracy,
            "orderbook_imbalance": enhanced.orderbook_imbalance,
            "whale_activity": enhanced.whale_activity,
            "flow_toxicity": enhanced.flow_toxicity,
            "rl_action": enhanced.rl_action,
            "rl_confidence": enhanced.rl_confidence,
        },
        "adjustments": {
            "position_size_multiplier": enhanced.position_size_multiplier,
            "stop_loss_adjustment": enhanced.stop_loss_adjustment,
            "take_profit_adjustment": enhanced.take_profit_adjustment,
        },
        "enhancements_applied": enhanced.enhancements_applied,
        "warnings": enhanced.warnings,
        "reasoning": enhanced.reasoning,
    }
