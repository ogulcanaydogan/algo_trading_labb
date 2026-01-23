"""
Strategy Selector.

Automatically selects and combines strategies based on market regime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Type

import pandas as pd

from .base import BaseStrategy, StrategySignal
from .ema_crossover import EMACrossoverStrategy
from .bollinger_bands import BollingerBandStrategy, BollingerBandConfig
from .macd_divergence import MACDDivergenceStrategy
from .rsi_mean_reversion import RSIMeanReversionStrategy
from .stochastic_divergence import StochasticDivergenceStrategy
from .keltner_channel import KeltnerChannelStrategy
from .grid_trading import GridTradingStrategy

from ..ml.regime_classifier import MarketRegimeClassifier, MarketRegime, RegimeAnalysis


@dataclass
class SelectedStrategy:
    """Result of strategy selection."""

    strategy: BaseStrategy
    regime: MarketRegime
    regime_confidence: float
    selection_reason: str


@dataclass
class CombinedSignal:
    """Combined signal from multiple strategies."""

    primary_signal: StrategySignal
    regime_analysis: RegimeAnalysis
    selected_strategy: str
    supporting_strategies: List[str]
    agreement_score: float  # How many strategies agree
    final_confidence: float
    position_size_multiplier: float

    def to_dict(self) -> Dict:
        return {
            "primary_signal": self.primary_signal.to_dict(),
            "regime": self.regime_analysis.regime.value,
            "regime_confidence": self.regime_analysis.confidence,
            "selected_strategy": self.selected_strategy,
            "supporting_strategies": self.supporting_strategies,
            "agreement_score": round(self.agreement_score, 2),
            "final_confidence": round(self.final_confidence, 4),
            "position_size_multiplier": round(self.position_size_multiplier, 2),
        }


class StrategySelector:
    """
    Intelligent Strategy Selector.

    Features:
    - Automatic market regime detection
    - Strategy selection based on regime
    - Optional multi-strategy voting
    - Position size adjustment based on confidence

    Regime to Strategy mapping:
    - Strong Bull/Bull: EMA Crossover (trend-following)
    - Sideways: RSI Mean Reversion + Bollinger Mean Reversion
    - Strong Bear/Bear: EMA Crossover (short bias) or stay flat
    - Volatile: Bollinger Breakout with reduced size
    """

    # Strategy classes for each regime
    REGIME_STRATEGIES: Dict[MarketRegime, List[Type[BaseStrategy]]] = {
        MarketRegime.STRONG_BULL: [
            EMACrossoverStrategy,
            KeltnerChannelStrategy,
            MACDDivergenceStrategy,
        ],
        MarketRegime.BULL: [EMACrossoverStrategy, KeltnerChannelStrategy, MACDDivergenceStrategy],
        MarketRegime.SIDEWAYS: [
            GridTradingStrategy,
            RSIMeanReversionStrategy,
            BollingerBandStrategy,
            StochasticDivergenceStrategy,
        ],
        MarketRegime.BEAR: [
            EMACrossoverStrategy,
            StochasticDivergenceStrategy,
            MACDDivergenceStrategy,
        ],
        MarketRegime.STRONG_BEAR: [EMACrossoverStrategy, StochasticDivergenceStrategy],
        MarketRegime.VOLATILE: [
            KeltnerChannelStrategy,
            BollingerBandStrategy,
            MACDDivergenceStrategy,
        ],
    }

    def __init__(
        self,
        use_multi_strategy: bool = True,
        min_agreement: float = 0.5,
        regime_classifier: Optional[MarketRegimeClassifier] = None,
    ):
        """
        Initialize strategy selector.

        Args:
            use_multi_strategy: Use multiple strategies and vote
            min_agreement: Minimum strategy agreement for signal
            regime_classifier: Custom regime classifier (or creates default)
        """
        self.use_multi_strategy = use_multi_strategy
        self.min_agreement = min_agreement
        self.regime_classifier = regime_classifier or MarketRegimeClassifier()

        # Initialize all strategies
        self._strategies: Dict[str, BaseStrategy] = {
            "ema_crossover": EMACrossoverStrategy(),
            "bollinger_mean_reversion": BollingerBandStrategy(
                BollingerBandConfig(mode="mean_reversion")
            ),
            "bollinger_breakout": BollingerBandStrategy(BollingerBandConfig(mode="breakout")),
            "macd_divergence": MACDDivergenceStrategy(),
            "rsi_mean_reversion": RSIMeanReversionStrategy(),
            "stochastic_divergence": StochasticDivergenceStrategy(),
            "keltner_channel": KeltnerChannelStrategy(),
            "grid_trading": GridTradingStrategy(),
        }

    def select_and_generate(self, ohlcv: pd.DataFrame) -> CombinedSignal:
        """
        Select best strategy and generate combined signal.

        Args:
            ohlcv: OHLCV DataFrame

        Returns:
            CombinedSignal with primary signal and supporting data
        """
        # Analyze market regime
        regime_analysis = self.regime_classifier.classify(ohlcv)

        # Select primary strategy
        selected = self._select_strategy(regime_analysis.regime)

        # Generate primary signal
        primary_signal = selected.strategy.generate_signal(ohlcv)

        # Get supporting signals if multi-strategy mode
        supporting_strategies = []
        agreement_score = 1.0

        if self.use_multi_strategy:
            signals = self._get_all_signals(ohlcv, regime_analysis.regime)
            supporting_strategies, agreement_score = self._calculate_agreement(
                primary_signal, signals
            )

        # Calculate final confidence
        final_confidence = self._calculate_final_confidence(
            primary_signal.confidence,
            regime_analysis.confidence,
            agreement_score,
        )

        # Get position size multiplier from regime
        regime_params = self.regime_classifier.get_strategy_parameters(regime_analysis.regime)
        position_multiplier = regime_params.get("position_size_multiplier", 1.0)

        # Reduce position if low agreement
        if agreement_score < self.min_agreement:
            position_multiplier *= 0.5

        return CombinedSignal(
            primary_signal=primary_signal,
            regime_analysis=regime_analysis,
            selected_strategy=selected.strategy.name,
            supporting_strategies=supporting_strategies,
            agreement_score=agreement_score,
            final_confidence=final_confidence,
            position_size_multiplier=position_multiplier,
        )

    def _select_strategy(self, regime: MarketRegime) -> SelectedStrategy:
        """Select the best strategy for the current regime."""
        strategy_classes = self.REGIME_STRATEGIES.get(regime, [EMACrossoverStrategy])

        # Get the primary strategy for this regime
        primary_class = strategy_classes[0]

        # Map class to instance
        strategy_map = {
            EMACrossoverStrategy: self._strategies["ema_crossover"],
            MACDDivergenceStrategy: self._strategies["macd_divergence"],
            RSIMeanReversionStrategy: self._strategies["rsi_mean_reversion"],
            StochasticDivergenceStrategy: self._strategies["stochastic_divergence"],
            KeltnerChannelStrategy: self._strategies["keltner_channel"],
            GridTradingStrategy: self._strategies["grid_trading"],
            BollingerBandStrategy: (
                self._strategies["bollinger_breakout"]
                if regime == MarketRegime.VOLATILE
                else self._strategies["bollinger_mean_reversion"]
            ),
        }

        strategy = strategy_map.get(primary_class, self._strategies["ema_crossover"])

        return SelectedStrategy(
            strategy=strategy,
            regime=regime,
            regime_confidence=0.0,  # Will be filled later
            selection_reason=f"Best strategy for {regime.value} regime",
        )

    def _get_all_signals(
        self,
        ohlcv: pd.DataFrame,
        regime: MarketRegime,
    ) -> Dict[str, StrategySignal]:
        """Get signals from all suitable strategies for this regime."""
        signals = {}

        # Get strategies suitable for this regime
        strategy_names = self._get_suitable_strategies(regime)

        for name in strategy_names:
            if name in self._strategies:
                try:
                    signals[name] = self._strategies[name].generate_signal(ohlcv)
                except Exception as e:
                    print(f"Strategy {name} failed: {e}")
                    continue

        return signals

    def _get_suitable_strategies(self, regime: MarketRegime) -> List[str]:
        """Get strategy names suitable for the regime."""
        regime_map = {
            MarketRegime.STRONG_BULL: ["ema_crossover", "keltner_channel", "macd_divergence"],
            MarketRegime.BULL: [
                "ema_crossover",
                "keltner_channel",
                "macd_divergence",
                "bollinger_breakout",
            ],
            MarketRegime.SIDEWAYS: [
                "grid_trading",
                "rsi_mean_reversion",
                "bollinger_mean_reversion",
                "stochastic_divergence",
            ],
            MarketRegime.BEAR: ["ema_crossover", "stochastic_divergence", "macd_divergence"],
            MarketRegime.STRONG_BEAR: ["ema_crossover", "stochastic_divergence"],
            MarketRegime.VOLATILE: ["keltner_channel", "bollinger_breakout", "macd_divergence"],
        }
        return regime_map.get(regime, ["ema_crossover"])

    def _calculate_agreement(
        self,
        primary: StrategySignal,
        all_signals: Dict[str, StrategySignal],
    ) -> tuple[List[str], float]:
        """Calculate agreement between strategies."""
        if not all_signals:
            return [], 1.0

        agreeing = []
        total = len(all_signals)

        for name, signal in all_signals.items():
            if signal.decision == primary.decision:
                agreeing.append(name)

        agreement_score = len(agreeing) / total if total > 0 else 0.0

        return agreeing, agreement_score

    def _calculate_final_confidence(
        self,
        signal_confidence: float,
        regime_confidence: float,
        agreement_score: float,
    ) -> float:
        """Calculate final combined confidence."""
        # Weighted combination
        weights = {
            "signal": 0.5,
            "regime": 0.3,
            "agreement": 0.2,
        }

        final = (
            signal_confidence * weights["signal"]
            + regime_confidence * weights["regime"]
            + agreement_score * weights["agreement"]
        )

        return min(final, 1.0)

    def get_available_strategies(self) -> List[str]:
        """Get list of available strategy names."""
        return list(self._strategies.keys())

    def get_strategy(self, name: str) -> Optional[BaseStrategy]:
        """Get a specific strategy by name."""
        return self._strategies.get(name)

    def run_strategy(self, name: str, ohlcv: pd.DataFrame) -> Optional[StrategySignal]:
        """Run a specific strategy by name."""
        strategy = self.get_strategy(name)
        if strategy:
            return strategy.generate_signal(ohlcv)
        return None
