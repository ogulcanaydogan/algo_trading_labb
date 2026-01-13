"""
Strategy Library Module.

Provides multiple trading strategies for different market conditions:
- EMA Crossover (trend-following)
- Bollinger Band Breakout (volatility)
- MACD Divergence (momentum)
- RSI Mean Reversion (range-bound)
- Ichimoku Cloud (trend-following)
- VWAP Deviation (mean reversion)
- Breakout (breakout trading)
- Stochastic Divergence (reversal detection)
- Keltner Channel (ATR-based breakouts)
- Grid Trading (range trading)
- Strategy Selector (auto-selects best strategy)
"""

from .base import BaseStrategy, StrategySignal
from .ema_crossover import EMACrossoverStrategy
from .bollinger_bands import BollingerBandStrategy, BollingerBandConfig
from .macd_divergence import MACDDivergenceStrategy
from .rsi_mean_reversion import RSIMeanReversionStrategy
from .ichimoku import IchimokuStrategy
from .vwap import VWAPStrategy
from .breakout import BreakoutStrategy
from .stochastic_divergence import StochasticDivergenceStrategy
from .keltner_channel import KeltnerChannelStrategy
from .grid_trading import GridTradingStrategy
from .selector import StrategySelector

__all__ = [
    "BaseStrategy",
    "StrategySignal",
    "EMACrossoverStrategy",
    "BollingerBandStrategy",
    "BollingerBandConfig",
    "MACDDivergenceStrategy",
    "RSIMeanReversionStrategy",
    "IchimokuStrategy",
    "VWAPStrategy",
    "BreakoutStrategy",
    "StochasticDivergenceStrategy",
    "KeltnerChannelStrategy",
    "GridTradingStrategy",
    "StrategySelector",
]
