"""
ML Signal Generator for Unified Trading Engine.

Provides ML-based signal generation that can be plugged into
the unified trading engine.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from bot.multi_timeframe import MultiTimeframeAnalyzer, confirm_signal_mtf
from bot.ml_performance_tracker import get_ml_tracker, track_prediction

# Intelligent Brain for regime-based strategy adaptation
try:
    from bot.intelligence import get_intelligent_brain, RegimeAdapter
    INTELLIGENT_BRAIN_AVAILABLE = True
except ImportError:
    INTELLIGENT_BRAIN_AVAILABLE = False

logger = logging.getLogger(__name__)


class MLSignalGenerator:
    """
    ML-based signal generator for the unified trading engine.

    Uses trained models to generate BUY/SELL signals with confidence scores.
    """

    def __init__(
        self,
        model_dir: Path = Path("data/models"),
        model_type: str = "gradient_boosting",
        confidence_threshold: float = 0.45,  # Lowered from 0.55 to allow more signals
        use_mtf_filter: bool = True,
        mtf_strict_mode: bool = False,
        regime_adaptive_threshold: bool = True,  # Adjust threshold based on market regime
    ):
        self.model_dir = model_dir
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.use_mtf_filter = use_mtf_filter
        self.mtf_strict_mode = mtf_strict_mode
        self.regime_adaptive_threshold = regime_adaptive_threshold

        self.models: Dict[str, Any] = {}
        self._price_cache: Dict[str, pd.DataFrame] = {}
        self._mtf_analyzer = MultiTimeframeAnalyzer()
        self._initialized = False
        self._current_regime: Optional[str] = None

    def initialize(self, symbols: List[str]) -> bool:
        """Initialize models for given symbols with intelligent fallback."""
        try:
            from bot.ml.predictor import MLPredictor
        except ImportError:
            logger.warning("MLPredictor not available, using fallback signals")
            self._initialized = True
            return True

        for symbol in symbols:
            loaded = False
            symbol_base = symbol.replace('/', '_')

            # Try multiple model name patterns
            model_patterns = [
                # Exact match
                f"{symbol_base}_{self.model_type}",
                # With 1h timeframe
                f"{symbol_base}_1h_{self.model_type}",
                # Alternative model types as fallback
                f"{symbol_base}_random_forest",
                f"{symbol_base}_gradient_boosting",
                f"{symbol_base}_xgboost",
                # With timeframe variants
                f"{symbol_base}_1h_random_forest",
                f"{symbol_base}_1h_gradient_boosting",
                f"{symbol_base}_1h_xgboost",
            ]

            for pattern in model_patterns:
                try:
                    # Determine model type from pattern
                    if "xgboost" in pattern:
                        mt = "xgboost"
                    elif "gradient_boosting" in pattern:
                        mt = "gradient_boosting"
                    else:
                        mt = "random_forest"

                    predictor = MLPredictor(
                        model_type=mt,
                        model_dir=str(self.model_dir),
                    )
                    if predictor.load(pattern):
                        self.models[symbol] = predictor
                        logger.info(f"Loaded model for {symbol} (pattern: {pattern})")
                        loaded = True
                        break
                except Exception:
                    continue

            if not loaded:
                # Last resort: scan directory for any matching model
                loaded = self._try_scan_for_model(symbol, symbol_base)

            if not loaded:
                logger.warning(f"No model found for {symbol}")

        self._initialized = True
        return len(self.models) > 0

    def _try_scan_for_model(self, symbol: str, symbol_base: str) -> bool:
        """Scan model directory for any matching model file."""
        try:
            from bot.ml.predictor import MLPredictor

            model_dir = Path(self.model_dir)
            if not model_dir.exists():
                return False

            # Look for model files matching the symbol
            for model_file in model_dir.glob(f"{symbol_base}*_model.pkl"):
                model_name = model_file.stem.replace("_model", "")

                # Determine model type from filename
                if "xgboost" in model_name:
                    mt = "xgboost"
                elif "gradient_boosting" in model_name:
                    mt = "gradient_boosting"
                else:
                    mt = "random_forest"

                predictor = MLPredictor(
                    model_type=mt,
                    model_dir=str(self.model_dir),
                )
                if predictor.load(model_name):
                    self.models[symbol] = predictor
                    logger.info(f"Found model for {symbol} via scan: {model_name}")
                    return True

            # Also check subdirectories
            for subdir in model_dir.iterdir():
                if subdir.is_dir() and symbol_base.lower() in subdir.name.lower():
                    # Look for model files inside
                    for model_file in subdir.glob("*model*.pkl"):
                        model_name = subdir.name

                        if "xgboost" in model_name:
                            mt = "xgboost"
                        elif "gradient_boosting" in model_name:
                            mt = "gradient_boosting"
                        else:
                            mt = "random_forest"

                        predictor = MLPredictor(
                            model_type=mt,
                            model_dir=str(subdir),
                        )
                        # Try loading from subdirectory
                        if predictor.load("model"):
                            self.models[symbol] = predictor
                            logger.info(f"Found model for {symbol} in subdir: {subdir.name}")
                            return True

            return False
        except Exception as e:
            logger.debug(f"Scan for model failed: {e}")
            return False

    def set_regime(self, regime: str) -> None:
        """Set the current market regime for adaptive thresholds."""
        self._current_regime = regime.lower() if regime else None

    def get_regime_strategy(self, prices: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Get regime-specific strategy parameters from the Intelligent Brain.

        Returns dict with:
            - regime: current detected regime
            - position_size_multiplier: 0.25-1.5x normal size
            - stop_loss_pct: adjusted stop loss %
            - take_profit_pct: adjusted take profit %
            - strategy_type: e.g., "momentum", "mean_reversion"
            - confidence_adjustment: adjustment to apply to signal confidence
        """
        default_strategy = {
            "regime": self._current_regime or "unknown",
            "position_size_multiplier": 1.0,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
            "strategy_type": "trend_following",
            "confidence_adjustment": 0.0,
        }

        if not INTELLIGENT_BRAIN_AVAILABLE:
            return default_strategy

        try:
            brain = get_intelligent_brain()

            # Detect regime using price data if available
            if prices is not None and len(prices) > 0:
                closes = prices["close"].values
                regime_info = brain.regime_adapter.detect_regime(closes)

                # Update internal regime
                if regime_info.get("regime"):
                    self._current_regime = regime_info["regime"]

            # Get strategy for current regime
            strategy = brain.regime_adapter.get_strategy(self._current_regime)

            if strategy:
                return {
                    "regime": self._current_regime or "unknown",
                    "position_size_multiplier": strategy.position_size_multiplier,
                    "stop_loss_pct": strategy.stop_loss_pct,
                    "take_profit_pct": strategy.take_profit_pct,
                    "strategy_type": strategy.strategy_type,
                    "confidence_adjustment": strategy.confidence_adjustment,
                }

            return default_strategy

        except Exception as e:
            logger.debug(f"Failed to get regime strategy: {e}")
            return default_strategy

    def _get_adaptive_threshold(self, signal_direction: str = "BUY") -> float:
        """
        Get confidence threshold based on market regime.

        In trending markets, lower the threshold for trend-following signals.
        In volatile markets, require higher confidence.
        """
        if not self.regime_adaptive_threshold or not self._current_regime:
            return self.confidence_threshold

        regime = self._current_regime
        base_threshold = self.confidence_threshold

        # Regime-based adjustments
        regime_adjustments = {
            # Strong trends - lower threshold for trend signals
            "strong_bull": {"BUY": -0.08, "SELL": 0.05},
            "bull": {"BUY": -0.05, "SELL": 0.02},
            "strong_bear": {"BUY": 0.05, "SELL": -0.08},
            "bear": {"BUY": 0.02, "SELL": -0.05},
            "crash": {"BUY": 0.10, "SELL": -0.10},  # Favor shorts in crash
            # Volatile/uncertain - require higher confidence
            "volatile": {"BUY": 0.05, "SELL": 0.05},
            "high_vol": {"BUY": 0.05, "SELL": 0.05},
            # Sideways - neutral
            "sideways": {"BUY": 0.0, "SELL": 0.0},
            "unknown": {"BUY": 0.02, "SELL": 0.02},
        }

        adjustment = regime_adjustments.get(regime, {}).get(signal_direction, 0.0)
        adjusted = max(0.35, min(0.70, base_threshold + adjustment))

        return adjusted

    def _fetch_prices(self, symbol: str, period: str = "60d") -> Optional[pd.DataFrame]:
        """Fetch price data for a symbol."""
        try:
            # Convert symbol format for Yahoo Finance
            yf_symbol = symbol.replace("/", "-")
            if yf_symbol.endswith("-USDT"):
                yf_symbol = yf_symbol.replace("-USDT", "-USD")

            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(period=period, interval="1h")

            if df.empty:
                return self._price_cache.get(symbol)

            # Standardize columns
            df.columns = [c.lower() for c in df.columns]
            df = df[["open", "high", "low", "close", "volume"]].copy()
            df = df.dropna()

            # Cache the data
            self._price_cache[symbol] = df
            return df

        except Exception as e:
            logger.warning(f"Failed to fetch {symbol}: {e}")
            return self._price_cache.get(symbol)

    async def generate_signal(
        self, symbol: str, current_price: float
    ) -> Optional[Dict[str, Any]]:
        """
        Generate trading signal for a symbol.

        Returns dict with:
            - action: "BUY", "SELL", or None
            - confidence: float 0-1
            - reason: str explanation
            - price: current price
            - regime: current market regime
            - regime_strategy: dict with position_size_multiplier, stop_loss_pct, etc.
        """
        if not self._initialized:
            self.initialize([symbol])

        # Get price data
        df = self._fetch_prices(symbol)
        if df is None or len(df) < 50:
            return None

        # Get regime-specific strategy parameters from Intelligent Brain
        regime_strategy = self.get_regime_strategy(df)

        # Use ML model if available
        if symbol in self.models:
            signal = await self._ml_signal(symbol, df, current_price)
        else:
            # Fallback to technical analysis
            signal = await self._technical_signal(symbol, df, current_price)

        # Apply multi-timeframe filter
        if signal and self.use_mtf_filter:
            signal = self._apply_mtf_filter(signal, df)

        # Enrich signal with regime strategy info
        if signal:
            signal["regime"] = regime_strategy["regime"]
            signal["regime_strategy"] = regime_strategy

            # Apply regime-based confidence adjustment
            if regime_strategy.get("confidence_adjustment"):
                original_conf = signal.get("confidence", 0.5)
                adjusted_conf = min(0.95, max(0.1, original_conf + regime_strategy["confidence_adjustment"]))
                signal["confidence"] = adjusted_conf
                signal["regime_confidence_adjustment"] = regime_strategy["confidence_adjustment"]

        return signal

    def _apply_mtf_filter(
        self, signal: Dict[str, Any], df: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """Apply multi-timeframe trend filter to boost or reject signals."""
        try:
            action = signal.get("action", "")
            if action not in ("BUY", "SELL"):
                return signal

            # Map action to signal direction
            direction = "LONG" if action == "BUY" else "SHORT"

            # Load data into MTF analyzer
            self._mtf_analyzer.load_data("symbol", df)

            # Get HTF trend filter result
            should_take, conf_adj, reason = self._mtf_analyzer.get_htf_trend_filter(
                direction, htf_timeframe="4h"
            )

            original_confidence = signal.get("confidence", 0.5)

            if not should_take:
                if self.mtf_strict_mode:
                    # Reject the signal entirely in strict mode
                    logger.info(f"MTF filter rejected {action}: {reason}")
                    return None
                else:
                    # Reduce confidence for counter-trend signals
                    new_confidence = max(0.35, original_confidence + conf_adj)
                    signal["confidence"] = new_confidence
                    signal["mtf_filtered"] = True
                    signal["mtf_reason"] = reason
                    signal["mtf_adjustment"] = conf_adj

                    # Use adaptive threshold for the specific signal direction
                    effective_threshold = self._get_adaptive_threshold(action)
                    if new_confidence < effective_threshold:
                        logger.info(
                            f"MTF filter dropped {action} below threshold "
                            f"({new_confidence:.2f} < {effective_threshold:.2f}): {reason}"
                        )
                        return None
            else:
                # Boost confidence for trend-aligned signals
                new_confidence = min(0.95, original_confidence + conf_adj)
                signal["confidence"] = new_confidence
                signal["mtf_filtered"] = True
                signal["mtf_reason"] = reason
                signal["mtf_adjustment"] = conf_adj

            logger.debug(
                f"MTF filter: {action} {original_confidence:.2f} -> {signal['confidence']:.2f} ({reason})"
            )
            return signal

        except Exception as e:
            logger.warning(f"MTF filter error: {e}")
            return signal  # Return original signal on error

    async def _ml_signal(
        self, symbol: str, df: pd.DataFrame, current_price: float
    ) -> Optional[Dict[str, Any]]:
        """Generate signal using ML model with regime-adaptive thresholds."""
        try:
            prediction = self.models[symbol].predict(df)

            prob_long = prediction.probability_long
            prob_short = prediction.probability_short

            # Get regime-adaptive thresholds
            long_threshold = self._get_adaptive_threshold("BUY")
            short_threshold = self._get_adaptive_threshold("SELL")

            # Determine action based on probabilities with adaptive thresholds
            if prob_long > prob_short and prob_long > long_threshold:
                action = "BUY"
                confidence = prob_long
                threshold_used = long_threshold
            elif prob_short > prob_long and prob_short > short_threshold:
                action = "SELL"
                confidence = prob_short
                threshold_used = short_threshold
            else:
                # Log near-miss signals for debugging
                if prob_long > 0.4 or prob_short > 0.4:
                    logger.debug(
                        f"{symbol}: No signal (L:{prob_long:.2f}<{long_threshold:.2f}, "
                        f"S:{prob_short:.2f}<{short_threshold:.2f}, regime:{self._current_regime})"
                    )
                return None  # No clear signal

            regime_info = f" [regime:{self._current_regime}]" if self._current_regime else ""

            # Track prediction for performance analysis
            try:
                prediction_id = track_prediction(
                    model_type=self.model_type,
                    symbol=symbol,
                    prediction=action.lower(),  # "buy" or "sell"
                    confidence=confidence,
                    market_condition=self._current_regime or "unknown",
                    volatility=50.0,  # Default, can be enhanced
                    predicted_return=None
                )
            except Exception as track_err:
                logger.debug(f"Failed to track prediction: {track_err}")
                prediction_id = None

            return {
                "action": action,
                "confidence": confidence,
                "reason": f"ML {self.model_type}: {action} ({confidence:.1%}){regime_info}",
                "price": current_price,
                "prob_long": prob_long,
                "prob_short": prob_short,
                "threshold_used": threshold_used,
                "regime": self._current_regime,
                "prediction_id": prediction_id,  # For tracking outcome
            }

        except Exception as e:
            logger.warning(f"ML prediction failed for {symbol}: {e}")
            return await self._technical_signal(symbol, df, current_price)

    async def _technical_signal(
        self, symbol: str, df: pd.DataFrame, current_price: float
    ) -> Optional[Dict[str, Any]]:
        """Generate signal using technical analysis with multiple indicators."""
        try:
            close = df["close"]
            high = df["high"]
            low = df["low"]

            # === EMA ===
            ema_fast = close.ewm(span=12).mean()
            ema_slow = close.ewm(span=26).mean()

            # === RSI ===
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            # === MACD ===
            ema_12 = close.ewm(span=12).mean()
            ema_26 = close.ewm(span=26).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9).mean()
            macd_histogram = macd_line - signal_line

            # === Bollinger Bands ===
            bb_period = 20
            bb_std = 2
            bb_middle = close.rolling(window=bb_period).mean()
            bb_std_dev = close.rolling(window=bb_period).std()
            bb_upper = bb_middle + (bb_std_dev * bb_std)
            bb_lower = bb_middle - (bb_std_dev * bb_std)
            bb_width = (bb_upper - bb_lower) / bb_middle

            # === Momentum ===
            momentum = close.pct_change(periods=10) * 100

            # Get latest values
            latest_ema_fast = ema_fast.iloc[-1]
            latest_ema_slow = ema_slow.iloc[-1]
            prev_ema_fast = ema_fast.iloc[-2]
            prev_ema_slow = ema_slow.iloc[-2]
            latest_rsi = rsi.iloc[-1]
            latest_macd = macd_line.iloc[-1]
            latest_signal = signal_line.iloc[-1]
            prev_macd = macd_line.iloc[-2]
            prev_signal = signal_line.iloc[-2]
            latest_histogram = macd_histogram.iloc[-1]
            prev_histogram = macd_histogram.iloc[-2]
            latest_close = close.iloc[-1]
            latest_bb_upper = bb_upper.iloc[-1]
            latest_bb_lower = bb_lower.iloc[-1]
            latest_bb_middle = bb_middle.iloc[-1]
            latest_momentum = momentum.iloc[-1]

            # Scoring system
            buy_score = 0
            sell_score = 0
            reasons = []

            # === EMA Crossover (weight: 2) ===
            if latest_ema_fast > latest_ema_slow and prev_ema_fast <= prev_ema_slow:
                buy_score += 2
                reasons.append("EMA bullish cross")
            elif latest_ema_fast < latest_ema_slow and prev_ema_fast >= prev_ema_slow:
                sell_score += 2
                reasons.append("EMA bearish cross")
            elif latest_ema_fast > latest_ema_slow:
                buy_score += 0.5
            else:
                sell_score += 0.5

            # === MACD Crossover (weight: 2) ===
            if latest_macd > latest_signal and prev_macd <= prev_signal:
                buy_score += 2
                reasons.append("MACD bullish cross")
            elif latest_macd < latest_signal and prev_macd >= prev_signal:
                sell_score += 2
                reasons.append("MACD bearish cross")

            # MACD Histogram momentum
            if latest_histogram > 0 and latest_histogram > prev_histogram:
                buy_score += 1
                reasons.append("MACD histogram rising")
            elif latest_histogram < 0 and latest_histogram < prev_histogram:
                sell_score += 1
                reasons.append("MACD histogram falling")

            # === RSI (weight: 1.5) ===
            if latest_rsi < 30:
                buy_score += 1.5
                reasons.append(f"RSI oversold ({latest_rsi:.0f})")
            elif latest_rsi > 70:
                sell_score += 1.5
                reasons.append(f"RSI overbought ({latest_rsi:.0f})")
            elif latest_rsi < 45:
                buy_score += 0.5
            elif latest_rsi > 55:
                sell_score += 0.5

            # === Bollinger Bands (weight: 1.5) ===
            if latest_close <= latest_bb_lower:
                buy_score += 1.5
                reasons.append("Price at lower BB")
            elif latest_close >= latest_bb_upper:
                sell_score += 1.5
                reasons.append("Price at upper BB")
            elif latest_close < latest_bb_middle:
                buy_score += 0.3
            else:
                sell_score += 0.3

            # === Momentum (weight: 1) ===
            if latest_momentum > 2:
                buy_score += 1
                reasons.append(f"Strong momentum (+{latest_momentum:.1f}%)")
            elif latest_momentum < -2:
                sell_score += 1
                reasons.append(f"Weak momentum ({latest_momentum:.1f}%)")

            # Determine action based on score
            min_score = 2.5  # Minimum score to trigger a signal

            if buy_score >= min_score and buy_score > sell_score:
                action = "BUY"
                confidence = min(0.5 + (buy_score / 10), 0.85)
            elif sell_score >= min_score and sell_score > buy_score:
                action = "SELL"
                confidence = min(0.5 + (sell_score / 10), 0.85)
            else:
                return None  # No clear signal

            # Track technical prediction
            try:
                prediction_id = track_prediction(
                    model_type="technical_analysis",
                    symbol=symbol,
                    prediction=action.lower(),
                    confidence=confidence,
                    market_condition=self._current_regime or "unknown",
                    volatility=50.0,
                    predicted_return=None
                )
            except Exception as track_err:
                logger.debug(f"Failed to track TA prediction: {track_err}")
                prediction_id = None

            return {
                "action": action,
                "confidence": confidence,
                "reason": f"Technical: {', '.join(reasons[:3])}",
                "price": current_price,
                "prediction_id": prediction_id,
                "indicators": {
                    "rsi": latest_rsi,
                    "macd": latest_macd,
                    "macd_signal": latest_signal,
                    "bb_position": (latest_close - latest_bb_lower) / (latest_bb_upper - latest_bb_lower) if (latest_bb_upper - latest_bb_lower) > 0 else 0.5,
                    "momentum": latest_momentum,
                    "buy_score": buy_score,
                    "sell_score": sell_score,
                }
            }

        except Exception as e:
            logger.warning(f"Technical analysis failed for {symbol}: {e}")
            return None


def create_signal_generator(
    symbols: List[str],
    model_type: str = "gradient_boosting",
    confidence_threshold: float = 0.55,
    use_mtf_filter: bool = True,
    mtf_strict_mode: bool = False,
) -> MLSignalGenerator:
    """Factory function to create and initialize signal generator."""
    generator = MLSignalGenerator(
        model_type=model_type,
        confidence_threshold=confidence_threshold,
        use_mtf_filter=use_mtf_filter,
        mtf_strict_mode=mtf_strict_mode,
    )
    generator.initialize(symbols)
    return generator
