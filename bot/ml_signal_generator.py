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
        confidence_threshold: float = 0.55,
    ):
        self.model_dir = model_dir
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold

        self.models: Dict[str, Any] = {}
        self._price_cache: Dict[str, pd.DataFrame] = {}
        self._initialized = False

    def initialize(self, symbols: List[str]) -> bool:
        """Initialize models for given symbols."""
        try:
            from bot.ml.predictor import MLPredictor
        except ImportError:
            logger.warning("MLPredictor not available, using fallback signals")
            self._initialized = True
            return True

        for symbol in symbols:
            try:
                model_name = f"{symbol.replace('/', '_')}_{self.model_type}"
                predictor = MLPredictor(
                    model_type=self.model_type,
                    model_dir=str(self.model_dir),
                )
                if predictor.load(model_name):
                    self.models[symbol] = predictor
                    logger.info(f"Loaded model for {symbol}")
                else:
                    logger.warning(f"No model found for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to load model for {symbol}: {e}")

        self._initialized = True
        return len(self.models) > 0

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
        """
        if not self._initialized:
            self.initialize([symbol])

        # Get price data
        df = self._fetch_prices(symbol)
        if df is None or len(df) < 50:
            return None

        # Use ML model if available
        if symbol in self.models:
            return await self._ml_signal(symbol, df, current_price)
        else:
            # Fallback to technical analysis
            return await self._technical_signal(symbol, df, current_price)

    async def _ml_signal(
        self, symbol: str, df: pd.DataFrame, current_price: float
    ) -> Optional[Dict[str, Any]]:
        """Generate signal using ML model."""
        try:
            prediction = self.models[symbol].predict(df)

            prob_long = prediction.probability_long
            prob_short = prediction.probability_short

            # Determine action based on probabilities
            if prob_long > prob_short and prob_long > self.confidence_threshold:
                action = "BUY"
                confidence = prob_long
            elif prob_short > prob_long and prob_short > self.confidence_threshold:
                action = "SELL"
                confidence = prob_short
            else:
                return None  # No clear signal

            return {
                "action": action,
                "confidence": confidence,
                "reason": f"ML {self.model_type}: {action} ({confidence:.1%})",
                "price": current_price,
                "prob_long": prob_long,
                "prob_short": prob_short,
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

            return {
                "action": action,
                "confidence": confidence,
                "reason": f"Technical: {', '.join(reasons[:3])}",
                "price": current_price,
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
) -> MLSignalGenerator:
    """Factory function to create and initialize signal generator."""
    generator = MLSignalGenerator(
        model_type=model_type,
        confidence_threshold=confidence_threshold,
    )
    generator.initialize(symbols)
    return generator
