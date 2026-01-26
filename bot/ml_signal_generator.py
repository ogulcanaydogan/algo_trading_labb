"""
ML Signal Generator for Unified Trading Engine.

Provides ML-based signal generation that can be plugged into
the unified trading engine. Supports ensemble predictions for
higher accuracy (target: 80%+).
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

# Model monitoring for calibration and drift-aware gating
try:
    from bot.ml.model_monitor import get_model_monitor

    MODEL_MONITOR_AVAILABLE = True
except ImportError:
    MODEL_MONITOR_AVAILABLE = False

# Intelligent Brain for regime-based strategy adaptation
try:
    from bot.intelligence import get_intelligent_brain, RegimeAdapter

    INTELLIGENT_BRAIN_AVAILABLE = True
except ImportError:
    INTELLIGENT_BRAIN_AVAILABLE = False

# Ensemble predictor for higher accuracy
try:
    from bot.ml.ensemble_predictor import EnsemblePredictor, create_ensemble_predictor

    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False

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
        confidence_threshold: float = 0.55,  # Balanced: filters noise while allowing quality signals
        use_mtf_filter: bool = True,
        mtf_strict_mode: bool = False,  # Relaxed MTF filtering - allow signals with confidence penalty
        regime_adaptive_threshold: bool = True,  # Adjust threshold based on market regime
        use_ensemble: bool = True,  # Use ensemble predictor for higher accuracy
        ensemble_voting_strategy: str = "performance",  # "majority", "weighted", "performance"
    ):
        self.model_dir = model_dir
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.use_mtf_filter = use_mtf_filter
        self.mtf_strict_mode = mtf_strict_mode
        self.regime_adaptive_threshold = regime_adaptive_threshold
        self.use_ensemble = use_ensemble and ENSEMBLE_AVAILABLE
        self.ensemble_voting_strategy = ensemble_voting_strategy

        self.models: Dict[str, Any] = {}
        self.ensemble_predictors: Dict[
            str, EnsemblePredictor
        ] = {}  # Ensemble predictors per symbol
        self._price_cache: Dict[str, pd.DataFrame] = {}
        self._mtf_analyzer = MultiTimeframeAnalyzer()
        self._initialized = False
        self._current_regime: Optional[str] = None

        self._model_monitor = None
        self._monitor_enabled = MODEL_MONITOR_AVAILABLE
        self._monitor_check_interval_seconds = 300
        self._monitor_min_samples = 50
        self._last_monitor_check: Optional[datetime] = None
        self._last_monitor_summary: Dict[str, Any] = {}

        if self._monitor_enabled:
            try:
                self._model_monitor = get_model_monitor()
            except Exception as e:
                self._monitor_enabled = False
                logger.debug(f"Model monitor unavailable: {e}")

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
            symbol_base = symbol.replace("/", "_")

            # Try ensemble predictor first if enabled (for higher accuracy)
            if self.use_ensemble:
                try:
                    ensemble = create_ensemble_predictor(
                        symbol=symbol,
                        model_dir=self.model_dir,
                        voting_strategy=self.ensemble_voting_strategy,
                    )
                    if ensemble is not None:
                        self.ensemble_predictors[symbol] = ensemble
                        logger.info(
                            f"Loaded ENSEMBLE predictor for {symbol} (voting: {self.ensemble_voting_strategy})"
                        )
                        loaded = True
                except Exception as e:
                    logger.debug(f"Ensemble load failed for {symbol}: {e}")

            # Fall back to single model if ensemble not available
            if not loaded:
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
                            logger.info(f"Loaded single model for {symbol} (pattern: {pattern})")
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
        total_loaded = len(self.ensemble_predictors) + len(self.models)
        logger.info(
            f"Signal generator initialized: {len(self.ensemble_predictors)} ensemble, {len(self.models)} single models"
        )
        return total_loaded > 0

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

    def _get_monitor_summary(self) -> Dict[str, Any]:
        """Get cached model monitoring summary."""
        if not self._monitor_enabled or not self._model_monitor:
            return {}

        now = datetime.now()
        if (
            self._last_monitor_check
            and self._last_monitor_summary
            and (now - self._last_monitor_check).total_seconds()
            < self._monitor_check_interval_seconds
        ):
            return self._last_monitor_summary

        try:
            summary = self._model_monitor.get_monitoring_summary()
            self._last_monitor_summary = summary or {}
            self._last_monitor_check = now
            return self._last_monitor_summary
        except Exception as e:
            logger.debug(f"Model monitor summary failed: {e}")
            return {}

    def _apply_calibration(self, raw_confidence: float) -> float:
        """Calibrate raw prediction confidence using model monitor."""
        if not self._monitor_enabled or not self._model_monitor:
            return raw_confidence

        try:
            calibrated = self._model_monitor.get_calibrated_prediction(raw_confidence)
            return float(calibrated)
        except Exception as e:
            logger.debug(f"Confidence calibration failed: {e}")
            return raw_confidence

    def _apply_monitor_gates(
        self,
        confidence: float,
        action: str,
    ) -> tuple[float, float, List[str]]:
        """Apply drift/performance-aware adjustments to confidence and thresholds."""
        summary = self._get_monitor_summary()
        if not summary:
            return confidence, 0.0, []

        notes: List[str] = []
        threshold_boost = 0.0
        confidence_penalty = 0.0

        drift_severity = summary.get("drift_severity", "none")
        should_retrain = summary.get("should_retrain", False)
        performance = summary.get("performance", {})
        window_size = performance.get("window_size", 0) or 0
        win_rate = performance.get("win_rate", 0) or 0
        trend = summary.get("performance_trend", {}).get("trend", "unknown")

        if drift_severity in ("critical", "high"):
            confidence_penalty += 0.08
            threshold_boost += 0.05
            notes.append(f"drift:{drift_severity}")
        elif drift_severity == "moderate":
            confidence_penalty += 0.04
            threshold_boost += 0.03
            notes.append("drift:moderate")
        elif drift_severity == "low":
            confidence_penalty += 0.02
            threshold_boost += 0.01
            notes.append("drift:low")

        if should_retrain:
            confidence_penalty += 0.05
            threshold_boost += 0.03
            notes.append("retrain:recommended")

        if window_size >= self._monitor_min_samples and win_rate < 0.45:
            confidence_penalty += 0.03
            threshold_boost += 0.02
            notes.append(f"win_rate:{win_rate:.2f}")

        if trend == "degrading":
            confidence_penalty += 0.02
            threshold_boost += 0.01
            notes.append("trend:degrading")

        adjusted_confidence = max(0.05, min(0.95, confidence - confidence_penalty))
        return adjusted_confidence, threshold_boost, notes

    @staticmethod
    def _classify_trend(trend_value: float) -> str:
        """Classify trend direction from normalized trend value."""
        if trend_value > 0.001:
            return "up"
        if trend_value < -0.001:
            return "down"
        return "neutral"

    @staticmethod
    def _classify_volatility(volatility_value: float) -> str:
        """Classify volatility regime from rolling return std."""
        if volatility_value >= 0.05:
            return "high"
        if volatility_value >= 0.02:
            return "normal"
        return "low"

    def _build_monitor_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Build lightweight feature set for monitoring and learning."""
        close = df["close"]
        volume = df["volume"] if "volume" in df.columns else None

        returns_1 = close.pct_change().iloc[-1]
        returns_4 = close.pct_change(4).iloc[-1] if len(close) > 4 else returns_1
        volatility = close.pct_change().rolling(24).std().iloc[-1]
        if np.isnan(volatility):
            volatility = close.pct_change().rolling(6).std().iloc[-1]

        ema_fast = close.ewm(span=12).mean().iloc[-1]
        ema_slow = close.ewm(span=26).mean().iloc[-1]
        last_close = close.iloc[-1]
        ema_trend = (ema_fast - ema_slow) / last_close if last_close else 0.0

        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_value = rsi.iloc[-1] if len(rsi) else 50.0

        bb_middle = close.rolling(window=20).mean().iloc[-1]
        bb_std = close.rolling(window=20).std().iloc[-1]
        bb_width = (4 * bb_std / bb_middle) if bb_middle and bb_std else 0.0

        volume_ratio = 1.0
        if volume is not None and len(volume) >= 20:
            vol_avg = volume.rolling(window=20).mean().iloc[-1]
            if vol_avg:
                volume_ratio = volume.iloc[-1] / vol_avg

        features = {
            "return_1": returns_1,
            "return_4": returns_4,
            "volatility_24": volatility,
            "ema_trend": ema_trend,
            "rsi": rsi_value,
            "bb_width": bb_width,
            "volume_ratio": volume_ratio,
        }

        return {
            key: float(np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0))
            for key, val in features.items()
        }

    def _detect_asset_type(self, symbol: str) -> str:
        """Detect asset type from symbol."""
        symbol_upper = symbol.upper()

        # Index patterns
        if any(idx in symbol_upper for idx in ["SPX500", "NAS100", "US30", "UK100", "DE30"]):
            return "index"

        # Commodity patterns
        if any(comm in symbol_upper for comm in ["XAU", "XAG", "WTICO", "BCO", "NATGAS", "XCU"]):
            return "commodity"

        # Forex patterns (currency pairs without crypto)
        forex_currencies = ["EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]
        if "/" in symbol and "USDT" not in symbol_upper:
            parts = symbol_upper.split("/")
            if len(parts) == 2 and any(curr in parts[0] for curr in forex_currencies):
                return "forex"

        # Default to crypto
        return "crypto"

    def _fetch_prices(self, symbol: str, period: str = "60d") -> Optional[pd.DataFrame]:
        """Fetch price data for a symbol using appropriate data source."""
        try:
            asset_type = self._detect_asset_type(symbol)

            # Use OANDA for forex, commodities, and indices
            if asset_type in ["forex", "commodity", "index"]:
                return self._fetch_oanda_prices(symbol)

            # Use Yahoo Finance for crypto
            return self._fetch_yfinance_prices(symbol, period)

        except Exception as e:
            logger.warning(f"Failed to fetch {symbol}: {e}")
            return self._price_cache.get(symbol)

    def _fetch_oanda_prices(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch price data from OANDA for forex/commodities/indices."""
        try:
            import os
            import requests

            api_key = os.getenv("OANDA_API_KEY")
            account_id = os.getenv("OANDA_ACCOUNT_ID")
            environment = os.getenv("OANDA_ENVIRONMENT", "live")

            if not api_key or not account_id:
                logger.warning(f"OANDA credentials not configured for {symbol}")
                return self._price_cache.get(symbol)

            # Convert symbol format for OANDA (EUR/USD -> EUR_USD)
            oanda_symbol = symbol.replace("/", "_")

            # Select API endpoint
            if environment == "practice":
                base_url = "https://api-fxpractice.oanda.com"
            else:
                base_url = "https://api-fxtrade.oanda.com"

            # Fetch last 500 hourly candles
            url = f"{base_url}/v3/instruments/{oanda_symbol}/candles"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            params = {
                "granularity": "H1",
                "count": 500,
                "price": "M",  # Mid prices
            }

            response = requests.get(url, headers=headers, params=params, timeout=30)

            if response.status_code != 200:
                logger.warning(f"OANDA API error for {symbol}: {response.status_code}")
                return self._price_cache.get(symbol)

            data = response.json()
            candles = data.get("candles", [])

            if not candles:
                return self._price_cache.get(symbol)

            # Convert to DataFrame
            rows = []
            for candle in candles:
                if candle.get("complete", False):
                    mid = candle.get("mid", {})
                    rows.append(
                        {
                            "timestamp": candle["time"],
                            "open": float(mid.get("o", 0)),
                            "high": float(mid.get("h", 0)),
                            "low": float(mid.get("l", 0)),
                            "close": float(mid.get("c", 0)),
                            "volume": int(candle.get("volume", 0)),
                        }
                    )

            if not rows:
                return self._price_cache.get(symbol)

            df = pd.DataFrame(rows)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
            df = df[["open", "high", "low", "close", "volume"]].copy()

            # Cache the data
            self._price_cache[symbol] = df
            logger.debug(f"Fetched {len(df)} candles from OANDA for {symbol}")
            return df

        except Exception as e:
            logger.warning(f"OANDA fetch failed for {symbol}: {e}")
            return self._price_cache.get(symbol)

    def _fetch_yfinance_prices(self, symbol: str, period: str = "60d") -> Optional[pd.DataFrame]:
        """Fetch price data from Yahoo Finance for crypto."""
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
            logger.warning(f"YFinance fetch failed for {symbol}: {e}")
            return self._price_cache.get(symbol)

    async def generate_signal(self, symbol: str, current_price: float) -> Optional[Dict[str, Any]]:
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

        # Use ensemble predictor if available (higher accuracy)
        if symbol in self.ensemble_predictors:
            signal = await self._ensemble_signal(symbol, df, current_price)
        # Fall back to single ML model
        elif symbol in self.models:
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
                adjusted_conf = min(
                    0.95, max(0.1, original_conf + regime_strategy["confidence_adjustment"])
                )
                signal["confidence"] = adjusted_conf
                signal["regime_confidence_adjustment"] = regime_strategy["confidence_adjustment"]

        return signal

    def _apply_mtf_filter(
        self, signal: Dict[str, Any], df: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """Apply multi-timeframe trend filter to boost or reject signals."""
        try:
            action = signal.get("action", "")
            if action not in ("BUY", "SHORT", "SELL"):
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
                    # Paper mode: minimal penalty for counter-trend, never reject
                    # Just note the MTF disagreement but allow all signals through
                    capped_adj = max(-0.05, conf_adj)  # Very small penalty
                    new_confidence = max(0.45, original_confidence + capped_adj)
                    signal["confidence"] = new_confidence
                    signal["mtf_filtered"] = True
                    signal["mtf_reason"] = reason
                    signal["mtf_adjustment"] = capped_adj
                    # No longer rejecting signals - let them through for paper trade accumulation
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

    async def _ensemble_signal(
        self, symbol: str, df: pd.DataFrame, current_price: float
    ) -> Optional[Dict[str, Any]]:
        """Generate signal using ensemble predictor for higher accuracy."""
        try:
            ensemble = self.ensemble_predictors[symbol]

            # Prepare features from DataFrame (use last row)
            # The ensemble expects the same features used during training
            features = self._extract_features(df)
            if features is None:
                logger.warning(f"Failed to extract features for {symbol}")
                return await self._ml_signal(symbol, df, current_price)

            # Get ensemble prediction
            prediction, confidence, details = ensemble.predict(features)

            # Map prediction to action
            # prediction: 1 (long), 0 (flat), -1 (short)
            if prediction == 1:
                action = "BUY"
            elif prediction == -1:
                action = "SHORT"  # Changed from SELL to open short positions
            else:
                return None  # Flat - no signal

            raw_confidence = confidence
            calibrated_confidence = self._apply_calibration(raw_confidence)
            adjusted_confidence, threshold_adj, monitor_notes = self._apply_monitor_gates(
                calibrated_confidence, action
            )

            threshold = self._get_adaptive_threshold(action) + threshold_adj
            threshold = max(0.35, min(0.85, threshold))

            # Check against threshold
            if adjusted_confidence < threshold:
                logger.debug(
                    f"{symbol}: Ensemble {action} rejected ({adjusted_confidence:.2f} < {threshold:.2f})"
                )
                return None

            regime_info = f" [regime:{self._current_regime}]" if self._current_regime else ""
            num_models = details.get("num_models", 0)

            monitor_features = self._build_monitor_features(df)
            trend = self._classify_trend(monitor_features.get("ema_trend", 0.0))
            volatility_regime = self._classify_volatility(
                monitor_features.get("volatility_24", 0.0)
            )

            # Track prediction for performance analysis
            try:
                prediction_id = track_prediction(
                    model_type="ensemble",
                    symbol=symbol,
                    prediction=action.lower(),
                    confidence=adjusted_confidence,
                    market_condition=self._current_regime or "unknown",
                    volatility=50.0,
                    predicted_return=None,
                )
            except Exception as track_err:
                logger.debug(f"Failed to track ensemble prediction: {track_err}")
                prediction_id = None

            return {
                "action": action,
                "confidence": adjusted_confidence,
                "raw_confidence": raw_confidence,
                "calibrated_confidence": calibrated_confidence,
                "monitor_threshold_adjustment": threshold_adj,
                "monitor_notes": monitor_notes,
                "reason": f"Ensemble ({num_models} models): {action} ({adjusted_confidence:.1%}){regime_info}",
                "price": current_price,
                "threshold_used": threshold,
                "regime": self._current_regime,
                "prediction_id": prediction_id,
                "ensemble_details": details,
                "model_type": "ensemble",
                "monitor_features": monitor_features,
                "rsi": monitor_features.get("rsi", 50.0),
                "trend": trend,
                "volatility": volatility_regime,
            }

        except Exception as e:
            logger.warning(f"Ensemble prediction failed for {symbol}: {e}")
            # Fall back to single model
            if symbol in self.models:
                return await self._ml_signal(symbol, df, current_price)
            return await self._technical_signal(symbol, df, current_price)

    def _extract_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract features matching the improved_training.py feature set (43 features)."""
        try:
            import json

            # Use the same feature engineering as improved_training.py
            df_features = self._engineer_training_features(df.copy())

            if len(df_features) == 0:
                logger.warning("Feature engineering returned empty DataFrame")
                return None

            # The 43 features from improved_training.py in exact order
            feature_names = [
                "ema_7", "ema_7_dist", "ema_14", "ema_14_dist", "ema_21", "ema_21_dist",
                "ema_50", "ema_50_dist", "ema_100", "ema_100_dist", "ema_200", "ema_200_dist",
                "rsi_7", "rsi_14", "rsi_28",
                "macd", "macd_signal", "macd_hist",
                "bb_20_mid", "bb_20_std", "bb_20_upper", "bb_20_lower", "bb_20_position",
                "bb_50_mid", "bb_50_std", "bb_50_upper", "bb_50_lower", "bb_50_position",
                "roc_3", "momentum_3", "roc_5", "momentum_5", "roc_10", "momentum_10", "roc_20", "momentum_20",
                "volume_sma_20", "volume_ratio", "volume_roc",
                "atr_14", "volatility_20", "high_low_ratio", "close_open_ratio"
            ]

            # Build feature array
            features = []
            for feat_name in feature_names:
                if feat_name in df_features.columns:
                    val = df_features[feat_name].iloc[-1]
                else:
                    val = 0.0
                features.append(val)

            X = np.array(features).reshape(1, -1)

            # Sanitize: replace inf/-inf with NaN, then fill with 0
            X = np.where(np.isinf(X), np.nan, X)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            logger.debug(f"Extracted {X.shape[1]} features for ensemble prediction")
            return X

        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            return None

    def _engineer_training_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features matching improved_training.py exactly.

        This ensures inference uses the same 43 features as training.
        """
        # Multiple timeframe EMAs
        for period in [7, 14, 21, 50, 100, 200]:
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            df[f'ema_{period}_dist'] = (df['close'] - df[f'ema_{period}']) / df[f'ema_{period}'] * 100

        # RSI features (manual calculation)
        for period in [7, 14, 28]:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = -delta.where(delta < 0, 0).rolling(period).mean()
            rs = gain / (loss + 1e-10)
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Bollinger Band features
        for period in [20, 50]:
            rolling = df['close'].rolling(period)
            df[f'bb_{period}_mid'] = rolling.mean()
            df[f'bb_{period}_std'] = rolling.std()
            df[f'bb_{period}_upper'] = df[f'bb_{period}_mid'] + 2 * df[f'bb_{period}_std']
            df[f'bb_{period}_lower'] = df[f'bb_{period}_mid'] - 2 * df[f'bb_{period}_std']
            df[f'bb_{period}_position'] = (df['close'] - df[f'bb_{period}_lower']) / (df[f'bb_{period}_upper'] - df[f'bb_{period}_lower'] + 1e-10)

        # Price momentum features
        for period in [3, 5, 10, 20]:
            df[f'roc_{period}'] = df['close'].pct_change(period) * 100
            df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)

        # Volume features
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_sma_20'] + 1e-10)
        df['volume_roc'] = df['volume'].pct_change(5) * 100

        # Volatility features
        df['atr_14'] = df['high'].rolling(14).max() - df['low'].rolling(14).min()
        df['volatility_20'] = df['close'].pct_change().rolling(20).std() * 100

        # Price action
        df['high_low_ratio'] = (df['high'] - df['low']) / (df['close'] + 1e-10)
        df['close_open_ratio'] = (df['close'] - df['open']) / (df['open'] + 1e-10)

        return df.dropna()

    async def _ml_signal(
        self, symbol: str, df: pd.DataFrame, current_price: float
    ) -> Optional[Dict[str, Any]]:
        """Generate signal using ML model with regime-adaptive thresholds."""
        try:
            prediction = self.models[symbol].predict(df)

            prob_long = prediction.probability_long
            prob_short = prediction.probability_short

            calibrated_long = self._apply_calibration(prob_long)
            calibrated_short = self._apply_calibration(prob_short)

            # Get regime-adaptive thresholds
            long_threshold = self._get_adaptive_threshold("BUY")
            short_threshold = self._get_adaptive_threshold("SELL")

            # Determine action based on probabilities with adaptive thresholds
            if calibrated_long > calibrated_short and calibrated_long > long_threshold:
                action = "BUY"
                confidence = calibrated_long
                threshold_used = long_threshold
            elif calibrated_short > calibrated_long and calibrated_short > short_threshold:
                action = "SHORT"  # Changed from SELL to open short positions
                confidence = calibrated_short
                threshold_used = short_threshold
            else:
                # Log near-miss signals for debugging
                if calibrated_long > 0.4 or calibrated_short > 0.4:
                    logger.debug(
                        f"{symbol}: No signal (L:{calibrated_long:.2f}<{long_threshold:.2f}, "
                        f"S:{calibrated_short:.2f}<{short_threshold:.2f}, regime:{self._current_regime})"
                    )
                return None  # No clear signal

            regime_info = f" [regime:{self._current_regime}]" if self._current_regime else ""

            adjusted_confidence, threshold_adj, monitor_notes = self._apply_monitor_gates(
                confidence, action
            )
            threshold_used = max(0.35, min(0.85, threshold_used + threshold_adj))
            if adjusted_confidence < threshold_used:
                logger.debug(
                    f"{symbol}: ML {action} rejected ({adjusted_confidence:.2f} < {threshold_used:.2f})"
                )
                return None

            monitor_features = self._build_monitor_features(df)
            trend = self._classify_trend(monitor_features.get("ema_trend", 0.0))
            volatility_regime = self._classify_volatility(
                monitor_features.get("volatility_24", 0.0)
            )

            # Track prediction for performance analysis
            try:
                prediction_id = track_prediction(
                    model_type=self.model_type,
                    symbol=symbol,
                    prediction=action.lower(),  # "buy" or "sell"
                    confidence=adjusted_confidence,
                    market_condition=self._current_regime or "unknown",
                    volatility=50.0,  # Default, can be enhanced
                    predicted_return=None,
                )
            except Exception as track_err:
                logger.debug(f"Failed to track prediction: {track_err}")
                prediction_id = None

            return {
                "action": action,
                "confidence": adjusted_confidence,
                "raw_confidence": prob_long if action == "BUY" else prob_short,
                "calibrated_confidence": confidence,
                "monitor_threshold_adjustment": threshold_adj,
                "monitor_notes": monitor_notes,
                "reason": f"ML {self.model_type}: {action} ({adjusted_confidence:.1%}){regime_info}",
                "price": current_price,
                "prob_long": prob_long,
                "prob_short": prob_short,
                "prob_long_calibrated": calibrated_long,
                "prob_short_calibrated": calibrated_short,
                "threshold_used": threshold_used,
                "regime": self._current_regime,
                "prediction_id": prediction_id,  # For tracking outcome
                "model_type": self.model_type,
                "monitor_features": monitor_features,
                "rsi": monitor_features.get("rsi", 50.0),
                "trend": trend,
                "volatility": volatility_regime,
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
                action = "SHORT"  # Changed from SELL to open short positions
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
                    predicted_return=None,
                )
            except Exception as track_err:
                logger.debug(f"Failed to track TA prediction: {track_err}")
                prediction_id = None

            monitor_features = self._build_monitor_features(df)
            trend = self._classify_trend(monitor_features.get("ema_trend", 0.0))
            volatility_regime = self._classify_volatility(
                monitor_features.get("volatility_24", 0.0)
            )

            return {
                "action": action,
                "confidence": confidence,
                "reason": f"Technical: {', '.join(reasons[:3])}",
                "price": current_price,
                "prediction_id": prediction_id,
                "model_type": "technical_analysis",
                "monitor_features": monitor_features,
                "rsi": latest_rsi,
                "trend": trend,
                "volatility": volatility_regime,
                "indicators": {
                    "rsi": latest_rsi,
                    "macd": latest_macd,
                    "macd_signal": latest_signal,
                    "bb_position": (latest_close - latest_bb_lower)
                    / (latest_bb_upper - latest_bb_lower)
                    if (latest_bb_upper - latest_bb_lower) > 0
                    else 0.5,
                    "momentum": latest_momentum,
                    "buy_score": buy_score,
                    "sell_score": sell_score,
                },
            }

        except Exception as e:
            logger.warning(f"Technical analysis failed for {symbol}: {e}")
            return None


def create_signal_generator(
    symbols: List[str],
    model_type: str = "gradient_boosting",
    confidence_threshold: float = 0.65,
    use_mtf_filter: bool = True,
    mtf_strict_mode: bool = True,
    use_ensemble: bool = True,
    ensemble_voting_strategy: str = "performance",
) -> MLSignalGenerator:
    """Factory function to create and initialize signal generator.

    Args:
        symbols: List of symbols to trade
        model_type: Fallback model type if ensemble unavailable
        confidence_threshold: Minimum confidence to generate signal (0.65 recommended for 60%+ win rate)
        use_mtf_filter: Enable multi-timeframe trend filtering
        mtf_strict_mode: Reject signals that don't align with higher timeframes (True improves quality)
        use_ensemble: Use ensemble predictor for higher accuracy (recommended)
        ensemble_voting_strategy: "majority", "weighted", or "performance" (recommended)
    """
    generator = MLSignalGenerator(
        model_type=model_type,
        confidence_threshold=confidence_threshold,
        use_mtf_filter=use_mtf_filter,
        mtf_strict_mode=mtf_strict_mode,
        use_ensemble=use_ensemble,
        ensemble_voting_strategy=ensemble_voting_strategy,
    )
    generator.initialize(symbols)
    return generator
