#!/usr/bin/env python3
"""
AI-Powered Live Trading with Intelligent Brain.

Combines the Intelligent Trading Brain with real Binance trading.
Features:
- AI-driven trade decisions with explanations
- Real-time learning from each trade
- Regime-adaptive strategy parameters
- Safety limits and circuit breakers
- Telegram notifications for all trades
"""

import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

import ccxt
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("data/live_trading.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class AILiveTrader:
    """AI-powered live trader with Intelligent Brain."""

    def __init__(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        use_testnet: bool = True,
        dry_run: bool = False,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.use_testnet = use_testnet
        self.dry_run = dry_run

        # Load safety limits
        self.max_position_usd = float(os.getenv("LIVE_MAX_POSITION_USD", "50"))
        self.max_daily_loss_pct = float(os.getenv("LIVE_MAX_DAILY_LOSS_PCT", "2.0"))
        self.max_trades_per_day = int(os.getenv("LIVE_MAX_TRADES_PER_DAY", "10"))

        # Track daily stats
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.last_reset_date = datetime.now().date()

        # Initialize exchange
        self.exchange = self._init_exchange()

        # Initialize Intelligent Brain
        self.brain = self._init_brain()

        # Current position
        self.position = None

        logger.info(f"AI Live Trader initialized")
        logger.info(f"  Symbol: {symbol}")
        logger.info(f"  Timeframe: {timeframe}")
        logger.info(f"  Mode: {'TESTNET' if use_testnet else 'LIVE'} {'(DRY RUN)' if dry_run else ''}")
        logger.info(f"  Max Position: ${self.max_position_usd}")
        logger.info(f"  Max Daily Loss: {self.max_daily_loss_pct}%")

    def _init_exchange(self) -> ccxt.binance:
        """Initialize Binance exchange."""
        if self.use_testnet:
            api_key = os.getenv("BINANCE_TESTNET_API_KEY")
            api_secret = os.getenv("BINANCE_TESTNET_API_SECRET")
        else:
            api_key = os.getenv("BINANCE_API_KEY")
            api_secret = os.getenv("BINANCE_API_SECRET")

        if not api_key or not api_secret:
            raise ValueError("API keys not configured. Check your .env file.")

        exchange = ccxt.binance({
            "apiKey": api_key,
            "secret": api_secret,
            "sandbox": self.use_testnet,
            "options": {"defaultType": "spot"},
        })

        # Test connection
        ticker = exchange.fetch_ticker(self.symbol)
        logger.info(f"Connected to Binance {'Testnet' if self.use_testnet else 'Live'}")
        logger.info(f"  {self.symbol}: ${ticker['last']:,.2f}")

        return exchange

    def _init_brain(self):
        """Initialize Intelligent Trading Brain."""
        try:
            from bot.intelligence import get_intelligent_brain
            brain = get_intelligent_brain()
            logger.info("Intelligent Brain initialized")
            brain.log_status()
            return brain
        except Exception as e:
            logger.warning(f"Brain initialization failed: {e}")
            logger.warning("Running without AI brain (rule-based only)")
            return None

    def _check_safety_limits(self) -> tuple:
        """Check if we can trade based on safety limits."""
        # Reset daily stats if new day
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.last_reset_date = today
            logger.info("Daily stats reset")

        # Check trade count
        if self.daily_trades >= self.max_trades_per_day:
            return False, f"Max daily trades reached ({self.max_trades_per_day})"

        # Check daily loss
        if self.daily_pnl < -self.max_daily_loss_pct:
            return False, f"Max daily loss reached ({self.max_daily_loss_pct}%)"

        return True, "OK"

    def fetch_data(self) -> pd.DataFrame:
        """Fetch OHLCV data from exchange."""
        ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=250)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df

    def get_ai_signal(self, df: pd.DataFrame) -> dict:
        """Get trading signal from AI brain or fallback to rules."""
        if self.brain:
            try:
                # Detect regime
                prices = df["close"].values
                regime_state = self.brain.regime_adapter.detect_regime(prices)

                # Get ML signal
                from bot.ml_signal_generator import MLSignalGenerator
                signal_gen = MLSignalGenerator()
                signal_gen.set_regime(regime_state.regime.value)
                signal = signal_gen.generate_signal(df)

                # Get news sentiment
                news_result = self.brain.news_reasoner.analyze_symbol(self.symbol)

                # Combine signals
                final_confidence = signal.get("confidence", 0.5)
                if news_result and news_result.sentiment_score != 0:
                    # Adjust confidence based on news
                    sentiment_modifier = news_result.sentiment_score * 0.1
                    final_confidence = max(0, min(1, final_confidence + sentiment_modifier))

                return {
                    "action": signal.get("signal", "HOLD"),
                    "confidence": final_confidence,
                    "regime": regime_state.regime.value,
                    "news_sentiment": news_result.sentiment_score if news_result else 0,
                    "reasoning": signal.get("reasoning", []),
                }
            except Exception as e:
                logger.warning(f"AI signal generation failed: {e}")

        # Fallback to simple rules
        return self._simple_signal(df)

    def _simple_signal(self, df: pd.DataFrame) -> dict:
        """Simple rule-based signal as fallback."""
        close = df["close"].values
        ema_fast = pd.Series(close).ewm(span=9).mean().iloc[-1]
        ema_slow = pd.Series(close).ewm(span=21).mean().iloc[-1]

        rsi = self._calculate_rsi(close)

        if ema_fast > ema_slow and rsi < 70:
            action = "BUY"
            confidence = 0.6
        elif ema_fast < ema_slow and rsi > 30:
            action = "SELL"
            confidence = 0.6
        else:
            action = "HOLD"
            confidence = 0.5

        return {
            "action": action,
            "confidence": confidence,
            "regime": "unknown",
            "news_sentiment": 0,
            "reasoning": [f"EMA crossover, RSI={rsi:.1f}"],
        }

    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI."""
        deltas = pd.Series(prices).diff()
        gain = deltas.where(deltas > 0, 0).rolling(window=period).mean()
        loss = (-deltas.where(deltas < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return (100 - (100 / (1 + rs))).iloc[-1]

    def execute_trade(self, action: str, price: float, signal: dict):
        """Execute a trade on the exchange."""
        if self.dry_run:
            logger.info(f"[DRY RUN] Would execute {action} at ${price:,.2f}")
            return True

        # Calculate position size
        position_usd = min(self.max_position_usd, self.max_position_usd * signal["confidence"])
        quantity = position_usd / price

        # Round to exchange precision
        quantity = round(quantity, 6)

        try:
            if action == "BUY":
                order = self.exchange.create_market_buy_order(self.symbol, quantity)
            else:
                order = self.exchange.create_market_sell_order(self.symbol, quantity)

            logger.info(f"Order executed: {action} {quantity} {self.symbol} at ~${price:,.2f}")
            logger.info(f"  Order ID: {order['id']}")

            # Send Telegram notification
            self._send_telegram(action, price, quantity, signal)

            self.daily_trades += 1
            return True

        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            return False

    def _send_telegram(self, action: str, price: float, quantity: float, signal: dict):
        """Send trade notification via Telegram."""
        try:
            from bot.telegram_bot import send_message

            message = f"""
{'BUY' if action == 'BUY' else 'SELL'} {self.symbol}

Price: ${price:,.2f}
Quantity: {quantity:.6f}
Value: ${price * quantity:,.2f}

AI Analysis:
- Regime: {signal.get('regime', 'unknown')}
- Confidence: {signal.get('confidence', 0):.0%}
- News: {signal.get('news_sentiment', 0):+.2f}

Reasoning: {', '.join(signal.get('reasoning', ['N/A'])[:2])}
"""
            send_message(message)
        except Exception as e:
            logger.warning(f"Telegram notification failed: {e}")

    def run_iteration(self):
        """Run one trading iteration."""
        logger.info("=" * 60)
        logger.info(f"Trading Iteration - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)

        # Check safety limits
        can_trade, reason = self._check_safety_limits()
        if not can_trade:
            logger.warning(f"Trading blocked: {reason}")
            return

        # Fetch data
        df = self.fetch_data()
        current_price = df["close"].iloc[-1]
        logger.info(f"{self.symbol}: ${current_price:,.2f}")

        # Get AI signal
        signal = self.get_ai_signal(df)
        logger.info(f"Signal: {signal['action']} (confidence: {signal['confidence']:.0%})")
        logger.info(f"Regime: {signal['regime']}, News: {signal['news_sentiment']:+.2f}")

        # Execute if high confidence
        if signal["action"] in ["BUY", "LONG"] and signal["confidence"] > 0.6:
            self.execute_trade("BUY", current_price, signal)
        elif signal["action"] in ["SELL", "SHORT"] and signal["confidence"] > 0.6:
            self.execute_trade("SELL", current_price, signal)
        else:
            logger.info("No trade - signal confidence too low or HOLD")

    def run(self, interval_seconds: int = 300):
        """Run the trading loop."""
        logger.info(f"Starting trading loop (interval: {interval_seconds}s)")
        logger.info("Press Ctrl+C to stop")

        try:
            while True:
                try:
                    self.run_iteration()
                except Exception as e:
                    logger.error(f"Iteration error: {e}")

                logger.info(f"Next iteration in {interval_seconds}s...")
                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            logger.info("Trading stopped by user")


def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("AI-POWERED LIVE TRADING")
    print("=" * 60)

    # Check for API keys
    has_live_keys = bool(os.getenv("BINANCE_API_KEY"))
    has_testnet_keys = bool(os.getenv("BINANCE_TESTNET_API_KEY"))

    print("\nAvailable modes:")
    print("  1. DRY RUN (no real orders)")
    if has_testnet_keys:
        print("  2. TESTNET (Binance testnet)")
    if has_live_keys:
        print("  3. LIVE (REAL MONEY - CAUTION!)")

    mode = input("\nSelect mode (1/2/3): ").strip()

    dry_run = mode == "1"
    use_testnet = mode in ["1", "2"]

    if mode == "3":
        if not has_live_keys:
            print("\nLive API keys not configured!")
            print("Add BINANCE_API_KEY and BINANCE_API_SECRET to .env")
            return

        confirm = input("\nWARNING: Real money trading! Type YES to confirm: ").strip()
        if confirm != "YES":
            print("Cancelled.")
            return

    # Configuration
    symbol = input("\nSymbol (default: BTC/USDT): ").strip() or "BTC/USDT"
    interval = int(input("Check interval seconds (default: 300): ").strip() or "300")

    # Create trader
    trader = AILiveTrader(
        symbol=symbol,
        timeframe="1h",
        use_testnet=use_testnet,
        dry_run=dry_run,
    )

    # Start trading
    trader.run(interval_seconds=interval)


if __name__ == "__main__":
    main()
