#!/usr/bin/env python3
"""
AI Prediction Demo - Works without Binance API keys
Uses Yahoo Finance for real market data
"""
import yfinance as yf
import pandas as pd
from bot.strategy import StrategyConfig, compute_indicators, generate_signal
from bot.ai import RuleBasedAIPredictor

print("="*60)
print("AI PREDICTION DEMO (No API keys needed)")
print("="*60)

# Get symbol
symbol_input = input("Symbol (BTC-USD, ETH-USD, default: BTC-USD): ").strip() or "BTC-USD"

print(f"\n📊 Fetching recent {symbol_input} data...")

# Fetch recent data from Yahoo Finance
ticker = yf.Ticker(symbol_input)
df = ticker.history(period="1mo", interval="1h")

if df.empty:
    print("❌ No data found.")
    exit(1)

# Convert to OHLCV format
ohlcv = pd.DataFrame({
    'open': df['Open'],
    'high': df['High'],
    'low': df['Low'],
    'close': df['Close'],
    'volume': df['Volume']
})

print(f"✅ Got {len(ohlcv)} hourly candles")
print(f"   Latest data: {ohlcv.index[-1]}")
print(f"   Current price: ${ohlcv['close'].iloc[-1]:,.2f}")

# Strategy configuration
config = StrategyConfig(
    symbol=symbol_input.replace("-", "/"),
    timeframe="1h",
    ema_fast=12,
    ema_slow=26,
    rsi_period=14,
)

# Compute indicators
print("\n🔍 Computing technical indicators...")
enriched = compute_indicators(ohlcv, config)

# Generate signal
print("🎯 Generating trading signal...")
signal = generate_signal(enriched, config)

print(f"\n📋 Current Signal:")
print(f"   Decision: {signal['decision']}")
print(f"   Confidence: {signal['confidence']:.1f}%")
print(f"   RSI: {signal['rsi']:.2f}")
print(f"   EMA Fast: ${signal['ema_fast']:,.2f}")
print(f"   EMA Slow: ${signal['ema_slow']:,.2f}")
print(f"   Reason: {signal['reason']}")

# AI Prediction
print("\n🤖 Running AI prediction...")
ai_predictor = RuleBasedAIPredictor(config)
prediction = ai_predictor.predict(enriched)

print(f"\n🎯 AI Prediction:")
print(f"   Recommended Action: {prediction.recommended_action}")
print(f"   Confidence: {prediction.confidence:.1f}%")
print(f"   Probability LONG: {prediction.probability_long:.1f}%")
print(f"   Probability SHORT: {prediction.probability_short:.1f}%")
print(f"   Probability FLAT: {prediction.probability_flat:.1f}%")
print(f"   Expected Move: {prediction.expected_move_pct:+.2f}%")
print(f"\n📝 Summary: {prediction.summary}")

print("\n📊 Feature Analysis:")
print(f"   EMA Gap: {prediction.features.ema_gap_pct:+.2f}%")
print(f"   Momentum: {prediction.features.momentum_pct:+.2f}%")
print(f"   RSI Distance: {prediction.features.rsi_distance_from_mid:+.2f}")
print(f"   Volatility: {prediction.features.volatility_pct:.2f}%")

print("\n" + "="*60)
print("✅ AI PREDICTION COMPLETE")
print("="*60)
print("\n💡 This analysis used real market data from Yahoo Finance")
print("   No Binance API keys needed!")
