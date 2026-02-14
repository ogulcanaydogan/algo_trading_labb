"""Quick sentiment impact test."""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

print("Fetching TSLA data...")
import yfinance as yf
ticker = yf.Ticker("TSLA")
df = ticker.history(period="180d", interval="1h")
df.columns = [c.lower() for c in df.columns]
print(f"Got {len(df)} rows")

# Technical features
close = df["close"]
df["return_1"] = close.pct_change(1)
df["return_5"] = close.pct_change(5)
df["ma_10_dist"] = (close - close.rolling(10).mean()) / close
df["ma_20_dist"] = (close - close.rolling(20).mean()) / close
delta = close.diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
df["rsi_norm"] = (100 - (100 / (1 + gain/(loss+1e-8)))) / 100 - 0.5
df["volatility"] = close.pct_change().rolling(10).std()
df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
df["target"] = (close.shift(-1) > close).astype(int)

# Synthetic sentiment
from bot.sentiment.feature_integration import generate_synthetic_sentiment_history
sentiment_df = generate_synthetic_sentiment_history(df, "TSLA", correlation_with_returns=0.2)
for col in sentiment_df.columns:
    df[col] = sentiment_df[col].values

df = df.dropna()
print(f"After cleaning: {len(df)} rows")

# Features
base_features = ["return_1", "return_5", "ma_10_dist", "ma_20_dist", "rsi_norm", "volatility", "volume_ratio"]
sentiment_features = ["sentiment_combined", "sentiment_momentum", "mention_volume", "social_sentiment"]

# Split
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

X_train_base = train_df[base_features].values
X_test_base = test_df[base_features].values
X_train_full = train_df[base_features + sentiment_features].values
X_test_full = test_df[base_features + sentiment_features].values
y_train = train_df["target"].values
y_test = test_df["target"].values

print(f"Train: {len(train_df)}, Test: {len(test_df)}")

# Train models
model_base = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
model_base.fit(X_train_base, y_train)
pred_base = model_base.predict(X_test_base)

model_full = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
model_full.fit(X_train_full, y_train)
pred_full = model_full.predict(X_test_full)

print()
print("=" * 60)
print("TSLA SENTIMENT IMPACT RESULTS")
print("=" * 60)
print()

# Results
acc_base = accuracy_score(y_test, pred_base)
acc_full = accuracy_score(y_test, pred_full)
prec_base = precision_score(y_test, pred_base, zero_division=0)
prec_full = precision_score(y_test, pred_full, zero_division=0)
f1_base = f1_score(y_test, pred_base, zero_division=0)
f1_full = f1_score(y_test, pred_full, zero_division=0)

print("Metric          Baseline        + Sentiment     Change")
print("-" * 60)
print(f"Accuracy        {acc_base:.4f}          {acc_full:.4f}          {(acc_full-acc_base)*100:+.2f}%")
print(f"Precision       {prec_base:.4f}          {prec_full:.4f}          {(prec_full-prec_base)*100:+.2f}%")
print(f"F1 Score        {f1_base:.4f}          {f1_full:.4f}          {(f1_full-f1_base)*100:+.2f}%")

print()
print("Top 10 Feature Importance (with sentiment):")
all_features = base_features + sentiment_features
imp = dict(zip(all_features, model_full.feature_importances_))
for name, val in sorted(imp.items(), key=lambda x: -x[1])[:10]:
    marker = "SENT" if name in sentiment_features else "TECH"
    print(f"  [{marker}] {name:25s} {val:.4f}")

# Summary
print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
gain = (acc_full - acc_base) * 100
if gain > 0:
    print(f"\n[OK] Sentiment features IMPROVED accuracy by {gain:.2f}%")
else:
    print(f"\n[--] Sentiment features changed accuracy by {gain:.2f}%")

sentiment_imp = sum(imp[f] for f in sentiment_features)
print(f"[INFO] Total sentiment feature importance: {sentiment_imp:.2%}")
