#!/usr/bin/env python3
"""
Optimized Deep Learning Model Retraining.

Uses the same 22 optimal features as ML models for consistency.
Trains LSTM and Transformer models with regularization.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Same 22 optimal features as ML models
OPTIMAL_FEATURES = [
    "ema_8_dist", "ema_21_dist", "ema_55_dist", "ema_100_dist",
    "rsi", "rsi_norm", "macd", "macd_signal", "macd_hist",
    "volatility", "volatility_ratio", "volume_ratio",
    "return_1", "return_3", "return_5", "return_10", "return_20",
    "bb_position", "bb_width", "atr", "momentum", "momentum_acc",
]

N_FEATURES = len(OPTIMAL_FEATURES)  # 22 features
SEQUENCE_LENGTH = 20
N_CLASSES = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMModel(nn.Module):
    """LSTM model for price direction prediction."""

    def __init__(self, input_size: int = N_FEATURES, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, N_CLASSES),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)


class TransformerModel(nn.Module):
    """Transformer model for price direction prediction."""

    def __init__(self, input_size: int = N_FEATURES, d_model: int = 64,
                 nhead: int = 4, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()

        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, SEQUENCE_LENGTH, d_model) * 0.1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, N_CLASSES),
        )

    def forward(self, x):
        x = self.input_proj(x) + self.pos_encoder
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.fc(x)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 22 optimal features."""
    features = pd.DataFrame(index=df.index)

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # EMA distances
    for period in [8, 21, 55, 100]:
        ema = close.ewm(span=period, adjust=False).mean()
        features[f"ema_{period}_dist"] = (close - ema) / ema

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    features["rsi"] = 100 - (100 / (1 + rs))
    features["rsi_norm"] = (features["rsi"] - 50) / 50

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    features["macd"] = ema12 - ema26
    features["macd_signal"] = features["macd"].ewm(span=9, adjust=False).mean()
    features["macd_hist"] = features["macd"] - features["macd_signal"]

    # Volatility
    features["volatility"] = close.pct_change().rolling(20).std()
    features["volatility_ratio"] = features["volatility"] / features["volatility"].rolling(50).mean()

    # Volume ratio
    features["volume_ratio"] = volume / volume.rolling(20).mean()

    # Returns
    for period in [1, 3, 5, 10, 20]:
        features[f"return_{period}"] = close.pct_change(period)

    # Bollinger Bands
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    features["bb_position"] = (close - lower) / (upper - lower + 1e-10)
    features["bb_width"] = (upper - lower) / sma20

    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    features["atr"] = tr.rolling(14).mean() / close

    # Momentum
    features["momentum"] = close.pct_change(10)
    features["momentum_acc"] = features["momentum"] - features["momentum"].shift(5)

    return features[OPTIMAL_FEATURES]


def create_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = SEQUENCE_LENGTH) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for LSTM/Transformer input."""
    X_seq, y_seq = [], []

    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i + seq_len])
        y_seq.append(y[i + seq_len])

    return np.array(X_seq), np.array(y_seq)


def fetch_data(symbol: str, days: int = 365) -> pd.DataFrame:
    """Fetch historical data using yfinance."""
    import yfinance as yf

    yf_symbol = symbol.replace("/USDT", "-USD").replace("/USD", "-USD")
    logger.info(f"Fetching {days} days for {symbol} ({yf_symbol})...")

    ticker = yf.Ticker(yf_symbol)
    df = ticker.history(period=f"{days}d", interval="1h")

    if df.empty:
        return None

    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    logger.info(f"  Got {len(df)} candles")
    return df


def prepare_data(symbol: str, days: int = 365) -> Tuple[DataLoader, DataLoader, StandardScaler]:
    """Prepare data for DL training."""
    df = fetch_data(symbol, days)

    if df is None or len(df) < 500:
        raise ValueError(f"Insufficient data for {symbol}")

    # Compute features
    X = compute_features(df)

    # Create labels
    future_return = df["close"].pct_change(6).shift(-6)
    y = pd.Series(index=df.index, dtype=int)
    y[future_return < -0.003] = 0
    y[(future_return >= -0.003) & (future_return <= 0.003)] = 1
    y[future_return > 0.003] = 2

    # Remove NaN
    valid_mask = X.notna().all(axis=1) & y.notna()
    X = X[valid_mask].values
    y = y[valid_mask].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create sequences
    X_seq, y_seq = create_sequences(X_scaled, y)

    # Split
    split_idx = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

    # Create dataloaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    return train_loader, test_loader, scaler


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 50,
    lr: float = 0.001,
) -> Tuple[nn.Module, Dict]:
    """Train a DL model."""
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_acc = 0
    best_state = None
    history = {"train_loss": [], "test_acc": []}

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        # Evaluation
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(X_batch)
                _, predicted = outputs.max(1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        acc = correct / total
        avg_loss = total_loss / len(train_loader)

        history["train_loss"].append(avg_loss)
        history["test_acc"].append(acc)

        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, acc={acc:.4f}")

    # Load best model
    if best_state:
        model.load_state_dict(best_state)

    logger.info(f"Best accuracy: {best_acc:.4f}")

    return model, {"best_acc": best_acc, "history": history}


def save_model(
    model: nn.Module,
    scaler: StandardScaler,
    model_type: str,
    symbol: str,
    metrics: Dict,
    output_dir: Path
):
    """Save trained model and metadata."""
    symbol_safe = symbol.replace("/", "_")

    # Save model
    model_path = output_dir / f"{symbol_safe}_dl_{model_type}.pt"
    torch.save(model.state_dict(), model_path)

    # Save scaler
    import pickle
    scaler_path = output_dir / f"{symbol_safe}_dl_{model_type}_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    # Save metadata
    meta = {
        "symbol": symbol,
        "model_type": f"dl_{model_type}",
        "architecture": model_type,
        "accuracy": float(metrics["best_acc"]),
        "n_features": N_FEATURES,
        "sequence_length": SEQUENCE_LENGTH,
        "feature_names": OPTIMAL_FEATURES,
        "trained_at": datetime.now().isoformat(),
        "training_version": "optimized_v2",
        "device": str(DEVICE),
    }

    meta_path = output_dir / f"{symbol_safe}_dl_{model_type}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Saved {model_type} model to {model_path}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Retrain DL models")
    parser.add_argument("--symbols", default="BTC/USDT,ETH/USDT")
    parser.add_argument("--output-dir", default="data/models")
    parser.add_argument("--epochs", type=int, default=50)

    args = parser.parse_args()

    symbols = args.symbols.split(",")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*60)
    logger.info("OPTIMIZED DL MODEL RETRAINING")
    logger.info(f"Features: {N_FEATURES}, Sequence: {SEQUENCE_LENGTH}")
    logger.info(f"Device: {DEVICE}")
    logger.info("="*60)

    results = {}

    for symbol in symbols:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training DL models for {symbol}")
        logger.info("="*60)

        try:
            train_loader, test_loader, scaler = prepare_data(symbol)
            results[symbol] = {}

            # Train LSTM
            logger.info("\nTraining LSTM...")
            lstm_model = LSTMModel(input_size=N_FEATURES)
            lstm_model, lstm_metrics = train_model(
                lstm_model, train_loader, test_loader, epochs=args.epochs
            )
            save_model(lstm_model, scaler, "lstm", symbol, lstm_metrics, output_dir)
            results[symbol]["lstm"] = lstm_metrics["best_acc"]

            # Train Transformer
            logger.info("\nTraining Transformer...")
            transformer_model = TransformerModel(input_size=N_FEATURES)
            transformer_model, transformer_metrics = train_model(
                transformer_model, train_loader, test_loader, epochs=args.epochs
            )
            save_model(transformer_model, scaler, "transformer", symbol, transformer_metrics, output_dir)
            results[symbol]["transformer"] = transformer_metrics["best_acc"]

        except Exception as e:
            logger.error(f"Failed for {symbol}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    logger.info("\n" + "="*60)
    logger.info("DL TRAINING SUMMARY")
    logger.info("="*60)

    for symbol, models in results.items():
        logger.info(f"\n{symbol}:")
        for model_type, acc in models.items():
            logger.info(f"  {model_type}: {acc:.4f}")


if __name__ == "__main__":
    main()
