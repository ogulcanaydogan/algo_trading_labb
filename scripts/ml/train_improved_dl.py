#!/usr/bin/env python3
"""
Improved Deep Learning Model Training.

Uses the same 22 optimal features as ML models with better architectures:
- Bidirectional LSTM with attention
- Transformer with positional encoding
- Proper regularization and early stopping
"""

import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Same 22 optimal features as ML models
OPTIMAL_FEATURES = [
    "ema_8_dist", "ema_21_dist", "ema_55_dist", "ema_100_dist",
    "rsi", "rsi_norm", "macd", "macd_signal", "macd_hist",
    "volatility", "volatility_ratio", "volume_ratio",
    "return_1", "return_3", "return_5", "return_10", "return_20",
    "bb_position", "bb_width", "atr", "momentum", "momentum_acc",
]


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
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    features["atr"] = tr.rolling(14).mean() / close

    # Momentum
    features["momentum"] = close.pct_change(10)
    features["momentum_acc"] = features["momentum"] - features["momentum"].shift(5)

    return features[OPTIMAL_FEATURES]


def create_trend_labels(df: pd.DataFrame) -> pd.Series:
    """Create trend-following labels with MTF confirmation."""
    close = df["close"]

    future_4h = close.pct_change(4).shift(-4).rolling(2).mean()
    future_8h = close.pct_change(8).shift(-8).rolling(2).mean()
    future_12h = close.pct_change(12).shift(-12).rolling(2).mean()

    future_return = (future_4h * 0.5 + future_8h * 0.3 + future_12h * 0.2)

    volatility = close.pct_change().rolling(20).std()
    vol_threshold = volatility * 1.5
    vol_threshold = vol_threshold.clip(lower=0.002, upper=0.015)

    labels = pd.Series(1, index=df.index)
    labels[future_return > vol_threshold] = 2
    labels[future_return < -vol_threshold] = 0

    return labels


def create_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for LSTM/Transformer."""
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X)):
        X_seq.append(X[i-seq_len:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


def fetch_data(symbol: str, days: int = 730) -> Optional[pd.DataFrame]:
    """Fetch historical data."""
    import yfinance as yf

    yf_symbol = symbol.replace("/USDT", "-USD").replace("/USD", "-USD")
    logger.info(f"Fetching {days} days for {symbol}...")

    ticker = yf.Ticker(yf_symbol)
    df = ticker.history(period=f"{days}d", interval="1h")

    if df.empty:
        return None

    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    logger.info(f"  Got {len(df)} candles")
    return df


def build_lstm_model(input_shape: Tuple[int, int], num_classes: int = 3):
    """Build improved LSTM model with attention."""
    import torch
    import torch.nn as nn

    class AttentionLSTM(nn.Module):
        def __init__(self, input_dim, hidden_dim=64, num_layers=2, num_classes=3, dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(
                input_dim, hidden_dim, num_layers,
                batch_first=True, bidirectional=True, dropout=dropout
            )
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
                nn.Softmax(dim=1)
            )
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            attention_weights = self.attention(lstm_out)
            context = torch.sum(attention_weights * lstm_out, dim=1)
            return self.classifier(context)

    return AttentionLSTM(input_shape[1], num_classes=num_classes)


def build_transformer_model(input_shape: Tuple[int, int], num_classes: int = 3):
    """Build improved Transformer model."""
    import torch
    import torch.nn as nn

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=100):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, :x.size(1)]

    class TransformerClassifier(nn.Module):
        def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, num_classes=3, dropout=0.3):
            super().__init__()
            self.input_proj = nn.Linear(input_dim, d_model)
            self.pos_encoder = PositionalEncoding(d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, num_classes)
            )

        def forward(self, x):
            x = self.input_proj(x)
            x = self.pos_encoder(x)
            x = self.transformer(x)
            x = x.mean(dim=1)  # Global average pooling
            return self.classifier(x)

    return TransformerClassifier(input_shape[1], num_classes=num_classes)


def train_model(
    model, X_train, y_train, X_val, y_val,
    epochs: int = 100, batch_size: int = 64, patience: int = 15, lr: float = 0.001
) -> Dict:
    """Train PyTorch model with early stopping."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    model = model.to(device)

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Class weights for imbalanced data
    class_counts = np.bincount(y_train)
    class_weights = torch.FloatTensor(1.0 / (class_counts + 1)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_acc = 0
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()
            val_preds = val_outputs.argmax(dim=1)
            val_acc = (val_preds == y_val_t).float().mean().item()

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            logger.info(f"  Epoch {epoch+1}: train_loss={train_loss/len(train_loader):.4f}, val_acc={val_acc:.4f}")

        if patience_counter >= patience:
            logger.info(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    return {"best_val_acc": best_val_acc, "epochs_trained": epoch + 1}


def save_model(model, symbol: str, model_type: str, accuracy: float, output_dir: Path):
    """Save PyTorch model."""
    import torch

    symbol_safe = symbol.replace("/", "_")
    model_path = output_dir / f"{symbol_safe}_{model_type}_model.pt"
    torch.save(model.state_dict(), model_path)

    meta = {
        "symbol": symbol,
        "model_type": model_type,
        "accuracy": float(accuracy),
        "n_features": len(OPTIMAL_FEATURES),
        "feature_names": OPTIMAL_FEATURES,
        "sequence_length": 20,
        "trained_at": datetime.now().isoformat(),
    }
    meta_path = output_dir / f"{symbol_safe}_{model_type}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"  Saved {model_type} model: {accuracy:.2%}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="BTC/USDT,ETH/USDT,SOL/USDT")
    parser.add_argument("--output-dir", default="data/models")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seq-len", type=int, default=20)
    args = parser.parse_args()

    symbols = args.symbols.split(",")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("IMPROVED DL MODEL TRAINING")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Using {len(OPTIMAL_FEATURES)} optimal features")
    logger.info("=" * 60)

    results = {}
    for symbol in symbols:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {symbol}")
        logger.info("=" * 60)

        try:
            df = fetch_data(symbol)
            if df is None or len(df) < 1000:
                logger.warning(f"Insufficient data for {symbol}")
                continue

            X = compute_features(df)
            y = create_trend_labels(df)

            valid = X.notna().all(axis=1) & y.notna()
            X, y = X[valid].values, y[valid].values

            # Normalize features
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            # Create sequences
            X_seq, y_seq = create_sequences(X, y, seq_len=args.seq_len)

            # Split data (time-series aware)
            split_idx = int(len(X_seq) * 0.8)
            X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
            y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]

            logger.info(f"Training samples: {len(X_train)}, Validation: {len(X_val)}")

            results[symbol] = {}

            # Train LSTM
            logger.info("Training Attention LSTM...")
            lstm_model = build_lstm_model((args.seq_len, len(OPTIMAL_FEATURES)))
            lstm_result = train_model(lstm_model, X_train, y_train, X_val, y_val, epochs=args.epochs)
            save_model(lstm_model, symbol, "lstm", lstm_result["best_val_acc"], output_dir)
            results[symbol]["lstm"] = lstm_result["best_val_acc"]

            # Train Transformer
            logger.info("Training Transformer...")
            transformer_model = build_transformer_model((args.seq_len, len(OPTIMAL_FEATURES)))
            transformer_result = train_model(transformer_model, X_train, y_train, X_val, y_val, epochs=args.epochs)
            save_model(transformer_model, symbol, "transformer", transformer_result["best_val_acc"], output_dir)
            results[symbol]["transformer"] = transformer_result["best_val_acc"]

        except Exception as e:
            logger.error(f"Failed for {symbol}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("DL TRAINING SUMMARY")
    logger.info("=" * 60)
    for symbol, models in results.items():
        logger.info(f"\n{symbol}:")
        for model_type, acc in models.items():
            logger.info(f"  {model_type}: {acc:.2%}")


if __name__ == "__main__":
    main()
