#!/usr/bin/env python3
"""
Simple Deep Learning Training Script.

Trains LSTM models for crypto symbols with CPU fallback.
"""

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_DIR = Path("data/models")
DATA_DIR = Path("data/training")


class SimpleLSTM(nn.Module):
    """Simple LSTM for price prediction."""

    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=3, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


class SimpleTransformer(nn.Module):
    """Simple Transformer for price prediction."""

    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, output_size=3, dropout=0.3):
        super().__init__()

        self.input_proj = nn.Linear(input_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_size)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        out = self.fc(x[:, -1, :])
        return out


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create technical features."""
    df = df.copy()

    close = df['close']
    high = df['high']
    low = df['low']
    volume = df.get('volume', pd.Series(1, index=df.index))
    returns = close.pct_change()

    # EMAs
    for p in [8, 21, 55]:
        df[f'ema_{p}'] = close.ewm(span=p).mean()
        df[f'ema_{p}_dist'] = (close - df[f'ema_{p}']) / df[f'ema_{p}']

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df['macd'] = (ema12 - ema26) / close * 100
    df['macd_signal'] = df['macd'].ewm(span=9).mean()

    # Volatility
    df['volatility'] = returns.rolling(20).std()

    # Volume
    df['volume_ratio'] = volume / (volume.rolling(20).mean() + 1)

    # Returns
    for p in [1, 3, 5, 10]:
        df[f'return_{p}'] = returns.rolling(p).sum()

    # Bollinger position
    bb_sma = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df['bb_position'] = (close - bb_sma) / (2 * bb_std + 1e-10)

    return df


def create_sequences(X: np.ndarray, y: np.ndarray, seq_length: int = 30):
    """Create sequences for LSTM."""
    X_seq, y_seq = [], []

    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length])

    return np.array(X_seq), np.array(y_seq)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 0.001
) -> dict:
    """Train the model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_acc = 0
    best_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()

        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Acc: {train_acc:.1f}%, Val Acc: {val_acc:.1f}%")

    # Load best state
    if best_state:
        model.load_state_dict(best_state)

    return {'best_val_accuracy': best_val_acc}


async def fetch_data(symbol: str, days: int = 730) -> pd.DataFrame:
    """Fetch data from exchange."""
    data_file = DATA_DIR / f"{symbol.replace('/', '_')}_extended.parquet"

    if data_file.exists():
        return pd.read_parquet(data_file)

    try:
        import ccxt
        exchange = ccxt.binance({'enableRateLimit': True})

        logger.info(f"Fetching {days} days of {symbol} data...")

        all_ohlcv = []
        since = exchange.parse8601((datetime.now() - timedelta(days=days)).isoformat())

        while True:
            ohlcv = exchange.fetch_ohlcv(symbol.replace("/", ""), '1h', since=since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            if len(ohlcv) < 1000:
                break

        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df.to_parquet(data_file)

        return df
    except Exception as e:
        logger.error(f"Failed to fetch {symbol}: {e}")
        return pd.DataFrame()


async def train_symbol(symbol: str, device: torch.device) -> dict:
    """Train models for a single symbol."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Training models for {symbol}")
    logger.info("=" * 60)

    # Fetch data
    df = await fetch_data(symbol)

    if df.empty or len(df) < 1000:
        logger.warning(f"Insufficient data for {symbol}")
        return {'symbol': symbol, 'success': False}

    # Create features
    df = create_features(df)

    # Create labels (3-class: DOWN, FLAT, UP)
    future_return = df['close'].shift(-24) / df['close'] - 1
    labels = pd.Series(1, index=df.index)  # Default: FLAT
    labels[future_return > 0.015] = 2  # UP
    labels[future_return < -0.015] = 0  # DOWN

    df['target'] = labels
    df = df.dropna()

    # Features
    feature_cols = [c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume', 'target']]

    X = df[feature_cols].values
    y = df['target'].values.astype(int)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create sequences
    seq_length = 30
    X_seq, y_seq = create_sequences(X_scaled, y, seq_length)

    if len(X_seq) < 500:
        logger.warning(f"Not enough sequences for {symbol}: {len(X_seq)}")
        return {'symbol': symbol, 'success': False}

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Create DataLoaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    results = {}
    symbol_clean = symbol.replace("/", "_")

    # Train LSTM
    logger.info("Training LSTM model...")
    lstm_model = SimpleLSTM(input_size=X_seq.shape[2], hidden_size=64, num_layers=2).to(device)
    lstm_result = train_model(lstm_model, train_loader, val_loader, device, epochs=50)

    # Test LSTM
    lstm_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = lstm_model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    lstm_test_acc = 100 * correct / total
    logger.info(f"LSTM Test Accuracy: {lstm_test_acc:.2f}%")

    # Save LSTM
    torch.save(lstm_model.state_dict(), MODEL_DIR / f"{symbol_clean}_lstm_model.pt")

    results['lstm'] = {
        'val_accuracy': lstm_result['best_val_accuracy'],
        'test_accuracy': lstm_test_acc
    }

    # Train Transformer
    logger.info("Training Transformer model...")
    transformer_model = SimpleTransformer(input_size=X_seq.shape[2], d_model=64, nhead=4, num_layers=2).to(device)
    transformer_result = train_model(transformer_model, train_loader, val_loader, device, epochs=50)

    # Test Transformer
    transformer_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = transformer_model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    transformer_test_acc = 100 * correct / total
    logger.info(f"Transformer Test Accuracy: {transformer_test_acc:.2f}%")

    # Save Transformer
    torch.save(transformer_model.state_dict(), MODEL_DIR / f"{symbol_clean}_transformer_model.pt")

    results['transformer'] = {
        'val_accuracy': transformer_result['best_val_accuracy'],
        'test_accuracy': transformer_test_acc
    }

    # Save scaler
    import joblib
    joblib.dump(scaler, MODEL_DIR / f"{symbol_clean}_dl_scaler.pkl")

    # Save metadata
    meta = {
        'symbol': symbol,
        'seq_length': seq_length,
        'input_size': X_seq.shape[2],
        'feature_cols': feature_cols,
        'results': results,
        'trained_at': datetime.now().isoformat()
    }
    with open(MODEL_DIR / f"{symbol_clean}_dl_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    logger.info(f"\nResults for {symbol}:")
    logger.info(f"  LSTM:        Val={lstm_result['best_val_accuracy']:.1f}%, Test={lstm_test_acc:.1f}%")
    logger.info(f"  Transformer: Val={transformer_result['best_val_accuracy']:.1f}%, Test={transformer_test_acc:.1f}%")

    return {
        'symbol': symbol,
        'success': True,
        **results
    }


async def main():
    """Train DL models for all crypto symbols."""
    logger.info("=" * 70)
    logger.info("DEEP LEARNING MODEL TRAINING")
    logger.info("=" * 70)

    # Use CPU for stability (MPS can be unstable)
    device = torch.device('cpu')
    logger.info(f"Using device: {device}")

    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']

    results = []
    for symbol in symbols:
        result = await train_symbol(symbol, device)
        results.append(result)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 70)

    for r in results:
        if r.get('success'):
            lstm_acc = r.get('lstm', {}).get('test_accuracy', 0)
            trans_acc = r.get('transformer', {}).get('test_accuracy', 0)
            logger.info(f"  {r['symbol']}: LSTM={lstm_acc:.1f}%, Transformer={trans_acc:.1f}%")
        else:
            logger.info(f"  {r['symbol']}: FAILED")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
