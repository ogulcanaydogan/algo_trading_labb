#!/usr/bin/env python3
"""
Regularized Deep Learning Training Script.

Addresses overfitting with:
1. Early stopping based on validation loss
2. Higher dropout (0.5)
3. Weight decay (L2 regularization)
4. Gradient clipping
5. Label smoothing
6. Class weights for imbalanced data
7. Data augmentation (noise injection)
8. Smaller sequence length
"""

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_DIR = Path("data/models")
DATA_DIR = Path("data/training")


class RegularizedLSTM(nn.Module):
    """LSTM with strong regularization."""

    def __init__(self, input_size, hidden_size=48, num_layers=1, output_size=3, dropout=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Smaller LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0  # No dropout between LSTM layers for single layer
        )

        # Heavy dropout and simpler FC
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Add input noise during training (data augmentation)
        if self.training:
            x = x + torch.randn_like(x) * 0.02

        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(out)
        return out


class RegularizedTransformer(nn.Module):
    """Transformer with strong regularization."""

    def __init__(self, input_size, d_model=48, nhead=4, num_layers=1, output_size=3, dropout=0.5):
        super().__init__()

        self.input_proj = nn.Linear(input_size, d_model)
        self.dropout1 = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,  # Smaller feedforward
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        # Add input noise during training
        if self.training:
            x = x + torch.randn_like(x) * 0.02

        x = self.input_proj(x)
        x = self.dropout1(x)
        x = self.transformer(x)
        out = self.dropout2(x[:, -1, :])
        out = self.fc(out)
        return out


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing."""

    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_pred = F.log_softmax(pred, dim=-1)

        # Create smoothed target
        with torch.no_grad():
            smooth_target = torch.zeros_like(pred)
            smooth_target.fill_(self.smoothing / (n_classes - 1))
            smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        # Apply class weights if provided
        if self.weight is not None:
            weight = self.weight[target]
            loss = (-smooth_target * log_pred).sum(dim=-1) * weight
        else:
            loss = (-smooth_target * log_pred).sum(dim=-1)

        return loss.mean()


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create technical features with normalization."""
    df = df.copy()

    close = df['close']
    high = df['high']
    low = df['low']
    volume = df.get('volume', pd.Series(1, index=df.index))
    returns = close.pct_change()

    # EMAs - normalized as distance from price
    for p in [8, 21, 55]:
        df[f'ema_{p}_dist'] = (close - close.ewm(span=p).mean()) / close

    # RSI - normalized to -1 to 1
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi_norm'] = (100 - (100 / (1 + rs)) - 50) / 50  # -1 to 1

    # MACD - normalized
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd = ema12 - ema26
    df['macd_norm'] = macd / close

    # Volatility - rolling std of returns
    df['volatility'] = returns.rolling(20).std()

    # Volume ratio (log)
    vol_ma = volume.rolling(20).mean()
    df['volume_ratio'] = np.log1p(volume / (vol_ma + 1))

    # Returns at different horizons
    for p in [1, 3, 5]:
        df[f'return_{p}'] = returns.rolling(p).sum()

    # Bollinger position (-1 to 1)
    bb_sma = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df['bb_position'] = (close - bb_sma) / (2 * bb_std + 1e-10)
    df['bb_position'] = df['bb_position'].clip(-2, 2) / 2  # Clip outliers

    # ATR normalized
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    df['atr_norm'] = tr.rolling(14).mean() / close

    return df


def create_sequences(X: np.ndarray, y: np.ndarray, seq_length: int = 20):
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
    class_weights: torch.Tensor,
    epochs: int = 100,
    lr: float = 0.001,
    patience: int = 15
) -> dict:
    """Train with early stopping and regularization."""

    criterion = LabelSmoothingCrossEntropy(smoothing=0.1, weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-6
    )

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    best_epoch = 0

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

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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
        avg_val_loss = val_loss / len(val_loader)

        scheduler.step(avg_val_loss)

        # Early stopping based on validation LOSS (not accuracy)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            best_epoch = epoch
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} - Train: {train_acc:.1f}%, Val: {val_acc:.1f}%, Val Loss: {avg_val_loss:.4f}")

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}. Best epoch: {best_epoch+1}")
            break

    # Load best state
    if best_state:
        model.load_state_dict(best_state)
        model.to(device)

    return {'best_val_loss': best_val_loss, 'best_epoch': best_epoch}


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
    """Train regularized models for a single symbol."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Training REGULARIZED models for {symbol}")
    logger.info("=" * 60)

    # Fetch data
    df = await fetch_data(symbol)

    if df.empty or len(df) < 1000:
        logger.warning(f"Insufficient data for {symbol}")
        return {'symbol': symbol, 'success': False}

    # Create features
    df = create_features(df)

    # Create labels (3-class: DOWN, FLAT, UP)
    # Use smaller threshold for more balanced classes
    future_return = df['close'].shift(-12) / df['close'] - 1  # 12h instead of 24h
    labels = pd.Series(1, index=df.index)  # Default: FLAT
    labels[future_return > 0.01] = 2  # UP (1% threshold)
    labels[future_return < -0.01] = 0  # DOWN

    df['target'] = labels
    df = df.dropna()

    # Check class distribution
    class_counts = df['target'].value_counts().sort_index()
    logger.info(f"Class distribution: DOWN={class_counts.get(0,0)}, FLAT={class_counts.get(1,0)}, UP={class_counts.get(2,0)}")

    # Features
    feature_cols = [c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume', 'target']]

    X = df[feature_cols].values
    y = df['target'].values.astype(int)

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1, 2]), y=y)
    class_weights_tensor = torch.FloatTensor(class_weights)
    logger.info(f"Class weights: {class_weights}")

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Clip outliers
    X_scaled = np.clip(X_scaled, -3, 3)

    # Create sequences (shorter for less overfitting)
    seq_length = 20  # Reduced from 30
    X_seq, y_seq = create_sequences(X_scaled, y, seq_length)

    if len(X_seq) < 500:
        logger.warning(f"Not enough sequences for {symbol}: {len(X_seq)}")
        return {'symbol': symbol, 'success': False}

    # Split (use more data for validation)
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, shuffle=False)

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Create DataLoaders (larger batch size for regularization)
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)

    results = {}
    symbol_clean = symbol.replace("/", "_")

    # Train Regularized LSTM
    logger.info("Training Regularized LSTM...")
    lstm_model = RegularizedLSTM(
        input_size=X_seq.shape[2],
        hidden_size=48,
        num_layers=1,
        dropout=0.5
    ).to(device)

    lstm_result = train_model(
        lstm_model, train_loader, val_loader, device,
        class_weights_tensor, epochs=100, patience=15
    )

    # Test LSTM
    lstm_model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = lstm_model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    lstm_test_acc = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
    logger.info(f"LSTM Test Accuracy: {lstm_test_acc:.2f}%")

    # Per-class accuracy
    for cls in [0, 1, 2]:
        mask = np.array(all_labels) == cls
        if mask.sum() > 0:
            cls_acc = 100 * np.mean(np.array(all_preds)[mask] == cls)
            logger.info(f"  Class {cls} accuracy: {cls_acc:.1f}%")

    # Save LSTM
    torch.save(lstm_model.state_dict(), MODEL_DIR / f"{symbol_clean}_lstm_regularized.pt")

    results['lstm'] = {
        'test_accuracy': lstm_test_acc,
        'best_epoch': lstm_result['best_epoch']
    }

    # Train Regularized Transformer
    logger.info("Training Regularized Transformer...")
    transformer_model = RegularizedTransformer(
        input_size=X_seq.shape[2],
        d_model=48,
        nhead=4,
        num_layers=1,
        dropout=0.5
    ).to(device)

    transformer_result = train_model(
        transformer_model, train_loader, val_loader, device,
        class_weights_tensor, epochs=100, patience=15
    )

    # Test Transformer
    transformer_model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = transformer_model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    transformer_test_acc = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
    logger.info(f"Transformer Test Accuracy: {transformer_test_acc:.2f}%")

    # Save Transformer
    torch.save(transformer_model.state_dict(), MODEL_DIR / f"{symbol_clean}_transformer_regularized.pt")

    results['transformer'] = {
        'test_accuracy': transformer_test_acc,
        'best_epoch': transformer_result['best_epoch']
    }

    # Save scaler
    import joblib
    joblib.dump(scaler, MODEL_DIR / f"{symbol_clean}_dl_regularized_scaler.pkl")

    # Save metadata
    meta = {
        'symbol': symbol,
        'seq_length': seq_length,
        'input_size': X_seq.shape[2],
        'feature_cols': feature_cols,
        'results': results,
        'regularization': {
            'dropout': 0.5,
            'weight_decay': 0.01,
            'label_smoothing': 0.1,
            'gradient_clipping': 1.0,
            'early_stopping_patience': 15
        },
        'trained_at': datetime.now().isoformat()
    }
    with open(MODEL_DIR / f"{symbol_clean}_dl_regularized_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    logger.info(f"\nResults for {symbol}:")
    logger.info(f"  LSTM:        Test={lstm_test_acc:.1f}% (best epoch: {lstm_result['best_epoch']+1})")
    logger.info(f"  Transformer: Test={transformer_test_acc:.1f}% (best epoch: {transformer_result['best_epoch']+1})")

    return {
        'symbol': symbol,
        'success': True,
        **results
    }


async def main():
    """Train regularized DL models."""
    logger.info("=" * 70)
    logger.info("REGULARIZED DEEP LEARNING MODEL TRAINING")
    logger.info("=" * 70)
    logger.info("Techniques: Early stopping, high dropout, weight decay,")
    logger.info("            label smoothing, gradient clipping, class weights")
    logger.info("=" * 70)

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
