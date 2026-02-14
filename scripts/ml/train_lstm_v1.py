#!/usr/bin/env python3
"""
LSTM Model v1 for Time Series Prediction

Compares LSTM deep learning with V6 ensemble models.
Uses same data loading and walk-forward validation as V6.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score

# Import V6 feature engineering
from bot.ml.v6_feature_extractor import build_v6_features, get_v6_feature_cols


# =============================================================================
# LSTM Model Architecture
# =============================================================================
class LSTMClassifier(nn.Module):
    """LSTM for binary classification with dropout regularization."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * self.num_directions, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state from all directions
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            h_final = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        else:
            h_final = h_n[-1, :, :]
        
        out = self.dropout(h_final)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out.squeeze()


# =============================================================================
# Data Loading (Same as V6)
# =============================================================================
def load_data(symbol: str, limit: int) -> pd.DataFrame:
    """Load OHLCV data from parquet files."""
    sym = symbol.replace("/", "_")
    for name in [f"{sym}_extended.parquet", f"{sym}_1h.parquet"]:
        path = PROJECT_ROOT / "data" / "training" / name
        if path.exists():
            df = pd.read_parquet(path)
            if "timestamp" in df.columns and df.index.name not in ("timestamp", "Datetime"):
                if df["timestamp"].dtype == "int64" and df["timestamp"].iloc[0] > 1e12:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                else:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
            elif df.index.name in ("timestamp", "Datetime"):
                df.index = pd.to_datetime(df.index)
                df.index.name = "timestamp"
            df.columns = [c.lower() for c in df.columns]
            ohlcv = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
            df = df[ohlcv].tail(limit)
            print(f"  Loaded {len(df)} bars from {name}")
            return df
    return pd.DataFrame()


# =============================================================================
# V6-Compatible Target Creation
# =============================================================================
def create_targets(
    df: pd.DataFrame,
    pred_horizon: int = 8,
    min_move: float = 0.015,
    vol_mult: float = 0.4,
) -> pd.DataFrame:
    """Create binary targets with adaptive threshold (same as V6)."""
    c = df["close"]
    
    # Forward return
    df = df.copy()
    df["fwd_return"] = c.pct_change(pred_horizon).shift(-pred_horizon)
    
    # Adaptive threshold
    vol = c.pct_change().rolling(24).std().fillna(min_move)
    thresh = np.maximum(vol * vol_mult, min_move)
    
    df["target"] = np.nan
    up_mask = df["fwd_return"] > thresh
    down_mask = df["fwd_return"] < -thresh
    df.loc[up_mask, "target"] = 1
    df.loc[down_mask, "target"] = 0
    
    n_up = (df["target"] == 1).sum()
    n_down = (df["target"] == 0).sum()
    n_total = len(df)
    
    print(f"  Target (horizon={pred_horizon}h): UP={n_up} ({n_up/n_total:.1%}), DOWN={n_down} ({n_down/n_total:.1%})")
    
    return df


# =============================================================================
# Sequence Creation for LSTM
# =============================================================================
def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_length: int = 24,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for LSTM input."""
    sequences = []
    labels = []
    
    for i in range(seq_length, len(X)):
        sequences.append(X[i - seq_length:i])
        labels.append(y[i])
    
    return np.array(sequences), np.array(labels)


# =============================================================================
# Training Function
# =============================================================================
def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).float()
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        preds = (outputs > 0.5).float()
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
    
    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).float()
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            all_probs.extend(outputs.cpu().numpy())
            all_preds.extend((outputs > 0.5).cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    return (
        total_loss / len(dataloader),
        np.array(all_preds),
        np.array(all_probs),
        np.array(all_labels),
    )


# =============================================================================
# Walk-Forward Validation (Same as V6)
# =============================================================================
MIN_SAMPLES_PER_FOLD = 50

def walk_forward_lstm(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 10,
    seq_length: int = 24,
    conf_thresh: float = 0.55,
    hidden_size: int = 64,
    num_layers: int = 2,
    epochs: int = 30,
    device: str = "cpu",
):
    """Walk-forward with EXPANDING window for LSTM."""
    n = len(X)
    test_size = max(n // (n_splits + 1), MIN_SAMPLES_PER_FOLD)
    results = []
    
    print(f"\n  Walk-Forward LSTM ({n_splits} folds, seq_length={seq_length}):")
    print(f"  {'Fold':<6} {'Train':<8} {'Test':<8} {'Acc':<8} {'AUC':<8} {'HC Acc':<8} {'Time':<8}")
    print(f"  {'-'*60}")
    
    for i in range(n_splits):
        fold_start = time.time()
        
        train_end = test_size * (i + 2)
        test_start = train_end
        test_end = min(test_start + test_size, n)
        
        if test_end <= test_start or test_end - test_start < 20:
            break
        
        # Get train/test data
        X_train = X[:train_end]
        y_train = y[:train_end]
        X_test = X[test_start:test_end]
        y_test = y[test_start:test_end]
        
        # Scale features
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        # Create sequences
        X_train_seq, y_train_seq = create_sequences(X_train_s, y_train, seq_length)
        X_test_seq, y_test_seq = create_sequences(X_test_s, y_test, seq_length)
        
        if len(X_train_seq) < 50 or len(X_test_seq) < 10:
            continue
        
        # Create model
        model = LSTMClassifier(
            input_size=X_train_s.shape[1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.3,
        ).to(device)
        
        # Compute class weights
        pos_weight = (y_train_seq == 0).sum() / (y_train_seq == 1).sum() if (y_train_seq == 1).sum() > 0 else 1.0
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
        criterion = nn.BCELoss()  # Use standard BCE since we have sigmoid
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_seq),
            torch.FloatTensor(y_train_seq),
        )
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test_seq),
            torch.FloatTensor(y_test_seq),
        )
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Train
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, _, _, _ = evaluate(model, test_loader, criterion, device)
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    break
        
        # Evaluate
        _, preds, probs, labels = evaluate(model, test_loader, criterion, device)
        
        acc = accuracy_score(labels, preds)
        auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
        
        # High confidence accuracy
        hc = np.maximum(probs, 1 - probs) >= conf_thresh
        hc_acc = accuracy_score(labels[hc], preds[hc]) if hc.sum() >= 5 else acc
        hc_n = int(hc.sum())
        
        test_samples = test_end - test_start
        is_valid = test_samples >= MIN_SAMPLES_PER_FOLD
        
        fold_time = time.time() - fold_start
        
        results.append({
            "acc": acc, "auc": auc, "hc_acc": hc_acc, "hc_n": hc_n,
            "test_samples": test_samples, "is_valid": is_valid,
            "time": fold_time,
        })
        
        print(f"  {i+1:<6} {train_end:<8} {test_samples:<8} {acc:<8.4f} {auc:<8.4f} {hc_acc:<8.4f} {fold_time:<8.1f}s")
    
    # Calculate stats from valid folds
    valid_results = [r for r in results if r["is_valid"]]
    valid_folds = len(valid_results)
    
    if valid_folds > 0:
        avg_acc = np.mean([r["acc"] for r in valid_results])
        avg_auc = np.mean([r["auc"] for r in valid_results])
        avg_hc = np.mean([r["hc_acc"] for r in valid_results])
        std_acc = np.std([r["acc"] for r in valid_results])
    else:
        avg_acc = np.mean([r["acc"] for r in results])
        avg_auc = np.mean([r["auc"] for r in results])
        avg_hc = np.mean([r["hc_acc"] for r in results])
        std_acc = np.std([r["acc"] for r in results])
    
    print(f"  {'-'*60}")
    print(f"  AVG (valid={valid_folds}/{len(results)}): acc={avg_acc:.4f}, auc={avg_auc:.4f}, hc={avg_hc:.4f}, std={std_acc:.4f}")
    
    return {
        "folds": results,
        "avg_acc": float(avg_acc),
        "avg_auc": float(avg_auc),
        "avg_hc_acc": float(avg_hc),
        "std": float(std_acc),
        "stable": bool(std_acc < 0.04),
        "valid_folds": valid_folds,
        "total_folds": len(results),
    }


# =============================================================================
# Main Training Pipeline
# =============================================================================
def train_lstm(
    symbol: str,
    pred_horizon: int = 8,
    lookback: int = 10000,
    min_move: float = 0.015,
    vol_mult: float = 0.4,
    conf_thresh: float = 0.55,
    seq_length: int = 24,
    hidden_size: int = 64,
    num_layers: int = 2,
    epochs: int = 50,
    batch_size: int = 32,
    asset_class: str = "stock",
) -> Dict:
    """Train LSTM model for a symbol."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*70}")
    print(f"  LSTM v1 Training: {symbol} [{asset_class}]")
    print(f"  horizon={pred_horizon}h, seq_length={seq_length}, device={device}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    # Load data
    df = load_data(symbol, lookback)
    if df.empty or len(df) < 500:
        return {"symbol": symbol, "status": "failed", "reason": "no_data"}
    
    # Build V6 features
    feat = build_v6_features(df, pred_horizon, asset_class)
    feat = create_targets(feat, pred_horizon, min_move, vol_mult)
    feature_cols = get_v6_feature_cols(feat)
    
    # Filter to signal bars
    signal = feat.dropna(subset=["target"])
    signal = signal.dropna(subset=feature_cols, how="any")
    
    if len(signal) < 500:
        print(f"  [FAIL] Only {len(signal)} samples, need 500+")
        return {"symbol": symbol, "status": "failed", "reason": "few_signals"}
    
    X = signal[feature_cols].values
    y = signal["target"].values.astype(int)
    X = np.nan_to_num(np.where(np.isinf(X), 0, X), nan=0.0)
    
    print(f"  Total samples: {len(X)}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Class balance: UP={y.sum()} ({y.mean():.1%}), DOWN={len(y)-y.sum()}")
    
    # Train/test split (80/20)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Create sequences
    X_train_seq, y_train_seq = create_sequences(X_train_s, y_train, seq_length)
    X_test_seq, y_test_seq = create_sequences(X_test_s, y_test, seq_length)
    
    print(f"  Sequence shape: train={X_train_seq.shape}, test={X_test_seq.shape}")
    
    # Build model
    model = LSTMClassifier(
        input_size=len(feature_cols),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0.3,
    ).to(device)
    
    print(f"\n  Model: {sum(p.numel() for p in model.parameters())} parameters")
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_seq),
        torch.FloatTensor(y_train_seq),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test_seq),
        torch.FloatTensor(y_test_seq),
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Training loop
    print(f"\n  Training ({epochs} epochs)...")
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_preds, val_probs, val_labels = evaluate(model, test_loader, criterion, device)
        
        val_acc = accuracy_score(val_labels, val_preds)
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 15:
                print(f"    Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Final evaluation
    _, y_pred, y_prob, y_true = evaluate(model, test_loader, criterion, device)
    
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    f1 = f1_score(y_true, y_pred)
    
    print(f"\n  === TEST RESULTS ===")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  AUC-ROC:  {auc:.4f}")
    print(f"  F1:       {f1:.4f}")
    
    # Confidence sweep
    print(f"\n  Confidence sweep:")
    print(f"  {'Thresh':<8} {'Acc':<8} {'Prec':<8} {'Count':<8}")
    print(f"  {'-'*36}")
    
    best_hc = {"thresh": 0.5, "acc": acc, "count": len(y_true)}
    for t in [0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.70]:
        mask = np.maximum(y_prob, 1 - y_prob) >= t
        if mask.sum() >= 10:
            a = accuracy_score(y_true[mask], y_pred[mask])
            p = precision_score(y_true[mask], y_pred[mask], zero_division=0)
            marker = " <" if abs(t - conf_thresh) < 0.02 else ""
            print(f"  {t:<8.0%} {a:<8.4f} {p:<8.4f} {int(mask.sum()):<8}{marker}")
            if abs(t - conf_thresh) < 0.02:
                best_hc = {"thresh": t, "acc": a, "count": int(mask.sum())}
    
    # Walk-forward validation
    wf = walk_forward_lstm(
        X, y, n_splits=10, seq_length=seq_length,
        conf_thresh=conf_thresh, hidden_size=hidden_size,
        num_layers=num_layers, epochs=30, device=device,
    )
    
    total_time = time.time() - start_time
    
    # Save model
    sym = symbol.replace("/", "_")
    out_dir = PROJECT_ROOT / "data" / "models_lstm_v1"
    out_dir.mkdir(exist_ok=True, parents=True)
    
    torch.save(model.state_dict(), out_dir / f"{sym}_lstm_v1.pt")
    
    # Save scaler and config
    import joblib
    joblib.dump(scaler, out_dir / f"{sym}_scaler_lstm_v1.pkl")
    
    with open(out_dir / f"{sym}_features_lstm_v1.json", "w") as f:
        json.dump(feature_cols, f)
    
    meta = {
        "symbol": symbol,
        "version": "lstm_v1",
        "asset_class": asset_class,
        "trained_at": datetime.now().isoformat(),
        "training_time_seconds": total_time,
        "config": {
            "horizon": pred_horizon,
            "seq_length": seq_length,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "min_move": min_move,
            "vol_mult": vol_mult,
            "confidence_threshold": conf_thresh,
        },
        "metrics": {
            "test_accuracy": float(acc),
            "test_auc": float(auc),
            "test_f1": float(f1),
            "hc_accuracy": best_hc["acc"],
            "hc_count": best_hc["count"],
            "total_samples": len(X),
            "train_samples": len(X_train_seq),
            "test_samples": len(X_test_seq),
        },
        "walk_forward": wf,
        "model_params": sum(p.numel() for p in model.parameters()),
    }
    
    with open(out_dir / f"{sym}_meta_lstm_v1.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"\n  Saved to {out_dir}")
    print(f"  Training time: {total_time:.1f}s")
    print(f"  OK {sym}: acc={acc:.4f}, hc={best_hc['acc']:.4f}, wf={wf['avg_acc']:.4f}")
    
    return {
        "symbol": symbol,
        "status": "success",
        "accuracy": float(acc),
        "auc": float(auc),
        "hc_accuracy": best_hc["acc"],
        "wf_accuracy": wf["avg_acc"],
        "wf_stable": wf["stable"],
        "training_time": total_time,
    }


# =============================================================================
# Comparison with V6
# =============================================================================
def load_v6_results(symbol: str) -> Dict:
    """Load V6 model results for comparison."""
    sym = symbol.replace("/", "_")
    meta_path = PROJECT_ROOT / "data" / "models_v6_improved" / f"{sym}_binary_meta_v6.json"
    
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {}


def compare_with_v6(lstm_result: Dict, v6_meta: Dict) -> Dict:
    """Compare LSTM results with V6."""
    comparison = {
        "symbol": lstm_result["symbol"],
        "lstm": {
            "test_accuracy": lstm_result["accuracy"],
            "test_auc": lstm_result.get("auc", 0),
            "hc_accuracy": lstm_result["hc_accuracy"],
            "wf_accuracy": lstm_result["wf_accuracy"],
            "wf_stable": lstm_result["wf_stable"],
            "training_time": lstm_result.get("training_time", 0),
        },
        "v6": {},
        "winner": "unknown",
    }
    
    if v6_meta:
        comparison["v6"] = {
            "test_accuracy": v6_meta.get("metrics", {}).get("test_accuracy", 0),
            "test_auc": v6_meta.get("metrics", {}).get("test_auc", 0),
            "hc_accuracy": v6_meta.get("metrics", {}).get("hc_accuracy", 0),
            "wf_accuracy": v6_meta.get("walk_forward", {}).get("avg_acc", 0),
            "wf_stable": v6_meta.get("walk_forward", {}).get("stable", False),
        }
        
        # Determine winner based on walk-forward accuracy (most important)
        lstm_wf = lstm_result["wf_accuracy"]
        v6_wf = comparison["v6"]["wf_accuracy"]
        
        if lstm_wf > v6_wf + 0.01:  # LSTM wins by >1%
            comparison["winner"] = "LSTM"
        elif v6_wf > lstm_wf + 0.01:  # V6 wins by >1%
            comparison["winner"] = "V6"
        else:
            comparison["winner"] = "TIE"
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description="Train LSTM v1 models")
    parser.add_argument("--symbols", nargs="+", default=["TSLA"], help="Symbols to train")
    parser.add_argument("--lookback", type=int, default=10000, help="Data lookback bars")
    parser.add_argument("--seq-length", type=int, default=24, help="Sequence length")
    parser.add_argument("--hidden-size", type=int, default=64, help="LSTM hidden size")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  LSTM v1 MODEL TRAINING")
    print("="*70)
    print(f"  Symbols: {args.symbols}")
    print(f"  Seq length: {args.seq_length}")
    print(f"  Hidden size: {args.hidden_size}")
    print(f"  Layers: {args.num_layers}")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("="*70)
    
    # Asset configs (same as V6)
    CONFIGS = {
        "TSLA": {"horizon": 8, "min_move": 0.015, "vol_mult": 0.4, "conf_thresh": 0.55, "asset_class": "stock"},
        "GOOGL": {"horizon": 8, "min_move": 0.010, "vol_mult": 0.4, "conf_thresh": 0.55, "asset_class": "stock"},
        "BTC_USDT": {"horizon": 3, "min_move": 0.015, "vol_mult": 0.4, "conf_thresh": 0.60, "asset_class": "crypto"},
        "ETH_USDT": {"horizon": 3, "min_move": 0.012, "vol_mult": 0.4, "conf_thresh": 0.58, "asset_class": "crypto"},
    }
    
    results = []
    comparisons = []
    
    for sym in args.symbols:
        cfg = CONFIGS.get(sym, {
            "horizon": 8, "min_move": 0.015, "vol_mult": 0.4,
            "conf_thresh": 0.55, "asset_class": "stock"
        })
        
        result = train_lstm(
            symbol=sym,
            pred_horizon=cfg["horizon"],
            lookback=args.lookback,
            min_move=cfg["min_move"],
            vol_mult=cfg["vol_mult"],
            conf_thresh=cfg["conf_thresh"],
            seq_length=args.seq_length,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            epochs=args.epochs,
            asset_class=cfg["asset_class"],
        )
        results.append(result)
        
        # Compare with V6
        if result["status"] == "success":
            v6_meta = load_v6_results(sym)
            comp = compare_with_v6(result, v6_meta)
            comparisons.append(comp)
    
    # Summary
    print("\n" + "="*70)
    print("  LSTM v1 vs V6 COMPARISON")
    print("="*70)
    
    for comp in comparisons:
        sym = comp["symbol"]
        lstm = comp["lstm"]
        v6 = comp["v6"]
        winner = comp["winner"]
        
        print(f"\n  {sym}:")
        print(f"    {'Metric':<20} {'LSTM':<12} {'V6':<12} {'Diff':<12}")
        print(f"    {'-'*56}")
        
        if v6:
            for metric in ["test_accuracy", "test_auc", "hc_accuracy", "wf_accuracy"]:
                lstm_val = lstm.get(metric, 0)
                v6_val = v6.get(metric, 0)
                diff = lstm_val - v6_val
                diff_str = f"{diff:+.4f}" if diff != 0 else "0"
                print(f"    {metric:<20} {lstm_val:<12.4f} {v6_val:<12.4f} {diff_str:<12}")
            
            print(f"    {'-'*56}")
            print(f"    Winner: {winner}")
            print(f"    Training time: {lstm.get('training_time', 0):.1f}s")
        else:
            print(f"    [No V6 results available for comparison]")
    
    # Save comparison
    out_dir = PROJECT_ROOT / "data" / "models_lstm_v1"
    out_dir.mkdir(exist_ok=True, parents=True)
    
    with open(out_dir / "comparison_lstm_vs_v6.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "comparisons": comparisons,
        }, f, indent=2)
    
    print(f"\n  Comparison saved to {out_dir / 'comparison_lstm_vs_v6.json'}")


if __name__ == "__main__":
    main()
