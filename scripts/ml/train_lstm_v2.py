#!/usr/bin/env python3
"""
LSTM Model v2 - Improved Architecture

Changes from v1:
- Bidirectional LSTM
- Larger sequence (48 steps)
- Attention mechanism
- Better regularization
- Feature selection
"""
from __future__ import annotations

import argparse
import json
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

from bot.ml.v6_feature_extractor import build_v6_features, get_v6_feature_cols


# =============================================================================
# Improved LSTM with Attention
# =============================================================================
class AttentionLayer(nn.Module):
    """Simple attention mechanism."""
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden_size)
        attn_weights = F.softmax(self.attention(lstm_output), dim=1)
        # Weighted sum
        context = torch.sum(attn_weights * lstm_output, dim=1)
        return context, attn_weights


class BiLSTMAttention(nn.Module):
    """Bidirectional LSTM with Attention."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True,
        )
        
        # Attention
        self.attention = AttentionLayer(hidden_size * 2)
        
        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden*2)
        
        # Attention
        context, _ = self.attention(lstm_out)  # (batch, hidden*2)
        
        # Classifier
        out = self.dropout(context)
        out = F.relu(self.bn1(self.fc1(out)))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = torch.sigmoid(self.fc3(out))
        
        return out.squeeze()


# =============================================================================
# Simple Transformer for comparison
# =============================================================================
class TransformerClassifier(nn.Module):
    """Simple Transformer for time series classification."""
    
    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
        seq_length: int = 48,
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_length, d_model) * 0.1)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classifier
        self.fc1 = nn.Linear(d_model, 32)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)
        
    def forward(self, x):
        # Project input
        x = self.input_proj(x)  # (batch, seq, d_model)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1), :]
        
        # Transformer
        x = self.transformer(x)  # (batch, seq, d_model)
        
        # Use last token for classification
        x = x[:, -1, :]  # (batch, d_model)
        
        # Classifier
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        
        return x.squeeze()


# =============================================================================
# Data Loading
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


def create_targets(df: pd.DataFrame, pred_horizon: int, min_move: float, vol_mult: float) -> pd.DataFrame:
    """Create binary targets with adaptive threshold."""
    c = df["close"]
    df = df.copy()
    df["fwd_return"] = c.pct_change(pred_horizon).shift(-pred_horizon)
    
    vol = c.pct_change().rolling(24).std().fillna(min_move)
    thresh = np.maximum(vol * vol_mult, min_move)
    
    df["target"] = np.nan
    df.loc[df["fwd_return"] > thresh, "target"] = 1
    df.loc[df["fwd_return"] < -thresh, "target"] = 0
    
    n_up = (df["target"] == 1).sum()
    n_down = (df["target"] == 0).sum()
    print(f"  Target: UP={n_up} ({n_up/len(df):.1%}), DOWN={n_down} ({n_down/len(df):.1%})")
    
    return df


def select_features(X, y, names, top_n=40):
    """Feature selection using MI + RF importance."""
    mi = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
    mi = mi / (mi.max() + 1e-8)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    rf_imp = rf.feature_importances_
    rf_imp = rf_imp / (rf_imp.max() + 1e-8)
    
    combined = 0.4 * mi + 0.6 * rf_imp
    ranked = sorted(zip(names, combined, range(len(names))), key=lambda x: x[1], reverse=True)
    
    selected = [(n, i) for n, _, i in ranked[:top_n]]
    selected_names = [n for n, _ in selected]
    selected_idx = [i for _, i in selected]
    
    print(f"  Feature selection: {len(names)} -> {top_n}")
    return selected_names, selected_idx


def create_sequences(X: np.ndarray, y: np.ndarray, seq_length: int = 48):
    """Create sequences for LSTM/Transformer input."""
    sequences, labels = [], []
    for i in range(seq_length, len(X)):
        sequences.append(X[i - seq_length:i])
        labels.append(y[i])
    return np.array(sequences), np.array(labels)


# =============================================================================
# Training
# =============================================================================
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).float()
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        correct += ((outputs > 0.5).float() == y_batch).sum().item()
        total += y_batch.size(0)
    
    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_probs, all_labels = [], [], []
    
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
    
    return total_loss / len(dataloader), np.array(all_preds), np.array(all_probs), np.array(all_labels)


def walk_forward(X, y, model_class, model_kwargs, n_splits=8, seq_length=48, epochs=30, device="cpu", conf_thresh=0.55):
    """Walk-forward validation."""
    n = len(X)
    test_size = max(n // (n_splits + 1), 50)
    results = []
    
    print(f"\n  Walk-Forward ({n_splits} folds):")
    print(f"  {'Fold':<6} {'Train':<8} {'Test':<8} {'Acc':<8} {'AUC':<8}")
    print(f"  {'-'*40}")
    
    for i in range(n_splits):
        train_end = test_size * (i + 2)
        test_start = train_end
        test_end = min(test_start + test_size, n)
        
        if test_end - test_start < 30:
            break
        
        # Scale and create sequences
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X[:train_end])
        X_test_s = scaler.transform(X[test_start:test_end])
        
        X_train_seq, y_train_seq = create_sequences(X_train_s, y[:train_end], seq_length)
        X_test_seq, y_test_seq = create_sequences(X_test_s, y[test_start:test_end], seq_length)
        
        if len(X_train_seq) < 50 or len(X_test_seq) < 10:
            continue
        
        # Train
        model = model_class(**model_kwargs).to(device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train_seq), torch.FloatTensor(y_train_seq)),
            batch_size=32, shuffle=True
        )
        test_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_test_seq), torch.FloatTensor(y_test_seq)),
            batch_size=32, shuffle=False
        )
        
        best_loss = float('inf')
        patience = 0
        for epoch in range(epochs):
            train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, _, _, _ = evaluate(model, test_loader, criterion, device)
            if val_loss < best_loss:
                best_loss = val_loss
                patience = 0
            else:
                patience += 1
                if patience >= 8:
                    break
        
        # Evaluate
        _, preds, probs, labels = evaluate(model, test_loader, criterion, device)
        acc = accuracy_score(labels, preds)
        auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
        
        results.append({"acc": acc, "auc": auc, "samples": test_end - test_start})
        print(f"  {i+1:<6} {train_end:<8} {test_end-test_start:<8} {acc:<8.4f} {auc:<8.4f}")
    
    avg_acc = np.mean([r["acc"] for r in results])
    avg_auc = np.mean([r["auc"] for r in results])
    std_acc = np.std([r["acc"] for r in results])
    
    print(f"  {'-'*40}")
    print(f"  AVG: acc={avg_acc:.4f}, auc={avg_auc:.4f}, std={std_acc:.4f}")
    
    return {"avg_acc": avg_acc, "avg_auc": avg_auc, "std": std_acc, "folds": len(results)}


# =============================================================================
# Main
# =============================================================================
def train_model(
    symbol: str,
    model_type: str = "bilstm",
    pred_horizon: int = 8,
    lookback: int = 10000,
    min_move: float = 0.015,
    vol_mult: float = 0.4,
    seq_length: int = 48,
    hidden_size: int = 64,
    epochs: int = 60,
    conf_thresh: float = 0.55,
    asset_class: str = "stock",
) -> Dict:
    """Train LSTM/Transformer model."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*70}")
    print(f"  {model_type.upper()} v2 Training: {symbol}")
    print(f"  horizon={pred_horizon}h, seq={seq_length}, device={device}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    # Load data
    df = load_data(symbol, lookback)
    if df.empty or len(df) < 500:
        return {"symbol": symbol, "status": "failed", "reason": "no_data"}
    
    # Build features
    feat = build_v6_features(df, pred_horizon, asset_class)
    feat = create_targets(feat, pred_horizon, min_move, vol_mult)
    feature_cols = get_v6_feature_cols(feat)
    
    # Filter valid samples
    signal = feat.dropna(subset=["target"]).dropna(subset=feature_cols, how="any")
    if len(signal) < 500:
        return {"symbol": symbol, "status": "failed", "reason": "few_signals"}
    
    X = signal[feature_cols].values
    y = signal["target"].values.astype(int)
    X = np.nan_to_num(np.where(np.isinf(X), 0, X), nan=0.0)
    
    print(f"  Samples: {len(X)}, Features: {len(feature_cols)}")
    print(f"  Class balance: UP={y.sum()} ({y.mean():.1%})")
    
    # Feature selection
    split = int(len(X) * 0.8)
    selected_names, selected_idx = select_features(X[:split], y[:split], feature_cols, top_n=40)
    X = X[:, selected_idx]
    
    # Scale and create sequences
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X[:split])
    X_test_s = scaler.transform(X[split:])
    
    X_train_seq, y_train_seq = create_sequences(X_train_s, y[:split], seq_length)
    X_test_seq, y_test_seq = create_sequences(X_test_s, y[split:], seq_length)
    
    print(f"  Train: {X_train_seq.shape}, Test: {X_test_seq.shape}")
    
    # Build model
    input_size = len(selected_idx)
    if model_type == "bilstm":
        model = BiLSTMAttention(input_size=input_size, hidden_size=hidden_size, num_layers=2, dropout=0.4).to(device)
    else:
        model = TransformerClassifier(input_size=input_size, d_model=hidden_size, nhead=4, num_layers=2, dropout=0.3, seq_length=seq_length).to(device)
    
    print(f"  Model: {sum(p.numel() for p in model.parameters())} parameters")
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_seq), torch.FloatTensor(y_train_seq)), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test_seq), torch.FloatTensor(y_test_seq)), batch_size=32, shuffle=False)
    
    # Train
    print(f"\n  Training ({epochs} epochs)...")
    best_loss, best_state, patience_counter = float('inf'), None, 0
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, _, _, _ = evaluate(model, test_loader, criterion, device)
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 12:
                print(f"    Early stopping at epoch {epoch+1}")
                break
    
    if best_state:
        model.load_state_dict(best_state)
    
    # Evaluate
    _, y_pred, y_prob, y_true = evaluate(model, test_loader, criterion, device)
    
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    f1 = f1_score(y_true, y_pred)
    
    print(f"\n  === TEST RESULTS ===")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  AUC-ROC:  {auc:.4f}")
    print(f"  F1:       {f1:.4f}")
    
    # Walk-forward
    model_class = BiLSTMAttention if model_type == "bilstm" else TransformerClassifier
    model_kwargs = {"input_size": input_size, "hidden_size": hidden_size}
    if model_type == "transformer":
        model_kwargs = {"input_size": input_size, "d_model": hidden_size, "seq_length": seq_length}
    
    wf = walk_forward(X, y, model_class, model_kwargs, n_splits=8, seq_length=seq_length, epochs=25, device=device, conf_thresh=conf_thresh)
    
    total_time = time.time() - start_time
    
    # Save
    sym = symbol.replace("/", "_")
    out_dir = PROJECT_ROOT / "data" / f"models_{model_type}_v2"
    out_dir.mkdir(exist_ok=True, parents=True)
    
    torch.save(model.state_dict(), out_dir / f"{sym}_{model_type}_v2.pt")
    
    import joblib
    joblib.dump(scaler, out_dir / f"{sym}_scaler_{model_type}_v2.pkl")
    
    with open(out_dir / f"{sym}_features_{model_type}_v2.json", "w") as f:
        json.dump(selected_names, f)
    
    meta = {
        "symbol": symbol,
        "model_type": model_type,
        "version": "v2",
        "trained_at": datetime.now().isoformat(),
        "training_time": total_time,
        "config": {
            "horizon": pred_horizon,
            "seq_length": seq_length,
            "hidden_size": hidden_size,
        },
        "metrics": {
            "test_accuracy": float(acc),
            "test_auc": float(auc),
            "test_f1": float(f1),
        },
        "walk_forward": wf,
    }
    
    with open(out_dir / f"{sym}_meta_{model_type}_v2.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"\n  Saved to {out_dir}")
    print(f"  Time: {total_time:.1f}s")
    
    return {
        "symbol": symbol,
        "model_type": model_type,
        "status": "success",
        "accuracy": float(acc),
        "auc": float(auc),
        "wf_accuracy": wf["avg_acc"],
        "training_time": total_time,
    }


def load_v6_results(symbol: str) -> Dict:
    """Load V6 results for comparison."""
    path = PROJECT_ROOT / "data" / "models_v6_improved" / f"{symbol}_binary_meta_v6.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=["TSLA"])
    parser.add_argument("--model-type", choices=["bilstm", "transformer"], default="bilstm")
    parser.add_argument("--seq-length", type=int, default=48)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()
    
    CONFIGS = {
        "TSLA": {"horizon": 8, "min_move": 0.015, "vol_mult": 0.4, "conf_thresh": 0.55, "asset_class": "stock"},
        "GOOGL": {"horizon": 8, "min_move": 0.010, "vol_mult": 0.4, "conf_thresh": 0.55, "asset_class": "stock"},
    }
    
    print(f"\n{'='*70}")
    print(f"  {args.model_type.upper()} v2 TRAINING")
    print(f"  Symbols: {args.symbols}")
    print(f"  Seq: {args.seq_length}, Hidden: {args.hidden_size}")
    print(f"{'='*70}")
    
    results = []
    for sym in args.symbols:
        cfg = CONFIGS.get(sym, {"horizon": 8, "min_move": 0.015, "vol_mult": 0.4, "conf_thresh": 0.55, "asset_class": "stock"})
        
        result = train_model(
            symbol=sym,
            model_type=args.model_type,
            pred_horizon=cfg["horizon"],
            min_move=cfg["min_move"],
            vol_mult=cfg["vol_mult"],
            seq_length=args.seq_length,
            hidden_size=args.hidden_size,
            epochs=args.epochs,
            conf_thresh=cfg["conf_thresh"],
            asset_class=cfg["asset_class"],
        )
        results.append(result)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"  COMPARISON: {args.model_type.upper()} v2 vs V6")
    print(f"{'='*70}")
    
    for r in results:
        if r["status"] != "success":
            continue
        
        sym = r["symbol"]
        v6 = load_v6_results(sym)
        
        print(f"\n  {sym}:")
        print(f"    {'Metric':<16} {args.model_type.upper():<12} {'V6':<12} {'Diff':<10}")
        print(f"    {'-'*50}")
        
        if v6:
            v6_acc = v6.get("metrics", {}).get("test_accuracy", 0)
            v6_auc = v6.get("metrics", {}).get("test_auc", 0)
            v6_wf = v6.get("walk_forward", {}).get("avg_acc", 0)
            
            for metric, dl_val, v6_val in [
                ("Test Acc", r["accuracy"], v6_acc),
                ("Test AUC", r["auc"], v6_auc),
                ("WF Acc", r["wf_accuracy"], v6_wf),
            ]:
                diff = dl_val - v6_val
                print(f"    {metric:<16} {dl_val:<12.4f} {v6_val:<12.4f} {diff:+.4f}")
            
            winner = "DL" if r["wf_accuracy"] > v6_wf + 0.01 else "V6" if v6_wf > r["wf_accuracy"] + 0.01 else "TIE"
            print(f"    Winner: {winner}")


if __name__ == "__main__":
    main()
