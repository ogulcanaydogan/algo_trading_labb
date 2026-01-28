"""
Centralized Data Storage Module.

Stores all trading data for persistence and server migration:
- Trade history (every executed trade)
- Portfolio snapshots (daily equity curves)
- Strategy performance metrics
- Signal history
- Model training results

All data stored in JSON format for easy backup and portability.
"""

import json
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

from bot.core.exceptions import DataError, handle_exceptions

logger = logging.getLogger(__name__)

# Base data directory
DATA_DIR = Path(__file__).parent.parent / "data"
STORAGE_DIR = DATA_DIR / "persistent_storage"


class DataStore:
    """
    Centralized data storage for all trading data.

    Stores:
    - trades: All executed trades with full details
    - snapshots: Daily portfolio snapshots
    - signals: Historical signal data
    - strategies: Strategy performance metrics
    - models: ML model training results
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = Path(storage_dir) if storage_dir else STORAGE_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Initialize storage files
        self.files = {
            "trades": self.storage_dir / "trade_history.json",
            "snapshots": self.storage_dir / "portfolio_snapshots.json",
            "signals": self.storage_dir / "signal_history.json",
            "strategies": self.storage_dir / "strategy_performance.json",
            "models": self.storage_dir / "model_registry.json",
            "config": self.storage_dir / "trading_config.json",
            "metadata": self.storage_dir / "metadata.json",
        }

        # Initialize files if they don't exist
        self._init_storage()

        logger.info(f"DataStore initialized at {self.storage_dir}")

    def _init_storage(self):
        """Initialize storage files with empty structures."""
        defaults = {
            "trades": {"trades": [], "last_updated": None},
            "snapshots": {"snapshots": [], "last_updated": None},
            "signals": {"signals": [], "last_updated": None},
            "strategies": {"strategies": {}, "last_updated": None},
            "models": {"models": {}, "last_updated": None},
            "config": {"config": {}, "last_updated": None},
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "version": "1.0",
                "last_backup": None,
                "total_trades": 0,
                "total_snapshots": 0,
            },
        }

        for key, filepath in self.files.items():
            if not filepath.exists():
                self._save_json(filepath, defaults.get(key, {}))

    @handle_exceptions(default={}, context="load_json")
    def _load_json(self, filepath: Path) -> Dict:
        """Load JSON file safely."""
        if filepath.exists():
            with open(filepath, "r") as f:
                return json.load(f)
        return {}

    def _save_json(self, filepath: Path, data: Dict):
        """Save JSON file safely with backup."""
        try:
            # Create backup of existing file
            if filepath.exists():
                backup_path = filepath.with_suffix(".json.bak")
                shutil.copy2(filepath, backup_path)

            # Write new data
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except (OSError, IOError) as e:
            logger.error(f"Error saving {filepath}: {e}")
            raise DataError(f"Failed to save data to {filepath}: {e}")
        except (TypeError, ValueError) as e:
            logger.error(f"Error serializing data for {filepath}: {e}")
            raise DataError(f"Failed to serialize data: {e}")

    # ==================== TRADE RECORDING ====================

    def record_trade(
        self,
        symbol: str,
        action: str,  # BUY, SELL
        quantity: float,
        price: float,
        market: str,  # crypto, commodity, stock
        regime: Optional[str] = None,
        confidence: Optional[float] = None,
        signal: Optional[str] = None,
        strategy: Optional[str] = None,
        pnl: Optional[float] = None,
        pnl_pct: Optional[float] = None,
        entry_price: Optional[float] = None,
        exit_reason: Optional[str] = None,
        dl_prediction: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Record a trade execution.

        Returns:
            trade_id: Unique identifier for the trade
        """
        trade_id = f"{market}_{symbol.replace('/', '_')}_{datetime.now().timestamp()}"

        trade = {
            "trade_id": trade_id,
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "price": price,
            "value": quantity * price,
            "market": market,
            "regime": regime,
            "confidence": confidence,
            "signal": signal,
            "strategy": strategy or "default",
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "entry_price": entry_price,
            "exit_reason": exit_reason,
            "dl_prediction": dl_prediction,
            "metadata": metadata or {},
        }

        # Load existing trades
        data = self._load_json(self.files["trades"])
        trades = data.get("trades", [])
        trades.append(trade)

        # Keep last 10000 trades
        if len(trades) > 10000:
            trades = trades[-10000:]

        data["trades"] = trades
        data["last_updated"] = datetime.now().isoformat()

        self._save_json(self.files["trades"], data)

        # Update metadata
        self._update_metadata("total_trades", len(trades))

        logger.info(f"Recorded trade: {action} {quantity} {symbol} @ ${price:.2f}")
        return trade_id

    def get_trades(
        self,
        market: Optional[str] = None,
        symbol: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """Get trades with optional filters."""
        data = self._load_json(self.files["trades"])
        trades = data.get("trades", [])

        # Apply filters
        if market:
            trades = [t for t in trades if t.get("market") == market]
        if symbol:
            trades = [t for t in trades if t.get("symbol") == symbol]
        if start_date:
            trades = [t for t in trades if t.get("timestamp", "") >= start_date]
        if end_date:
            trades = [t for t in trades if t.get("timestamp", "") <= end_date]

        # Sort by timestamp (newest first)
        trades.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        return trades[:limit]

    def get_trade_summary(self) -> Dict:
        """Get summary statistics of all trades."""
        data = self._load_json(self.files["trades"])
        trades = data.get("trades", [])

        if not trades:
            return {"total_trades": 0}

        total_pnl = sum(t.get("pnl", 0) or 0 for t in trades)
        wins = sum(1 for t in trades if (t.get("pnl") or 0) > 0)
        losses = sum(1 for t in trades if (t.get("pnl") or 0) < 0)

        by_market = {}
        for t in trades:
            m = t.get("market", "unknown")
            if m not in by_market:
                by_market[m] = {"count": 0, "pnl": 0}
            by_market[m]["count"] += 1
            by_market[m]["pnl"] += t.get("pnl", 0) or 0

        return {
            "total_trades": len(trades),
            "total_pnl": total_pnl,
            "wins": wins,
            "losses": losses,
            "win_rate": (wins / len(trades) * 100) if trades else 0,
            "by_market": by_market,
            "first_trade": trades[-1].get("timestamp") if trades else None,
            "last_trade": trades[0].get("timestamp") if trades else None,
        }

    # ==================== PORTFOLIO SNAPSHOTS ====================

    def record_snapshot(
        self,
        total_value: float,
        cash_balance: float,
        positions: Dict[str, Dict],
        pnl: float,
        pnl_pct: float,
        market_values: Optional[Dict[str, float]] = None,
    ):
        """Record a portfolio snapshot."""
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "total_value": total_value,
            "cash_balance": cash_balance,
            "positions_count": len(positions),
            "positions": positions,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "market_values": market_values or {},
        }

        data = self._load_json(self.files["snapshots"])
        snapshots = data.get("snapshots", [])

        # Only keep one snapshot per hour (to avoid too much data)
        current_hour = datetime.now().strftime("%Y-%m-%d %H:00")
        snapshots = [
            s for s in snapshots if not s.get("timestamp", "").startswith(current_hour[:13])
        ]
        snapshots.append(snapshot)

        # Keep last 30 days of hourly snapshots (720 snapshots)
        if len(snapshots) > 720:
            snapshots = snapshots[-720:]

        data["snapshots"] = snapshots
        data["last_updated"] = datetime.now().isoformat()

        self._save_json(self.files["snapshots"], data)
        self._update_metadata("total_snapshots", len(snapshots))

    def get_snapshots(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """Get portfolio snapshots."""
        data = self._load_json(self.files["snapshots"])
        snapshots = data.get("snapshots", [])

        if start_date:
            snapshots = [s for s in snapshots if s.get("date", "") >= start_date]
        if end_date:
            snapshots = [s for s in snapshots if s.get("date", "") <= end_date]

        return snapshots[-limit:]

    def get_equity_curve(self, days: int = 30) -> List[Dict]:
        """Get equity curve data for charting."""
        data = self._load_json(self.files["snapshots"])
        snapshots = data.get("snapshots", [])

        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        recent = [s for s in snapshots if s.get("timestamp", "") >= cutoff]

        return [
            {
                "timestamp": s["timestamp"],
                "value": s["total_value"],
                "pnl": s["pnl"],
            }
            for s in recent
        ]

    # ==================== SIGNAL HISTORY ====================

    def record_signal(
        self,
        symbol: str,
        market: str,
        signal: str,  # LONG, SHORT, FLAT
        regime: str,
        confidence: float,
        price: float,
        dl_prediction: Optional[Dict] = None,
    ):
        """Record a trading signal."""
        signal_data = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "market": market,
            "signal": signal,
            "regime": regime,
            "confidence": confidence,
            "price": price,
            "dl_prediction": dl_prediction,
        }

        data = self._load_json(self.files["signals"])
        signals = data.get("signals", [])
        signals.append(signal_data)

        # Keep last 5000 signals
        if len(signals) > 5000:
            signals = signals[-5000:]

        data["signals"] = signals
        data["last_updated"] = datetime.now().isoformat()

        self._save_json(self.files["signals"], data)

    def get_signals(
        self,
        symbol: Optional[str] = None,
        market: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """Get historical signals."""
        data = self._load_json(self.files["signals"])
        signals = data.get("signals", [])

        if symbol:
            signals = [s for s in signals if s.get("symbol") == symbol]
        if market:
            signals = [s for s in signals if s.get("market") == market]

        return signals[-limit:]

    # ==================== STRATEGY PERFORMANCE ====================

    def update_strategy_performance(
        self,
        strategy_id: str,
        metrics: Dict[str, Any],
    ):
        """Update strategy performance metrics."""
        data = self._load_json(self.files["strategies"])
        strategies = data.get("strategies", {})

        if strategy_id not in strategies:
            strategies[strategy_id] = {
                "created_at": datetime.now().isoformat(),
                "history": [],
            }

        strategies[strategy_id]["last_updated"] = datetime.now().isoformat()
        strategies[strategy_id]["current_metrics"] = metrics

        # Keep history of metrics (daily)
        today = datetime.now().strftime("%Y-%m-%d")
        history = strategies[strategy_id].get("history", [])
        history = [h for h in history if h.get("date") != today]
        history.append({"date": today, "metrics": metrics})
        strategies[strategy_id]["history"] = history[-365:]  # Keep 1 year

        data["strategies"] = strategies
        data["last_updated"] = datetime.now().isoformat()

        self._save_json(self.files["strategies"], data)

    def get_strategy_performance(self, strategy_id: Optional[str] = None) -> Dict:
        """Get strategy performance data."""
        data = self._load_json(self.files["strategies"])
        strategies = data.get("strategies", {})

        if strategy_id:
            return strategies.get(strategy_id, {})
        return strategies

    # ==================== MODEL REGISTRY ====================

    def register_model(
        self,
        model_id: str,
        model_type: str,  # lstm, transformer, ensemble
        symbol: str,
        metrics: Dict[str, float],
        hyperparameters: Dict[str, Any],
        file_path: Optional[str] = None,
    ):
        """Register a trained ML model."""
        data = self._load_json(self.files["models"])
        models = data.get("models", {})

        models[model_id] = {
            "model_id": model_id,
            "model_type": model_type,
            "symbol": symbol,
            "created_at": datetime.now().isoformat(),
            "metrics": metrics,
            "hyperparameters": hyperparameters,
            "file_path": file_path,
            "active": True,
        }

        data["models"] = models
        data["last_updated"] = datetime.now().isoformat()

        self._save_json(self.files["models"], data)
        logger.info(f"Registered model: {model_id}")

    def get_models(self, model_type: Optional[str] = None, symbol: Optional[str] = None) -> Dict:
        """Get registered models."""
        data = self._load_json(self.files["models"])
        models = data.get("models", {})

        if model_type:
            models = {k: v for k, v in models.items() if v.get("model_type") == model_type}
        if symbol:
            models = {k: v for k, v in models.items() if v.get("symbol") == symbol}

        return models

    # ==================== CONFIG STORAGE ====================

    def save_config(self, config_name: str, config_data: Dict):
        """Save a configuration."""
        data = self._load_json(self.files["config"])
        configs = data.get("config", {})

        configs[config_name] = {
            "data": config_data,
            "saved_at": datetime.now().isoformat(),
        }

        data["config"] = configs
        data["last_updated"] = datetime.now().isoformat()

        self._save_json(self.files["config"], data)

    def get_config(self, config_name: str) -> Optional[Dict]:
        """Get a saved configuration."""
        data = self._load_json(self.files["config"])
        configs = data.get("config", {})
        return configs.get(config_name, {}).get("data")

    # ==================== BACKUP / EXPORT ====================

    def _update_metadata(self, key: str, value: Any):
        """Update metadata."""
        data = self._load_json(self.files["metadata"])
        data[key] = value
        data["last_updated"] = datetime.now().isoformat()
        self._save_json(self.files["metadata"], data)

    def create_backup(self, backup_dir: Optional[Path] = None) -> str:
        """
        Create a full backup of all data.

        Returns:
            Path to the backup directory
        """
        backup_dir = backup_dir or self.storage_dir.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"backup_{timestamp}"
        backup_path.mkdir(parents=True, exist_ok=True)

        # Copy all storage files
        for name, filepath in self.files.items():
            if filepath.exists():
                shutil.copy2(filepath, backup_path / filepath.name)

        # Also backup ML models directory
        models_dir = DATA_DIR / "models"
        if models_dir.exists():
            shutil.copytree(models_dir, backup_path / "models", dirs_exist_ok=True)

        # Update metadata
        self._update_metadata("last_backup", timestamp)

        logger.info(f"Created backup at {backup_path}")
        return str(backup_path)

    def export_all(self, export_path: Optional[Path] = None) -> str:
        """
        Export all data to a single JSON file for server migration.

        Returns:
            Path to the export file
        """
        export_path = (
            export_path
            or self.storage_dir.parent / f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "version": "1.0",
        }

        for name, filepath in self.files.items():
            export_data[name] = self._load_json(filepath)

        with open(export_path, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info(f"Exported all data to {export_path}")
        return str(export_path)

    def import_data(self, import_path: Path):
        """
        Import data from an export file.

        Args:
            import_path: Path to the export JSON file
        """
        with open(import_path, "r") as f:
            import_data = json.load(f)

        for name, filepath in self.files.items():
            if name in import_data and name != "metadata":
                self._save_json(filepath, import_data[name])

        logger.info(f"Imported data from {import_path}")

    def get_storage_stats(self) -> Dict:
        """Get storage statistics."""
        stats = {
            "storage_dir": str(self.storage_dir),
            "files": {},
        }

        total_size = 0
        for name, filepath in self.files.items():
            if filepath.exists():
                size = filepath.stat().st_size
                total_size += size
                stats["files"][name] = {
                    "path": str(filepath),
                    "size_bytes": size,
                    "size_kb": round(size / 1024, 2),
                }

        stats["total_size_kb"] = round(total_size / 1024, 2)
        stats["total_size_mb"] = round(total_size / 1024 / 1024, 2)

        # Add metadata
        metadata = self._load_json(self.files["metadata"])
        stats["metadata"] = metadata

        return stats


# Global instance
_data_store: Optional[DataStore] = None


def get_data_store() -> DataStore:
    """Get the global DataStore instance."""
    global _data_store
    if _data_store is None:
        _data_store = DataStore()
    return _data_store


# Convenience functions
def record_trade(**kwargs) -> str:
    """Record a trade (convenience function)."""
    return get_data_store().record_trade(**kwargs)


def record_snapshot(**kwargs):
    """Record a portfolio snapshot (convenience function)."""
    return get_data_store().record_snapshot(**kwargs)


def record_signal(**kwargs):
    """Record a signal (convenience function)."""
    return get_data_store().record_signal(**kwargs)


def create_backup() -> str:
    """Create a backup (convenience function)."""
    return get_data_store().create_backup()


def export_all() -> str:
    """Export all data (convenience function)."""
    return get_data_store().export_all()
