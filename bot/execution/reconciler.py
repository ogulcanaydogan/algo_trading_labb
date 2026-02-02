"""
Reconciliation and Idempotency Module.

Provides crash-safe trade processing and state reconstruction:
- Idempotent trade processing (safe to replay)
- Restart-safe state reconstruction
- Position reconciliation with exchange
- Order state recovery
- Transaction logging for audit trail

Critical for real-money deployment to ensure no duplicate trades
or lost state after system restarts.
"""

import hashlib
import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class ReconciliationStatus(Enum):
    """Status of reconciliation check."""

    MATCHED = "matched"  # Local matches exchange
    DISCREPANCY = "discrepancy"  # Local differs from exchange
    MISSING_LOCAL = "missing_local"  # On exchange but not locally
    MISSING_EXCHANGE = "missing_exchange"  # Locally but not on exchange
    PENDING = "pending"  # Waiting for confirmation


class TransactionState(Enum):
    """State of a tracked transaction."""

    PENDING = "pending"  # Submitted, awaiting confirmation
    CONFIRMED = "confirmed"  # Confirmed by exchange
    PROCESSED = "processed"  # Fully processed by learning systems
    FAILED = "failed"  # Transaction failed
    ROLLED_BACK = "rolled_back"  # Transaction was rolled back


@dataclass
class TransactionRecord:
    """Record of a trade transaction for idempotency."""

    transaction_id: str  # Unique ID (hash of key fields)
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    timestamp: datetime
    state: TransactionState
    exchange_order_id: Optional[str] = None
    fill_id: Optional[str] = None
    processed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PositionRecord:
    """Local position record for reconciliation."""

    symbol: str
    quantity: float
    average_entry_price: float
    unrealized_pnl: float
    last_updated: datetime
    open_orders: List[str] = field(default_factory=list)


@dataclass
class ReconciliationResult:
    """Result of position reconciliation."""

    symbol: str
    status: ReconciliationStatus
    local_position: Optional[PositionRecord]
    exchange_position: Optional[Dict[str, Any]]
    discrepancy_details: Optional[str] = None
    recommended_action: Optional[str] = None


@dataclass
class ReconcilerConfig:
    """Configuration for the reconciler."""

    # Database path
    db_path: Path = field(default_factory=lambda: Path("data/reconciliation.db"))

    # Transaction retention
    retention_days: int = 30
    max_pending_age_minutes: int = 60  # Max time before pending is considered stale

    # Reconciliation thresholds
    quantity_tolerance: float = 0.0001  # Acceptable quantity difference
    price_tolerance: float = 0.01  # Acceptable price difference (1%)

    # Idempotency window
    idempotency_window_seconds: int = 3600  # 1 hour window for duplicate detection

    # Checkpoint interval
    checkpoint_interval_seconds: int = 60


class Reconciler:
    """
    Reconciliation and idempotency manager.

    Ensures:
    1. No duplicate trade processing (idempotency)
    2. Safe restart with state reconstruction
    3. Position reconciliation with exchange
    4. Full audit trail

    Thread-safe singleton pattern.
    """

    _instance: Optional["Reconciler"] = None
    _lock = RLock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Optional[ReconcilerConfig] = None):
        if self._initialized:
            return

        self._config = config or ReconcilerConfig()
        self._lock = RLock()

        # In-memory caches
        self._transaction_cache: Dict[str, TransactionRecord] = {}
        self._position_cache: Dict[str, PositionRecord] = {}
        self._processed_ids: Set[str] = set()  # Quick lookup for idempotency

        # Initialize database
        self._init_database()

        # Load state
        self._load_state()

        self._initialized = True
        logger.info("Reconciler initialized")

    def _init_database(self) -> None:
        """Initialize SQLite database for persistence."""
        self._config.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(str(self._config.db_path)) as conn:
            cursor = conn.cursor()

            # Transaction log table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS transactions (
                    transaction_id TEXT PRIMARY KEY,
                    order_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    state TEXT NOT NULL,
                    exchange_order_id TEXT,
                    fill_id TEXT,
                    processed_at TEXT,
                    error_message TEXT,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Position snapshot table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    quantity REAL NOT NULL,
                    average_entry_price REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL,
                    last_updated TEXT NOT NULL,
                    open_orders TEXT
                )
            """
            )

            # Checkpoint table for recovery
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    state_json TEXT NOT NULL
                )
            """
            )

            # Indices for performance
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON transactions(timestamp)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_transactions_state ON transactions(state)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_transactions_symbol ON transactions(symbol)"
            )

            conn.commit()

    def _load_state(self) -> None:
        """Load state from database on startup."""
        with sqlite3.connect(str(self._config.db_path)) as conn:
            cursor = conn.cursor()

            # Load recent transactions into cache
            cutoff = datetime.now() - timedelta(
                seconds=self._config.idempotency_window_seconds
            )
            cursor.execute(
                """
                SELECT transaction_id, order_id, symbol, side, quantity, price,
                       timestamp, state, exchange_order_id, fill_id, processed_at,
                       error_message, metadata
                FROM transactions
                WHERE timestamp > ?
            """,
                (cutoff.isoformat(),),
            )

            for row in cursor.fetchall():
                record = TransactionRecord(
                    transaction_id=row[0],
                    order_id=row[1],
                    symbol=row[2],
                    side=row[3],
                    quantity=row[4],
                    price=row[5],
                    timestamp=datetime.fromisoformat(row[6]),
                    state=TransactionState(row[7]),
                    exchange_order_id=row[8],
                    fill_id=row[9],
                    processed_at=(
                        datetime.fromisoformat(row[10]) if row[10] else None
                    ),
                    error_message=row[11],
                    metadata=json.loads(row[12]) if row[12] else {},
                )
                self._transaction_cache[record.transaction_id] = record

                if record.state in (
                    TransactionState.CONFIRMED,
                    TransactionState.PROCESSED,
                ):
                    self._processed_ids.add(record.transaction_id)

            # Load positions
            cursor.execute(
                """
                SELECT symbol, quantity, average_entry_price, unrealized_pnl,
                       last_updated, open_orders
                FROM positions
            """
            )

            for row in cursor.fetchall():
                record = PositionRecord(
                    symbol=row[0],
                    quantity=row[1],
                    average_entry_price=row[2],
                    unrealized_pnl=row[3],
                    last_updated=datetime.fromisoformat(row[4]),
                    open_orders=json.loads(row[5]) if row[5] else [],
                )
                self._position_cache[record.symbol] = record

            logger.info(
                f"Loaded {len(self._transaction_cache)} transactions, "
                f"{len(self._position_cache)} positions"
            )

    def generate_transaction_id(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        timestamp: datetime,
        order_id: Optional[str] = None,
    ) -> str:
        """
        Generate a deterministic transaction ID for idempotency.

        Same inputs will always produce same ID, enabling duplicate detection.
        """
        # Create deterministic hash from key fields
        key_string = f"{symbol}|{side}|{quantity:.8f}|{price:.8f}|{timestamp.isoformat()}"
        if order_id:
            key_string += f"|{order_id}"

        return hashlib.sha256(key_string.encode()).hexdigest()[:32]

    def is_duplicate(self, transaction_id: str) -> bool:
        """Check if a transaction has already been processed."""
        with self._lock:
            return transaction_id in self._processed_ids

    def check_idempotency(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        timestamp: datetime,
        order_id: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a trade can be processed (not a duplicate).

        Returns (can_process, transaction_id).
        """
        transaction_id = self.generate_transaction_id(
            symbol, side, quantity, price, timestamp, order_id
        )

        with self._lock:
            if transaction_id in self._processed_ids:
                logger.warning(f"Duplicate transaction detected: {transaction_id}")
                return False, transaction_id

            return True, transaction_id

    def begin_transaction(
        self,
        transaction_id: str,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        timestamp: datetime,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TransactionRecord:
        """
        Begin tracking a new transaction.

        Must be called before submitting order to exchange.
        """
        with self._lock:
            record = TransactionRecord(
                transaction_id=transaction_id,
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                timestamp=timestamp,
                state=TransactionState.PENDING,
                metadata=metadata or {},
            )

            self._transaction_cache[transaction_id] = record

            # Persist immediately
            self._persist_transaction(record)

            logger.debug(f"Transaction started: {transaction_id}")
            return record

    def confirm_transaction(
        self,
        transaction_id: str,
        exchange_order_id: str,
        fill_id: Optional[str] = None,
    ) -> bool:
        """
        Confirm a transaction was accepted by exchange.

        Called after receiving exchange confirmation.
        """
        with self._lock:
            if transaction_id not in self._transaction_cache:
                logger.error(f"Unknown transaction: {transaction_id}")
                return False

            record = self._transaction_cache[transaction_id]
            record.state = TransactionState.CONFIRMED
            record.exchange_order_id = exchange_order_id
            record.fill_id = fill_id

            self._persist_transaction(record)

            logger.debug(f"Transaction confirmed: {transaction_id}")
            return True

    def mark_processed(self, transaction_id: str) -> bool:
        """
        Mark a transaction as fully processed.

        Called after all learning systems have processed the trade.
        """
        with self._lock:
            if transaction_id not in self._transaction_cache:
                logger.error(f"Unknown transaction: {transaction_id}")
                return False

            record = self._transaction_cache[transaction_id]
            record.state = TransactionState.PROCESSED
            record.processed_at = datetime.now()

            self._processed_ids.add(transaction_id)
            self._persist_transaction(record)

            logger.debug(f"Transaction processed: {transaction_id}")
            return True

    def fail_transaction(self, transaction_id: str, error: str) -> bool:
        """Mark a transaction as failed."""
        with self._lock:
            if transaction_id not in self._transaction_cache:
                logger.error(f"Unknown transaction: {transaction_id}")
                return False

            record = self._transaction_cache[transaction_id]
            record.state = TransactionState.FAILED
            record.error_message = error

            self._persist_transaction(record)

            logger.warning(f"Transaction failed: {transaction_id} - {error}")
            return True

    def rollback_transaction(self, transaction_id: str, reason: str) -> bool:
        """Roll back a transaction (e.g., after exchange rejection)."""
        with self._lock:
            if transaction_id not in self._transaction_cache:
                logger.error(f"Unknown transaction: {transaction_id}")
                return False

            record = self._transaction_cache[transaction_id]
            record.state = TransactionState.ROLLED_BACK
            record.error_message = f"Rolled back: {reason}"

            self._persist_transaction(record)

            logger.warning(f"Transaction rolled back: {transaction_id} - {reason}")
            return True

    def _persist_transaction(self, record: TransactionRecord) -> None:
        """Persist a transaction record to database."""
        with sqlite3.connect(str(self._config.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO transactions
                (transaction_id, order_id, symbol, side, quantity, price,
                 timestamp, state, exchange_order_id, fill_id, processed_at,
                 error_message, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    record.transaction_id,
                    record.order_id,
                    record.symbol,
                    record.side,
                    record.quantity,
                    record.price,
                    record.timestamp.isoformat(),
                    record.state.value,
                    record.exchange_order_id,
                    record.fill_id,
                    record.processed_at.isoformat() if record.processed_at else None,
                    record.error_message,
                    json.dumps(record.metadata) if record.metadata else None,
                ),
            )
            conn.commit()

    def update_position(
        self,
        symbol: str,
        quantity: float,
        average_entry_price: float,
        unrealized_pnl: float = 0.0,
        open_orders: Optional[List[str]] = None,
    ) -> None:
        """Update local position record."""
        with self._lock:
            record = PositionRecord(
                symbol=symbol,
                quantity=quantity,
                average_entry_price=average_entry_price,
                unrealized_pnl=unrealized_pnl,
                last_updated=datetime.now(),
                open_orders=open_orders or [],
            )

            self._position_cache[symbol] = record

            # Persist to database
            with sqlite3.connect(str(self._config.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO positions
                    (symbol, quantity, average_entry_price, unrealized_pnl,
                     last_updated, open_orders)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        symbol,
                        quantity,
                        average_entry_price,
                        unrealized_pnl,
                        record.last_updated.isoformat(),
                        json.dumps(open_orders or []),
                    ),
                )
                conn.commit()

    def get_local_position(self, symbol: str) -> Optional[PositionRecord]:
        """Get local position record."""
        with self._lock:
            return self._position_cache.get(symbol)

    def reconcile_position(
        self,
        symbol: str,
        exchange_position: Dict[str, Any],
    ) -> ReconciliationResult:
        """
        Reconcile local position with exchange position.

        Args:
            symbol: Trading symbol
            exchange_position: Position data from exchange
                Expected keys: quantity, average_price, unrealized_pnl

        Returns:
            ReconciliationResult with status and details
        """
        with self._lock:
            local = self._position_cache.get(symbol)

            if local is None and not exchange_position:
                return ReconciliationResult(
                    symbol=symbol,
                    status=ReconciliationStatus.MATCHED,
                    local_position=None,
                    exchange_position=None,
                )

            if local is None:
                return ReconciliationResult(
                    symbol=symbol,
                    status=ReconciliationStatus.MISSING_LOCAL,
                    local_position=None,
                    exchange_position=exchange_position,
                    discrepancy_details=f"Position exists on exchange but not locally",
                    recommended_action="Sync position from exchange",
                )

            if not exchange_position or exchange_position.get("quantity", 0) == 0:
                if abs(local.quantity) > self._config.quantity_tolerance:
                    return ReconciliationResult(
                        symbol=symbol,
                        status=ReconciliationStatus.MISSING_EXCHANGE,
                        local_position=local,
                        exchange_position=exchange_position,
                        discrepancy_details=f"Local shows {local.quantity} but exchange shows 0",
                        recommended_action="Clear local position or investigate",
                    )
                else:
                    return ReconciliationResult(
                        symbol=symbol,
                        status=ReconciliationStatus.MATCHED,
                        local_position=local,
                        exchange_position=exchange_position,
                    )

            # Compare quantities
            exchange_qty = exchange_position.get("quantity", 0)
            qty_diff = abs(local.quantity - exchange_qty)

            if qty_diff > self._config.quantity_tolerance:
                return ReconciliationResult(
                    symbol=symbol,
                    status=ReconciliationStatus.DISCREPANCY,
                    local_position=local,
                    exchange_position=exchange_position,
                    discrepancy_details=f"Quantity mismatch: local={local.quantity}, exchange={exchange_qty}",
                    recommended_action="Sync with exchange data",
                )

            # Compare prices if both have positions
            exchange_price = exchange_position.get("average_price", 0)
            if local.average_entry_price > 0 and exchange_price > 0:
                price_diff_pct = abs(
                    local.average_entry_price - exchange_price
                ) / local.average_entry_price

                if price_diff_pct > self._config.price_tolerance:
                    return ReconciliationResult(
                        symbol=symbol,
                        status=ReconciliationStatus.DISCREPANCY,
                        local_position=local,
                        exchange_position=exchange_position,
                        discrepancy_details=f"Price mismatch: local={local.average_entry_price:.4f}, "
                        f"exchange={exchange_price:.4f}",
                        recommended_action="Verify fill prices and sync",
                    )

            return ReconciliationResult(
                symbol=symbol,
                status=ReconciliationStatus.MATCHED,
                local_position=local,
                exchange_position=exchange_position,
            )

    def get_pending_transactions(self) -> List[TransactionRecord]:
        """Get all pending transactions that may need recovery."""
        with self._lock:
            cutoff = datetime.now() - timedelta(
                minutes=self._config.max_pending_age_minutes
            )

            pending = []
            for record in self._transaction_cache.values():
                if record.state == TransactionState.PENDING:
                    if record.timestamp < cutoff:
                        # Stale pending transaction
                        logger.warning(
                            f"Stale pending transaction: {record.transaction_id}"
                        )
                    pending.append(record)

            return pending

    def get_unprocessed_transactions(self) -> List[TransactionRecord]:
        """Get confirmed transactions that haven't been fully processed."""
        with self._lock:
            return [
                record
                for record in self._transaction_cache.values()
                if record.state == TransactionState.CONFIRMED
            ]

    def create_checkpoint(self) -> None:
        """Create a state checkpoint for recovery."""
        with self._lock:
            state = {
                "timestamp": datetime.now().isoformat(),
                "positions": {
                    symbol: {
                        "quantity": p.quantity,
                        "average_entry_price": p.average_entry_price,
                        "unrealized_pnl": p.unrealized_pnl,
                        "last_updated": p.last_updated.isoformat(),
                    }
                    for symbol, p in self._position_cache.items()
                },
                "processed_count": len(self._processed_ids),
                "pending_count": len(
                    [
                        t
                        for t in self._transaction_cache.values()
                        if t.state == TransactionState.PENDING
                    ]
                ),
            }

            with sqlite3.connect(str(self._config.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO checkpoints (timestamp, state_json) VALUES (?, ?)",
                    (datetime.now().isoformat(), json.dumps(state)),
                )
                conn.commit()

            logger.debug("Checkpoint created")

    def get_last_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Get the most recent checkpoint."""
        with sqlite3.connect(str(self._config.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT timestamp, state_json
                FROM checkpoints
                ORDER BY id DESC
                LIMIT 1
            """
            )
            row = cursor.fetchone()

            if row:
                return {
                    "timestamp": row[0],
                    "state": json.loads(row[1]),
                }
            return None

    def cleanup_old_transactions(self) -> int:
        """Clean up old transaction records."""
        with self._lock:
            cutoff = datetime.now() - timedelta(days=self._config.retention_days)

            with sqlite3.connect(str(self._config.db_path)) as conn:
                cursor = conn.cursor()

                # Count before deletion
                cursor.execute(
                    "SELECT COUNT(*) FROM transactions WHERE timestamp < ?",
                    (cutoff.isoformat(),),
                )
                count = cursor.fetchone()[0]

                # Delete old records
                cursor.execute(
                    "DELETE FROM transactions WHERE timestamp < ?",
                    (cutoff.isoformat(),),
                )

                # Also clean up old checkpoints
                cursor.execute(
                    "DELETE FROM checkpoints WHERE timestamp < ?",
                    (cutoff.isoformat(),),
                )

                conn.commit()

            # Clean memory cache
            self._transaction_cache = {
                k: v
                for k, v in self._transaction_cache.items()
                if v.timestamp >= cutoff
            }

            logger.info(f"Cleaned up {count} old transaction records")
            return count

    def get_status(self) -> Dict[str, Any]:
        """Get reconciler status for monitoring."""
        with self._lock:
            pending = [
                t
                for t in self._transaction_cache.values()
                if t.state == TransactionState.PENDING
            ]
            confirmed = [
                t
                for t in self._transaction_cache.values()
                if t.state == TransactionState.CONFIRMED
            ]

            return {
                "cached_transactions": len(self._transaction_cache),
                "processed_ids": len(self._processed_ids),
                "cached_positions": len(self._position_cache),
                "pending_transactions": len(pending),
                "unprocessed_confirmed": len(confirmed),
                "positions": {
                    symbol: {
                        "quantity": p.quantity,
                        "avg_price": p.average_entry_price,
                        "pnl": p.unrealized_pnl,
                    }
                    for symbol, p in self._position_cache.items()
                },
            }


# Singleton accessor
_instance: Optional[Reconciler] = None


def get_reconciler(config: Optional[ReconcilerConfig] = None) -> Reconciler:
    """Get or create the singleton Reconciler instance."""
    global _instance
    if _instance is None:
        _instance = Reconciler(config=config)
    return _instance


def reset_reconciler() -> None:
    """Reset the singleton instance (for testing)."""
    global _instance
    if _instance is not None:
        _instance._initialized = False
    _instance = None
    Reconciler._instance = None
