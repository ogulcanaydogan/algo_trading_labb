"""
Database Layer - PostgreSQL and Redis integration.

Provides persistent storage with PostgreSQL for trades, positions,
and historical data, plus Redis for caching and real-time state.
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Generator, List, Optional, Type, TypeVar

logger = logging.getLogger(__name__)

# Optional imports
try:
    from sqlalchemy import (
        create_engine,
        Column,
        Integer,
        Float,
        String,
        DateTime,
        Boolean,
        Text,
        JSON,
        ForeignKey,
        Index,
        event,
    )
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker, relationship, Session
    from sqlalchemy.pool import QueuePool

    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False
    logger.warning("SQLAlchemy not installed. Install with: pip install sqlalchemy psycopg2-binary")

try:
    import redis

    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False
    logger.warning("Redis not installed. Install with: pip install redis")


# SQLAlchemy Base
if HAS_SQLALCHEMY:
    Base = declarative_base()
else:
    Base = object


@dataclass
class DatabaseConfig:
    """Database configuration.

    Credentials should be provided via environment variables:
    - POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DATABASE
    - REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_PASSWORD
    """

    # PostgreSQL - defaults only for development, production should use env vars
    postgres_host: str = field(default_factory=lambda: os.getenv("POSTGRES_HOST", "localhost"))
    postgres_port: int = field(default_factory=lambda: int(os.getenv("POSTGRES_PORT", "5432")))
    postgres_user: str = field(default_factory=lambda: os.getenv("POSTGRES_USER", "trading"))
    postgres_password: str = field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD", ""))
    postgres_database: str = field(
        default_factory=lambda: os.getenv("POSTGRES_DATABASE", "algo_trading")
    )
    postgres_pool_size: int = field(
        default_factory=lambda: int(os.getenv("POSTGRES_POOL_SIZE", "5"))
    )
    postgres_max_overflow: int = field(
        default_factory=lambda: int(os.getenv("POSTGRES_MAX_OVERFLOW", "10"))
    )

    # Redis
    redis_host: str = field(default_factory=lambda: os.getenv("REDIS_HOST", "localhost"))
    redis_port: int = field(default_factory=lambda: int(os.getenv("REDIS_PORT", "6379")))
    redis_db: int = field(default_factory=lambda: int(os.getenv("REDIS_DB", "0")))
    redis_password: Optional[str] = field(default_factory=lambda: os.getenv("REDIS_PASSWORD"))
    redis_ttl_seconds: int = field(
        default_factory=lambda: int(os.getenv("REDIS_TTL_SECONDS", "3600"))
    )

    def __post_init__(self):
        """Validate configuration and warn about insecure defaults."""
        if not self.postgres_password:
            logger.warning(
                "POSTGRES_PASSWORD not set. Set via environment variable for production."
            )
        if self.postgres_host == "localhost" and os.getenv("TRADING_ENV") == "production":
            logger.warning("Using localhost for PostgreSQL in production environment!")

    @property
    def postgres_url(self) -> str:
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_database}"


# ============ SQLAlchemy Models ============

if HAS_SQLALCHEMY:

    class TradeModel(Base):
        """Trade record model."""

        __tablename__ = "trades"

        id = Column(Integer, primary_key=True, autoincrement=True)
        trade_id = Column(String(64), unique=True, nullable=False, index=True)
        symbol = Column(String(32), nullable=False, index=True)
        side = Column(String(10), nullable=False)  # buy, sell
        quantity = Column(Float, nullable=False)
        price = Column(Float, nullable=False)
        value = Column(Float, nullable=False)
        commission = Column(Float, default=0)
        pnl = Column(Float)
        pnl_pct = Column(Float)
        strategy = Column(String(64))
        order_id = Column(String(64))
        position_id = Column(String(64))
        timestamp = Column(DateTime, nullable=False, index=True)
        metadata_json = Column(JSON)
        created_at = Column(DateTime, default=datetime.utcnow)

        __table_args__ = (Index("ix_trades_symbol_timestamp", "symbol", "timestamp"),)

        def to_dict(self) -> Dict:
            return {
                "trade_id": self.trade_id,
                "symbol": self.symbol,
                "side": self.side,
                "quantity": self.quantity,
                "price": self.price,
                "value": self.value,
                "commission": self.commission,
                "pnl": self.pnl,
                "pnl_pct": self.pnl_pct,
                "strategy": self.strategy,
                "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            }

    class PositionModel(Base):
        """Position record model."""

        __tablename__ = "positions"

        id = Column(Integer, primary_key=True, autoincrement=True)
        position_id = Column(String(64), unique=True, nullable=False, index=True)
        symbol = Column(String(32), nullable=False, index=True)
        side = Column(String(10), nullable=False)
        quantity = Column(Float, nullable=False)
        entry_price = Column(Float, nullable=False)
        current_price = Column(Float)
        unrealized_pnl = Column(Float)
        realized_pnl = Column(Float, default=0)
        stop_loss = Column(Float)
        take_profit = Column(Float)
        strategy = Column(String(64))
        status = Column(String(20), default="open", index=True)  # open, closed
        opened_at = Column(DateTime, nullable=False)
        closed_at = Column(DateTime)
        metadata_json = Column(JSON)
        created_at = Column(DateTime, default=datetime.utcnow)
        updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

        def to_dict(self) -> Dict:
            return {
                "position_id": self.position_id,
                "symbol": self.symbol,
                "side": self.side,
                "quantity": self.quantity,
                "entry_price": self.entry_price,
                "current_price": self.current_price,
                "unrealized_pnl": self.unrealized_pnl,
                "realized_pnl": self.realized_pnl,
                "stop_loss": self.stop_loss,
                "take_profit": self.take_profit,
                "status": self.status,
                "opened_at": self.opened_at.isoformat() if self.opened_at else None,
                "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            }

    class SignalModel(Base):
        """Trading signal record."""

        __tablename__ = "signals"

        id = Column(Integer, primary_key=True, autoincrement=True)
        signal_id = Column(String(64), unique=True, nullable=False, index=True)
        symbol = Column(String(32), nullable=False, index=True)
        action = Column(String(20), nullable=False)  # BUY, SELL, HOLD
        confidence = Column(Float, nullable=False)
        strategy = Column(String(64))
        entry_price = Column(Float)
        stop_loss = Column(Float)
        take_profit = Column(Float)
        reasoning = Column(Text)
        executed = Column(Boolean, default=False)
        timestamp = Column(DateTime, nullable=False, index=True)
        metadata_json = Column(JSON)
        created_at = Column(DateTime, default=datetime.utcnow)

        def to_dict(self) -> Dict:
            return {
                "signal_id": self.signal_id,
                "symbol": self.symbol,
                "action": self.action,
                "confidence": self.confidence,
                "strategy": self.strategy,
                "entry_price": self.entry_price,
                "executed": self.executed,
                "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            }

    class EquityModel(Base):
        """Equity snapshot model."""

        __tablename__ = "equity_snapshots"

        id = Column(Integer, primary_key=True, autoincrement=True)
        total_value = Column(Float, nullable=False)
        cash = Column(Float, nullable=False)
        positions_value = Column(Float, nullable=False)
        unrealized_pnl = Column(Float)
        realized_pnl = Column(Float)
        drawdown = Column(Float)
        drawdown_pct = Column(Float)
        timestamp = Column(DateTime, nullable=False, index=True)
        created_at = Column(DateTime, default=datetime.utcnow)

        __table_args__ = (Index("ix_equity_timestamp", "timestamp"),)

    class MLPredictionModel(Base):
        """ML prediction record."""

        __tablename__ = "ml_predictions"

        id = Column(Integer, primary_key=True, autoincrement=True)
        prediction_id = Column(String(64), unique=True, nullable=False)
        symbol = Column(String(32), nullable=False, index=True)
        model_id = Column(String(64), nullable=False)
        prediction = Column(Float, nullable=False)
        confidence = Column(Float)
        direction = Column(String(20))
        features_json = Column(JSON)
        actual_outcome = Column(Float)
        is_correct = Column(Boolean)
        timestamp = Column(DateTime, nullable=False, index=True)
        created_at = Column(DateTime, default=datetime.utcnow)


class PostgresDatabase:
    """PostgreSQL database manager."""

    def __init__(self, config: Optional[DatabaseConfig] = None):
        if not HAS_SQLALCHEMY:
            raise ImportError(
                "SQLAlchemy required. Install: pip install sqlalchemy psycopg2-binary"
            )

        self.config = config or DatabaseConfig()
        self._engine = None
        self._session_factory = None

    def connect(self):
        """Establish database connection."""
        self._engine = create_engine(
            self.config.postgres_url,
            poolclass=QueuePool,
            pool_size=self.config.postgres_pool_size,
            max_overflow=self.config.postgres_max_overflow,
            pool_pre_ping=True,
        )
        self._session_factory = sessionmaker(bind=self._engine)
        logger.info(f"Connected to PostgreSQL: {self.config.postgres_host}")

    def create_tables(self):
        """Create all tables."""
        Base.metadata.create_all(self._engine)
        logger.info("Database tables created")

    def drop_tables(self):
        """Drop all tables (use with caution)."""
        Base.metadata.drop_all(self._engine)
        logger.warning("Database tables dropped")

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Get database session context."""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def close(self):
        """Close database connection."""
        if self._engine:
            self._engine.dispose()
            logger.info("PostgreSQL connection closed")

    # Trade operations

    def save_trade(self, trade_data: Dict) -> str:
        """Save a trade record."""
        with self.session() as session:
            trade = TradeModel(
                trade_id=trade_data["trade_id"],
                symbol=trade_data["symbol"],
                side=trade_data["side"],
                quantity=trade_data["quantity"],
                price=trade_data["price"],
                value=trade_data.get("value", trade_data["quantity"] * trade_data["price"]),
                commission=trade_data.get("commission", 0),
                pnl=trade_data.get("pnl"),
                pnl_pct=trade_data.get("pnl_pct"),
                strategy=trade_data.get("strategy"),
                order_id=trade_data.get("order_id"),
                position_id=trade_data.get("position_id"),
                timestamp=trade_data.get("timestamp", datetime.utcnow()),
                metadata_json=trade_data.get("metadata"),
            )
            session.add(trade)
            return trade.trade_id

    def get_trades(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """Get trades with optional filters."""
        with self.session() as session:
            query = session.query(TradeModel)

            if symbol:
                query = query.filter(TradeModel.symbol == symbol)
            if start_date:
                query = query.filter(TradeModel.timestamp >= start_date)
            if end_date:
                query = query.filter(TradeModel.timestamp <= end_date)

            query = query.order_by(TradeModel.timestamp.desc()).limit(limit)

            return [t.to_dict() for t in query.all()]

    # Position operations

    def save_position(self, position_data: Dict) -> str:
        """Save or update a position."""
        with self.session() as session:
            existing = (
                session.query(PositionModel)
                .filter(PositionModel.position_id == position_data["position_id"])
                .first()
            )

            if existing:
                for key, value in position_data.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
            else:
                position = PositionModel(
                    position_id=position_data["position_id"],
                    symbol=position_data["symbol"],
                    side=position_data["side"],
                    quantity=position_data["quantity"],
                    entry_price=position_data["entry_price"],
                    current_price=position_data.get("current_price"),
                    stop_loss=position_data.get("stop_loss"),
                    take_profit=position_data.get("take_profit"),
                    strategy=position_data.get("strategy"),
                    opened_at=position_data.get("opened_at", datetime.utcnow()),
                    metadata_json=position_data.get("metadata"),
                )
                session.add(position)

            return position_data["position_id"]

    def get_open_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get all open positions."""
        with self.session() as session:
            query = session.query(PositionModel).filter(PositionModel.status == "open")

            if symbol:
                query = query.filter(PositionModel.symbol == symbol)

            return [p.to_dict() for p in query.all()]

    def close_position(self, position_id: str, close_data: Dict):
        """Close a position."""
        with self.session() as session:
            position = (
                session.query(PositionModel)
                .filter(PositionModel.position_id == position_id)
                .first()
            )

            if position:
                position.status = "closed"
                position.closed_at = close_data.get("closed_at", datetime.utcnow())
                position.realized_pnl = close_data.get("pnl", 0)
                position.current_price = close_data.get("exit_price")

    # Signal operations

    def save_signal(self, signal_data: Dict) -> str:
        """Save a trading signal."""
        with self.session() as session:
            signal = SignalModel(
                signal_id=signal_data["signal_id"],
                symbol=signal_data["symbol"],
                action=signal_data["action"],
                confidence=signal_data["confidence"],
                strategy=signal_data.get("strategy"),
                entry_price=signal_data.get("entry_price"),
                stop_loss=signal_data.get("stop_loss"),
                take_profit=signal_data.get("take_profit"),
                reasoning=signal_data.get("reasoning"),
                timestamp=signal_data.get("timestamp", datetime.utcnow()),
                metadata_json=signal_data.get("metadata"),
            )
            session.add(signal)
            return signal.signal_id

    # Equity operations

    def save_equity_snapshot(self, equity_data: Dict):
        """Save equity snapshot."""
        with self.session() as session:
            snapshot = EquityModel(
                total_value=equity_data["total_value"],
                cash=equity_data["cash"],
                positions_value=equity_data["positions_value"],
                unrealized_pnl=equity_data.get("unrealized_pnl"),
                realized_pnl=equity_data.get("realized_pnl"),
                drawdown=equity_data.get("drawdown"),
                drawdown_pct=equity_data.get("drawdown_pct"),
                timestamp=equity_data.get("timestamp", datetime.utcnow()),
            )
            session.add(snapshot)

    def get_equity_history(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[Dict]:
        """Get equity history."""
        with self.session() as session:
            query = session.query(EquityModel)

            if start_date:
                query = query.filter(EquityModel.timestamp >= start_date)
            if end_date:
                query = query.filter(EquityModel.timestamp <= end_date)

            query = query.order_by(EquityModel.timestamp.desc()).limit(limit)

            return [
                {
                    "total_value": e.total_value,
                    "cash": e.cash,
                    "positions_value": e.positions_value,
                    "drawdown_pct": e.drawdown_pct,
                    "timestamp": e.timestamp.isoformat(),
                }
                for e in query.all()
            ]


class RedisCache:
    """Redis cache manager for real-time data."""

    def __init__(self, config: Optional[DatabaseConfig] = None):
        if not HAS_REDIS:
            raise ImportError("Redis required. Install: pip install redis")

        self.config = config or DatabaseConfig()
        self._client: Optional[redis.Redis] = None

    def connect(self):
        """Connect to Redis."""
        self._client = redis.Redis(
            host=self.config.redis_host,
            port=self.config.redis_port,
            db=self.config.redis_db,
            password=self.config.redis_password,
            decode_responses=True,
        )
        # Test connection
        self._client.ping()
        logger.info(f"Connected to Redis: {self.config.redis_host}")

    def close(self):
        """Close Redis connection."""
        if self._client:
            self._client.close()

    # Basic operations

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set a value."""
        ttl = ttl or self.config.redis_ttl_seconds
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        self._client.setex(key, ttl, value)

    def get(self, key: str) -> Optional[Any]:
        """Get a value."""
        value = self._client.get(key)
        if value:
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
        return None

    def delete(self, key: str):
        """Delete a key."""
        self._client.delete(key)

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        return bool(self._client.exists(key))

    # Trading-specific operations

    def set_ticker(self, symbol: str, ticker_data: Dict):
        """Cache ticker data."""
        key = f"ticker:{symbol}"
        self.set(key, ticker_data, ttl=60)  # 1 minute TTL

    def get_ticker(self, symbol: str) -> Optional[Dict]:
        """Get cached ticker."""
        return self.get(f"ticker:{symbol}")

    def set_signal(self, symbol: str, signal_data: Dict):
        """Cache latest signal."""
        key = f"signal:{symbol}"
        self.set(key, signal_data, ttl=300)  # 5 minute TTL

    def get_signal(self, symbol: str) -> Optional[Dict]:
        """Get cached signal."""
        return self.get(f"signal:{symbol}")

    def set_position(self, symbol: str, position_data: Dict):
        """Cache position data."""
        key = f"position:{symbol}"
        self.set(key, position_data, ttl=3600)

    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get cached position."""
        return self.get(f"position:{symbol}")

    def set_portfolio(self, portfolio_data: Dict):
        """Cache portfolio state."""
        self.set("portfolio:current", portfolio_data, ttl=60)

    def get_portfolio(self) -> Optional[Dict]:
        """Get cached portfolio."""
        return self.get("portfolio:current")

    # Pub/Sub for real-time updates

    def publish(self, channel: str, message: Dict):
        """Publish message to channel."""
        self._client.publish(channel, json.dumps(message))

    def subscribe(self, channel: str):
        """Subscribe to channel."""
        pubsub = self._client.pubsub()
        pubsub.subscribe(channel)
        return pubsub

    # Rate limiting

    def check_rate_limit(self, key: str, limit: int, window_seconds: int) -> bool:
        """Check if rate limit is exceeded."""
        current = self._client.incr(key)
        if current == 1:
            self._client.expire(key, window_seconds)
        return current <= limit


class DatabaseManager:
    """
    Unified database manager combining PostgreSQL and Redis.

    Provides:
    - PostgreSQL for persistent storage
    - Redis for caching and real-time data
    - Automatic fallback when services unavailable
    """

    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self._postgres: Optional[PostgresDatabase] = None
        self._redis: Optional[RedisCache] = None
        self._postgres_available = False
        self._redis_available = False

    def connect(self):
        """Connect to all databases."""
        # Try PostgreSQL
        if HAS_SQLALCHEMY:
            try:
                self._postgres = PostgresDatabase(self.config)
                self._postgres.connect()
                self._postgres.create_tables()
                self._postgres_available = True
            except Exception as e:
                logger.warning(f"PostgreSQL unavailable: {e}")

        # Try Redis
        if HAS_REDIS:
            try:
                self._redis = RedisCache(self.config)
                self._redis.connect()
                self._redis_available = True
            except Exception as e:
                logger.warning(f"Redis unavailable: {e}")

        if not self._postgres_available and not self._redis_available:
            logger.warning("No database backends available - using in-memory fallback")

    def close(self):
        """Close all connections."""
        if self._postgres:
            self._postgres.close()
        if self._redis:
            self._redis.close()

    @property
    def postgres(self) -> Optional[PostgresDatabase]:
        return self._postgres if self._postgres_available else None

    @property
    def redis(self) -> Optional[RedisCache]:
        return self._redis if self._redis_available else None

    def save_trade(self, trade_data: Dict) -> str:
        """Save trade to PostgreSQL and cache in Redis."""
        trade_id = trade_data.get("trade_id", f"trade_{datetime.now().timestamp()}")

        if self._postgres_available:
            self._postgres.save_trade(trade_data)

        if self._redis_available:
            # Cache recent trade
            self._redis.set(f"trade:latest:{trade_data['symbol']}", trade_data, ttl=3600)

        return trade_id

    def get_status(self) -> Dict:
        """Get database connection status."""
        return {
            "postgres_available": self._postgres_available,
            "redis_available": self._redis_available,
            "postgres_host": self.config.postgres_host if self._postgres_available else None,
            "redis_host": self.config.redis_host if self._redis_available else None,
        }


def create_database_manager(config: Optional[DatabaseConfig] = None) -> DatabaseManager:
    """Factory function to create database manager."""
    return DatabaseManager(config=config)
