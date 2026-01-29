"""Initial schema for trading database.

Revision ID: 001_initial
Revises:
Create Date: 2026-01-28

This migration creates the initial database schema for the trading system.
Tables include:
- trades: Record of all executed trades
- positions: Current open positions
- equity_snapshots: Equity curve data points
- trade_signals: ML/TA signals that generated trades
- audit_events: Audit trail for all system events
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create initial database schema."""

    # Trades table - records all executed trades
    op.create_table(
        "trades",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("trade_id", sa.String(36), nullable=False, unique=True),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("side", sa.String(10), nullable=False),  # buy/sell
        sa.Column("quantity", sa.Float(), nullable=False),
        sa.Column("price", sa.Float(), nullable=False),
        sa.Column("fee", sa.Float(), nullable=True, default=0),
        sa.Column("pnl", sa.Float(), nullable=True),
        sa.Column("strategy", sa.String(50), nullable=True),
        sa.Column("signal_id", sa.String(36), nullable=True),
        sa.Column("order_type", sa.String(20), nullable=True),  # market/limit
        sa.Column("execution_time_ms", sa.Float(), nullable=True),
        sa.Column("slippage_pct", sa.Float(), nullable=True),
        sa.Column("market", sa.String(20), nullable=True),  # crypto/stock/commodity
        sa.Column("exchange", sa.String(20), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("metadata_json", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_trades_symbol", "trades", ["symbol"])
    op.create_index("ix_trades_created_at", "trades", ["created_at"])
    op.create_index("ix_trades_market", "trades", ["market"])

    # Positions table - current open positions
    op.create_table(
        "positions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False, unique=True),
        sa.Column("quantity", sa.Float(), nullable=False),
        sa.Column("entry_price", sa.Float(), nullable=False),
        sa.Column("current_price", sa.Float(), nullable=True),
        sa.Column("side", sa.String(10), nullable=False),  # long/short
        sa.Column("unrealized_pnl", sa.Float(), nullable=True),
        sa.Column("stop_loss", sa.Float(), nullable=True),
        sa.Column("take_profit", sa.Float(), nullable=True),
        sa.Column("entry_time", sa.DateTime(), nullable=False),
        sa.Column("last_updated", sa.DateTime(), nullable=False),
        sa.Column("market", sa.String(20), nullable=True),
        sa.Column("strategy", sa.String(50), nullable=True),
        sa.Column("signal_confidence", sa.Float(), nullable=True),
        sa.Column("metadata_json", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_positions_market", "positions", ["market"])

    # Equity snapshots - for equity curve
    op.create_table(
        "equity_snapshots",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("equity", sa.Float(), nullable=False),
        sa.Column("cash", sa.Float(), nullable=True),
        sa.Column("positions_value", sa.Float(), nullable=True),
        sa.Column("unrealized_pnl", sa.Float(), nullable=True),
        sa.Column("realized_pnl", sa.Float(), nullable=True),
        sa.Column("drawdown_pct", sa.Float(), nullable=True),
        sa.Column("market", sa.String(20), nullable=True),  # total/crypto/stock
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_equity_snapshots_timestamp", "equity_snapshots", ["timestamp"])
    op.create_index("ix_equity_snapshots_market", "equity_snapshots", ["market"])

    # Trade signals - ML/TA signals
    op.create_table(
        "trade_signals",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("signal_id", sa.String(36), nullable=False, unique=True),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("signal_type", sa.String(20), nullable=False),  # buy/sell/hold
        sa.Column("strength", sa.Float(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("model_type", sa.String(50), nullable=True),  # ensemble/technical/ml
        sa.Column("model_version", sa.String(20), nullable=True),
        sa.Column("prediction_id", sa.String(36), nullable=True),
        sa.Column("regime", sa.String(20), nullable=True),
        sa.Column("trend", sa.String(20), nullable=True),
        sa.Column("volatility", sa.String(20), nullable=True),
        sa.Column("rsi", sa.Float(), nullable=True),
        sa.Column("executed", sa.Boolean(), default=False),
        sa.Column("execution_trade_id", sa.String(36), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("metadata_json", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_trade_signals_symbol", "trade_signals", ["symbol"])
    op.create_index("ix_trade_signals_created_at", "trade_signals", ["created_at"])
    op.create_index("ix_trade_signals_model_type", "trade_signals", ["model_type"])

    # Audit events - system audit trail
    op.create_table(
        "audit_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("event_id", sa.String(36), nullable=False, unique=True),
        sa.Column("event_type", sa.String(50), nullable=False),
        sa.Column("severity", sa.String(20), nullable=False),  # info/warning/critical
        sa.Column("actor", sa.String(50), nullable=True),
        sa.Column("action", sa.Text(), nullable=False),
        sa.Column("resource", sa.String(100), nullable=True),
        sa.Column("resource_id", sa.String(50), nullable=True),
        sa.Column("ip_address", sa.String(45), nullable=True),
        sa.Column("user_agent", sa.String(255), nullable=True),
        sa.Column("correlation_id", sa.String(36), nullable=True),
        sa.Column("checksum", sa.String(64), nullable=True),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("details_json", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_audit_events_event_type", "audit_events", ["event_type"])
    op.create_index("ix_audit_events_timestamp", "audit_events", ["timestamp"])
    op.create_index("ix_audit_events_severity", "audit_events", ["severity"])
    op.create_index("ix_audit_events_correlation_id", "audit_events", ["correlation_id"])

    # Risk events - risk management events
    op.create_table(
        "risk_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("event_id", sa.String(36), nullable=False, unique=True),
        sa.Column("event_type", sa.String(50), nullable=False),
        sa.Column("symbol", sa.String(20), nullable=True),
        sa.Column("market", sa.String(20), nullable=True),
        sa.Column("trigger_value", sa.Float(), nullable=True),
        sa.Column("threshold_value", sa.Float(), nullable=True),
        sa.Column("action_taken", sa.String(100), nullable=True),
        sa.Column("positions_affected", sa.Integer(), nullable=True),
        sa.Column("pnl_impact", sa.Float(), nullable=True),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("metadata_json", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_risk_events_event_type", "risk_events", ["event_type"])
    op.create_index("ix_risk_events_timestamp", "risk_events", ["timestamp"])


def downgrade() -> None:
    """Drop all tables."""
    op.drop_table("risk_events")
    op.drop_table("audit_events")
    op.drop_table("trade_signals")
    op.drop_table("equity_snapshots")
    op.drop_table("positions")
    op.drop_table("trades")
