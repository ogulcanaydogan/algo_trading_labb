"""
On-Chain Analytics - Blockchain data analysis for trading signals.

Tracks whale movements, exchange flows, and on-chain metrics
for crypto trading signals.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Deque, Dict, List, Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WhaleTransaction:
    """Large wallet transaction."""
    tx_hash: str
    asset: str
    amount: float
    usd_value: float
    from_address: str
    to_address: str
    from_type: str  # "exchange", "whale", "unknown"
    to_type: str
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_exchange_inflow(self) -> bool:
        """Transaction moving to exchange (potential sell)."""
        return self.to_type == "exchange" and self.from_type != "exchange"

    @property
    def is_exchange_outflow(self) -> bool:
        """Transaction moving from exchange (potential accumulation)."""
        return self.from_type == "exchange" and self.to_type != "exchange"

    def to_dict(self) -> Dict:
        return {
            "tx_hash": self.tx_hash[:16] + "...",
            "asset": self.asset,
            "amount": self.amount,
            "usd_value": self.usd_value,
            "from_type": self.from_type,
            "to_type": self.to_type,
            "is_exchange_inflow": self.is_exchange_inflow,
            "is_exchange_outflow": self.is_exchange_outflow,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ExchangeFlow:
    """Exchange inflow/outflow data."""
    asset: str
    exchange: str
    inflow: float
    outflow: float
    net_flow: float  # Negative = outflow (bullish)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def flow_ratio(self) -> float:
        """Inflow/Outflow ratio (>1 = more inflow = bearish)."""
        return self.inflow / self.outflow if self.outflow > 0 else 2.0

    def to_dict(self) -> Dict:
        return {
            "asset": self.asset,
            "exchange": self.exchange,
            "inflow": self.inflow,
            "outflow": self.outflow,
            "net_flow": self.net_flow,
            "flow_ratio": round(self.flow_ratio, 4),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class OnChainMetrics:
    """Aggregated on-chain metrics."""
    asset: str
    active_addresses: int
    transaction_count: int
    avg_transaction_value: float
    total_exchange_balance: float
    exchange_balance_change_24h: float
    whale_transaction_count: int
    whale_net_flow: float
    nvt_ratio: float  # Network Value to Transactions
    mvrv_ratio: float  # Market Value to Realized Value
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "asset": self.asset,
            "active_addresses": self.active_addresses,
            "transaction_count": self.transaction_count,
            "avg_transaction_value": round(self.avg_transaction_value, 2),
            "total_exchange_balance": self.total_exchange_balance,
            "exchange_balance_change_24h_pct": round(self.exchange_balance_change_24h * 100, 2),
            "whale_transaction_count": self.whale_transaction_count,
            "whale_net_flow": self.whale_net_flow,
            "nvt_ratio": round(self.nvt_ratio, 2),
            "mvrv_ratio": round(self.mvrv_ratio, 4),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class OnChainSignal:
    """Trading signal from on-chain analysis."""
    asset: str
    signal: Literal["BULLISH", "BEARISH", "NEUTRAL"]
    strength: float  # 0 to 1
    indicators: Dict[str, str]  # indicator -> signal
    whale_activity: str  # "accumulating", "distributing", "neutral"
    exchange_flow: str  # "inflow", "outflow", "neutral"
    reasoning: List[str]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "asset": self.asset,
            "signal": self.signal,
            "strength": round(self.strength, 4),
            "indicators": self.indicators,
            "whale_activity": self.whale_activity,
            "exchange_flow": self.exchange_flow,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class OnChainConfig:
    """On-chain analytics configuration."""
    # Whale thresholds
    whale_threshold_btc: float = 100  # BTC
    whale_threshold_eth: float = 1000  # ETH
    whale_threshold_usd: float = 1000000  # $1M

    # Signal thresholds
    exchange_flow_threshold: float = 0.01  # 1% change
    whale_accumulation_threshold: float = 0.6  # 60% outflow = accumulation

    # History
    history_size: int = 100


class OnChainAnalytics:
    """
    Analyze on-chain data for trading signals.

    Features:
    - Whale transaction tracking
    - Exchange flow analysis
    - Network activity metrics
    - Signal generation from on-chain data
    """

    def __init__(self, config: Optional[OnChainConfig] = None):
        self.config = config or OnChainConfig()
        self._transactions: Dict[str, Deque[WhaleTransaction]] = {}
        self._exchange_flows: Dict[str, Deque[ExchangeFlow]] = {}
        self._metrics: Dict[str, OnChainMetrics] = {}

    def add_whale_transaction(self, tx: WhaleTransaction):
        """Add a whale transaction."""
        asset = tx.asset

        if asset not in self._transactions:
            self._transactions[asset] = deque(maxlen=self.config.history_size)

        self._transactions[asset].append(tx)
        logger.debug(f"Whale tx: {tx.amount} {asset} ({tx.from_type} -> {tx.to_type})")

    def add_transaction(
        self,
        tx_hash: str,
        asset: str,
        amount: float,
        usd_value: float,
        from_address: str,
        to_address: str,
        from_type: str = "unknown",
        to_type: str = "unknown",
    ) -> Optional[WhaleTransaction]:
        """Add transaction if it qualifies as whale transaction."""
        # Check if whale-sized
        is_whale = usd_value >= self.config.whale_threshold_usd

        if asset.upper() == "BTC":
            is_whale = is_whale or amount >= self.config.whale_threshold_btc
        elif asset.upper() == "ETH":
            is_whale = is_whale or amount >= self.config.whale_threshold_eth

        if not is_whale:
            return None

        tx = WhaleTransaction(
            tx_hash=tx_hash,
            asset=asset,
            amount=amount,
            usd_value=usd_value,
            from_address=from_address,
            to_address=to_address,
            from_type=from_type,
            to_type=to_type,
        )

        self.add_whale_transaction(tx)
        return tx

    def add_exchange_flow(self, flow: ExchangeFlow):
        """Add exchange flow data."""
        key = f"{flow.asset}_{flow.exchange}"

        if key not in self._exchange_flows:
            self._exchange_flows[key] = deque(maxlen=self.config.history_size)

        self._exchange_flows[key].append(flow)

    def update_metrics(self, metrics: OnChainMetrics):
        """Update on-chain metrics for an asset."""
        self._metrics[metrics.asset] = metrics

    def get_whale_activity(
        self,
        asset: str,
        hours: int = 24,
    ) -> Dict[str, Any]:
        """
        Analyze whale activity for an asset.

        Args:
            asset: Asset symbol
            hours: Analysis window

        Returns:
            Whale activity summary
        """
        transactions = self._transactions.get(asset, deque())
        cutoff = datetime.now() - timedelta(hours=hours)

        recent = [tx for tx in transactions if tx.timestamp > cutoff]

        if not recent:
            return {
                "transaction_count": 0,
                "total_volume": 0,
                "net_exchange_flow": 0,
                "activity": "neutral",
            }

        total_volume = sum(tx.usd_value for tx in recent)
        exchange_inflow = sum(tx.usd_value for tx in recent if tx.is_exchange_inflow)
        exchange_outflow = sum(tx.usd_value for tx in recent if tx.is_exchange_outflow)
        net_flow = exchange_inflow - exchange_outflow

        # Determine activity
        if total_volume == 0:
            activity = "neutral"
        elif exchange_outflow / total_volume > self.config.whale_accumulation_threshold:
            activity = "accumulating"
        elif exchange_inflow / total_volume > self.config.whale_accumulation_threshold:
            activity = "distributing"
        else:
            activity = "neutral"

        return {
            "transaction_count": len(recent),
            "total_volume": total_volume,
            "exchange_inflow": exchange_inflow,
            "exchange_outflow": exchange_outflow,
            "net_exchange_flow": net_flow,
            "activity": activity,
        }

    def get_exchange_flow_summary(
        self,
        asset: str,
        hours: int = 24,
    ) -> Dict[str, Any]:
        """
        Get aggregated exchange flow summary.

        Args:
            asset: Asset symbol
            hours: Analysis window

        Returns:
            Exchange flow summary
        """
        cutoff = datetime.now() - timedelta(hours=hours)

        total_inflow = 0
        total_outflow = 0
        exchanges_with_inflow = 0
        exchanges_with_outflow = 0

        for key, flows in self._exchange_flows.items():
            if not key.startswith(f"{asset}_"):
                continue

            recent = [f for f in flows if f.timestamp > cutoff]
            if not recent:
                continue

            period_inflow = sum(f.inflow for f in recent)
            period_outflow = sum(f.outflow for f in recent)

            total_inflow += period_inflow
            total_outflow += period_outflow

            if period_inflow > period_outflow:
                exchanges_with_inflow += 1
            else:
                exchanges_with_outflow += 1

        net_flow = total_inflow - total_outflow
        total = total_inflow + total_outflow

        if total == 0:
            flow_sentiment = "neutral"
        elif net_flow / total > self.config.exchange_flow_threshold:
            flow_sentiment = "inflow"  # Bearish
        elif net_flow / total < -self.config.exchange_flow_threshold:
            flow_sentiment = "outflow"  # Bullish
        else:
            flow_sentiment = "neutral"

        return {
            "total_inflow": total_inflow,
            "total_outflow": total_outflow,
            "net_flow": net_flow,
            "flow_sentiment": flow_sentiment,
            "exchanges_with_net_inflow": exchanges_with_inflow,
            "exchanges_with_net_outflow": exchanges_with_outflow,
        }

    def generate_signal(self, asset: str) -> OnChainSignal:
        """
        Generate trading signal from on-chain data.

        Args:
            asset: Asset symbol

        Returns:
            OnChainSignal with analysis
        """
        indicators = {}
        reasoning = []
        bullish_score = 0
        bearish_score = 0

        # Whale activity
        whale_data = self.get_whale_activity(asset)
        whale_activity = whale_data["activity"]

        if whale_activity == "accumulating":
            indicators["whale_activity"] = "bullish"
            bullish_score += 0.3
            reasoning.append("Whales accumulating (moving off exchanges)")
        elif whale_activity == "distributing":
            indicators["whale_activity"] = "bearish"
            bearish_score += 0.3
            reasoning.append("Whales distributing (moving to exchanges)")
        else:
            indicators["whale_activity"] = "neutral"

        # Exchange flow
        flow_data = self.get_exchange_flow_summary(asset)
        exchange_flow = flow_data["flow_sentiment"]

        if exchange_flow == "outflow":
            indicators["exchange_flow"] = "bullish"
            bullish_score += 0.3
            reasoning.append("Net exchange outflow (reducing sell pressure)")
        elif exchange_flow == "inflow":
            indicators["exchange_flow"] = "bearish"
            bearish_score += 0.3
            reasoning.append("Net exchange inflow (increasing sell pressure)")
        else:
            indicators["exchange_flow"] = "neutral"

        # On-chain metrics
        metrics = self._metrics.get(asset)
        if metrics:
            # Exchange balance change
            if metrics.exchange_balance_change_24h < -self.config.exchange_flow_threshold:
                indicators["exchange_balance"] = "bullish"
                bullish_score += 0.2
                reasoning.append(f"Exchange balance down {metrics.exchange_balance_change_24h*100:.1f}%")
            elif metrics.exchange_balance_change_24h > self.config.exchange_flow_threshold:
                indicators["exchange_balance"] = "bearish"
                bearish_score += 0.2
                reasoning.append(f"Exchange balance up {metrics.exchange_balance_change_24h*100:.1f}%")

            # MVRV ratio
            if metrics.mvrv_ratio < 1.0:
                indicators["mvrv"] = "bullish"
                bullish_score += 0.1
                reasoning.append(f"MVRV below 1 ({metrics.mvrv_ratio:.2f}) - undervalued")
            elif metrics.mvrv_ratio > 3.0:
                indicators["mvrv"] = "bearish"
                bearish_score += 0.1
                reasoning.append(f"MVRV above 3 ({metrics.mvrv_ratio:.2f}) - overvalued")

            # NVT ratio
            if metrics.nvt_ratio < 50:
                indicators["nvt"] = "bullish"
                bullish_score += 0.1
                reasoning.append(f"Low NVT ({metrics.nvt_ratio:.0f}) - high network usage")
            elif metrics.nvt_ratio > 100:
                indicators["nvt"] = "bearish"
                bearish_score += 0.1
                reasoning.append(f"High NVT ({metrics.nvt_ratio:.0f}) - low network usage")

        # Determine signal
        net_score = bullish_score - bearish_score

        if net_score > 0.2:
            signal = "BULLISH"
            strength = min(1.0, net_score)
        elif net_score < -0.2:
            signal = "BEARISH"
            strength = min(1.0, abs(net_score))
        else:
            signal = "NEUTRAL"
            strength = 0

        if not reasoning:
            reasoning.append("Insufficient on-chain data")

        return OnChainSignal(
            asset=asset,
            signal=signal,
            strength=strength,
            indicators=indicators,
            whale_activity=whale_activity,
            exchange_flow=exchange_flow,
            reasoning=reasoning,
        )

    def get_recent_whale_transactions(
        self,
        asset: Optional[str] = None,
        limit: int = 20,
    ) -> List[WhaleTransaction]:
        """Get recent whale transactions."""
        if asset:
            transactions = list(self._transactions.get(asset, []))
        else:
            transactions = []
            for asset_txs in self._transactions.values():
                transactions.extend(list(asset_txs))

        transactions.sort(key=lambda tx: tx.timestamp, reverse=True)
        return transactions[:limit]

    def get_summary(self) -> Dict:
        """Get on-chain analytics summary."""
        total_whale_txs = sum(len(txs) for txs in self._transactions.values())
        assets_tracked = list(self._transactions.keys())

        recent_signals = {}
        for asset in assets_tracked:
            signal = self.generate_signal(asset)
            recent_signals[asset] = signal.signal

        return {
            "assets_tracked": len(assets_tracked),
            "total_whale_transactions": total_whale_txs,
            "signals": recent_signals,
            "assets": assets_tracked,
        }


def create_onchain_analytics(config: Optional[OnChainConfig] = None) -> OnChainAnalytics:
    """Factory function to create on-chain analytics."""
    return OnChainAnalytics(config=config)
