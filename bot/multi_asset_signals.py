"""
Multi-Asset Signal Aggregator.
Combines signals from multiple assets and generates portfolio-level recommendations.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import numpy as np

class Signal(str, Enum):
    """Trading signals."""
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"

@dataclass
class AssetSignal:
    """Signal from a single asset."""
    symbol: str
    signal: Signal
    confidence: float  # 0.0 to 1.0
    reason: str
    timestamp: datetime
    features: Dict[str, float]  # EMA, RSI, volatility, etc.

@dataclass
class PortfolioSignal:
    """Portfolio-level signal combining multiple assets."""
    action: str  # "REBALANCE", "HEDGE", "DIVERSIFY", "CONSOLIDATE"
    assets_affected: List[str]
    priority: float  # 0.0 to 1.0
    recommended_trades: Dict[str, float]  # symbol -> quantity_delta
    reason: str
    timestamp: datetime

class MultiAssetSignalAggregator:
    """Aggregates signals across multiple assets."""
    
    def __init__(self):
        self.asset_signals: Dict[str, AssetSignal] = {}
        self.portfolio_signals: List[PortfolioSignal] = []
    
    def add_asset_signal(self, asset_signal: AssetSignal):
        """Add or update signal for an asset."""
        self.asset_signals[asset_signal.symbol] = asset_signal
    
    def get_portfolio_sentiment(self) -> Tuple[float, Dict[str, int]]:
        """
        Calculate overall portfolio sentiment.
        Returns: (sentiment_score, signal_counts)
        
        sentiment_score: -1.0 (bearish) to +1.0 (bullish)
        """
        if not self.asset_signals:
            return 0.0, {}
        
        signal_counts = {"LONG": 0, "SHORT": 0, "FLAT": 0}
        signal_scores = {"LONG": 1.0, "SHORT": -1.0, "FLAT": 0.0}
        
        total_score = 0
        total_confidence = 0
        
        for signal in self.asset_signals.values():
            signal_counts[signal.signal.value] += 1
            score = signal_scores[signal.signal.value] * signal.confidence
            total_score += score
            total_confidence += signal.confidence
        
        if total_confidence == 0:
            return 0.0, signal_counts
        
        sentiment = total_score / total_confidence
        return np.clip(sentiment, -1.0, 1.0), signal_counts
    
    def identify_correlation_risk(self, correlation_matrix: np.ndarray, symbols: List[str], threshold: float = 0.8) -> List[Tuple[str, str, float]]:
        """
        Identify highly correlated asset pairs (diversification risk).
        Returns list of (symbol1, symbol2, correlation)
        """
        risks = []
        
        for i, sym1 in enumerate(symbols):
            for j in range(i + 1, len(symbols)):
                sym2 = symbols[j]
                corr = abs(correlation_matrix[i, j])
                
                if corr > threshold:
                    risks.append((sym1, sym2, corr))
        
        return sorted(risks, key=lambda x: x[2], reverse=True)
    
    def generate_hedging_signal(self, portfolio_sentiment: float, drawdown_pct: float) -> Optional[PortfolioSignal]:
        """
        Generate hedging signal if portfolio is exposed to risk.
        """
        if drawdown_pct < 5 or portfolio_sentiment > -0.3:
            return None
        
        # Suggest hedges: reduce longs or add shorts
        return PortfolioSignal(
            action="HEDGE",
            assets_affected=[s for s in self.asset_signals.keys()],
            priority=min(drawdown_pct / 10, 1.0),  # 0-1 scale
            recommended_trades={},  # Specific trades determined by portfolio manager
            reason=f"Drawdown at {drawdown_pct:.1f}%, sentiment {portfolio_sentiment:.2f}. Consider hedging.",
            timestamp=datetime.utcnow(),
        )
    
    def generate_diversification_signal(self, correlation_risks: List[Tuple[str, str, float]], target_hhi: float, current_hhi: float) -> Optional[PortfolioSignal]:
        """
        Generate diversification signal if concentration/correlation too high.
        """
        if not correlation_risks or current_hhi < target_hhi:
            return None
        
        # Suggest reducing correlated positions
        affected_symbols = set()
        for sym1, sym2, _ in correlation_risks[:3]:  # Top 3 risks
            affected_symbols.add(sym1)
            affected_symbols.add(sym2)
        
        return PortfolioSignal(
            action="DIVERSIFY",
            assets_affected=list(affected_symbols),
            priority=0.5,
            recommended_trades={},
            reason=f"High correlation detected. HHI {current_hhi:.0f} vs target {target_hhi:.0f}. Rebalance to reduce concentration.",
            timestamp=datetime.utcnow(),
        )
    
    def generate_rebalancing_signal(self, allocation_drifts: Dict[str, float], threshold_pct: float) -> Optional[PortfolioSignal]:
        """
        Generate rebalancing signal if allocations drift too far.
        allocation_drifts: {symbol: drift_pct}
        """
        exceeds_threshold = {s: d for s, d in allocation_drifts.items() if abs(d) > threshold_pct}
        
        if not exceeds_threshold:
            return None
        
        return PortfolioSignal(
            action="REBALANCE",
            assets_affected=list(exceeds_threshold.keys()),
            priority=max(abs(d) for d in exceeds_threshold.values()) / 10,  # 0-1 scale
            recommended_trades={},
            reason=f"Allocation drift detected. {len(exceeds_threshold)} assets exceed {threshold_pct}% threshold.",
            timestamp=datetime.utcnow(),
        )
    
    def get_consensus_signal(self) -> Tuple[Signal, float]:
        """
        Get consensus signal across all assets.
        Returns (consensus_signal, confidence)
        """
        if not self.asset_signals:
            return Signal.FLAT, 0.0
        
        sentiment, counts = self.get_portfolio_sentiment()
        
        # Determine signal based on sentiment
        if sentiment > 0.3:
            signal = Signal.LONG
            confidence = sentiment
        elif sentiment < -0.3:
            signal = Signal.SHORT
            confidence = abs(sentiment)
        else:
            signal = Signal.FLAT
            confidence = 1 - abs(sentiment)
        
        return signal, min(confidence, 1.0)
    
    def get_signals_summary(self) -> Dict:
        """Get comprehensive signals summary."""
        sentiment, counts = self.get_portfolio_sentiment()
        consensus_signal, consensus_conf = self.get_consensus_signal()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "consensus_signal": consensus_signal.value,
            "consensus_confidence": consensus_conf,
            "portfolio_sentiment": sentiment,
            "signal_distribution": counts,
            "asset_signals": {
                symbol: {
                    "signal": sig.signal.value,
                    "confidence": sig.confidence,
                    "reason": sig.reason,
                }
                for symbol, sig in self.asset_signals.items()
            },
        }
