"""
Alpaca Adapter - Stock Trading Integration.

Integrates with Alpaca Markets API for commission-free stock trading.
Supports paper trading and live trading modes.
"""

import logging
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

import requests

from bot.execution_adapter import ExecutionAdapter, Order, OrderSide, OrderStatus, OrderResult, Position, Balance

logger = logging.getLogger(__name__)


class AlpacaAdapter(ExecutionAdapter):
    """
    Alpaca Markets execution adapter for stock trading.
    
    Features:
    - Commission-free stock trading
    - Paper trading sandbox
    - Real-time market data
    - REST + WebSocket APIs
    """
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        is_paper: bool = True,
        base_url: Optional[str] = None
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.is_paper = is_paper
        
        # Set base URL
        if base_url:
            self.base_url = base_url
        elif is_paper:
            self.base_url = "https://paper-api.alpaca.markets"
            self.data_url = "https://data.alpaca.markets"
        else:
            self.base_url = "https://api.alpaca.markets"
            self.data_url = "https://data.alpaca.markets"
        
        self._session = requests.Session()
        self._session.headers.update({
            'APCA-API-KEY-ID': api_key,
            'APCA-API-SECRET-KEY': api_secret
        })
        
        logger.info(f"Alpaca adapter initialized: {'PAPER' if is_paper else 'LIVE'}")
    
    async def initialize(self) -> bool:
        """Initialize connection and verify credentials."""
        try:
            account = self._get_account()
            logger.info(f"Alpaca account verified: {account['account_number']}")
            logger.info(f"Buying power: ${float(account['buying_power']):.2f}")
            return True
        except Exception as e:
            logger.error(f"Alpaca initialization failed: {e}")
            return False

    async def connect(self) -> bool:
        """Connect to Alpaca (alias for initialize)."""
        return await self.initialize()

    async def disconnect(self) -> None:
        """Disconnect from Alpaca."""
        self._session.close()
        logger.info("Alpaca adapter disconnected")

    async def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        price = await self.get_market_price(symbol)
        return price if price else 0.0

    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol."""
        positions = await self.get_positions()
        for pos in positions:
            if pos.symbol == symbol.replace('/USD', ''):
                return pos
        return None

    async def get_all_positions(self) -> List[Position]:
        """Get all open positions."""
        return await self.get_positions()
    
    def _get_account(self) -> Dict:
        """Get account information."""
        response = self._session.get(f"{self.base_url}/v2/account")
        response.raise_for_status()
        return response.json()
    
    async def get_balance(self) -> Balance:
        """Get account balance."""
        try:
            account = self._get_account()
            return Balance(
                total=float(account['equity']),
                available=float(account['buying_power']),
                in_positions=float(account['equity']) - float(account['cash']),
                unrealized_pnl=float(account.get('unrealized_pl', 0)),
                currency="USD"
            )
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return Balance(total=0, available=0, in_positions=0, unrealized_pnl=0, currency="USD")
    
    async def get_positions(self) -> List[Position]:
        """Get open positions."""
        try:
            response = self._session.get(f"{self.base_url}/v2/positions")
            response.raise_for_status()
            positions_data = response.json()
            
            positions = []
            for pos in positions_data:
                position = Position(
                    symbol=pos['symbol'],
                    side=OrderSide.BUY if float(pos['qty']) > 0 else OrderSide.SELL,
                    quantity=abs(float(pos['qty'])),
                    entry_price=float(pos['avg_entry_price']),
                    current_price=float(pos['current_price']),
                    unrealized_pnl=float(pos['unrealized_pl']),
                    unrealized_pnl_pct=float(pos['unrealized_plpc']) * 100
                )
                positions.append(position)
            
            return positions
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    async def place_order(self, order: Order) -> OrderResult:
        """Place an order and return OrderResult."""
        try:
            # Alpaca uses different symbol format (no /USD suffix)
            symbol = order.symbol.replace('/USD', '')

            order_data = {
                'symbol': symbol,
                'qty': order.quantity,
                'side': order.side.value.lower(),  # "buy" or "sell"
                'type': 'market',
                'time_in_force': 'day'
            }

            response = self._session.post(
                f"{self.base_url}/v2/orders",
                json=order_data
            )
            response.raise_for_status()

            result = response.json()
            logger.info(f"Alpaca order executed: {result['id']} - {order.side.value} {order.quantity} {symbol}")

            return OrderResult(
                success=True,
                order_id=result['id'],
                status=OrderStatus.FILLED,
                filled_quantity=order.quantity,
                average_price=float(result.get('filled_avg_price', 0)),
                error_message=None
            )

        except Exception as e:
            logger.error(f"Failed to execute order: {e}")
            return OrderResult(
                success=False,
                order_id="",
                status=OrderStatus.REJECTED,
                filled_quantity=0,
                average_price=0,
                error_message=str(e)
            )

    async def execute_order(self, order: Order) -> bool:
        """Execute a market order (legacy method)."""
        result = await self.place_order(order)
        return result.success
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order."""
        try:
            response = self._session.delete(f"{self.base_url}/v2/orders/{order_id}")
            response.raise_for_status()
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return False

    async def get_order_status(self, order_id: str, symbol: str) -> OrderStatus:
        """Get order status."""
        try:
            response = self._session.get(f"{self.base_url}/v2/orders/{order_id}")
            response.raise_for_status()
            order_data = response.json()

            status_map = {
                'new': OrderStatus.PENDING,
                'accepted': OrderStatus.PENDING,
                'pending_new': OrderStatus.PENDING,
                'partially_filled': OrderStatus.OPEN,
                'filled': OrderStatus.FILLED,
                'done_for_day': OrderStatus.FILLED,
                'canceled': OrderStatus.CANCELLED,
                'expired': OrderStatus.CANCELLED,
                'replaced': OrderStatus.CANCELLED,
                'pending_cancel': OrderStatus.CANCELLED,
                'pending_replace': OrderStatus.OPEN,
                'rejected': OrderStatus.REJECTED,
                'suspended': OrderStatus.REJECTED,
                'calculated': OrderStatus.PENDING
            }

            return status_map.get(order_data['status'], OrderStatus.OPEN)
        except Exception as e:
            logger.error(f"Failed to get order status: {e}")
            return OrderStatus.PENDING
    
    async def get_market_price(self, symbol: str) -> Optional[float]:
        """Get current market price."""
        try:
            # Alpaca uses different symbol format
            symbol = symbol.replace('/USD', '')
            
            response = self._session.get(
                f"{self.data_url}/v2/stocks/{symbol}/quotes/latest"
            )
            response.raise_for_status()
            quote = response.json()
            
            # Use mid price (average of bid and ask)
            bid = float(quote['quote']['bp'])
            ask = float(quote['quote']['ap'])
            return (bid + ask) / 2
        except Exception as e:
            logger.error(f"Failed to get market price for {symbol}: {e}")
            return None
    
    async def close_position(self, symbol: str) -> bool:
        """Close a position."""
        try:
            symbol = symbol.replace('/USD', '')
            response = self._session.delete(f"{self.base_url}/v2/positions/{symbol}")
            response.raise_for_status()
            logger.info(f"Position closed: {symbol}")
            return True
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return False
    
    async def get_historical_data(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> Optional[list]:
        """Get historical OHLCV data."""
        try:
            symbol = symbol.replace('/USD', '')
            
            # Map timeframe to Alpaca format
            timeframe_map = {
                "1m": "1Min",
                "5m": "5Min",
                "15m": "15Min",
                "1h": "1Hour",
                "1d": "1Day"
            }
            alpaca_timeframe = timeframe_map.get(timeframe, "1Hour")
            
            response = self._session.get(
                f"{self.data_url}/v2/stocks/{symbol}/bars",
                params={
                    'timeframe': alpaca_timeframe,
                    'limit': limit
                }
            )
            response.raise_for_status()
            data = response.json()
            
            # Convert to standard OHLCV format
            bars = []
            for bar in data['bars']:
                bars.append({
                    'timestamp': bar['t'],
                    'open': float(bar['o']),
                    'high': float(bar['h']),
                    'low': float(bar['l']),
                    'close': float(bar['c']),
                    'volume': float(bar['v'])
                })
            
            return bars
        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            return None


def create_alpaca_adapter(is_paper: bool = True) -> Optional[AlpacaAdapter]:
    """
    Factory function to create Alpaca adapter from environment variables.
    
    Required env vars:
    - ALPACA_API_KEY
    - ALPACA_API_SECRET
    - ALPACA_PAPER_MODE (true/false)
    """
    import os
    
    api_key = os.getenv('ALPACA_API_KEY')
    api_secret = os.getenv('ALPACA_API_SECRET')
    
    if not api_key or not api_secret:
        logger.error("Alpaca credentials not found in environment")
        return None
    
    return AlpacaAdapter(
        api_key=api_key,
        api_secret=api_secret,
        is_paper=is_paper
    )
