"""
Gerçek Trading İşlemleri Yöneticisi

Bu modül Binance Testnet veya gerçek borsada al-sat işlemlerini yönetir.
"""

import logging
from typing import Optional, Dict, Literal
from datetime import datetime, timezone
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OrderResult:
    """Emir sonucu"""
    success: bool
    order_id: Optional[str] = None
    price: float = 0.0
    quantity: float = 0.0
    side: str = ""  # BUY veya SELL
    error: Optional[str] = None
    timestamp: Optional[datetime] = None


class TradingManager:
    """
    Gerçek işlemleri yöneten sınıf
    
    Bu sınıf Binance Testnet veya gerçek borsada emir göndermeyi yönetir.
    Risk yönetimi ve pozisyon kontrolü içerir.
    """
    
    def __init__(
        self,
        exchange_client,
        symbol: str = "BTC/USDT",
        max_position_size: float = 0.1,  # Maksimum pozisyon büyüklüğü (BTC)
        min_order_size: float = 0.001,   # Minimum emir büyüklüğü
        dry_run: bool = True,            # True = sadece log, gerçek emir gönderme
    ):
        self.client = exchange_client
        self.symbol = symbol
        self.max_position_size = max_position_size
        self.min_order_size = min_order_size
        self.dry_run = dry_run
        
        self.current_position: Optional[Dict] = None
        self.pending_orders: Dict[str, Dict] = {}
        
        logger.info(
            f"TradingManager başlatıldı | Symbol: {symbol} | Dry Run: {dry_run}"
        )
    
    def open_position(
        self,
        direction: Literal["LONG", "SHORT"],
        size: float,
        stop_loss: float,
        take_profit: float,
        signal_info: Optional[Dict] = None,
    ) -> OrderResult:
        """
        Yeni pozisyon aç
        
        Args:
            direction: LONG veya SHORT
            size: Pozisyon büyüklüğü (BTC)
            stop_loss: Stop loss fiyatı
            take_profit: Take profit fiyatı
            signal_info: Sinyal bilgileri (opsiyonel)
        
        Returns:
            OrderResult: İşlem sonucu
        """
        # Mevcut pozisyon kontrolü
        if self.current_position:
            logger.warning("Zaten açık bir pozisyon var!")
            return OrderResult(
                success=False,
                error="Existing position already open"
            )
        
        # Pozisyon büyüklüğü kontrolü
        if size > self.max_position_size:
            logger.warning(f"Pozisyon çok büyük: {size} > {self.max_position_size}")
            size = self.max_position_size
        
        if size < self.min_order_size:
            logger.warning(f"Pozisyon çok küçük: {size} < {self.min_order_size}")
            return OrderResult(
                success=False,
                error=f"Position size too small: {size}"
            )
        
        # Market fiyatını al
        try:
            ticker = self._get_ticker()
            current_price = ticker["last"]
        except Exception as e:
            logger.error(f"Fiyat alınamadı: {e}")
            return OrderResult(success=False, error=str(e))
        
        # Emir gönder
        side = "BUY" if direction == "LONG" else "SELL"
        
        if self.dry_run:
            logger.info(
                f"[DRY RUN] {side} emri: {size} {self.symbol} @ ${current_price:.2f}"
            )
            logger.info(f"[DRY RUN] Stop Loss: ${stop_loss:.2f} | Take Profit: ${take_profit:.2f}")
            
            self.current_position = {
                "direction": direction,
                "entry_price": current_price,
                "size": size,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "entry_time": datetime.now(timezone.utc),
                "signal_info": signal_info,
            }
            
            return OrderResult(
                success=True,
                order_id=f"DRY_RUN_{datetime.now().timestamp()}",
                price=current_price,
                quantity=size,
                side=side,
                timestamp=datetime.now(timezone.utc),
            )
        else:
            # Gerçek emir gönderimi
            try:
                order = self.client.client.create_market_order(
                    symbol=self.symbol,
                    side=side,
                    amount=size,
                )
                
                logger.info(f"✅ {side} emri gerçekleşti: {order['id']}")
                
                # Stop loss ve take profit emirlerini gönder
                self._place_stop_loss(direction, size, stop_loss)
                self._place_take_profit(direction, size, take_profit)
                
                self.current_position = {
                    "direction": direction,
                    "entry_price": order.get("price", current_price),
                    "size": size,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "entry_time": datetime.now(timezone.utc),
                    "order_id": order["id"],
                    "signal_info": signal_info,
                }
                
                return OrderResult(
                    success=True,
                    order_id=order["id"],
                    price=order.get("price", current_price),
                    quantity=size,
                    side=side,
                    timestamp=datetime.now(timezone.utc),
                )
                
            except Exception as e:
                logger.error(f"❌ Emir gönderilemedi: {e}")
                return OrderResult(success=False, error=str(e))
    
    def close_position(self, reason: str = "Manual close") -> OrderResult:
        """
        Mevcut pozisyonu kapat
        
        Args:
            reason: Kapatma nedeni
            
        Returns:
            OrderResult: İşlem sonucu
        """
        if not self.current_position:
            logger.warning("Kapatılacak pozisyon yok!")
            return OrderResult(success=False, error="No position to close")
        
        direction = self.current_position["direction"]
        size = self.current_position["size"]
        
        # Ters yönde emir gönder
        side = "SELL" if direction == "LONG" else "BUY"
        
        try:
            ticker = self._get_ticker()
            current_price = ticker["last"]
        except Exception as e:
            logger.error(f"Fiyat alınamadı: {e}")
            return OrderResult(success=False, error=str(e))
        
        if self.dry_run:
            entry_price = self.current_position["entry_price"]
            if direction == "LONG":
                pnl = (current_price - entry_price) * size
                pnl_pct = (current_price / entry_price - 1) * 100
            else:
                pnl = (entry_price - current_price) * size
                pnl_pct = (entry_price / current_price - 1) * 100
            
            logger.info(
                f"[DRY RUN] Pozisyon kapatıldı: {side} {size} @ ${current_price:.2f} | "
                f"P&L: ${pnl:.2f} ({pnl_pct:.2f}%) | Neden: {reason}"
            )
            
            self.current_position = None
            
            return OrderResult(
                success=True,
                order_id=f"DRY_RUN_{datetime.now().timestamp()}",
                price=current_price,
                quantity=size,
                side=side,
                timestamp=datetime.now(timezone.utc),
            )
        else:
            try:
                # Bekleyen stop/tp emirlerini iptal et
                self._cancel_pending_orders()
                
                # Pozisyonu kapat
                order = self.client.client.create_market_order(
                    symbol=self.symbol,
                    side=side,
                    amount=size,
                )
                
                logger.info(f"✅ Pozisyon kapatıldı: {order['id']} | Neden: {reason}")
                
                self.current_position = None
                
                return OrderResult(
                    success=True,
                    order_id=order["id"],
                    price=order.get("price", current_price),
                    quantity=size,
                    side=side,
                    timestamp=datetime.now(timezone.utc),
                )
                
            except Exception as e:
                logger.error(f"❌ Pozisyon kapatılamadı: {e}")
                return OrderResult(success=False, error=str(e))
    
    def check_position_exit(self, current_price: float) -> Optional[str]:
        """
        Pozisyon çıkış kontrolü (stop loss / take profit)
        
        Args:
            current_price: Güncel fiyat
            
        Returns:
            Çıkış nedeni veya None
        """
        if not self.current_position:
            return None
        
        direction = self.current_position["direction"]
        stop_loss = self.current_position["stop_loss"]
        take_profit = self.current_position["take_profit"]
        
        if direction == "LONG":
            if current_price <= stop_loss:
                return "Stop Loss Hit"
            elif current_price >= take_profit:
                return "Take Profit Hit"
        else:  # SHORT
            if current_price >= stop_loss:
                return "Stop Loss Hit"
            elif current_price <= take_profit:
                return "Take Profit Hit"
        
        return None
    
    def get_position_info(self) -> Optional[Dict]:
        """Mevcut pozisyon bilgisini döndür"""
        if not self.current_position:
            return None
        
        try:
            ticker = self._get_ticker()
            current_price = ticker["last"]
            
            entry_price = self.current_position["entry_price"]
            size = self.current_position["size"]
            direction = self.current_position["direction"]
            
            if direction == "LONG":
                unrealized_pnl = (current_price - entry_price) * size
                unrealized_pnl_pct = (current_price / entry_price - 1) * 100
            else:
                unrealized_pnl = (entry_price - current_price) * size
                unrealized_pnl_pct = (entry_price / current_price - 1) * 100
            
            return {
                **self.current_position,
                "current_price": current_price,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_pnl_pct": unrealized_pnl_pct,
            }
            
        except Exception as e:
            logger.error(f"Pozisyon bilgisi alınamadı: {e}")
            return self.current_position
    
    def _get_ticker(self) -> Dict:
        """Güncel fiyat bilgisini al"""
        return self.client.client.fetch_ticker(self.symbol)
    
    def _place_stop_loss(self, direction: str, size: float, stop_price: float):
        """Stop loss emri gönder"""
        try:
            side = "SELL" if direction == "LONG" else "BUY"
            order = self.client.client.create_order(
                symbol=self.symbol,
                type="STOP_LOSS_LIMIT",
                side=side,
                amount=size,
                price=stop_price,
                params={"stopPrice": stop_price},
            )
            self.pending_orders[order["id"]] = order
            logger.info(f"Stop Loss emri gönderildi: {order['id']} @ ${stop_price:.2f}")
        except Exception as e:
            logger.error(f"Stop Loss emri gönderilemedi: {e}")
    
    def _place_take_profit(self, direction: str, size: float, tp_price: float):
        """Take profit emri gönder"""
        try:
            side = "SELL" if direction == "LONG" else "BUY"
            order = self.client.client.create_limit_order(
                symbol=self.symbol,
                side=side,
                amount=size,
                price=tp_price,
            )
            self.pending_orders[order["id"]] = order
            logger.info(f"Take Profit emri gönderildi: {order['id']} @ ${tp_price:.2f}")
        except Exception as e:
            logger.error(f"Take Profit emri gönderilemedi: {e}")
    
    def _cancel_pending_orders(self):
        """Bekleyen emirleri iptal et"""
        for order_id in list(self.pending_orders.keys()):
            try:
                self.client.client.cancel_order(order_id, self.symbol)
                logger.info(f"Emir iptal edildi: {order_id}")
                del self.pending_orders[order_id]
            except Exception as e:
                logger.error(f"Emir iptal edilemedi {order_id}: {e}")
