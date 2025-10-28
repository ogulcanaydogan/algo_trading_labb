"""
Canlı Trading Scripti (Testnet veya Gerçek Borsa)

Bu script ile stratejinizi canlı olarak çalıştırabilirsiniz.
DRY RUN modu ile önce güvenli test yapmanız önerilir.
"""

import os
import sys
import time
from pathlib import Path

# Proje root'unu path'e ekle
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from bot.exchange import ExchangeClient
from bot.strategy import StrategyConfig, compute_indicators, generate_signal, calculate_position_size
from bot.trading import TradingManager

# .env dosyasını yükle
load_dotenv()


def run_live_trading():
    """Canlı trading'i başlat"""
    
    print("="*60)
    print("CANLI TRADING MODU")
    print("="*60)
    print("⚠️  DİKKAT: Bu script gerçek işlem yapar!")
    print("⚠️  Önce DRY RUN modu ile test etmeniz önerilir.\n")
    
    # Mod seçimi
    print("Trading Modu:")
    print("1. DRY RUN (sadece log, gerçek emir yok)")
    print("2. TESTNET (Binance testnet, gerçek emir)")
    print("3. LIVE (GERÇEK BORSA - DİKKAT!)")
    
    mode = input("Seçiminiz (1/2/3): ").strip()
    
    if mode not in ["1", "2", "3"]:
        print("❌ Geçersiz seçim!")
        return
    
    if mode == "3":
        confirm = input("⚠️  GERÇEK BORSADA İŞLEM YAPACAKSINIZ! Emin misiniz? (YES yazın): ").strip()
        if confirm != "YES":
            print("❌ İşlem iptal edildi.")
            return
    
    dry_run = mode == "1"
    use_testnet = mode == "2"
    
    # Konfigürasyon
    symbol = input("\nSymbol (default: BTC/USDT): ").strip() or "BTC/USDT"
    timeframe = input("Timeframe (default: 5m): ").strip() or "5m"
    loop_interval = int(input("Loop interval (saniye) (default: 60): ").strip() or "60")
    
    # Strateji parametreleri
    print("\n📊 Strateji Parametreleri:")
    risk_pct = float(input("  Risk per trade % (default: 1.0): ").strip() or "1.0")
    stop_loss_pct = float(input("  Stop Loss % (default: 2.0): ").strip() or "2.0") / 100
    take_profit_pct = float(input("  Take Profit % (default: 4.0): ").strip() or "4.0") / 100
    
    config = StrategyConfig(
        symbol=symbol,
        timeframe=timeframe,
        risk_per_trade_pct=risk_pct,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
    )
    
    # Exchange client oluştur
    if use_testnet:
        api_key = os.getenv("BINANCE_TESTNET_API_KEY")
        api_secret = os.getenv("BINANCE_TESTNET_API_SECRET")
        if not api_key or not api_secret:
            print("❌ Testnet API anahtarları .env dosyasında bulunamadı!")
            return
        print("\n🔄 Binance Testnet'e bağlanılıyor...")
        exchange = ExchangeClient(
            exchange_id="binance",
            api_key=api_key,
            api_secret=api_secret,
            testnet=True,
        )
    else:
        api_key = os.getenv("EXCHANGE_API_KEY")
        api_secret = os.getenv("EXCHANGE_API_SECRET")
        if not api_key or not api_secret:
            print("❌ API anahtarları .env dosyasında bulunamadı!")
            return
        print("\n🔄 Binance'e bağlanılıyor...")
        exchange = ExchangeClient(
            exchange_id="binance",
            api_key=api_key,
            api_secret=api_secret,
        )
    
    # Trading manager oluştur
    trading_manager = TradingManager(
        exchange_client=exchange,
        symbol=symbol,
        dry_run=dry_run,
    )
    
    print(f"\n✅ Bağlantı başarılı!")
    print(f"📊 Trading başlatılıyor...")
    print(f"   Symbol: {symbol}")
    print(f"   Timeframe: {timeframe}")
    print(f"   Mode: {'DRY RUN' if dry_run else 'TESTNET' if use_testnet else 'LIVE'}")
    print(f"   Loop Interval: {loop_interval}s")
    print("\n🔄 Trading loop başladı... (Ctrl+C ile durdurun)\n")
    
    iteration = 0
    
    try:
        while True:
            iteration += 1
            start_time = time.time()
            
            print(f"\n{'='*60}")
            print(f"İTERASYON #{iteration} - {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*60}")
            
            try:
                # 1. Veri çek
                print("📊 Veri çekiliyor...")
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=250)
                current_price = float(ohlcv.iloc[-1]["close"])
                print(f"   Güncel fiyat: ${current_price:,.2f}")
                
                # 2. Mevcut pozisyon kontrolü
                position_info = trading_manager.get_position_info()
                if position_info:
                    print(f"\n📈 Mevcut Pozisyon:")
                    print(f"   Direction: {position_info['direction']}")
                    print(f"   Entry: ${position_info['entry_price']:,.2f}")
                    print(f"   Current: ${position_info.get('current_price', current_price):,.2f}")
                    print(f"   P&L: ${position_info.get('unrealized_pnl', 0):,.2f} "
                          f"({position_info.get('unrealized_pnl_pct', 0):.2f}%)")
                    print(f"   Stop Loss: ${position_info['stop_loss']:,.2f}")
                    print(f"   Take Profit: ${position_info['take_profit']:,.2f}")
                    
                    # Exit kontrolü
                    exit_reason = trading_manager.check_position_exit(current_price)
                    if exit_reason:
                        print(f"\n🚪 Pozisyon kapatılıyor: {exit_reason}")
                        result = trading_manager.close_position(reason=exit_reason)
                        if result.success:
                            print(f"✅ Pozisyon kapatıldı: {result.order_id}")
                        else:
                            print(f"❌ Hata: {result.error}")
                else:
                    print("\n📊 Açık pozisyon yok")
                    
                    # 3. Sinyal üret
                    print("\n🔍 Sinyal analizi...")
                    enriched = compute_indicators(ohlcv, config)
                    signal = generate_signal(enriched, config)
                    
                    print(f"   Karar: {signal['decision']}")
                    print(f"   Güven: {signal['confidence']:.2%}")
                    print(f"   RSI: {signal['rsi']:.2f}")
                    print(f"   EMA Fast: ${signal['ema_fast']:,.2f}")
                    print(f"   EMA Slow: ${signal['ema_slow']:,.2f}")
                    print(f"   Sebep: {signal['reason']}")
                    
                    # 4. Pozisyon aç (eğer sinyal varsa)
                    if signal["decision"] != "FLAT" and signal["confidence"] > 0.4:
                        print(f"\n🚀 {signal['decision']} pozisyonu açılıyor...")
                        
                        # Pozisyon büyüklüğünü hesapla
                        # Basit örnek: Balance'ı testnet'ten al
                        try:
                            balance_info = exchange.client.fetch_balance()
                            usdt_balance = balance_info.get("USDT", {}).get("free", 10000)
                        except:
                            usdt_balance = 10000  # Varsayılan
                        
                        size = calculate_position_size(
                            usdt_balance,
                            config.risk_per_trade_pct,
                            current_price,
                            config.stop_loss_pct,
                        )
                        
                        # BTC cinsinden size'a çevir
                        size_btc = size / current_price
                        
                        # Stop loss ve take profit hesapla
                        if signal["decision"] == "LONG":
                            stop_loss = current_price * (1 - config.stop_loss_pct)
                            take_profit = current_price * (1 + config.take_profit_pct)
                        else:
                            stop_loss = current_price * (1 + config.stop_loss_pct)
                            take_profit = current_price * (1 - config.take_profit_pct)
                        
                        result = trading_manager.open_position(
                            direction=signal["decision"],
                            size=size_btc,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            signal_info=signal,
                        )
                        
                        if result.success:
                            print(f"✅ Pozisyon açıldı: {result.order_id}")
                        else:
                            print(f"❌ Hata: {result.error}")
                    else:
                        print("\n⏸️  İşlem yapılmadı (sinyal yok veya güven düşük)")
                
            except Exception as e:
                print(f"\n❌ Loop hatası: {e}")
                import traceback
                traceback.print_exc()
            
            # Bekleme
            elapsed = time.time() - start_time
            sleep_time = max(1, loop_interval - int(elapsed))
            print(f"\n⏳ Sonraki iterasyon için {sleep_time} saniye bekleniyor...")
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Trading durduruldu!")
        
        # Açık pozisyon varsa sor
        if trading_manager.current_position:
            close_pos = input("\n❓ Açık pozisyonu kapatmak ister misiniz? (y/n): ").strip().lower()
            if close_pos == "y":
                result = trading_manager.close_position(reason="Manual stop")
                if result.success:
                    print("✅ Pozisyon kapatıldı")
                else:
                    print(f"❌ Hata: {result.error}")


if __name__ == "__main__":
    try:
        run_live_trading()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
