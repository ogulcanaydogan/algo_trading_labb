"""
Backtest Çalıştırma Scripti

Bu script ile stratejinizi geçmiş verilerle test edebilirsiniz.
"""

import os
import sys
from pathlib import Path

# Proje root'unu path'e ekle
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from bot.exchange import ExchangeClient, PaperExchangeClien
from bot.strategy import StrategyConfig
from bot.backtesting import Backtester, save_backtest_results

# .env dosyasını yükle
load_dotenv()


def run_backtest():
    """Backtest'i çalıştır"""

    print("="*60)
    print("BACKTEST MODU")
    print("="*60)

    # Konfigürasyon
    symbol = input("Symbol (default: BTC/USDT): ").strip() or "BTC/USDT"
    timeframe = input("Timeframe (default: 1h): ").strip() or "1h"
    lookback = int(input("Kaç mum geriye gidilsin? (default: 1000): ").strip() or "1000")
    initial_balance = float(input("Başlangıç bakiyesi ($) (default: 10000): ").strip() or "10000")

    # Strateji parametreleri
    print("\n📊 Strateji Parametreleri:")
    ema_fast = int(input("  EMA Fast (default: 12): ").strip() or "12")
    ema_slow = int(input("  EMA Slow (default: 26): ").strip() or "26")
    rsi_period = int(input("  RSI Period (default: 14): ").strip() or "14")
    risk_pct = float(input("  Risk per trade % (default: 1.0): ").strip() or "1.0")
    stop_loss_pct = float(input("  Stop Loss % (default: 2.0): ").strip() or "2.0") / 100
    take_profit_pct = float(input("  Take Profit % (default: 4.0): ").strip() or "4.0") / 100

    # Strategy config oluştur
    config = StrategyConfig(
        symbol=symbol,
        timeframe=timeframe,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        rsi_period=rsi_period,
        risk_per_trade_pct=risk_pct,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
    )

    # Veri kaynağını seç
    print("\n📡 Veri Kaynağı:")
    print("1. Binance Testnet (gerçek veriler)")
    print("2. Paper Exchange (sentetik veriler)")
    data_source = input("Seçiminiz (1/2): ").strip()

    if data_source == "1":
        # Binance Testne
        api_key = os.getenv("BINANCE_TESTNET_API_KEY")
        api_secret = os.getenv("BINANCE_TESTNET_API_SECRET")

        if not api_key or not api_secret:
            print("❌ Testnet API anahtarları .env dosyasında bulunamadı!")
            return

        print("🔄 Binance Testnet'ten veri çekiliyor...")
        exchange = ExchangeClient(
            exchange_id="binance",
            api_key=api_key,
            api_secret=api_secret,
            testnet=True,
        )
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=lookback)
    else:
        # Paper Exchange
        print("🔄 Sentetik veri üretiliyor...")
        exchange = PaperExchangeClient(symbol=symbol, timeframe=timeframe)
        ohlcv = exchange.fetch_ohlcv(limit=lookback)

    print(f"✅ {len(ohlcv)} mum verisi alındı")
    print(f"   Başlangıç: {ohlcv.index[0]}")
    print(f"   Bitiş: {ohlcv.index[-1]}")

    # Backtest'i çalıştır
    backtester = Backtester(
        strategy_config=config,
        initial_balance=initial_balance,
    )

    result = backtester.run(ohlcv)

    # Sonuçları göster
    result.print_summary()

    # Son 10 trade'i göster
    if result.trades:
        print("\n📊 Son 10 Trade:")
        print("-" * 80)
        for trade in result.trades[-10:]:
            emoji = "✅" if trade.pnl > 0 else "❌"
            print(
                f"{emoji} {trade.direction:5s} | "
                f"Entry: ${trade.entry_price:8.2f} | "
                f"Exit: ${trade.exit_price:8.2f} | "
                f"P&L: ${trade.pnl:8.2f} ({trade.pnl_pct:6.2f}%) | "
                f"{trade.exit_reason}"
            )
        print("-" * 80)

    # Sonuçları kayde
    save_choice = input("\n💾 Sonuçları JSON dosyasına kaydetmek ister misiniz? (y/n): ").strip().lower()
    if save_choice == "y":
        filename = input("Dosya adı (default: backtest_results.json): ").strip() or "backtest_results.json"
        save_backtest_results(result, filename)

    print("\n✅ Backtest tamamlandı!")


if __name__ == "__main__":
    try:
        run_backtest()
    except KeyboardInterrupt:
        print("\n\n⚠️  İşlem kullanıcı tarafından durduruldu")
    except Exception as e:
        print(f"\n❌ Hata: {e}")
        import traceback
        traceback.print_exc()
