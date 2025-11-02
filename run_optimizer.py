"""
Strateji Optimizasyonu Scripti

Bu script, geçmiş veriler üzerinde rastgele arama (random search) ile
EMA/RSI ve risk parametrelerini optimize eder. Amaç fonksiyonu olarak
Sharpe, PnL veya WinRate seçilebilir ve Max Drawdown'a ceza uygulanır.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

from bot.exchange import ExchangeClient, PaperExchangeClient
from bot.strategy import StrategyConfig
from bot.optimizer import random_search_optimize, results_to_dataframe


def run_optimizer():
    load_dotenv()

    print("=" * 60)
    print("STRATEJI OPTIMIZASYONU")
    print("=" * 60)

    symbol = input("Symbol (default: BTC/USDT): ").strip() or "BTC/USDT"
    timeframe = input("Timeframe (default: 1h): ").strip() or "1h"
    lookback = int(input("Kaç mum geriye gidilsin? (default: 2000): ").strip() or "2000")
    initial_balance = float(input("Başlangıç bakiyesi ($) (default: 10000): ").strip() or "10000")

    print("\n🎯 Amaç fonksiyonu:")
    print("1. Sharpe (DD cezalılı)")
    print("2. PnL (DD cezalılı)")
    print("3. WinRate (DD cezalılı)")
    obj_choice = input("Seçiminiz (1/2/3) [default 1]: ").strip() or "1"
    objective = {"1": "sharpe", "2": "pnl", "3": "winrate"}.get(obj_choice, "sharpe")
    mdd_weight = float(input("Max Drawdown ceza katsayısı (default: 0.5): ").strip() or "0.5")
    n_trials = int(input("Deneme sayısı (default: 50): ").strip() or "50")
    min_trades = int(input("Min işlem adedi filtresi (default: 5): ").strip() or "5")

    base_cfg = StrategyConfig(symbol=symbol, timeframe=timeframe)

    print("\n📡 Veri Kaynağı:")
    print("1. Binance Testnet (gerçek veriler)")
    print("2. Paper Exchange (sentetik veriler)")
    data_source = input("Seçiminiz (1/2): ").strip()

    if data_source == "1":
        api_key = os.getenv("BINANCE_TESTNET_API_KEY")
        api_secret = os.getenv("BINANCE_TESTNET_API_SECRET")
        if not api_key or not api_secret:
            print("❌ Testnet API anahtarları .env dosyasında bulunamadı!")
            return
        print("🔄 Binance Testnet'ten veri çekiliyor...")
        ex = ExchangeClient(
            exchange_id="binance",
            api_key=api_key,
            api_secret=api_secret,
            testnet=True,
        )
        ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=lookback)
    else:
        print("🔄 Sentetik veri üretiliyor...")
        ex = PaperExchangeClient(symbol=symbol, timeframe=timeframe)
        ohlcv = ex.fetch_ohlcv(limit=lookback)

    print(f"✅ {len(ohlcv)} mum verisi hazır. Optimizasyon başlıyor...")

    best, all_results = random_search_optimize(
        ohlcv=ohlcv,
        base_config=base_cfg,
        n_trials=n_trials,
        seed=42,
        initial_balance=initial_balance,
        objective=objective,
        mdd_weight=mdd_weight,
        min_trades=min_trades,
    )

    print("\n" + "-" * 60)
    print("EN İYİ PARAMETRELER")
    print("-" * 60)
    for k, v in best.params.items():
        print(f"{k:18s}: {v}")
    print("-" * 60)
    print(
        f"Sharpe: {best.sharpe_ratio:.3f} | PnL%: {best.total_pnl_pct:.2f} | WinRate: {best.win_rate*100:.1f}% | "
        f"PF: {best.profit_factor:.2f} | MDD%: {best.max_drawdown_pct:.2f} | Trades: {best.total_trades}"
    )

    # İlk 10 sonucu tablo olarak göster
    try:
        import pandas as _pd  # noqa: F401
        df = results_to_dataframe(all_results[:10])
        print("\nTOP 10 SONUÇ:")
        # pretty print limited columns
        shown = df[[
            "param_ema_fast", "param_ema_slow", "param_rsi_period",
            "param_rsi_overbought", "param_rsi_oversold",
            "param_risk_per_trade_pct", "param_stop_loss_pct", "param_take_profit_pct",
            "sharpe_ratio", "total_pnl_pct", "win_rate", "profit_factor", "max_drawdown_pct", "total_trades",
        ]]
        print(shown.to_string(index=False, justify="center", col_space=12))
    except Exception:
        pass

    save = input("\n💾 Sonuçları CSV'e kaydetmek ister misiniz? (y/n): ").strip().lower()
    if save == "y":
        out = input("Dosya adı (default: optimization_results.csv): ").strip() or "optimization_results.csv"
        df_all = results_to_dataframe(all_results)
        df_all.to_csv(out, index=False)
        print(f"✅ Kaydedildi: {out}")

    print("\n🏁 Optimizasyon tamamlandı.")


if __name__ == "__main__":
    try:
        run_optimizer()
    except KeyboardInterrupt:
        print("\n\n⚠️  İşlem kullanıcı tarafından durduruldu")
    except Exception as e:
        print(f"\n❌ Hata: {e}")
        import traceback
        traceback.print_exc()
