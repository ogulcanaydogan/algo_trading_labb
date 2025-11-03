"""
Sürekli Otomatik Optimizasyon Servisi

Belirli aralıklarla geçmiş veriyi çekip strateji parametrelerini
optimize eder ve en iyi ayarları data/strategy_config.json dosyasına yazar.
Bot her döngüde bu dosyayı okuyup parametreleri canlıca uygular.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, Union

from dotenv import load_dotenv

from bot.exchange import ExchangeClient, PaperExchangeClient
from bot.optimizer import random_search_optimize
from bot.strategy import StrategyConfig


def fetch_data(symbol: str, timeframe: str, lookback: int):
    use_testnet = os.getenv("BINANCE_TESTNET_ENABLED", "false").lower() == "true"
    if use_testnet:
        api_key = os.getenv("BINANCE_TESTNET_API_KEY")
        api_secret = os.getenv("BINANCE_TESTNET_API_SECRET")
        if not api_key or not api_secret:
            print("❌ Testnet API anahtarları bulunamadı, sentetik veriye geçiliyor.")
            use_testnet = False
        else:
            ex = ExchangeClient("binance", api_key=api_key, api_secret=api_secret, testnet=True)
            return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=lookback)

    # Paper fallback
    ex = PaperExchangeClient(symbol=symbol, timeframe=timeframe)
    return ex.fetch_ohlcv(limit=lookback)


def write_strategy_config(
    data_dir: Path, base: StrategyConfig, params: Dict[str, Union[int, float, str]]
) -> Path:
    out_path = data_dir / "strategy_config.json"
    payload: Dict[str, Union[int, float, str]] = {
        # keep base values, override with best params
        "symbol": base.symbol,
        "timeframe": base.timeframe,
        "ema_fast": int(params.get("ema_fast", base.ema_fast)),
        "ema_slow": int(params.get("ema_slow", base.ema_slow)),
        "rsi_period": int(params.get("rsi_period", base.rsi_period)),
        "rsi_overbought": float(params.get("rsi_overbought", base.rsi_overbought)),
        "rsi_oversold": float(params.get("rsi_oversold", base.rsi_oversold)),
        "risk_per_trade_pct": float(params.get("risk_per_trade_pct", base.risk_per_trade_pct)),
        "stop_loss_pct": float(params.get("stop_loss_pct", base.stop_loss_pct)),
        "take_profit_pct": float(params.get("take_profit_pct", base.take_profit_pct)),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path


def run_service():
    load_dotenv()

    data_dir = Path(os.getenv("DATA_DIR", "./data"))
    symbol = os.getenv("SYMBOL", "BTC/USDT")
    timeframe = os.getenv("TIMEFRAME", "1h")
    lookback = int(os.getenv("OPTIMIZE_LOOKBACK", "2000"))
    initial_balance = float(os.getenv("STARTING_BALANCE", "10000"))
    interval_minutes = int(os.getenv("AUTO_OPTIMIZE_INTERVAL_MINUTES", "180"))
    n_trials = int(os.getenv("AUTO_OPTIMIZE_TRIALS", "60"))
    objective = os.getenv("AUTO_OPTIMIZE_OBJECTIVE", "sharpe").lower()
    mdd_weight = float(os.getenv("AUTO_OPTIMIZE_MDD_WEIGHT", "0.5"))
    min_trades = int(os.getenv("AUTO_OPTIMIZE_MIN_TRADES", "5"))

    print("=" * 60)
    print("AUTO OPTIMIZER SERVIS")
    print("=" * 60)
    print(
        "symbol=%s timeframe=%s lookback=%d trials=%d interval=%dm obj=%s"
        % (symbol, timeframe, lookback, n_trials, interval_minutes, objective)
    )

    base_cfg = StrategyConfig(symbol=symbol, timeframe=timeframe)

    while True:
        started = time.time()
        try:
            ohlcv = fetch_data(symbol, timeframe, lookback)
            best, _ = random_search_optimize(
                ohlcv=ohlcv,
                base_config=base_cfg,
                n_trials=n_trials,
                seed=42,
                initial_balance=initial_balance,
                objective=objective,
                mdd_weight=mdd_weight,
                min_trades=min_trades,
            )
            out = write_strategy_config(data_dir, base_cfg, best.params)
            print(
                "✅ Güncellendi: %s | Sharpe=%.3f PnL%%=%.2f WinRate=%.1f%% MDD%%=%.2f"
                % (
                    out,
                    best.sharpe_ratio,
                    best.total_pnl_pct,
                    best.win_rate * 100,
                    best.max_drawdown_pct,
                )
            )
        except Exception as exc:
            print(f"❌ Optimizasyon hatası: {exc}")

        # If the user set the run-once flag, exit after first successful run (useful for debugging)
        run_once = os.getenv("AUTO_OPTIMIZE_RUN_ONCE", "false").lower() in ("1", "true", "yes")
        if run_once:
            print("🔁 AUTO_OPTIMIZE_RUN_ONCE=true — servis tek seferlik çalıştı, çıkılıyor.")
            break

        elapsed = time.time() - started
        sleep_for = max(30, interval_minutes * 60 - int(elapsed))
        time.sleep(sleep_for)


if __name__ == "__main__":
    try:
        run_service()
    except KeyboardInterrupt:
        print("\n\n⚠️  Servis durduruldu")
    except Exception as e:
        print(f"\n❌ Hata: {e}")
        import traceback
        traceback.print_exc()
