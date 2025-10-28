# Algo Trading Lab

Algo Trading Lab; çoklu varlıklar için sinyal üretebilen, risk yönetimi yapan ve ileride hem paper-trading hem de gerçek işlemleri destekleyecek şekilde tasarlanmış modüler bir trading bot iskeletidir.

## Özellikler
- Python tabanlı bot döngüsü (EMA crossover + RSI onayı) ve JSON tabanlı state saklama.
- Paper trading modu için sentetik veri üreticisi; ileride ccxt ile gerçek borsa entegrasyonuna hazır.
- FastAPI servisi aracılığıyla `/status`, `/signals`, `/equity` endpoint’leri.
- Docker + docker-compose ile 7/24 çalışacak şekilde konteynerleştirme.
- İlerleyen fazlarda self-supervised learning modelinin entegre edilebilmesi için ayrıştırılmış strateji ve state katmanı.

## Dizin Yapısı
```
algo_trading_lab/
├── bot/
│   ├── bot.py          # Ana loop ve risk yönetimi
│   ├── exchange.py     # ccxt wrapper + paper-exchange mock
│   ├── state.py        # JSON tabanlı state/signals/equity saklama
│   └── strategy.py     # EMA/RSI stratejisi ve pozisyon boyutu hesapları
├── api/
│   ├── api.py          # FastAPI uygulaması
│   └── schemas.py      # Pydantic response şemaları
├── data/               # State dosyaları (docker volume ile paylaşılır)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

## Başlangıç

### Binance Spot Testnet Kurulumu
1. https://testnet.binance.vision/ adresine gidin ve API anahtarı oluşturun
2. API Key ve Secret Key'i kopyalayın
3. `.env` dosyasını düzenleyin:
   ```bash
   cp .env.example .env
   ```
4. `.env` içerisinde testnet bilgilerini güncelleyin:
   ```bash
   BINANCE_TESTNET_ENABLED=true
   BINANCE_TESTNET_API_KEY=your_api_key_here
   BINANCE_TESTNET_API_SECRET=your_secret_key_here
   PAPER_MODE=false  # Testnet kullanmak için false yapın
   ```

### Test Bağlantısı
Binance testnet bağlantınızı test etmek için:
```bash
python test_binance_testnet.py
```

### Ortam Değişkenleri
1. Ortam değişkenlerini düzenleyin:
   ```bash
   cp .env.example .env
   # .env içindeki değerleri ihtiyacınıza göre güncelleyin
   ```
2. Konteynerleri ayağa kaldırın:
   ```bash
   docker-compose up --build
   ```
3. FastAPI arayüzü varsayılan olarak `http://localhost:8000/docs` adresinde çalışır.

## Lokal Geliştirme
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export $(grep -v '^#' .env | xargs)  # veya dotenv kullanın
python -m bot.bot  # bot döngüsünü başlatır
uvicorn api.api:app --reload
```

## Örnek State Çıktısı
```json
{
  "timestamp": "2025-10-28T16:32:00Z",
  "symbol": "BTC/USDT",
  "position": "LONG",
  "entry_price": 67321.5,
  "unrealized_pnl_pct": 0.42,
  "last_signal": "LONG",
  "confidence": 0.66,
  "rsi": 54.2,
  "ema_fast": 67310.1,
  "ema_slow": 67190.7,
  "risk_per_trade_pct": 0.5
}
```

## Notlar
- `requirements.txt` dosyası temel bağımlılıkları içerir. SSL/ML entegrasyonu için PyTorch ve PyTorch Lightning ek olarak kurulmalıdır (platforma göre whl dosyaları değişir).
- **Testnet Kullanımı**: `.env` dosyasında `BINANCE_TESTNET_ENABLED=true` ve `PAPER_MODE=false` yaparak Binance Spot Testnet'i kullanabilirsiniz.
- **Production Kullanımı**: Gerçek borsa kullanımı için `.env` dosyasındaki `PAPER_MODE=false`, `BINANCE_TESTNET_ENABLED=false` ve `EXCHANGE_API_KEY`, `EXCHANGE_API_SECRET` alanlarını güncelleyin.
- Çoklu enstrüman desteği için `docker-compose` içerisine aynı imajdan türetilmiş yeni servisler eklenebilir veya bot loop'u parametre alacak şekilde genişletilebilir.

## High Frequency Trading (HFT) için Öneriler
- Binance Futures Testnet kullanın (daha gerçekçi): https://testnet.binancefuture.com
- REST API yerine WebSocket ile order book ve trade stream'leri dinleyin
- Latency optimizasyonu için sunucunuzu Binance'e yakın bir bölgede çalıştırın
- Rate limit ve order matching test edilmelidir

