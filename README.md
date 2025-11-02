# Algo Trading Lab

Algo Trading Lab; çoklu varlıklar için sinyal üretebilen, risk yönetimi yapan ve ileride hem paper-trading hem de gerçek işlemleri destekleyecek şekilde tasarlanmış modüler bir trading bot iskeletidir.

## Özellikler
- Python tabanlı bot döngüsü (EMA crossover + RSI onayı) ve JSON tabanlı state saklama.
- Paper trading modu için sentetik veri üreticisi; ileride ccxt ile gerçek borsa entegrasyonuna hazır.
- *Yeni:* Hisse, endeks, altın ve emtia gibi kripto dışı varlıklardan veri çekebilen (opsiyonel `yfinance`) piyasa veri katmanı.
- *Yeni:* Portföy seviyesinde çoklu varlık çalıştırıcısı; her enstrüman için ayrı risk parametreleri ve veri klasörü ile eş zamanlı bot döngüleri.
- FastAPI servisi aracılığıyla `/status`, `/signals`, `/equity`, `/strategy` endpoint’leri ve dahili web dashboard'u.
- Yapay zekâ katmanı için `/ai/prediction` (tahmin) ve `/ai/question` (soru-cevap) endpoint’leri ile dashboard üzerindeki AI Insights bölümü.
- Trump gibi politik aktörlerin kararları ve Fed faiz beklentileri gibi makro başlıkları skorlayan makro motoru; `/macro/insights` endpoint’i ve dashboard üzerindeki **Macro & News Pulse** paneli ile son katalizörleri takip eder.
- Docker + docker-compose ile 7/24 çalışacak şekilde konteynerleştirme.
- İlerleyen fazlarda self-supervised learning modelinin entegre edilebilmesi için ayrıştırılmış strateji ve state katmanı.

## Dizin Yapısı
```
algo_trading_lab/
├── bot/
│   ├── ai.py           # Heuristik AI tahmincisi ve soru-cevap motoru
│   ├── bot.py          # Ana loop ve risk yönetimi
│   ├── market_data.py  # ccxt/yfinance/paper veri sağlayıcıları
│   ├── exchange.py     # ccxt wrapper + paper-exchange mock
│   ├── state.py        # JSON tabanlı state/signals/equity saklama
│   ├── strategy.py     # EMA/RSI stratejisi ve pozisyon boyutu hesapları
│   ├── portfolio.py    # Çoklu varlık portföy koşucusu
│   ├── backtesting.py  # Backtest motoru
│   └── trading.py      # Gerçek işlem yöneticisi
├── api/
│   ├── api.py          # FastAPI uygulaması
│   └── schemas.py      # Pydantic response şemaları
├── data/               # State dosyaları (docker volume ile paylaşılır)
├── test_binance_testnet.py  # Testnet bağlantı testi
├── run_backtest.py     # Backtest çalıştırma scripti
├── run_live_trading.py # Canlı trading scripti
├── run_portfolio.py    # Çoklu varlık botunu başlatan script
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

## 🎯 Strateji Testi ve Al-Sat Kararları

### 1. Backtest (Geçmiş Veri Testi)
Stratejinizi geçmiş verilerle test edin:

```bash
python run_backtest.py
```

Bu script ile:
- Geçmiş verilerde stratejinizi test edebilirsiniz
- Win rate, profit factor, max drawdown gibi metrikleri görebilirsiniz
- Farklı parametrelerle deneme yapabilirsiniz
- Sonuçları JSON dosyasına kaydedebilirsiniz

**Örnek Çıktı:**
```
============================================================
BACKTEST SONUÇLARI
============================================================
Başlangıç Bakiyesi: $10,000.00
Bitiş Bakiyesi: $11,250.00
Toplam P&L: $1,250.00 (12.50%)

Toplam İşlem: 45
Kazanan: 28 | Kaybeden: 17
Win Rate: 62.22%
Ortalama Kazanç: $120.50
Ortalama Kayıp: $65.30
Profit Factor: 1.85
Max Drawdown: $450.00 (4.50%)
Sharpe Ratio: 1.42
============================================================
```

### 2. Canlı Trading (Testnet veya Gerçek)
Stratejinizi canlı olarak çalıştırın:

```bash
python run_live_trading.py
```

**3 Mod Seçeneği:**
1. **DRY RUN**: Sadece log tutar, gerçek emir göndermez (güvenli test)
2. **TESTNET**: Binance testnet'te gerçek emir gönderir (test parası)
3. **LIVE**: GERÇEK BORSADA işlem yapar (DİKKAT!)

**Önerilen İş Akışı:**
```
1. Backtest ile stratejiyi test et
   └─> Win rate > %55 ve Profit Factor > 1.5 ise devam et
   
2. DRY RUN modunda canlı veri ile test et (1-2 gün)
   └─> Sinyaller mantıklı mı kontrol et
   
3. TESTNET modunda gerçek emirlerle test et (1 hafta)
   └─> Emir gönderimi, stop loss, take profit çalışıyor mu?
   
4. Küçük sermaye ile LIVE teste geç
   └─> Risk yönetimini doğrula

5. Tam sermaye ile production
```

### 3. Portföy Botu (Çoklu Varlık)
Kripto dışı varlıkları (hisse, ETF, altın, endeks vb.) aynı loop içinde takip etmek için yeni portföy koşucusunu kullanın.

1. Örnek konfigürasyonu çoğaltın:
   ```bash
   cp data/portfolio.sample.json data/portfolio.json
   ```
2. Dosya içindeki `assets` listesine istediğiniz sembolleri ekleyin. `asset_type` alanı `crypto`, `equity`, `commodity`, `forex` gibi değerler alabilir. Yahoo Finance ile veri çekilecekse `data_symbol` alanına ilgili ticker'ı (`GC=F`, `^GSPC`, `AAPL` vb.) yazın.
3. Toplam sermayeyi (`portfolio_capital`) ve her varlığın payını (`allocation_pct`) belirleyin. Boş bırakılanlar kalan yüzdeyi eşit böler.
4. Botu başlatın:
   ```bash
   python run_portfolio.py --config data/portfolio.json
   ```

> **Not:** Hisse/emtia verisi çekebilmek için `pip install yfinance` kurulu olmalıdır. Makro duyarlılık motoru her varlık için `macro_symbol` tanımlanırsa ilgili katalizörleri ayrı ayrı raporlar.

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
4. Yönetim dashboard'una `http://localhost:8000/dashboard` adresinden erişebilirsiniz.
   - Bot henüz çalışmıyorsa bile `/dashboard/preview` (veya `?demo=1` parametresi) ile canlı önizlemeyi görebilirsiniz.
   - Dashboard üzerindeki **AI Insights** bölümü, `/ai/prediction` ve `/ai/question` endpoint'lerinden gelen verilerle modelin önerdiği aksiyonu, olasılık dağılımını ve açıklamasını gösterir.
   - **Decision Playbook** bölümü, botun ne zaman LONG/SHORT olacağını ve risk yönetimini nasıl yaptığını `/strategy` endpoint'inden aldığı verilere göre özetler.

## Dashboard nasıl görünüyor?
Dashboard, tek sayfalık bir arayüz içinde aşağıdaki bloklarla organize edilmiştir:
- **Üst durum şeridi:** Seçili sembol, pozisyon, giriş fiyatı, gerçekleşmemiş PnL ve bot çalışma modunu gösteren renk kodlu kartlar.
- **Signal Stream:** Sağ tarafta son sinyaller, emir özetleri ve AI tahminlerine ait kısa açıklamalar kronolojik olarak akar.
- **Equity & Risk:** Orta bölümde equity eğrisi, günlük PnL şeridi ve risk parametreleri yan yana yer alır. Preview modunda örnek veri, canlı modda state dosyasındaki gerçek değerler gösterilir.
- **AI Insights:** AI aksiyonu, olasılıklar, açıklayıcı özellikler (EMA açığı, momentum vb.) ve kısa anlatım kutucuğu.
- **Decision Playbook:** EMA/RSI eşiklerini, stop-loss/take-profit örneklerini ve pozisyon boyutu formülünü, canlı strateji konfigürasyonuna göre açıklar.
- **Macro & News Pulse:** Trump/Fed gibi katalizörleri, makro bias skorunu, faiz görünümünü ve siyasi risk özetlerini listeler.
- **Assistant formu:** Dashboard alt kısmındaki form ile `/ai/question` endpoint’ine soru gönderebilir, cevapları gerçek zamanlı görebilirsiniz; preview modunda örnek sorular hazır gelir.

`/dashboard/preview` rotası bu bileşenlerin tamamını örnek veriyle render eder; bu sayede botu başlatmadan arayüzü inceleyebilir ve tasarımı özelleştirebilirsiniz. Daha ayrıntılı bir bölümlendirme ve ASCII yerleşim krokisi için [docs/ui_walkthrough.md](docs/ui_walkthrough.md) dosyasına göz atabilirsiniz.

## Neleri geliştirebilirim?
Aşağıdaki alanlar ilk etapta kolayca genişletilebilir:
1. **Görsel tema ve marka kimliği:** `api/dashboard.html` içinde Tailwind-esintili yardımcı sınıflar bulunuyor; kendi renk paletinizi eklemek için `<style>` bloklarındaki CSS değiştirilebilir veya harici bir CSS dosyası eklenebilir.
2. **Grafik kütüphaneleri:** Şu an lightweight SVG grafikleri kullanılıyor. Highcharts, Plotly veya TradingView widget’ını ekleyerek daha detaylı grafikler sunabilirsiniz.
3. **Çoklu enstrüman desteği:** Dashboard’daki sembol seçiciyi genişleterek aynı anda birden fazla varlık için sinyal/equity görüntüleme imkânı ekleyebilirsiniz.
4. **Bildirim ve uyarılar:** WebSocket/Server-Sent Events kanalıyla yeni sinyaller veya kritik makro olaylar için tarayıcı bildirimleri göndermek mümkün.
5. **Kullanıcı yönetimi:** FastAPI tarafında auth katmanı ekleyip dashboard’u parola korumalı hale getirebilirsiniz.

## High Frequency Trading (HFT) yol haritası
HFT’ye yaklaşırken aşağıdaki teknik geliştirmeler önemlidir:
1. **Düşük gecikmeli veri akışı:** REST çağrıları yerine Binance WebSocket (ccxt.pro veya python-binance) kullanarak milisaniye seviyesinde fiyat güncellemeleri alın.
2. **Asenkron bot döngüsü:** `bot/bot.py` içinde veri alma, sinyal hesaplama ve emir gönderme adımlarını `asyncio` tabanlı hale getirip aynı anda birden fazla varlık için concurrency sağlayın.
3. **Order book izleme:** Yalnızca OHLCV yerine seviye-2 order book verilerini okuyup mikro yapı sinyalleri (spread, imbalance) üretin.
4. **Risk guardrail’leri:** HFT’de hatalar hızlı büyür; latency, başarısız emir sayısı veya art arda zarar limitleri için otomatik circuit breaker’lar ekleyin.
5. **Performans ölçümü:** Prometheus metrikleriyle ortalama latency, fill oranı, kayma (slippage) ve PnL dağılımını takip edin; Grafana veya özel dashboard’a gerçek zamanlı grafikler ekleyin.
6. **Backtest & simülasyon:** vectorbt/backtrader ile saniyelik/dakikalık veri üzerinde HFT stratejisi senaryolarını simüle edip gerçek ortamla kıyaslayın.

Bu yol haritasındaki adımlar, mevcut mimariye kademeli olarak entegre edilerek UI’nın sunduğu içgörüleri milisaniye ölçekli karar destek sistemine dönüştürmenize yardımcı olur.

## AI Destekli Tahmin ve Soru-Cevap
- **AI Prediction (`GET /ai/prediction`)**: Son loop’taki yapay zekâ değerlendirmesini döndürür. Yanıt, önerilen aksiyon (`LONG`/`SHORT`/`FLAT`), güven skoru, uzun/kısa/düz olasılıkları, beklenen hareket yüzdesi ve kullanılan ana özelliklerin hızlı özetini içerir.
- **AI Question (`POST /ai/question`)**: JSON gövdesinde `{ "question": "When should I buy?" }` benzeri bir istekle stratejiye dair sorular sorabilirsiniz. Motor, güncel state ve AI tahminini kullanarak yanıt verir.
- Dashboard’daki formu kullanarak aynı soru-cevap deneyimini tarayıcıdan da test edebilirsiniz; preview modunda örnek yanıtlar simüle edilir.
- Sorularınıza `macro`, `Trump`, `Fed`, `rates` gibi anahtar kelimeler eklerseniz AI motoru makro modülden gelen öngörüleri de yanıtına dahil eder.

## Makro & Haber Farkındalığı
- Bot döngüsü her turda `bot/macro.py` içindeki `MacroSentimentEngine` ile makro/politik olay listesini değerlendirir. Varsayılan olarak Trump’ın tarifeleri ve Fed toplantı rehberliği gibi örnek olaylar gelir; kendi olaylarınızı `data/macro_events.json` benzeri bir dosyayla genişletebilirsiniz.
- Özel olaylar eklemek için JSON listesi kullanın. Örnek yapı `data/macro_events.sample.json` içinde yer alır:
  ```json
  [
    {
      "title": "Trump announces new tariff schedule",
      "category": "politics",
      "sentiment": "bearish",
      "impact": "high",
      "actor": "Donald Trump",
      "summary": "Tariff threats raise volatility across risk assets.",
      "assets": { "BTC/USDT": -0.2, "ETH/USDT": -0.15 }
    },
    {
      "title": "FOMC statement",
      "category": "central_bank",
      "sentiment": "dovish",
      "impact": "medium",
      "interest_rate_expectation": "Fed signals a cautious path with one cut pencilled in for Q4."
    }
  ]
  ```
- Dosyayı botun eriştiği `DATA_DIR` altında `macro_events.json` adıyla saklayın ve `.env` içinde `MACRO_EVENTS_PATH=data/macro_events.json` şeklinde işaret edin. `MACRO_REFRESH_SECONDS` ile yükleme aralığını (varsayılan 300 sn) değiştirebilirsiniz.
- `/macro/insights` endpoint’i ve dashboard’daki **Macro & News Pulse** paneli; özet makro bias skorunu, güven seviyesini, faiz beklentilerini ve son katalizör listesini JSON veya görsel olarak sunar. Bu sinyaller AI tahminine ağırlık olarak eklenir, böylece haber akışı LONG/SHORT kararlarını güçlendirebilir veya zayıflatabilir.

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
  "risk_per_trade_pct": 0.5,
  "ai_action": "LONG",
  "ai_confidence": 0.72,
  "ai_probability_long": 0.72,
  "ai_probability_short": 0.18,
  "ai_probability_flat": 0.1,
  "ai_expected_move_pct": 0.64,
  "ai_summary": "Model leans upside with 72.0% confidence driven by EMA spread 0.48% and momentum 0.35%. Expected move: 0.64%",
  "ai_features": {
    "ema_gap_pct": 0.48,
    "momentum_pct": 0.35,
    "rsi_distance_from_mid": 8.5,
    "volatility_pct": 0.62
  },
  "macro_bias": -0.18,
  "macro_confidence": 0.58,
  "macro_summary": "Macro bias is bearish (-0.18) based on 3 tracked catalysts. Fed watch: Fed likely to keep rates unchanged but watch core inflation prints. Political risk: Donald Trump: Potential tariff escalation keeps risk assets cautious.",
  "macro_drivers": [
    "Trump vows fresh tariffs review (bearish, high impact)",
    "US payrolls surprise to upside (hawkish, high impact)"
  ],
  "macro_interest_rate_outlook": "Fed likely to keep rates unchanged but watch core inflation prints.",
  "macro_political_risk": "Donald Trump: Potential tariff escalation keeps risk assets cautious.",
  "macro_events": [
    {
      "title": "Trump vows fresh tariffs review",
      "category": "politics",
      "sentiment": "bearish",
      "impact": "high",
      "actor": "Donald Trump"
    },
    {
      "title": "Fed officials guide for data-dependent path",
      "category": "central_bank",
      "impact": "medium",
      "interest_rate_expectation": "Fed likely to keep rates unchanged but watch core inflation prints."
    }
  ]
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


## Backend ve Frontend Açıkları
Proje hangi alanlarda henüz eksik diye hızlıca bakmak için [`docs/backend_frontend_gaps.md`](docs/backend_frontend_gaps.md) dosyasına göz atın. Bu doküman hem sunucu tarafında (borsa entegrasyonu, risk yönetimi, dağıtım) hem de arayüz tarafında (component mimarisi, gerçek zamanlı veri akışı, erişilebilirlik) tamamlanması gereken somut maddeleri kontrol listesi şeklinde sunar.
