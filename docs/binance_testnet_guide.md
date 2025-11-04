# Binance Testnet Entegrasyon Rehberi

Bu rehber, Algo Trading Lab projesinde Binance test ortamının (https://testnet.binance.vision) hangi rolü oynadığını ve botun farklı katmanlarına nasıl bağlandığını açıklar. Amaç, gerçek paraya geçmeden önce yüksek frekanslı stratejiler dâhil tüm iş akışını güvenli şekilde doğrulamaktır.

## 1. Neden Binance Testnet?
- **Gerçekçi emir akışı:** Testnet, Spot ve Futures marketlerde gerçek API yüzeyini taklit eder. Böylece `bot/exchange.py` içindeki ccxt tabanlı istemci aynı kodla hem paper mode hem de testnet modunda kullanılabilir.
- **Risk almadan deney:** API limitleri, emir reddi sebepleri ve latency davranışı gerçek ortama yakındır; varlık fiyatları ise simüle edilir. Bu sayede strateji döngüsü, risk kontrolleri ve AI karar mantığı canlı ortamı etkilemeden test edilir.
- **HFT prova zemini:** WebSocket akışları (ccxt.pro veya python-binance) ve emirlere verilen yanıt süreleri testnet’te ölçülebilir, bu da ileride düşük gecikmeli optimizasyonlar için veri sağlar.

## 2. Ortam Değişkenleri
`.env` dosyanıza aşağıdaki değişkenleri ekleyerek testnet anahtarlarınızı projeye tanıtın:

```ini
BINANCE_TESTNET_API_KEY=xxx
BINANCE_TESTNET_SECRET=yyy
# Spot yerine perpetual/futures testnet kullanacaksanız true yapın.
BINANCE_TESTNET_USE_FUTURES=true
# Paper mode kapatılarak emirler testnet'e yönlendirilir.
PAPER_MODE=false
```

Ek olarak Futures tarafı için kaldıraç, pozisyon modu gibi ayarları ccxt istemcisi üzerinden yapılandırabilirsiniz. Spot testnet’te kaldıraç olmadığı için risk hesaplamalarını `.env` içindeki `RISK_PER_TRADE_PCT` gibi parametrelerle hizalayın.

## 3. exchange.py İçindeki Rolü
`bot/exchange.py` dosyasında iki temel client bulunur:
1. **PaperExchangeClient:** Varsayılan mock istemci olup emirleri yerel state’e yazar.
2. **CcxtExchangeClient:** ccxt kullanarak gerçek veya testnet API’lerine bağlanır.

Testnet kullanmak için `CcxtExchangeClient` seçilmeli ve ccxt oturumuna `{'options': {'defaultType': 'future'}, 'urls': {'api': binance.urls['apiTest']}}` gibi testnet URL’leri verilmelidir. ccxt bunu `set_sandbox_mode(True)` ile otomatik yapar. Bot konfigürasyonu için örnek kod parçası:

```python
from ccxt import binance

client = binance({
    "apiKey": os.environ["BINANCE_TESTNET_API_KEY"],
    "secret": os.environ["BINANCE_TESTNET_SECRET"],
    "enableRateLimit": True,
})
client.set_sandbox_mode(True)
```

`bot.bot.TradingBot` init aşamasında bu istemciyi alarak emir gönderimlerini testnet’e iletir. Paper/testnet ayrımı `PAPER_MODE` ve `BINANCE_TESTNET_*` bayrakları üzerinden yapılabilir.

## 4. Veri Beslemesi ve HFT Hazırlığı
- **WebSocket:** HFT hedefi varsa REST yerine WebSocket seçin. ccxt’nin `watch_trades`, `watch_order_book` metodları testnet URL’leriyle çalışır. Bot döngüsünü `asyncio` tabanlı hale getirerek WebSocket event’lerini kullanabilirsiniz.
- **Order Book Analitiği:** Testnet verisi ile spread, depth, imbalance ölçümleri yapıp `bot/strategy.py` içinde sinyal kurallarınızı genişletebilirsiniz.
- **Latency Ölçümü:** `bot/state.py` içine emir gönderim ve fill sürelerini yazdırarak gerçek ortama geçmeden performans bütçenizi ölçün.

## 5. Risk Yönetimi
Testnet’te bile risk kuralları (max pozisyon büyüklüğü, stop-loss, take-profit) uygulanmalıdır. `bot/strategy.py` ve `bot/state.py` dosyalarında yer alan pozisyon boyutu formülleri testnet fiyatlarını referans alır. Futures testnet’te kaldıraç etkisini hesaba katmak için:

- `LEVERAGE` değişkenini `.env` içinde tutup pozisyon boyutunu `balance * risk_pct * leverage / price` şeklinde uyarlayın.
- Unrealized PnL raporlamasında mark price’ı kullanmak için ccxt `fetch_positions` çıktısından `markPrice` alanını çekebilirsiniz.

## 6. Dashboard ve API Katmanına Etkisi
- `/status`, `/signals`, `/equity` endpoint’leri testnet verisinden türeyen state’i gösterecektir.
- Dashboard’daki **Macro & News Pulse** ve **AI Insights** bölümleri gerçek zamanlı testnet sonuçlarıyla eşleşen anlatımlar sunar; böylece politika/FAED haberleri ile emir sonuçlarını birlikte gözlemleyebilirsiniz.
- Testnet modunda çalıştığınızı kullanıcıya göstermek için `state.json` içine `"exchange": "binance-testnet"` alanı ekleyebilir, dashboard’da renk kodu kullanabilirsiniz.

## 7. Üretime Geçiş İçin Kontrol Listesi
1. Testnet’te pozisyon açma/kapama senaryolarını, stop-loss/TP tetiklenmelerini doğrulayın.
2. API rate limit ihlallerini ve hata kodlarını loglayın; gerektiğinde otomatik retry/backoff ekleyin.
3. Paper mode ↔️ Testnet ↔️ Gerçek mod geçişlerini `Makefile` veya CLI komutlarıyla şablonlaştırın.
4. Güvenlik için prod anahtarlarını `.env` yerine AWS Secrets Manager benzeri araçlarda tutun.
5. Gerçek ortama geçmeden önce kaldıraç, risk parametreleri ve varlık listesi için yönetim paneli veya config dosyası üzerinden onay adımı ekleyin.

Bu adımlar, Binance test ortamını Algo Trading Lab içinde köprü olarak kullanıp hem strateji arayüzlerini hem de yüksek frekanslı yürütme altyapısını üretim öncesi olgunlaştırmanıza yardımcı olacaktır.
