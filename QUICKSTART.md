# 🚀 Hızlı Başlangıç Kılavuzu

## 1. Kurulum ve İlk Test (5 dakika)

```bash
# Depoyu klonla veya indir
cd algo_trading_lab

# Sanal ortam oluştur ve aktive et
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Bağımlılıkları kur
pip install -r requirements.txt

# .env dosyasını oluştur
cp .env.example .env
```

### Binance Testnet API Anahtarı Al
1. https://testnet.binance.vision/ adresine git
2. Giriş yap ve API anahtarı oluştur (TRADE, USER_DATA, USER_STREAM izinleri)
3. API Key ve Secret Key'i kopyala
4. `.env` dosyasını aç ve anahtarları ekle:

```bash
BINANCE_TESTNET_ENABLED=true
BINANCE_TESTNET_API_KEY=senin_api_key
BINANCE_TESTNET_API_SECRET=senin_secret_key
PAPER_MODE=false
```

### Bağlantıyı Test Et
```bash
python test_binance_testnet.py
```

✅ Başarılı çıktı:
```
✅ Client created successfully
✅ Successfully fetched 10 candles
✅ Account access successful
✅ All tests passed! Testnet connection is working.
```

---

## 2. Stratejiyi Backtest ile Test Et (10 dakika)

```bash
python run_backtest.py
```

**Örnek Girişler:**
- Symbol: `BTC/USDT`
- Timeframe: `1h`
- Kaç mum: `1000`
- Başlangıç bakiyesi: `10000`
- EMA Fast: `12`
- EMA Slow: `26`
- RSI Period: `14`
- Risk per trade %: `1.0`
- Stop Loss %: `2.0`
- Take Profit %: `4.0`
- Veri kaynağı: `1` (Binance Testnet)

### Sonuçları Değerlendir

**İyi Sonuçlar:**
- ✅ Win Rate > %55
- ✅ Profit Factor > 1.5
- ✅ Max Drawdown < %15
- ✅ Sharpe Ratio > 1.0

**Kötü Sonuçlar:**
- ❌ Win Rate < %45
- ❌ Profit Factor < 1.0
- ❌ Max Drawdown > %30

➡️ Kötü sonuçlar varsa parametreleri değiştir ve tekrar test et.

---

## 3. DRY RUN ile Canlı Test (1 gün)

```bash
python run_live_trading.py
```

**Seçimler:**
- Trading Modu: `1` (DRY RUN)
- Symbol: `BTC/USDT`
- Timeframe: `5m`
- Loop interval: `60` (saniye)

Bu modda:
- ✅ Gerçek veri kullanır
- ✅ Sinyalleri görürsün
- ✅ Log tutar
- ❌ Gerçek emir göndermez

**Ne izlemeli:**
1. Sinyaller mantıklı mı?
2. Stop loss ve take profit seviyeleri uygun mu?
3. Çok sık işlem yapıyor mu?
4. RSI ve EMA doğru çalışıyor mu?

---

## 4. TESTNET ile Gerçek Emir Testi (1 hafta)

```bash
python run_live_trading.py
```

**Seçimler:**
- Trading Modu: `2` (TESTNET)
- Diğer ayarlar aynı

Bu modda:
- ✅ Gerçek emir gönderir (test parası ile)
- ✅ Stop loss ve take profit emirleri çalışır
- ✅ Emir iptal ve pozisyon kapatma test edilir

**Ne izlemeli:**
1. Emirler düzgün gönderiliyor mu?
2. Stop loss tetikleniyor mu?
3. Take profit çalışıyor mu?
4. Hata mesajı var mı?

---

## 5. Parametre Optimizasyonu

Farklı parametrelerle backtest çalıştır ve en iyi kombinasyonu bul:

| Parametre | Deneme Değerleri |
|-----------|------------------|
| EMA Fast | 8, 12, 16 |
| EMA Slow | 21, 26, 34 |
| RSI Period | 7, 14, 21 |
| Risk % | 0.5, 1.0, 2.0 |
| Stop Loss % | 1.0, 2.0, 3.0 |
| Take Profit % | 2.0, 4.0, 6.0 |

**Örnek Test Matrisi:**
```bash
# Test 1: Hızlı EMA
EMA Fast: 8, EMA Slow: 21 -> Backtest çalıştır

# Test 2: Standart EMA
EMA Fast: 12, EMA Slow: 26 -> Backtest çalıştır

# Test 3: Yavaş EMA
EMA Fast: 16, EMA Slow: 34 -> Backtest çalıştır

# En iyi sonucu veren kombinasyonu seç
```

---

## 6. Production'a Geçiş (İsteğe Bağlı)

⚠️ **DİKKAT**: Gerçek para kullanacaksınız!

### Önce Küçük Başla
1. Binance'de gerçek API anahtarı oluştur
2. `.env` dosyasını güncelle:
```bash
BINANCE_TESTNET_ENABLED=false
EXCHANGE_API_KEY=gerçek_api_key
EXCHANGE_API_SECRET=gerçek_secret
```

3. İlk trade'de **çok küçük pozisyon** kullan
4. 1 hafta izle
5. Başarılı olursa yavaşça artır

---

## 📊 Metrik Tablosu

| Metrik | İyi | Orta | Kötü |
|--------|-----|------|------|
| Win Rate | >60% | 50-60% | <50% |
| Profit Factor | >2.0 | 1.5-2.0 | <1.5 |
| Max Drawdown | <10% | 10-20% | >20% |
| Sharpe Ratio | >1.5 | 1.0-1.5 | <1.0 |

---

## 🆘 Sorun Giderme

### Hata: "Import could not be resolved"
```bash
# Sanal ortamı aktive ettin mi?
source .venv/bin/activate

# Bağımlılıkları kur
pip install -r requirements.txt
```

### Hata: "API keys not found"
`.env` dosyasının doğru yerde olduğundan emin ol (proje root'unda)

### Hata: "Not enough data for indicator calculation"
Daha fazla mum verisi çek (lookback değerini artır)

### Test parası yok
Binance testnet'te otomatik test parası verilir, fakat bazen sıfırlanabilir. Yeni hesap oluşturarak tekrar dene.

---

## 📚 Sonraki Adımlar

1. **WebSocket Entegrasyonu**: Daha hızlı veri için WebSocket kullan
2. **Çoklu Timeframe**: Farklı timeframe'lerden sinyal al
3. **Machine Learning**: Model entegre et
4. **Dashboard**: Streamlit veya Dash ile görsel arayüz ekle
5. **Alarm Sistemi**: Telegram veya email bildirimleri

---

## 💡 İpuçları

- 🔥 **Risk Yönetimi**: Tek işlemde toplam bakiyenin %1-2'sinden fazlasını riske atma
- ⏰ **Sabırlı Ol**: İyi fırsatları bekle, her sinyale girme
- 📊 **Backtest Önemli**: Backtest olmadan canlı trading'e geçme
- 🛡️ **Stop Loss Kullan**: Her zaman stop loss belirle
- 📝 **Log Tut**: Tüm işlemleri kaydet ve analiz et
- 🔄 **Sürekli İyileştir**: Sonuçları düzenli olarak gözden geçir

---

## 📞 Yardım

Sorular için:
1. README.md dosyasını oku
2. Kod içindeki docstring'lere bak
3. Backtest sonuçlarını analiz et
4. Testnet'te önce dene

**Başarılar! 🚀**
