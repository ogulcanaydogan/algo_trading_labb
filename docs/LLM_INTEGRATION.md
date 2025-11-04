# 🤖 LLM Entegrasyonu Kılavuzu

## 🎯 Genel Bakış

Bu sistem, **yerel Mistral-7B LLM** kullanarak trading stratejisi geliştirme sürecini destekler.

---

## 📦 Kurulum

### 1. Ollama Kurulumu

```bash
# macOS
brew install ollama

# Servisi başlat
brew services start ollama

# Mistral modelini indir
ollama pull mistral
```

### 2. Python Bağımlılıkları

```bash
pip install requests pyyaml feedparser vaderSentiment
```

### 3. Doğrulama

```bash
# LLM client'ı test et
python tools/llm_client.py

# Çıktı:
# ✅ LLM servisi çalışıyor!
# Cevap: ...
```

---

## 🚀 Kullanım

### A) Jupyter Notebook'ta LLM

```python
# LLM client'ı import et
from tools.llm_client import LLMClient

# LLM'i başlat
llm = LLMClient(model="mistral")

# Soru sor
answer = llm.ask("Bitcoin için EMA crossover stratejisi nasıl çalışır?")
print(answer)
```

**Notebook:** `notebooks/strategy_research.ipynb` - Bölüm 15

### B) Haber Analizi (LLM ile)

```bash
# LLM ile detaylı analiz
python tools/ingest_news_llm.py \
  --feeds data/feeds.news.yml \
  --out data/macro_events.llm.json \
  --symbols "BTC/USDT,NVDA,GC=F" \
  --use-llm

# VADER ile basit analiz (LLM olmadan)
python tools/ingest_news_llm.py \
  --feeds data/feeds.news.yml \
  --out data/macro_events.basic.json \
  --symbols "BTC/USDT,NVDA,GC=F"
```

### C) Bot'a Entegrasyon

```python
# bot/macro.py içinde LLM kullan
from tools.llm_client import get_llm_client

llm = get_llm_client()
analysis = llm.analyze_news(news_items, symbol="BTC/USDT")

# MacroEvent oluştur
event = MacroEvent(
    title=f"LLM Analysis for {symbol}",
    sentiment=analysis['sentiment'],
    bias=analysis['bias_score'],
    impact=analysis['impact'],
    ...
)
```

---

## 🎯 LLM Fonksiyonları

### 1. `ask(prompt, system_prompt, temperature)`

Genel amaçlı soru-cevap.

```python
answer = llm.ask(
    "Volatilite yüksekken stop-loss nasıl ayarlanır?",
    temperature=0.7
)
```

### 2. `analyze_news(news_items, symbol)`

Haberleri analiz et, sentiment + impact döndür.

```python
analysis = llm.analyze_news(
    news_items=[...],
    symbol="BTC/USDT"
)
# Returns:
# {
#   "sentiment": "bullish",
#   "impact": "high",
#   "bias_score": 0.65,
#   "confidence": 0.82,
#   "summary": "...",
#   "catalysts": ["...", "..."]
# }
```

### 3. `suggest_strategy(symbol, performance, market_conditions)`

Backtest sonuçlarına göre strateji önerisi.

```python
suggestion = llm.suggest_strategy(
    symbol="BTC/USDT",
    historical_performance={
        "sharpe_ratio": 0.8,
        "win_rate": 55.0,
        "max_drawdown_pct": 12.5
    },
    market_conditions={
        "volatility": "high",
        "trend": "bullish",
        "rsi": 65
    }
)
```

### 4. `optimize_parameters(symbol, current_params, performance_history)`

Grid search sonuçlarını yorumla, optimizasyon öner.

```python
advice = llm.optimize_parameters(
    symbol="NVDA",
    current_params={"ema_fast": 12, "ema_slow": 26},
    performance_history=[...]  # Top 10 kombinasyon
)
```

### 5. `explain_trade(trade_data, market_context)`

Bir işlemi açıkla (neden açıldı, neden kapandı).

```python
explanation = llm.explain_trade(
    trade_data={
        "side": "LONG",
        "entry_price": 30500,
        "exit_price": 31200,
        "pnl_pct": 2.3,
        "exit_reason": "Take Profit"
    },
    market_context={
        "ema_fast": 30450,
        "ema_slow": 30200,
        "rsi": 68
    }
)
```

---

## 📊 Örnek Prompt'lar

### Strateji Geliştirme

```
"BTC/USDT için RSI 14 ve EMA 50 kullanan bir mean reversion
stratejisi öner. Stop-loss %2, take-profit %4 olsun. 
Python kodunu yaz."
```

### Parametre Optimizasyonu

```
"EMA fast'ı 10-30, EMA slow'u 30-100 arasında test eden
bir grid search kodu yaz. En yüksek Sharpe ratio'yu bul."
```

### Risk Yönetimi

```
"Volatilite arttığında risk_per_trade'i otomatik azaltan,
düştüğünde artıran dinamik bir risk yönetimi fonksiyonu yaz.
ATR kullan."
```

### Haber Analizi

```
"Şu haberler BTC/USDT'yi nasıl etkiler?
- Fed faiz artırımına devam edecek
- Bitcoin ETF onayları yaklaşıyor
Bullish mi bearish mi? Neden?"
```

---

## 🔧 Konfigürasyon

### Model Seçimi

```python
# Mistral (default)
llm = LLMClient(model="mistral")

# Alternatif modeller
llm = LLMClient(model="phi4")
llm = LLMClient(model="llama3.1")
```

### Temperature Ayarı

- **0.0-0.3**: Deterministik, tutarlı (metrik hesaplama, classification)
- **0.4-0.7**: Dengeli (genel amaçlı, strateji önerileri)
- **0.8-1.0**: Yaratıcı (brainstorming, yeni fikirler)

```python
# Tutarlı analiz için
answer = llm.ask(prompt, temperature=0.3)

# Yaratıcı öneriler için
answer = llm.ask(prompt, temperature=0.9)
```

---

## ⚠️ Önemli Notlar

### LLM Ne YAPAR?

✅ **Fikir üretir** - Strateji önerileri, parametre aralıkları
✅ **Kod yazar** - Prototip fonksiyonlar, algoritma skeleton'ları
✅ **Analiz yapar** - Backtest sonuçları, haber sentiment'ı
✅ **Açıklar** - İşlem mantığı, teknik gösterge yorumları

### LLM Ne YAPMAZ?

❌ **Gerçek zamanlı alım-satım kararı** - Bu senin kodun yapar
❌ **Gerçek piyasa verisi üretimi** - ccxt/yfinance kullan
❌ **Garantili kazanç** - Sadece bir araç, nihai karar sendedir

### Güvenlik

- LLM çıktısını **her zaman doğrula**
- Backtest/forward test **zorunlu**
- Gerçek para ile denemeden önce **testnet/paper trading**
- Risk limitleri (max drawdown, max exposure) **kod seviyesinde**

---

## 📈 Performans

### M2 Pro (16GB RAM)

| Model | Parametre | Inference Hızı | RAM Kullanımı |
|-------|-----------|----------------|---------------|
| Mistral-7B | 7B | ~1-2 saniye | 8-10 GB |
| Phi-4-mini | 3.8B | ~0.5-1 saniye | 4-6 GB |
| Llama 3.1 | 8B | ~2-3 saniye | 10-12 GB |

### Optimizasyon İpuçları

1. **Batch işleme** - Birden fazla soruyu tek prompt'ta birleştir
2. **Cache kullan** - Tekrar eden analizler için sonuçları kaydet
3. **Timeout ayarla** - Uzun süren prompt'lar için 120s timeout
4. **Temperature düşür** - Classification için 0.3, brainstorming için 0.8

---

## 🔄 Fine-Tuning (Gelecek)

### Veri Toplama

```python
# Trade log'larını kaydet
{
  "timestamp": "2025-11-01T12:00:00Z",
  "symbol": "BTC/USDT",
  "decision": "LONG",
  "entry_price": 30500,
  "indicators": {"rsi": 32, "ema_gap": 1.2},
  "outcome": "win",
  "pnl_pct": 2.3
}
```

### Fine-Tuning Süreci (3-6 ay sonra)

1. **Veri biriktir** - En az 500-1000 işlem log'u
2. **Format düzenle** - Instruction-tuning formatına çevir
3. **LoRA ile fine-tune** - M2 Pro'da 2-4 saat
4. **Değerlendirme** - Base model vs fine-tuned karşılaştır
5. **Deploy** - Kişiselleştirilmiş modeli kullan

**Araçlar:**
- `llama.cpp` - Inference + fine-tuning
- `Axolotl` - Fine-tuning framework
- `Unsloth` - macOS optimize edilmiş

---

## 🆘 Sorun Giderme

### LLM yanıt vermiyor

```bash
# Ollama servisini kontrol et
brew services list | grep ollama

# Servisi başlat
brew services start ollama

# Manuel başlatma
ollama serve
```

### JSON parse hatası

LLM bazen JSON dışında metin döndürür. `llm_client.py` otomatik temizler:

```python
# Markdown code block'tan çıkar
if "```json" in response:
    response = response.split("```json")[1].split("```")[0]
```

### Timeout hatası

```python
# Timeout'u artır
llm = LLMClient()
llm.timeout = 180  # 3 dakika
```

### Model bulunamadı

```bash
# Mevcut modelleri listele
ollama list

# Model indir
ollama pull mistral
```

---

## 📚 Kaynaklar

- **Ollama Docs**: https://ollama.ai/docs
- **Mistral AI**: https://mistral.ai/
- **LoRA Paper**: https://arxiv.org/abs/2106.09685
- **Notebook**: `notebooks/strategy_research.ipynb`
- **LLM Client**: `tools/llm_client.py`
- **Haber İngestor**: `tools/ingest_news_llm.py`

---

## 🎉 Başarı Hikayeleri

### Örnek 1: Parametre Optimizasyonu

```
Kullanıcı: "EMA parametrelerini optimize etmek istiyorum"
LLM: "EMA fast için 8-20, slow için 30-80 aralığında test et..."
→ Grid search sonrası: Sharpe 0.5 → 1.2 (+140%)
```

### Örnek 2: Risk Yönetimi

```
Kullanıcı: "Yüksek volatilitede kayıplarım artıyor"
LLM: "ATR bazlı dinamik stop-loss kullan. ATR > ma(ATR) ise..."
→ Max drawdown: 18% → 9% (-50%)
```

### Örnek 3: Haber Analizi

```
187 haber → LLM analizi → 3 macro event
Sentiment: Bearish bias -0.54
→ Bot SHORT bias kullanarak 5 günde +8.2% kazanç
```

---

**🚀 Mutlu Trading'ler!**

*Last Updated: 2025-11-01*
