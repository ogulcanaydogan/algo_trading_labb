# 🤖 LLM Trading Assistant - Quick Start

## ✅ Tamamlanan Kurulum

- [x] Ollama kuruldu
- [x] Mistral-7B modeli indirildi
- [x] LLM client (`tools/llm_client.py`) oluşturuldu
- [x] Notebook entegrasyonu tamamlandı
- [x] Haber analizi sistemi hazır

---

## 🚀 Hızlı Başlangıç

### 1. LLM'i Test Et

```bash
python tools/llm_client.py
```

**Beklenen Çıktı:**
```
✅ LLM servisi çalışıyor!
Cevap: [Mistral'dan gelen yanıt]
```

### 2. Notebook'u Aç

```bash
jupyter notebook notebooks/strategy_research.ipynb
```

**Bölüm 15'e git** → LLM entegrasyonu

### 3. Haber Analizi Yap

```bash
# LLM ile detaylı analiz
python tools/ingest_news_llm.py \
  --feeds data/feeds.news.yml \
  --out data/macro_events.llm.json \
  --symbols "BTC/USDT,NVDA,GC=F" \
  --use-llm
```

---

## 📊 LLM Fonksiyonları

| Fonksiyon | Açıklama | Notebook Hücresi |
|-----------|----------|------------------|
| `ask()` | Genel soru-cevap | 15-D |
| `analyze_news()` | Haber sentiment analizi | - |
| `suggest_strategy()` | Strateji iyileştirme önerisi | 15-A |
| `optimize_parameters()` | Parametre optimizasyon analizi | 15-B |
| `explain_trade()` | İşlem açıklaması | 15-C |

---

## 🎯 Kullanım Örnekleri

### Strateji Analizi

```python
from tools.llm_client import LLMClient

llm = LLMClient()

# Backtest sonuçlarını analiz et
suggestion = llm.suggest_strategy(
    symbol="BTC/USDT",
    historical_performance={
        "sharpe_ratio": 0.8,
        "win_rate": 55.0
    },
    market_conditions={
        "volatility": "high",
        "trend": "bullish"
    }
)

print(suggestion)
```

### Haber Analizi

```python
analysis = llm.analyze_news(
    news_items=[{"title": "Fed faiz artırdı"}, ...],
    symbol="BTC/USDT"
)

print(f"Sentiment: {analysis['sentiment']}")
print(f"Bias: {analysis['bias_score']}")
```

### Serbest Soru

```python
answer = llm.ask("Volatilite yüksekken stop-loss nasıl ayarlanır?")
print(answer)
```

---

## 📁 Dosya Yapısı

```
algo_trading_lab/
├── tools/
│   ├── llm_client.py              # LLM API client
│   └── ingest_news_llm.py         # LLM destekli haber analizi
├── notebooks/
│   └── strategy_research.ipynb    # Bölüm 15: LLM entegrasyonu
├── data/
│   ├── feeds.news.yml             # RSS feed kaynakları
│   ├── macro_events.llm.json      # LLM analiz sonuçları
│   └── macro_events.basic.json    # VADER analiz sonuçları
└── docs/
    └── LLM_INTEGRATION.md         # Detaylı dokümantasyon
```

---

## ⚙️ Konfigürasyon

### Model Değiştirme

```python
# Mistral (default, önerilen)
llm = LLMClient(model="mistral")

# Daha küçük, daha hızlı
llm = LLMClient(model="phi4")
```

### Temperature Ayarı

```python
# Deterministik (classification, metrics)
llm.ask(prompt, temperature=0.3)

# Dengeli (genel kullanım)
llm.ask(prompt, temperature=0.7)

# Yaratıcı (brainstorming)
llm.ask(prompt, temperature=0.9)
```

---

## 🔄 Sonraki Adımlar

### Şimdi Yapabileceklerin

1. ✅ **Notebook'u çalıştır** → Backtest sonuçlarını LLM'e analiz ettir
2. ✅ **Haber analizi** → Günlük haberleri LLM ile değerlendir
3. ✅ **Strateji geliştirme** → LLM ile parametre optimizasyonu

### Gelecekte (3-6 ay sonra)

4. 🔜 **Veri topla** → Trade log'larını kaydet (500-1000 işlem)
5. 🔜 **Fine-tune** → Kendi verilerinle modeli özelleştir (LoRA)
6. 🔜 **Deploy** → Kişiselleştirilmiş stratejist modeli

---

## ⚠️ Önemli Hatırlatmalar

### ✅ LLM Yapabilir:

- Fikir üretme
- Kod yazma
- Analiz yapma
- Açıklama getirme

### ❌ LLM Yapamaz:

- Gerçek zamanlı alım-satım kararı
- Gerçek piyasa verisi üretme
- Garantili kazanç sağlama

### 🛡️ Güvenlik:

- Her zaman backtest/forward test
- Paper trading ile doğrula
- Risk limitleri kod seviyesinde
- LLM sadece danışman, karar senin

---

## 🆘 Sorun mu var?

### LLM yanıt vermiyor?

```bash
brew services restart ollama
```

### Model bulunamadı?

```bash
ollama list
ollama pull mistral
```

### Hata alıyorum

Detaylı dokümantasyona bak:
```bash
cat docs/LLM_INTEGRATION.md
```

---

## 📚 Kaynaklar

- **Detaylı Docs**: `docs/LLM_INTEGRATION.md`
- **LLM Client**: `tools/llm_client.py`
- **Notebook**: `notebooks/strategy_research.ipynb`
- **Architecture**: `ARCHITECTURE.md`

---

**🎉 Hazırsın! LLM destekli trading stratejisi geliştirmeye başla!**

```bash
# İlk adımın:
jupyter notebook notebooks/strategy_research.ipynb
```
