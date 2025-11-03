"""
LLM Client for Strategy Development and News Analysis
Connects to local Ollama instance for trading strategy assistance
"""
import requests
import json
from typing import Optional, Dict, Any, Lis
from datetime import datetime


class LLMClient:
    """Client for interacting with local LLM (Ollama)"""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "mistral"):
        self.base_url = base_url
        self.model = model
        self.timeout = 120  # 2 dakika timeou

    def ask(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.7) -> str:
        """
        LLM'e soru sor ve cevap al

        Args:
            prompt: Kullanıcı sorusu
            system_prompt: Sistem rolü (opsiyonel)
            temperature: Yaratıcılık seviyesi (0.0-1.0)

        Returns:
            LLM'in cevabı
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }

        if system_prompt:
            payload["system"] = system_promp

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeou
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()

        except requests.exceptions.RequestException as e:
            return f"❌ LLM bağlantı hatası: {e}"

    def analyze_news(self, news_items: List[Dict[str, Any]], symbol: str) -> Dict[str, Any]:
        """
        Haberleri analiz et ve sentiment + impact döndür

        Args:
            news_items: Haber listesi (title, summary vb.)
            symbol: Sembol (BTC/USDT, NVDA, vb.)

        Returns:
            {
                "sentiment": "bullish" | "bearish" | "neutral",
                "impact": "low" | "medium" | "high" | "critical",
                "bias_score": -1.0 to 1.0,
                "confidence": 0.0 to 1.0,
                "summary": "Özet açıklama",
                "catalysts": ["katalizör 1", "katalizör 2", ...]
            }
        """
        # Haberleri metne dönüştür
        news_text = "\n".join([
            f"- {item.get('title', '')} ({item.get('published', 'N/A')})"
            for item in news_items[-10:]  # Son 10 haber
        ])

        system_prompt = """Sen bir finansal analist ve trading uzmanısın.
Haberleri analiz edip bir varlığın fiyatına muhtemel etkisini değerlendiriyorsun.
Cevabını JSON formatında ver."""

        prompt = f"""
Aşağıdaki haberler {symbol} sembolü ile ilgili:

{news_text}

Lütfen bu haberleri analiz et ve şu JSON formatında cevap ver:

{{
  "sentiment": "bullish veya bearish veya neutral",
  "impact": "low veya medium veya high veya critical",
  "bias_score": -1.0 ile 1.0 arası sayı (negatif=bearish, pozitif=bullish),
  "confidence": 0.0 ile 1.0 arası (analiz güvenirliği),
  "summary": "Kısa özet (max 2 cümle)",
  "catalysts": ["ana katalizör 1", "ana katalizör 2"]
}}

SADECE JSON döndür, başka açıklama ekleme.
"""

        response = self.ask(prompt, system_prompt=system_prompt, temperature=0.3)

        # JSON parse e
        try:
            # JSON'u çıkar (eğer markdown code block içindeyse)
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            result = json.loads(response)
            return resul
        except json.JSONDecodeError:
            # Parse edilemezse default değerler
            return {
                "sentiment": "neutral",
                "impact": "low",
                "bias_score": 0.0,
                "confidence": 0.3,
                "summary": "LLM analizi parse edilemedi",
                "catalysts": []
            }

    def suggest_strategy(self,
                        symbol: str,
                        historical_performance: Dict[str, Any],
                        market_conditions: Dict[str, Any]) -> str:
        """
        Mevcut performans ve piyasa koşullarına göre strateji önerisi

        Args:
            symbol: Sembol
            historical_performance: Backtest sonuçları
            market_conditions: Piyasa durumu (volatilite, trend, vb.)

        Returns:
            Strateji önerisi metni
        """
        system_prompt = """Sen bir algoritmik trading stratejisti ve quantitative analistisin.
Backtest sonuçlarını ve piyasa koşullarını analiz edip strateji iyileştirmeleri öneriyorsun."""

        prompt = f"""
Sembol: {symbol}

Mevcut Performans:
- Sharpe Ratio: {historical_performance.get('sharpe_ratio', 'N/A')}
- Win Rate: {historical_performance.get('win_rate', 'N/A')}%
- Max Drawdown: {historical_performance.get('max_drawdown_pct', 'N/A')}%
- Total Return: {historical_performance.get('total_pnl_pct', 'N/A')}%
- Total Trades: {historical_performance.get('total_trades', 'N/A')}

Piyasa Koşulları:
- Volatilite: {market_conditions.get('volatility', 'N/A')}
- Trend: {market_conditions.get('trend', 'N/A')}
- RSI: {market_conditions.get('rsi', 'N/A')}

Bu performansı iyileştirmek için:
1. Hangi parametreleri ayarlamalıyım?
2. Hangi ek göstergeleri eklemeliyim?
3. Risk yönetimi nasıl optimize edilir?
4. Bu piyasa koşullarında hangi strateji tipi daha uygun? (trend-following, mean-reversion, vb.)

Lütfen somut, uygulanabilir öneriler ver.
"""

        return self.ask(prompt, system_prompt=system_prompt, temperature=0.7)

    def optimize_parameters(self,
                           symbol: str,
                           current_params: Dict[str, Any],
                           performance_history: List[Dict[str, Any]]) -> str:
        """
        Parametre optimizasyonu önerisi

        Args:
            symbol: Sembol
            current_params: Mevcut parametreler
            performance_history: Farklı parametre kombinasyonlarının performansı

        Returns:
            Optimizasyon önerisi
        """
        system_prompt = """Sen bir parametre optimizasyon uzmanısın.
Grid search veya Bayesian optimization sonuçlarını yorumlayıp en iyi yaklaşımı öneriyorsun."""

        # En iyi 5 kombinasyonu al
        top_5 = sorted(performance_history, key=lambda x: x.get('sharpe_ratio', 0), reverse=True)[:5]

        prompt = f"""
Sembol: {symbol}

Mevcut Parametreler:
{json.dumps(current_params, indent=2)}

En İyi 5 Kombinasyon:
{json.dumps(top_5, indent=2)}

Bu sonuçlara bakarak:
1. Hangi parametreler performansı en çok etkiliyor?
2. Parametreler arasında nasıl bir ilişki var?
3. Overfitting riski var mı?
4. Önerilen yeni parametre aralıkları neler?

Somut sayısal öneriler ver.
"""

        return self.ask(prompt, system_prompt=system_prompt, temperature=0.5)

    def explain_trade(self,
                     trade_data: Dict[str, Any],
                     market_context: Dict[str, Any]) -> str:
        """
        Bir işlemin neden açıldığını/kapatıldığını açıkla

        Args:
            trade_data: İşlem detayları
            market_context: Piyasa durumu

        Returns:
            Açıklama metni
        """
        system_prompt = """Sen bir trading educator'ısın.
İşlemleri açık ve anlaşılır şekilde açıklıyorsun."""

        prompt = f"""
İşlem Detayları:
- Side: {trade_data.get('side', 'N/A')}
- Entry: ${trade_data.get('entry_price', 'N/A')}
- Exit: ${trade_data.get('exit_price', 'N/A')}
- P&L: {trade_data.get('pnl_pct', 'N/A')}%
- Exit Reason: {trade_data.get('exit_reason', 'N/A')}

Piyasa Durumu:
- EMA Fast: {market_context.get('ema_fast', 'N/A')}
- EMA Slow: {market_context.get('ema_slow', 'N/A')}
- RSI: {market_context.get('rsi', 'N/A')}
- Fiyat: ${market_context.get('price', 'N/A')}

Bu işlemi neden açtık ve neden bu şekilde kapandı?
Teknik analizle açıkla (2-3 cümle).
"""

        return self.ask(prompt, system_prompt=system_prompt, temperature=0.5)

    def health_check(self) -> bool:
        """LLM servisinin çalışıp çalışmadığını kontrol et"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False


# Global instance
_llm_client = None

def get_llm_client(model: str = "mistral") -> LLMClient:
    """Singleton LLM client al"""
    global _llm_clien
    if _llm_client is None:
        _llm_client = LLMClient(model=model)
    return _llm_clien


if __name__ == "__main__":
    # Tes
    client = LLMClient()

    print("🔍 LLM Health Check...")
    if client.health_check():
        print("✅ LLM servisi çalışıyor!")
    else:
        print("❌ LLM servisi yanıt vermiyor. 'ollama serve' çalıştırın.")
        exit(1)

    print("\n🤖 Test Sorusu...")
    response = client.ask("Bitcoin için EMA crossover stratejisi ne zaman long pozisyon açar? Kısa açıkla.")
    print(f"Cevap: {response}")

    print("\n📰 Haber Analizi Test...")
    test_news = [
        {"title": "Fed faiz artırımına devam edeceğini açıkladı", "published": "2025-11-01"},
        {"title": "Bitcoin ETF onayları yaklaşıyor", "published": "2025-11-01"},
    ]
    analysis = client.analyze_news(test_news, "BTC/USDT")
    print(f"Analiz: {json.dumps(analysis, indent=2, ensure_ascii=False)}")
