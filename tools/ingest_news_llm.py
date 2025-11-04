"""
LLM-Enhanced News Ingestion Tool
Haberleri RSS'ten çeker, LLM ile analiz eder ve macro events dosyasına yazar
"""
import argparse
import json
from datetime import datetime
from typing import Any, Dict, List

import feedparser
import yaml

from llm_client import LLMClient


def load_feeds(feeds_file: str) -> List[str]:
    """RSS feed URL'lerini yükle"""
    with open(feeds_file, 'r') as f:
        config = yaml.safe_load(f)
    return config.get('feeds', [])


def fetch_news(feed_urls: List[str], limit: int = 50) -> List[Dict[str, Any]]:
    """RSS feedlerinden haberleri çek"""
    all_entries = []

    for url in feed_urls:
        print(f"📡 Fetching {url}...")
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:limit]:
                all_entries.append({
                    'title': entry.get('title', ''),
                    'summary': entry.get('summary', entry.get('description', '')),
                    'published': entry.get('published', datetime.now().isoformat()),
                    'link': entry.get('link', ''),
                    'source': url
                })
        except Exception as e:
            print(f"  ⚠️  Hata: {e}")

    print(f"✅ Toplam {len(all_entries)} haber çekildi")
    return all_entries


def analyze_with_llm(
    news_items: List[Dict[str, Any]],
    symbols: List[str],
    llm: LLMClient,
) -> List[Dict[str, Any]]:
    """
    Her sembol için haberleri LLM ile analiz et
    """
    macro_events = []

    for symbol in symbols:
        print(f"\n🤖 {symbol} için LLM analizi...")

        # LLM'e analiz yaptır
        analysis = llm.analyze_news(news_items, symbol)

        # MacroEvent formatına çevir
        event = {
            "title": f"Market Analysis for {symbol}",
            "timestamp": datetime.now().isoformat(),
            "category": "macro",
            "sentiment": analysis.get("sentiment", "neutral"),
            "impact": analysis.get("impact", "medium"),
            "bias": analysis.get("bias_score", 0.0),
            "actor": "LLM Analysis",
            "assets": [symbol],
            "interest_rate_expectation": "neutral",
            "summary": analysis.get("summary", ""),
            "catalysts": analysis.get("catalysts", []),
            "confidence": analysis.get("confidence", 0.5),
            "source": "LLM-Enhanced News Analysis"
        }

        macro_events.append(event)

        # Sonuçları göster
        print(f"  📊 Sentiment: {event['sentiment']}")
        print(f"  ⚡ Impact: {event['impact']}")
        print(f"  📈 Bias Score: {event['bias']:.2f}")
        print(f"  💯 Confidence: {event['confidence']:.2f}")
        print(f"  📝 Summary: {event['summary'][:100]}...")

    return macro_events


def classify_news_basic(
    news_items: List[Dict[str, Any]],
    symbols: List[str],
) -> List[Dict[str, Any]]:
    """
    Basit keyword-based sınıflandırma (LLM olmadan fallback)
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()

    macro_events = []

    for item in news_items:
        title = item['title']

        # VADER sentiment
        scores = analyzer.polarity_scores(title)
        compound = scores['compound']

        # Sentiment mapping
        if compound > 0.3:
            sentiment = "bullish"
            bias = 0.6
        elif compound < -0.3:
            sentiment = "bearish"
            bias = -0.6
        else:
            sentiment = "neutral"
            bias = 0.0

        # Impact (basit keyword matching)
        impact_keywords = {
            'critical': ['crash', 'collapse', 'crisis', 'emergency'],
            'high': ['fed', 'rate', 'inflation', 'gdp', 'unemployment'],
            'medium': ['earnings', 'report', 'forecast', 'outlook'],
        }

        impact = "low"
        for level, keywords in impact_keywords.items():
            if any(kw in title.lower() for kw in keywords):
                impact = level
                break

        # Affected assets (basit matching)
        affected = []
        for symbol in symbols:
            symbol_keywords = symbol.replace('/', ' ').replace('-', ' ').split()
            if any(kw.lower() in title.lower() for kw in symbol_keywords):
                affected.append(symbol)

        if not affected:
            affected = symbols  # Tüm varlıkları etkiler

        event = {
            "title": title,
            "timestamp": item.get('published', datetime.now().isoformat()),
            "category": "news",
            "sentiment": sentiment,
            "impact": impact,
            "bias": bias,
            "actor": "News",
            "assets": affected,
            "interest_rate_expectation": "neutral",
            "summary": item.get('summary', title)[:200],
            "link": item.get('link', ''),
            "source": item.get('source', 'Unknown')
        }

        macro_events.append(event)

    return macro_events


def main():
    parser = argparse.ArgumentParser(description='LLM-Enhanced News Ingestion')
    parser.add_argument('--feeds', required=True, help='YAML file with RSS feeds')
    parser.add_argument('--out', required=True, help='Output JSON file')
    parser.add_argument('--symbols', required=True, help='Comma-separated symbols (e.g., BTC/USDT,NVDA,GC=F)')
    parser.add_argument('--use-llm', action='store_true', help='Use LLM for analysis (default: basic VADER)')
    parser.add_argument('--limit', type=int, default=50, help='Max news per feed')

    args = parser.parse_args()

    # Sembolleri parse et
    symbols = [s.strip() for s in args.symbols.split(',')]

    print("="*60)
    print("📰 LLM-ENHANCED NEWS INGESTION")
    print("="*60)
    print(f"📋 Symbols: {', '.join(symbols)}")
    print(f"🤖 LLM Analysis: {'Enabled' if args.use_llm else 'Disabled (VADER only)'}")
    print("="*60)

    # RSS feedlerini yükle
    feed_urls = load_feeds(args.feeds)
    print(f"\n📡 {len(feed_urls)} RSS feed yüklendi")

    # Haberleri çek
    news_items = fetch_news(feed_urls, limit=args.limit)

    if not news_items:
        print("❌ Hiç haber çekilemedi!")
        return

    # Analiz yap
    if args.use_llm:
        # LLM ile analiz
        llm = LLMClient()

        if not llm.health_check():
            print("\n⚠️  LLM servisi çalışmıyor!")
            print("💡 'ollama serve' komutu ile başlatın veya --use-llm olmadan çalıştırın")
            print("🔄 VADER ile devam ediliyor...\n")
            macro_events = classify_news_basic(news_items, symbols)
        else:
            print("\n✅ LLM servisi aktif, analiz başlıyor...\n")
            macro_events = analyze_with_llm(news_items, symbols, llm)
    else:
        # Basit VADER analizi
        print("\n📊 VADER sentiment analizi yapılıyor...\n")
        macro_events = classify_news_basic(news_items, symbols)

    # JSON olarak kaydet
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(macro_events, f, indent=2, ensure_ascii=False)

    print("\n" + "="*60)
    print(f"✅ {len(macro_events)} event yazıldı: {args.out}")
    print("="*60)

    # Özet istatistikler
    sentiments = {}
    impacts = {}
    for event in macro_events:
        sent = event['sentiment']
        imp = event['impact']
        sentiments[sent] = sentiments.get(sent, 0) + 1
        impacts[imp] = impacts.get(imp, 0) + 1

    print("\n📊 Sentiment Dağılımı:")
    for sent, count in sentiments.items():
        print(f"  {sent:10s}: {count}")

    print("\n⚡ Impact Dağılımı:")
    for imp, count in impacts.items():
        print(f"  {imp:10s}: {count}")


if __name__ == '__main__':
    main()
