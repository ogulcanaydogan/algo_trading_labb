"""Test live sentiment data fetching."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.sentiment import NewsSentimentFetcher, get_combined_features

print("=" * 60)
print("LIVE SENTIMENT DATA TEST")
print("=" * 60)
print()

symbols = ["TSLA", "NVDA", "AAPL"]

for symbol in symbols:
    print(f"--- {symbol} ---")
    try:
        features = get_combined_features(symbol)
        
        score_24h = features.get("sentiment_score_24h", 0)
        social = features.get("social_sentiment", 0)
        combined = features.get("sentiment_combined", 0)
        volume = features.get("mention_volume", 0)
        
        print(f"  Sentiment Score (24h): {score_24h:+.3f}")
        print(f"  Social Sentiment:      {social:+.3f}")
        print(f"  Combined Score:        {combined:+.3f}")
        print(f"  Mention Volume:        {volume:.2f}x")
        
        # Interpret
        if combined > 0.3:
            signal = "BULLISH"
        elif combined < -0.3:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"
        print(f"  Signal: {signal}")
        
    except Exception as e:
        print(f"  Error: {e}")
    
    print()

# Show sample headlines
print("=" * 60)
print("SAMPLE HEADLINES")
print("=" * 60)
print()

fetcher = NewsSentimentFetcher()
items = fetcher.fetch_news("TSLA", hours_back=24)

print(f"Found {len(items)} news items for TSLA\n")
for item in items[:5]:
    if item.sentiment_score > 0.1:
        emoji = "[+]"
    elif item.sentiment_score < -0.1:
        emoji = "[-]"
    else:
        emoji = "[=]"
    
    score_str = f"{item.sentiment_score:+.2f}"
    title = item.title[:55] + "..." if len(item.title) > 55 else item.title
    print(f"{emoji} {score_str} {title}")
    print(f"    Source: {item.source}")
    print()
