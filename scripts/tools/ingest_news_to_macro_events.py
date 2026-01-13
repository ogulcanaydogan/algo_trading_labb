from __future__ import annotations
"""
Ingest RSS headlines and convert them into macro events usable by MacroSentimentEngine.

Usage:
  python tools/ingest_news_to_macro_events.py \
    --feeds feeds.sample.yml \
    --out data/macro_events.news.json \
    --symbols BTC/USDT,ETH/USDT,AAPL

Notes:
- Requires `feedparser` and `vaderSentiment` (installed via requirements.txt)
- Designed to run periodically via cron or a background process.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import feedparser  # type: ignore
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
import yaml  # type: ignore


def load_feeds(path: Path) -> List[str]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    feeds = doc.get("feeds") or []
    return [str(u) for u in feeds if u]


def classify_category(title: str) -> str:
    lower = title.lower()
    if any(k in lower for k in ["fed", "ecb", "rate", "inflation", "cpi", "jobs"]):
        return "macro"
    if any(k in lower for k in ["war", "election", "tariff", "sanction", "geopolit"]):
        return "politics"
    if any(k in lower for k in ["bitcoin", "crypto", "binance", "coinbase", "ethereum"]):
        return "crypto"
    return "general"


def infer_assets(title: str, symbols: Iterable[str]) -> Dict[str, float]:
    lower = title.lower()
    mapping: Dict[str, float] = {}
    for s in symbols:
        key = s.lower()
        if any(k in lower for k in key.split("/")):
            mapping[s] = 0.1
    if not mapping:
        mapping["*"] = 0.0
    return mapping


def run(feeds_file: Path, out_path: Path, symbols: List[str]) -> None:
    urls = load_feeds(feeds_file)
    analyzer = SentimentIntensityAnalyzer()
    events = []
    for url in urls:
        parsed = feedparser.parse(url)
        for entry in parsed.entries[:50]:  # limit per feed
            title = str(entry.get("title") or "").strip()
            if not title:
                continue
            published = entry.get("published") or entry.get("updated")
            timestamp = None
            if published:
                try:
                    timestamp = datetime(*entry.published_parsed[:6]).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    )  # type: ignore[attr-defined]
                except Exception:
                    timestamp = None
            score = analyzer.polarity_scores(title)["compound"]
            sentiment = "positive" if score > 0.15 else "negative" if score < -0.15 else "neutral"
            impact = "high" if abs(score) > 0.6 else "medium" if abs(score) > 0.3 else "low"
            events.append(
                {
                    "title": title,
                    "category": classify_category(title),
                    "sentiment": sentiment,
                    "impact": impact,
                    "timestamp": timestamp,
                    "source": parsed.feed.get("title") if hasattr(parsed, "feed") else None,
                    "assets": infer_assets(title, symbols),
                }
            )
    out_path.write_text(json.dumps(events, indent=2), encoding="utf-8")
    print(f"Wrote {len(events)} events -> {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert RSS feeds into macro events JSON.")
    p.add_argument("--feeds", default="data/feeds.sample.yml", help="YAML file listing RSS feed URLs")
    p.add_argument("--out", default="data/macro_events.news.json", help="Output JSON file path")
    p.add_argument("--symbols", default="BTC/USDT,ETH/USDT,AAPL", help="Comma-separated list for asset inference")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(Path(args.feeds), Path(args.out), [s.strip() for s in args.symbols.split(",") if s.strip()])
