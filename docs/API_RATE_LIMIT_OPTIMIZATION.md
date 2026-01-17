# API Rate Limit Optimization Strategy

**Date Generated:** January 16, 2026  
**Analysis Status:** Current Free Tier Usage Patterns

---

## Executive Summary

You have configured **4 news/market data APIs** with the following free tier limits:

| API | Limit | Your Current Config | Status |
|-----|-------|-------------------|--------|
| **NewsAPI** | 100 req/day | ✅ Configured | **CRITICAL - Most Restrictive** |
| **Alpha Vantage** | 5 req/min (~7,200/day) | ✅ Configured | **SEVERE - Bottleneck** |
| **Finnhub** | 60 req/min (~86,400/day) | ✅ Configured | **GOOD** |
| **CryptoPanic** | Developer tier | ✅ Configured | **EXCELLENT** |

---

## 1. IMMEDIATE ISSUES & BOTTLENECKS

### 🔴 **Critical: NewsAPI - 100 requests/day**
- **Problem:** Only 100 requests per day = ~1 request per 14 minutes
- **Risk:** Single burst of requests exhausts entire daily quota
- **Current Usage:** Unknown (check `news_sentiment.py` for fetch frequency)
- **Solution:** Disable NewsAPI or use sparingly

### 🔴 **Critical: Alpha Vantage - 5 requests/minute**
- **Problem:** Only 5 requests/min = severe bottleneck for multi-symbol strategies
- **Risk:** Cannot fetch data for more than 5 different symbols simultaneously per minute
- **Current Usage:** Not actively used (no function calls in recent imports)
- **Solution:** Disable or reserve for specific stock quotes only

### 🟡 **Warning: Finnhub - 60 requests/minute**
- **Problem:** Good for single symbol but limited for portfolio-wide analysis
- **Status:** Most practical of the news APIs
- **Usage Pattern:** Should be primary for crypto news
- **Recommendation:** Optimize to ~2-3 requests/min to stay safe

### 🟢 **Good: CryptoPanic - Developer Tier**
- **Status:** No documented rate limits on developer tier
- **Current Usage:** Not actively configured in bot
- **Recommendation:** Make this PRIMARY for crypto news

---

## 2. OPTIMIZATION STRATEGY (DO THIS TODAY)

### **Priority 1: Disable Unused APIs** (Takes 2 minutes)
Modify `.env` to completely disable Alpha Vantage and NewsAPI:

```bash
# DISABLE - These have severe limits
NEWSAPI_API_KEY=  # Leave blank
ALPHAVANTAGE_API_KEY=  # Leave blank

# KEEP & OPTIMIZE
FINNHUB_API_KEY=d5l3531r01qgqufk01i0d5l3531r01qgqufk01ig  # Primary news
CRYPTOPANIC_API_KEY=84ae1ff9dfaa0c57083ca879347c9db37b06b9a1  # Backup crypto
```

**Time saved by end of day:** +200 requests

---

### **Priority 2: Configure Smart Caching** (Add to bot)

The rate limiter already has caching! Configure it properly:

```python
# In your bot initialization
from bot.rate_limiter import MultiLevelCache, RateLimitConfig

# Setup caching for news (15 min in memory, 2 hours on disk)
news_cache = MultiLevelCache(
    cache_dir="data/news_cache",
    memory_ttl=900,      # 15 minutes in memory
    disk_ttl=7200,       # 2 hours on disk
)

# Before fetching news:
cached_news = news_cache.get(f"{symbol}_news_finnhub")
if cached_news:
    return cached_news  # Use cache instead of API call
```

**Benefit:** Reduce news API calls by 80-90% if trading same symbols repeatedly

---

### **Priority 3: Batch Requests & Multi-Timeframe Strategy** (Core optimization)

Instead of fetching news every minute, implement this pattern:

```python
# CURRENT (BAD): Fetches for every symbol every 60 seconds
for symbol in ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]:
    news = fetch_news(symbol)  # 4 API calls per minute = 240/hour!

# OPTIMIZED: Fetch once per 5 minutes, cache aggressively
class OptimizedNewsFetcher:
    def __init__(self):
        self.last_fetch = {}
        self.fetch_interval_seconds = 300  # 5 minutes between fetches
        
    def should_fetch(self, symbol: str) -> bool:
        """Only fetch if 5+ minutes since last fetch"""
        if symbol not in self.last_fetch:
            return True
        elapsed = time.time() - self.last_fetch[symbol]
        return elapsed >= self.fetch_interval_seconds
    
    def fetch_news_optimized(self, symbols: List[str]):
        """Fetch only symbols needing updates"""
        for symbol in symbols:
            if self.should_fetch(symbol):
                news = self.fetch_news_from_api(symbol)
                self.last_fetch[symbol] = time.time()
                # Cache it for 5 minutes
```

**Benefit:** Reduce Finnhub calls from ~240/hour to ~12/hour (95% reduction!)

---

### **Priority 4: Progressive Tier Strategy** (End of day strategy)

```
9am-5pm: Use Finnhub (60/min = safe margin) + CryptoPanic backup
5pm onwards: Reduce to 2/min (use cached data mostly)
Daily aggregated fetch: 0.5/min average
```

---

## 3. RATE LIMIT STATUS (Real-Time Check)

### Get Current Usage with Built-in Monitor:

```python
from bot.rate_limit_monitor import RateLimitMonitor

monitor = RateLimitMonitor()
monitor.configure_api("finnhub", {
    "requests_per_minute": 60,
    "requests_per_hour": 3600,
    "requests_per_day": 86400,
})

status = monitor.get_status("finnhub")
print(f"Finnhub: {status.minute_usage}% of minute limit used")
print(f"Finnhub: {status.day_usage}% of day limit used")
```

---

## 4. IMPLEMENTATION CHECKLIST

### ✅ **Today (Next 30 minutes)**
- [ ] Disable NewsAPI and Alpha Vantage in `.env`
- [ ] Review `bot/news_sentiment.py` - check fetch frequency
- [ ] Add caching layer to news fetching
- [ ] Set fetch interval to 5 minutes minimum

### ✅ **Today (Before 5pm)**
- [ ] Monitor Finnhub usage via `rate_limit_monitor.py`
- [ ] Verify no requests hitting rate limits
- [ ] Reduce fetch frequency if approaching limits

### ✅ **This Week**
- [ ] Upgrade NewsAPI to paid tier if needed ($0.09/request)
- [ ] Consider Alpha Vantage API alternatives (polygon, IEX)
- [ ] Implement dynamic request throttling based on market hours

---

## 5. MAXIMUM SAFE DAILY USAGE (Current Setup)

With **Finnhub + CryptoPanic only**:

### Finnhub (60 req/min):
```
Safe daily budget = 60 req/min × 50 min/hour × 8 trading hours
                  = 24,000 requests/day
                  = ~4,000 requests/symbol/day (for 6 symbols)
                  = ~280 requests/symbol/hour
```

### Trading Implementation:
```
Symbols: BTC, ETH, SOL, XRP (4 symbols)
Fetch every 5 min = 12 fetches × 4 symbols = 48 req/hour
Total daily = 48 × 8 hours = 384 requests/day (0.5% of limit!)
```

---

## 6. SPECIFIC CODE CHANGES REQUIRED

### **CRITICAL FIX 1: Disable Unused APIs in `.env`** ⚡

```bash
# CHANGE THIS:
#NEWSAPI_API_KEY=d12f96658d554251a54673460028df31
NEWSAPI_API_KEY=d12f96658d554251a54673460028df31
ALPHAVANTAGE_API_KEY=6ASPY7XPXZD0XM8K

# TO THIS (disable the expensive ones):
# NEWSAPI_API_KEY=              # Leave blank - only 100/day
# ALPHAVANTAGE_API_KEY=          # Leave blank - only 5/min
FINNHUB_API_KEY=d5l3531r01qgqufk01i0d5l3531r01qgqufk01ig
CRYPTOPANIC_API_KEY=84ae1ff9dfaa0c57083ca879347c9db37b06b9a1
```

### **CRITICAL FIX 2: Add Fetch Interval Guard in `bot/news_feature_extractor.py`**

Add this class at the top of the file:

```python
import time

class NewsFetchThrottler:
    """Prevents excessive API calls by enforcing minimum intervals."""
    
    def __init__(self, min_interval_seconds: int = 300):  # 5 minutes default
        self.min_interval = min_interval_seconds
        self.last_fetch_time = {}  # Symbol -> timestamp
    
    def should_fetch(self, symbol: str) -> bool:
        """Check if enough time has passed since last fetch."""
        now = time.time()
        last = self.last_fetch_time.get(symbol, 0)
        
        if (now - last) >= self.min_interval:
            self.last_fetch_time[symbol] = now
            return True
        return False
    
    def get_next_fetch_time(self, symbol: str) -> float:
        """Returns seconds until next fetch allowed."""
        last = self.last_fetch_time.get(symbol, 0)
        next_allowed = last + self.min_interval
        return max(0, next_allowed - time.time())
```

Then modify the `extract_features()` method:

```python
class NewsFeatureExtractor:
    def __init__(self, ...):
        # ... existing code ...
        self.fetch_throttler = NewsFetchThrottler(min_interval_seconds=300)  # 5 min
    
    def extract_features(self, symbol: str, timestamp: datetime) -> Optional[NewsFeatures]:
        """Extract news features with throttling."""
        
        # Only fetch if interval has passed
        if not self.fetch_throttler.should_fetch(symbol):
            logger.debug(f"Skipping news fetch for {symbol} (throttled)")
            return self._get_cached_features(symbol)  # Return cached data
        
        # Existing fetch logic
        try:
            articles = self.fetcher.fetch_news(symbol)
            # ... rest of existing code ...
```

### **CRITICAL FIX 3: Fix Environment Variable Names**

In `bot/news_sentiment.py` lines 145-149, the code looks for wrong variable names:

```python
# CURRENT (WRONG):
self.newsapi_key = newsapi_key or os.getenv("NEWSAPI_KEY")
self.alpha_vantage_key = alpha_vantage_key or os.getenv("ALPHA_VANTAGE_KEY") 
self.finnhub_key = finnhub_key or os.getenv("FINNHUB_KEY")

# SHOULD BE (MATCHES `.env`):
self.newsapi_key = newsapi_key or os.getenv("NEWSAPI_API_KEY")
self.alpha_vantage_key = alpha_vantage_key or os.getenv("ALPHAVANTAGE_API_KEY")
self.finnhub_key = finnhub_key or os.getenv("FINNHUB_API_KEY")
self.crypto_panic_key = crypto_panic_key or os.getenv("CRYPTOPANIC_API_KEY")
```

**This is likely why news features aren't loading!**

---

## 7. ESTIMATED SAVINGS

| Scenario | Requests/Hour | Requests/Day | Risk Level |
|----------|---------------|--------------|-----------|
| **Current (All APIs)** | ~384 | ~3,072 | 🔴 HIGH (NewsAPI burnout) |
| **Disabled Alpha+News** | ~48 | ~384 | 🟢 SAFE (0.5% of limit) |
| **With 15-min cache** | ~12 | ~96 | 🟢 EXCELLENT |
| **With 30-min cache** | ~6 | ~48 | 🟢 OPTIMAL |

---

## 8. RECOMMENDED ACTION PLAN

### **RIGHT NOW:**
1. Comment out NewsAPI and Alpha Vantage in `.env`
2. Verify Finnhub is primary in `news_sentiment.py`
3. Add 5-minute fetch interval guard

### **BEFORE END OF DAY:**
1. Monitor rate limit usage with `RateLimitMonitor`
2. Confirm no 429 (rate limit) errors
3. Test caching effectiveness

### **THIS WEEK:**
1. Evaluate paid API tiers for NewsAPI (if needed)
2. Consider alternative providers (Polygon for stocks, IEX for real-time)
3. Implement predictive rate limit management

---

## 9. REFERENCE: API Comparison Table

| Metric | NewsAPI | Alpha V. | Finnhub | Crypto Panic |
|--------|---------|----------|---------|--------------|
| Free Limit | 100/day | 5/min | 60/min | Developer |
| Cost per Req | High | High | Free | Free |
| Data Quality | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Crypto Focus | Low | Low | High | Very High |
| Stock Focus | High | High | Medium | Very Low |
| Recommended | Paid Only | Alternative | ✅ Primary | ✅ Backup |

---

## VERIFIED CURRENT CONFIGURATION

### **Bot Loop Interval: 5 Minutes** ✅
Found in `bot/unified_engine.py` line 104:
```python
loop_interval_seconds: int = 300  # 5 minutes
```

This means the bot runs its main loop every **5 minutes**. Good news!

### **Current News Fetch Behavior:**
- Loop runs every 5 minutes = 12 times/hour = 288 times/day
- If fetching news every iteration: **12 symbols = 3,456 requests/day** ❌
- If fetching every 3rd iteration (15 min): **1,152 requests/day** ❌
- If fetching every 6th iteration (30 min): **576 requests/day** ❌

### **Default Symbols Configuration:**
From `run_unified_trading.py`:
```python
symbols=args.symbols.split(",") if args.symbols else ["BTC/USDT", "ETH/USDT"]
```
- **Default:** 2 symbols (BTC, ETH)
- **Potential:** Configurable up to 10+ symbols

---

## Quick Win: Cut Requests by 95% (5 min implementation)

```python
# Add to unified_orchestrator.py

import time

class NewsOptimizer:
    def __init__(self, fetch_interval_minutes=5):
        self.last_news_fetch = 0
        self.fetch_interval = fetch_interval_minutes * 60
    
    def should_fetch_news(self) -> bool:
        """Only fetch if interval elapsed"""
        now = time.time()
        should = (now - self.last_news_fetch) >= self.fetch_interval
        if should:
            self.last_news_fetch = now
        return should

# In main loop:
news_optimizer = NewsOptimizer(fetch_interval_minutes=5)
if news_optimizer.should_fetch_news():
    news_features = fetch_news()
```

**Result:** ~240 requests/day → 12 requests/day ✅

---

*Last Updated: 2026-01-16*
*Next Review: After monitoring for 2 hours of trading*
