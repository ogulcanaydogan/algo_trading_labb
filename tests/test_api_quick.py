#!/usr/bin/env python3
"""Quick smoke test for API endpoints"""
import sys
import subprocess
import time
import requests

def main():
    print("🚀 Starting API server...")
    # Start uvicorn in background
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.api:app", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to start
    time.sleep(3)
    
    try:
        print("\n✅ Testing endpoints...")
        
        # Test /status
        print("\n📊 GET /status")
        resp = requests.get("http://127.0.0.1:8000/status", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            print(f"   Symbol: {data.get('symbol')}")
            print(f"   Balance: ${data.get('balance'):,.2f}")
            print(f"   Position: {data.get('position')}")
            print(f"   Last Signal: {data.get('last_signal')}")
            print(f"   AI Action: {data.get('ai_action')}")
        else:
            print(f"   ❌ Failed: {resp.status_code}")
            
        # Test /signals
        print("\n📈 GET /signals")
        resp = requests.get("http://127.0.0.1:8000/signals?limit=3", timeout=5)
        if resp.status_code == 200:
            signals = resp.json()
            print(f"   Found {len(signals)} signals")
            for sig in signals[:3]:
                print(f"   - {sig.get('decision')} @ {sig.get('timestamp')}")
        else:
            print(f"   ❌ Failed: {resp.status_code}")
            
        # Test /strategy
        print("\n⚙️  GET /strategy")
        resp = requests.get("http://127.0.0.1:8000/strategy", timeout=5)
        if resp.status_code == 200:
            strat = resp.json()
            print(f"   Symbol: {strat.get('symbol')}")
            print(f"   Timeframe: {strat.get('timeframe')}")
            print(f"   EMA: {strat.get('ema_fast')}/{strat.get('ema_slow')}")
            print(f"   RSI: {strat.get('rsi_period')}")
        else:
            print(f"   ❌ Failed: {resp.status_code}")
            
        # Test dashboard
        print("\n🎨 GET /dashboard")
        resp = requests.get("http://127.0.0.1:8000/dashboard", timeout=5)
        if resp.status_code == 200:
            print(f"   ✅ Dashboard loaded ({len(resp.text)} bytes)")
        else:
            print(f"   ❌ Failed: {resp.status_code}")
            
        print("\n✅ All tests passed!")
        print("\n🌐 Dashboard available at: http://127.0.0.1:8000/dashboard")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        print("\n🛑 Stopping server...")
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    main()
