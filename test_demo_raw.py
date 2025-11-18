#!/usr/bin/env python3
"""
Test Binance Demo Trading API with raw HTTP requests
"""
import os
import time
import hmac
import hashlib
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("BINANCE_TESTNET_API_KEY")
API_SECRET = os.getenv("BINANCE_TESTNET_API_SECRET")
BASE_URL = "https://demo-api.binance.com"

print("🔄 Testing Binance Demo Trading API with raw requests...")
print(f"API Key: {API_KEY[:10]}...{API_KEY[-10:]}")
print(f"Base URL: {BASE_URL}\n")

# Test 1: Public endpoint (no auth needed)
print("📊 Test 1: Public endpoint (server time)")
try:
    response = requests.get(f"{BASE_URL}/api/v3/time")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")
except Exception as e:
    print(f"❌ Error: {e}\n")

# Test 2: Account endpoint (requires auth)
print("📊 Test 2: Account info (authenticated)")
try:
    timestamp = int(time.time() * 1000)
    params = f"timestamp={timestamp}"
    signature = hmac.new(
        API_SECRET.encode('utf-8'),
        params.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    url = f"{BASE_URL}/api/v3/account?{params}&signature={signature}"
    headers = {
        "X-MBX-APIKEY": API_KEY
    }
    
    response = requests.get(url, headers=headers)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")
    
    if response.status_code == 200:
        print("✅ SUCCESS! Demo Trading API is working!")
    else:
        print("❌ Authentication failed")
        
except Exception as e:
    print(f"❌ Error: {e}\n")

# Test 3: Market data endpoint
print("📊 Test 3: Market data (BTCUSDT ticker)")
try:
    response = requests.get(f"{BASE_URL}/api/v3/ticker/24hr?symbol=BTCUSDT")
    print(f"Status: {response.status_code}")
    data = response.json()
    if response.status_code == 200:
        print(f"BTC Price: ${float(data['lastPrice']):,.2f}")
        print("✅ Market data working!")
    else:
        print(f"Response: {data}")
except Exception as e:
    print(f"❌ Error: {e}")
