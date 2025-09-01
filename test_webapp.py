#!/usr/bin/env python3
"""
Simple test script to verify the FastAPI web application is working correctly
"""

import requests
import json
import time

def test_fastapi_webapp():
    """Test the FastAPI web application"""
    base_url = "http://127.0.0.1:5000"
    
    print("Testing Real Estate Scraper FastAPI Web App...")
    print("=" * 55)
    
    # Test 1: Check if main page loads
    try:
        response = requests.get(base_url, timeout=10)
        if response.status_code == 200:
            print("✅ Main page loads successfully")
        else:
            print(f"❌ Main page failed with status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to connect to web app: {e}")
        return False
    
    # Test 2: Check if API docs are available
    try:
        response = requests.get(f"{base_url}/docs", timeout=10)
        if response.status_code == 200:
            print("✅ API documentation is available at /docs")
        else:
            print(f"⚠️  API docs not accessible (status {response.status_code})")
    except requests.exceptions.RequestException as e:
        print(f"⚠️  API docs not accessible: {e}")
    
    # Test 3: Test API endpoint with Menzili data
    menzili_url = "https://www.menzili.tn/annonce/belle-villa-avec-piscine-privEe-en-zone-touristique-djerba-medenine-djerbamidoun-142024"
    
    try:
        api_data = {
            "provider": "menzili",
            "url": menzili_url
        }
        
        print("🔄 Testing Menzili API scraping endpoint...")
        response = requests.post(
            f"{base_url}/api/scrape",
            json=api_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                print("✅ Menzili API scraping works successfully")
                data = result.get("data", {})
                print(f"   - Property type: {data.get('property_type', 'N/A')}")
                print(f"   - Price: {data.get('price', 'N/A')} {data.get('currency', '')}")
                print(f"   - Location: {data.get('address', 'N/A')}")
                print(f"   - Rooms: {data.get('room_count', 'N/A')}")
                print(f"   - Photos: {len(data.get('photos', []))} found")
            else:
                print(f"❌ Menzili API returned error: {result}")
                return False
        else:
            print(f"❌ Menzili API failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Menzili API test failed: {e}")
        return False
    
    # Test 4: Test Any Website scraper
    try:
        api_data = {
            "provider": "any_website",
            "url": "https://example.com/property"
        }
        
        print("🔄 Testing Any Website API scraping endpoint...")
        response = requests.post(
            f"{base_url}/api/scrape",
            json=api_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                print("✅ Any Website API scraping works successfully")
                data = result.get("data", {})
                print(f"   - Property type: {data.get('property_type', 'N/A')}")
                print(f"   - Price: {data.get('price', 'N/A')} {data.get('currency', '')}")
                print(f"   - Source: {data.get('source', 'N/A')}")
            else:
                print(f"❌ Any Website API returned error: {result}")
                return False
        else:
            print(f"❌ Any Website API failed with status {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Any Website API test failed: {e}")
        return False
    
    print("\n🎉 All tests passed! FastAPI web application is working correctly.")
    print(f"\n🌐 Web Interface: {base_url}")
    print(f"📚 API Documentation: {base_url}/docs")
    print(f"📖 Alternative API Docs: {base_url}/redoc")
    return True

if __name__ == "__main__":
    # Give the FastAPI app a moment to start if just launched
    print("Waiting for FastAPI app to start...")
    time.sleep(3)
    test_fastapi_webapp()
